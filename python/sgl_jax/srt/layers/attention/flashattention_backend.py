import logging
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from jax.tree_util import register_pytree_node_class

from sgl_jax.srt.kernels.ragged_paged_attention.ragged_paged_attention import (
    ragged_paged_attention,
)
from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.speculative.eagle_util import EagleDraftInput
from sgl_jax.srt.utils import cdiv
from sgl_jax.srt.utils.jax_utils import device_array
from sgl_jax.srt.utils.profiling_utils import named_scope

logger = logging.getLogger(__name__)

_LOCAL_WINDOW_ATTN_KERNELS = None
_LOCAL_WINDOW_IMPORT_ERROR = None


def _get_local_window_attn_kernels():
    global _LOCAL_WINDOW_ATTN_KERNELS, _LOCAL_WINDOW_IMPORT_ERROR
    if _LOCAL_WINDOW_ATTN_KERNELS is not None:
        return _LOCAL_WINDOW_ATTN_KERNELS
    if _LOCAL_WINDOW_IMPORT_ERROR is not None:
        return None

    try:
        from pattern.local_window_attention import (
            batched_local_window_flash_attention,
            batched_local_window_flash_attention_trainable,
            local_window_attention_baseline,
        )

        _LOCAL_WINDOW_ATTN_KERNELS = {
            "baseline": local_window_attention_baseline,
            "pallas": batched_local_window_flash_attention,
            "trainable": batched_local_window_flash_attention_trainable,
        }
        logger.info(
            "Enabled local window attention override from thesis2026 for selected layer(s) in FlashAttention."
        )
        return _LOCAL_WINDOW_ATTN_KERNELS
    except Exception as e:  # pragma: no cover - runtime environment dependent
        _LOCAL_WINDOW_IMPORT_ERROR = e
        logger.warning(
            "Failed to import local window attention kernels from pattern.local_window_attention: %s",
            e,
        )
        return None


def _resolve_local_window_impl() -> str:
    impl = os.getenv("SGL_JAX_LOCAL_WINDOW_IMPL", "pallas").strip().lower()
    if impl in ("baseline", "pallas", "trainable"):
        return impl
    logger.warning("Unknown SGL_JAX_LOCAL_WINDOW_IMPL=%s, falling back to pallas.", impl)
    return "pallas"


def _resolve_effective_batch(batch_tot: int, requested: int) -> int:
    eff = max(1, min(batch_tot, requested))
    while eff > 1 and (batch_tot % eff) != 0:
        eff -= 1
    return eff


_LOCAL_WINDOW_DEBUG = os.getenv("SGL_JAX_LOCAL_WINDOW_DEBUG", "0") == "1"
_LOCAL_WINDOW_DEBUG_LOGGED_EVENTS: set[tuple[str, int]] = set()
_LOCAL_WINDOW_DEBUG_SKIP_LOGGED: set[tuple[int, str, str]] = set()


def _local_window_debug(
    event: str,
    layer_id: int,
    mode: ForwardMode,
    impl: str,
    details: str = "",
) -> None:
    if not _LOCAL_WINDOW_DEBUG:
        return

    if event in ("attempt", "success"):
        key = (event, layer_id)
        if key in _LOCAL_WINDOW_DEBUG_LOGGED_EVENTS:
            return
        _LOCAL_WINDOW_DEBUG_LOGGED_EVENTS.add(key)

    if event == "skip":
        key = (layer_id, str(mode), details)
        if key in _LOCAL_WINDOW_DEBUG_SKIP_LOGGED:
            return
        _LOCAL_WINDOW_DEBUG_SKIP_LOGGED.add(key)

    suffix = f" details={details}" if details else ""
    logger.warning(
        "[LOCAL_WINDOW_DEBUG] event=%s layer_id=%s mode=%s impl=%s%s",
        event,
        layer_id,
        mode,
        impl,
        suffix,
    )


def _should_try_local_window(layer_id: int, mode: ForwardMode) -> bool:
    enabled = os.getenv("SGL_JAX_ENABLE_LOCAL_WINDOW_FIRST_LAYER", "0") == "1"
    if not enabled:
        return False

    target_layer = int(os.getenv("SGL_JAX_LOCAL_WINDOW_LAYER_ID", "0"))
    should = layer_id == target_layer and mode == ForwardMode.EXTEND

    if _LOCAL_WINDOW_DEBUG and layer_id == target_layer and not should:
        _local_window_debug(
            "skip",
            layer_id,
            mode,
            _resolve_local_window_impl(),
            "reason=mode_not_extend" if mode != ForwardMode.EXTEND else "reason=layer_not_target",
        )

    return should


@register_pytree_node_class
@dataclass
class FlashAttentionMetadata:
    """Metadata to be init once in the model forward pass,
    each layer's forward pass can reuse the metadata.

    For each init metadata function, we will try set up them in below order
    """

    num_seqs: jax.Array = None
    cu_q_lens: jax.Array = None
    cu_kv_lens: jax.Array = None
    page_indices: jax.Array = None
    seq_lens: jax.Array = None
    distribution: jax.Array = None
    custom_mask: jax.Array = None

    def tree_flatten(self):
        children = (
            self.num_seqs,
            self.cu_q_lens,
            self.cu_kv_lens,
            self.page_indices,
            self.seq_lens,
            self.distribution,
            self.custom_mask,
        )

        aux_data = {}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls.__new__(cls)

        obj.num_seqs = children[0]
        obj.cu_q_lens = children[1]
        obj.cu_kv_lens = children[2]
        obj.page_indices = children[3]
        obj.seq_lens = children[4]
        obj.distribution = children[5]
        obj.custom_mask = children[6]

        return obj


@dataclass
class FlashAttention(AttentionBackend):
    """Native Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
        head_dim,
        vmem_limit_bytes: int = 64 * (1 << 20),  # 64MB
        page_size: int = 1,
        kv_partition_axis: str = "tensor",
        mesh: jax.sharding.Mesh = None,
    ):
        self.vmem_limit_bytes = vmem_limit_bytes
        self.num_heads = num_attn_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_attn_heads
        self.head_dim = head_dim
        self.page_size = page_size
        self.kv_partition_axis = kv_partition_axis
        self.forward_metadata = nnx.data(FlashAttentionMetadata())
        self.mesh = mesh

    def get_forward_metadata(
        self,
        batch: ModelWorkerBatch,
    ):
        """Return the metadata for a forward pass."""
        metadata = FlashAttentionMetadata()

        indices = np.arange(0, len(batch.cache_loc), self.page_size)
        selected_cache_locs = batch.cache_loc[indices]
        page_indices = (selected_cache_locs // self.page_size).astype(np.int32)

        if batch.forward_mode == ForwardMode.EXTEND:
            cu_q_lens = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(batch.extend_seq_lens, dtype=np.int32),
                ]
            )
        elif batch.forward_mode == ForwardMode.DECODE:
            cu_q_lens = np.concatenate(
                [
                    np.array([0], dtype=np.int32),
                    np.cumsum(np.ones(len(batch.seq_lens), dtype=np.int32)),
                ]
            )
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        seq_lens = np.copy(batch.seq_lens)

        aligned_seq_lens = (
            (batch.seq_lens + self.page_size - 1) // self.page_size
        ) * self.page_size
        cu_kv_lens = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(aligned_seq_lens),
            ]
        )

        num_seqs = np.sum(batch.seq_lens > 0, dtype=np.int32).reshape(
            1,
        )

        # Construct distribution for V2 kernel: [decode_end, prefill_end, mixed_end]
        if batch.forward_mode == ForwardMode.DECODE:
            # All sequences are decode/mixed mode
            distribution = np.array([0, 0, num_seqs.item()], dtype=np.int32)
        elif batch.forward_mode == ForwardMode.EXTEND:
            # All sequences are prefill mode
            distribution = np.array([0, num_seqs.item(), num_seqs.item()], dtype=np.int32)
        else:
            raise ValueError(f"Invalid forward mode: {batch.forward_mode}")

        (
            metadata.num_seqs,
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (num_seqs, cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
        )
        return metadata

    def get_eagle_forward_metadata(self, batch: ModelWorkerBatch):
        """Return the metadata for a forward pass."""
        # below code is for verify and draft extend phase
        metadata = FlashAttentionMetadata()
        indices = np.arange(0, len(batch.cache_loc), self.page_size)
        selected_cache_locs = batch.cache_loc[indices]
        page_indices = (selected_cache_locs // self.page_size).astype(np.int32)

        if batch.forward_mode == ForwardMode.TARGET_VERIFY:
            # convert custom_mask from bool to int32, because dma not support bool type
            if batch.spec_info.custom_mask.dtype == jnp.bool:
                # FIXME(pc) rm this dtype convert
                logger.warning(
                    "batch.spec_info.custom_mask type is  %s, it may make performance very low",
                    batch.spec_info.custom_mask.dtype,
                )
                metadata.custom_mask = batch.spec_info.custom_mask.astype(jnp.int32)
            else:
                metadata.custom_mask = batch.spec_info.custom_mask
        else:
            metadata.custom_mask = None

        if batch.forward_mode.is_target_verify():
            padded_batch_size = len(batch.seq_lens)
            real_batch_size = batch.real_bs
            q_lens = np.array([batch.spec_info.draft_token_num] * real_batch_size, dtype=np.int32)
            extend_seq_lens = np.pad(q_lens, (0, padded_batch_size - real_batch_size))
        else:
            extend_seq_lens = batch.extend_seq_lens
        cu_q_lens = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(extend_seq_lens),
            ]
        )

        seq_lens = np.copy(batch.seq_lens)

        if batch.forward_mode.is_target_verify():
            seq_lens += extend_seq_lens
            aligned_seq_lens = ((seq_lens + self.page_size - 1) // self.page_size) * self.page_size
        else:
            aligned_seq_lens = (
                (batch.seq_lens + self.page_size - 1) // self.page_size
            ) * self.page_size
        cu_kv_lens = np.concatenate(
            [
                np.array([0], dtype=np.int32),
                np.cumsum(aligned_seq_lens),
            ]
        )

        if batch.forward_mode == ForwardMode.DRAFT_EXTEND:
            # Reconstruct page_indices properly respecting ragged allocation
            page_indices_list = []
            offset = 0
            allocate_lens = batch.spec_info.allocate_lens
            # Ensure it's accessible as array
            if hasattr(allocate_lens, "device"):
                allocate_lens = jax.device_get(allocate_lens)

            num_pages_per_seq = aligned_seq_lens // self.page_size

            for i in range(batch.real_bs):
                alloc_len = (
                    (int(allocate_lens[i]) + self.page_size - 1) // self.page_size
                ) * self.page_size
                needed_pages = int(num_pages_per_seq[i])

                if needed_pages > 0:
                    # Get the slice of cache_loc for this request
                    # We assume batch.cache_loc is ordered and packed according to allocate_lens
                    req_cache_loc = batch.cache_loc[offset : offset + alloc_len]

                    # Select the first token of each page
                    # The tokens are at indices 0, page_size, 2*page_size...
                    # We need `needed_pages` entries.

                    indices = np.arange(needed_pages) * self.page_size
                    selected = req_cache_loc[indices]
                    page_indices_list.extend(selected // self.page_size)

                offset += alloc_len

            page_indices = np.pad(
                np.array(page_indices_list, dtype=np.int32),
                (0, page_indices.shape[0] - len(page_indices_list)),
            )

        num_seqs = np.sum(batch.seq_lens > 0, dtype=np.int32).reshape(
            1,
        )
        # Construct distribution for V2 kernel: [decode_end, prefill_end, mixed_end]

        # All sequences are prefill mode
        distribution = np.array([0, num_seqs.item(), num_seqs.item()], dtype=np.int32)

        num_seqs = np.array(num_seqs)
        cu_q_lens = np.array(cu_q_lens)
        cu_kv_lens = np.array(cu_kv_lens)
        page_indices = np.array(page_indices)
        seq_lens = np.array(seq_lens)
        (
            metadata.num_seqs,
            metadata.cu_q_lens,
            metadata.cu_kv_lens,
            metadata.page_indices,
            metadata.seq_lens,
            metadata.distribution,
        ) = device_array(
            (num_seqs, cu_q_lens, cu_kv_lens, page_indices, seq_lens, distribution),
            sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
        )
        return metadata

    def get_eagle_multi_step_metadata(self, batch: ModelWorkerBatch):

        indices = np.arange(0, len(batch.cache_loc), self.page_size)
        # NOTE: Use original_selected_cache_locs as the source of truth for all steps
        # to avoid the bug where selected_cache_locs is overwritten by truncated data in loops.
        original_selected_cache_locs = batch.cache_loc[indices]
        assert batch.forward_mode is ForwardMode.DECODE

        page_indices = []
        cu_kv_lens = []
        seq_lens = np.copy(batch.seq_lens)

        # Vectorized preparation
        real_bs = batch.real_bs
        current_seq_lens = batch.seq_lens[:real_bs]
        allocate_lens = batch.spec_info.allocate_lens[:real_bs]

        draft_allocs = allocate_lens - current_seq_lens

        alloc_tokens = current_seq_lens + draft_allocs
        alloc_pages = cdiv(alloc_tokens, self.page_size)

        # src_starts (offset2) is constant across steps
        src_starts = np.concatenate(([0], np.cumsum(alloc_pages)[:-1]))

        full_size = len(original_selected_cache_locs)
        seq_lens_list = []
        for speculative_step_id in range(batch.speculative_num_steps):
            seq_lens = batch.seq_lens + (speculative_step_id)
            seq_lens[batch.real_bs :] = 0
            seq_lens_list.append(seq_lens)
            aligned_seq_lens = ((seq_lens + self.page_size - 1) // self.page_size) * self.page_size
            cu_kv_lens.append(
                np.concatenate(
                    [
                        np.array([0], dtype=np.int32),
                        np.cumsum(aligned_seq_lens),
                    ]
                )
            )

            # Vectorized calculation of spec_pages
            step_spec_tokens = (
                current_seq_lens + (speculative_step_id) * batch.speculative_eagle_topk
            )
            step_spec_pages = cdiv(step_spec_tokens, self.page_size)

            total_spec_pages = np.sum(step_spec_pages)
            dst_starts = np.concatenate(([0], np.cumsum(step_spec_pages)[:-1]))

            # Vectorized Gather
            repeats = step_spec_pages
            gather_indices = np.repeat(src_starts, repeats) + (
                np.arange(total_spec_pages) - np.repeat(dst_starts, repeats)
            )

            gathered_locs = original_selected_cache_locs[gather_indices]

            # Reconstruct the full array (sparse/padded)
            result_locs = np.zeros(full_size, dtype=original_selected_cache_locs.dtype)
            result_locs[:total_spec_pages] = gathered_locs

            page_indices_cur_step = (result_locs // self.page_size).astype(np.int32)

            # FIXME Handle padding, this will be move to precompile
            TARGET_PADDING = 16384
            if page_indices_cur_step.shape[0] < TARGET_PADDING:
                padding_size = TARGET_PADDING - page_indices_cur_step.shape[0]
                # Use np.pad to keep it on CPU/Numpy until device_array call
                page_indices_cur_step = np.pad(page_indices_cur_step, (0, padding_size))

            page_indices.append(page_indices_cur_step)

        if batch.spec_algorithm.is_none():
            raise RuntimeError("should not reach here")
        else:
            assert isinstance(batch.spec_info, EagleDraftInput)
            # it is same across every step
            cu_q_lens = np.arange(
                0,
                len(batch.seq_lens) * batch.speculative_eagle_topk + 1,
                step=batch.speculative_eagle_topk,
                dtype=np.int32,
            )
        num_seqs = np.sum(batch.seq_lens > 0, dtype=np.int32).reshape(
            1,
        )

        distribution = np.array([0, 0, num_seqs.item()], dtype=np.int32)
        metadata = []
        for i in range(batch.speculative_num_steps):
            metadata_tmp = FlashAttentionMetadata()
            (
                metadata_tmp.num_seqs,
                metadata_tmp.cu_q_lens,
                metadata_tmp.cu_kv_lens,
                metadata_tmp.page_indices,
                metadata_tmp.seq_lens,
                metadata_tmp.distribution,
            ) = device_array(
                (
                    num_seqs,
                    cu_q_lens,
                    cu_kv_lens[i],
                    page_indices[i],
                    seq_lens_list[i],
                    distribution,
                ),
                sharding=(NamedSharding(self.mesh, P()) if jax.process_count() == 1 else None),
            )
            metadata.append(metadata_tmp)
        return metadata

    def tree_flatten(self):
        children = (self.forward_metadata,)
        aux_data = {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "vmem_limit_bytes": self.vmem_limit_bytes,
            "head_dim": self.head_dim,
            "page_size": self.page_size,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        obj = cls(
            aux_data["num_heads"],
            aux_data["num_kv_heads"],
            aux_data["head_dim"],
            aux_data["vmem_limit_bytes"],
            aux_data["page_size"],
        )

        obj.forward_metadata = children[0]

        return obj

    @named_scope
    def __call__(
        self,
        q: jax.Array,  # [total_tokens, num_heads, head_dim]
        k: jax.Array,  # [total_tokens, num_heads, head_dim]
        v: jax.Array,  # [total_tokens, num_heads, head_dim]
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        causal: int = 1,
    ):
        """
        Args:
            q, k, v: Input tensors of shape [total_tokens, num_heads, head_dim]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            attention_mask: Optional attention mask
            is_causal: Whether to apply causal masking
        Returns:
            Output tensor of shape [total_tokens, hidden_size]
        """
        if forward_batch is not None and token_to_kv_pool is not None:
            kv_cache_fused = self._get_fused_kv_cache(
                forward_batch, token_to_kv_pool, layer.layer_id
            )
        else:
            kv_cache_fused = jnp.zeros((0, self.num_kv_heads * 2, self.head_dim), dtype=q.dtype)
        scale = (
            1.0 / jnp.sqrt(layer.head_dim)
            if (layer is None or layer.scaling is None)
            else layer.scaling
        )

        # Prepare fused KV cache for paged format: [num_pages, page_size, num_kv_heads * 2, head_dim]
        total_tokens = kv_cache_fused.shape[0]
        num_pages = total_tokens // self.page_size
        kv_cache_fused_paged = kv_cache_fused.reshape(
            num_pages, self.page_size, -1, (self.head_dim + 127) // 128 * 128
        )
        if self.forward_metadata.custom_mask is not None:
            causal = 0
        # Select page indices and remap to SWA pool if KV cache supports it
        page_indices_arg = self.forward_metadata.page_indices
        if hasattr(token_to_kv_pool, "remap_cache_loc") and self.page_size == 1:
            page_indices_arg = token_to_kv_pool.remap_cache_loc(page_indices_arg, layer.layer_id)

        in_specs = (
            P(None, self.kv_partition_axis),  # queries
            P(None, self.kv_partition_axis),  # keys (new tokens)
            P(None, self.kv_partition_axis),  # values (new tokens)
            P(None, None, self.kv_partition_axis, None),  # kv_cache_fused (head interleaved)
            P(),  # kv_lens
            P(),  # page_indices
            P(),  # cu_q_lens
            P(),  # cu_kv_lens
            P(),  # distribution
            P(),  # custom_mask
        )
        out_specs = (
            P(None, self.kv_partition_axis),  # attention output
            P(
                None, self.kv_partition_axis, None
            ),  # updated kv_cache_fused (head interleaved) - 3D: [total_tokens, num_kv_heads*2, head_dim]
        )

        def _ragged_paged_attention_with_fused_kv(*args):
            queries, keys, values, kv_cache_fused = args[:4]
            other_args = args[4:]

            # Call fused KV kernel with head interleaving
            result, updated_kv_cache_fused = ragged_paged_attention(
                queries,
                keys,
                values,
                kv_cache_fused,
                *other_args,
                causal=causal,
                sm_scale=scale,
                sliding_window=layer.sliding_window_size,
                soft_cap=layer.logit_cap,
                xai_temperature_len=(
                    layer.xai_temperature_len if layer.xai_temperature_len > 0 else None
                ),
                vmem_limit_bytes=self.vmem_limit_bytes,
            )

            return result, updated_kv_cache_fused

        (
            attn_output,
            updated_kv_cache_fused,
        ) = jax.shard_map(  # Fused KV kernel handles cache updates internally
            _ragged_paged_attention_with_fused_kv,
            in_specs=in_specs,
            out_specs=out_specs,
            check_vma=False,
        )(
            q.reshape(q.shape[0], -1, self.head_dim),
            k.reshape(k.shape[0], -1, self.head_dim),
            v.reshape(v.shape[0], -1, self.head_dim),
            kv_cache_fused_paged,
            self.forward_metadata.seq_lens,
            page_indices_arg,
            self.forward_metadata.cu_q_lens,
            self.forward_metadata.cu_kv_lens,
            self.forward_metadata.distribution,
            self.forward_metadata.custom_mask,
        )
        if _should_try_local_window(layer.layer_id, forward_batch.forward_mode):
            local_window_kernels = _get_local_window_attn_kernels()
            local_window_impl = _resolve_local_window_impl()

            if local_window_kernels is not None:
                if forward_batch.batch_size != 1:
                    _local_window_debug(
                        "skip",
                        layer.layer_id,
                        forward_batch.forward_mode,
                        local_window_impl,
                        "reason=batch_size_not_1",
                    )
                else:
                    local_window_size = int(os.getenv("SGL_JAX_LOCAL_WINDOW_SIZE", "128"))
                    local_window_block_size = int(os.getenv("SGL_JAX_LOCAL_WINDOW_BLOCK_SIZE", "16"))
                    requested_effective_batch = int(
                        os.getenv("SGL_JAX_LOCAL_WINDOW_EFFECTIVE_BATCH", "8")
                    )

                    q_heads = q.reshape(q.shape[0], -1, self.head_dim)
                    k_heads = k.reshape(k.shape[0], -1, self.head_dim)
                    v_heads = v.reshape(v.shape[0], -1, self.head_dim)

                    q_t = jnp.transpose(q_heads, (1, 0, 2))
                    k_t = jnp.transpose(k_heads, (1, 0, 2))
                    v_t = jnp.transpose(v_heads, (1, 0, 2))

                    _local_window_debug(
                        "attempt",
                        layer.layer_id,
                        forward_batch.forward_mode,
                        local_window_impl,
                        f"window={local_window_size},block={local_window_block_size},q_len={q_t.shape[1]},batch={q_t.shape[0]},backend=fa",
                    )

                    q_len = q_t.shape[1]
                    k_local = k_t[:, :q_len, :]
                    v_local = v_t[:, :q_len, :]

                    try:
                        if local_window_impl == "baseline":
                            local_out = jax.vmap(
                                local_window_kernels["baseline"],
                                in_axes=(0, 0, 0, None),
                            )(q_t, k_local, v_local, local_window_size)
                        else:
                            if local_window_size % local_window_block_size != 0:
                                raise ValueError(
                                    f"window_size ({local_window_size}) must be divisible by block_size ({local_window_block_size})"
                                )

                            pad_tokens = (-q_len) % local_window_block_size
                            if pad_tokens:
                                pad_cfg = ((0, 0), (0, pad_tokens), (0, 0))
                                q_local = jnp.pad(q_t, pad_cfg)
                                k_local = jnp.pad(k_local, pad_cfg)
                                v_local = jnp.pad(v_local, pad_cfg)
                            else:
                                q_local = q_t

                            eff = _resolve_effective_batch(q_local.shape[0], requested_effective_batch)
                            if local_window_impl == "trainable":
                                local_out = local_window_kernels["trainable"](
                                    q_local,
                                    k_local,
                                    v_local,
                                    local_window_size,
                                    local_window_block_size,
                                    eff,
                                )
                            else:
                                local_out = local_window_kernels["pallas"](
                                    q_local,
                                    k_local,
                                    v_local,
                                    local_window_size,
                                    local_window_block_size,
                                    eff,
                                )

                            local_out = local_out[:, :q_len, :]

                        local_out = jnp.transpose(local_out, (1, 0, 2))
                        local_out = jnp.where(jnp.isfinite(local_out), local_out, attn_output)
                        attn_output = local_out
                        _local_window_debug(
                            "success",
                            layer.layer_id,
                            forward_batch.forward_mode,
                            local_window_impl,
                            f"q_len={q_len},backend=fa",
                        )
                    except Exception as e:
                        _local_window_debug(
                            "fallback",
                            layer.layer_id,
                            forward_batch.forward_mode,
                            local_window_impl,
                            repr(e),
                        )
                        logger.warning(
                            "FlashAttention local-window %s kernel failed for layer %s, keeping ragged attention output: %s",
                            local_window_impl,
                            layer.layer_id,
                            e,
                        )

        pad_width = (self.head_dim + 127) // 128 * 128 - self.head_dim
        if pad_width > 0:
            updated_kv_cache_fused = jnp.pad(
                updated_kv_cache_fused,
                ((0, 0), (0, 0), (0, pad_width)),
                mode="constant",
            )

        return (
            attn_output.reshape(q.shape[0], -1),
            updated_kv_cache_fused,
        )

    def _get_fused_kv_cache(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        layer_id: int,
    ) -> jax.Array:
        return token_to_kv_pool.get_fused_kv_buffer(layer_id)

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        num_page_per_req = cdiv(max_context_len, page_size)
        res = 1024 * 1024 // 2 // num_page_per_req // 4
        assert (
            res > 0
        ), f"max running requests: {res} must larger than 0, please increase page size or decrease max context length"
        return res
