import logging
import os

import jax
import jax.numpy as jnp
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.layers.attention.base_attn_backend import AttentionBackend
from sgl_jax.srt.layers.radix_attention import AttentionType, RadixAttention
from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sgl_jax.srt.utils.jax_utils import is_tpu_runtime
from sgl_jax.srt.utils.profiling_utils import named_scope


logger = logging.getLogger(__name__)

_LOCAL_WINDOW_ATTN_KERNELS = None
_LOCAL_WINDOW_IMPORT_ERROR = None


def _get_local_window_attn_kernels():
    """Lazy import of thesis2026 local window attention implementations."""
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
            "Enabled local window attention override from thesis2026 for selected layer(s)."
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
    logger.warning(
        "Unknown SGL_JAX_LOCAL_WINDOW_IMPL=%s, falling back to pallas.",
        impl,
    )
    return "pallas"


def _resolve_effective_batch(batch_tot: int, requested: int) -> int:
    eff = max(1, min(batch_tot, requested))
    while eff > 1 and (batch_tot % eff) != 0:
        eff -= 1
    return eff


def _should_try_local_window(layer_id: int, mode: ForwardMode) -> bool:
    if os.getenv("SGL_JAX_ENABLE_LOCAL_WINDOW_FIRST_LAYER", "0") != "1":
        return False

    target_layer = int(os.getenv("SGL_JAX_LOCAL_WINDOW_LAYER_ID", "0"))
    return layer_id == target_layer and mode == ForwardMode.EXTEND


class NativeAttention(AttentionBackend):
    """Native Attention layer for variable-length sequences using ForwardBatch."""

    def __init__(
        self,
        num_attn_heads,
        num_kv_heads,
        mesh,
    ):
        self.num_heads = num_attn_heads
        if num_kv_heads is not None:
            self.num_kv_heads = num_kv_heads
        else:
            self.num_kv_heads = num_attn_heads
        self.mesh = mesh
        self.kv_sharding = NamedSharding(self.mesh, P(None, "tensor", None))

    def tree_flatten(self):
        children = ()
        aux_data = {
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "mesh": self.mesh,
        }
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(num_attn_heads=aux_data["num_heads"], num_kv_heads=aux_data["num_kv_heads"])

    def get_forward_metadata(self, batch: ModelWorkerBatch):
        """Init the metadata for a forward pass and return it."""
        return None

    @named_scope
    def __call__(
        self,
        q: jax.Array,
        k: jax.Array,
        v: jax.Array,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
    ):
        """
        Args:
            q, k, v: Input tensors of shape [total_tokens, hidden_size]
            forward_batch: ForwardBatch object containing seq_lens and batch_size
            is_causal: Whether to apply causal masking
        Returns:
            Tuple of (output tensor of shape [total_tokens, hidden_size], k, v)
        """
        # TODO(pc) support tree based native attention backend
        k_buffer, v_buffer, kv_fused = self._get_and_update_kv_cache(
            k, v, forward_batch, token_to_kv_pool, self.kv_sharding, layer.layer_id
        )

        scale = 1.0 / jnp.sqrt(layer.head_dim) if layer.scaling is None else layer.scaling

        is_causal = True
        if (
            forward_batch.forward_mode == ForwardMode.DECODE
            or layer.attn_type == AttentionType.ENCODER_ONLY
        ):
            is_causal = False

        # Get xai_temperature_len from the layer if it exists and pass it down.
        xai_temp_len = getattr(layer, "xai_temperature_len", None)

        attn_output = forward_attention(
            layer.layer_id,
            q,
            k_buffer,
            v_buffer,
            forward_batch.seq_lens,
            forward_batch.cache_loc,
            forward_batch.extend_prefix_lens,
            forward_batch.extend_seq_lens,
            layer.q_head_num,
            layer.kv_head_num,
            scale,
            is_causal,
            forward_batch.forward_mode,
            self.kv_sharding,
            xai_temperature_len=xai_temp_len,
        )

        # Return full fused KV buffer for this layer so that caller can persist it outside JIT
        return attn_output, kv_fused

    def _get_and_update_kv_cache(
        self,
        k: jax.Array,
        v: jax.Array,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        kv_sharding: jax.NamedSharding,
        layer_id: int,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        """
        Get the kv cache from the forward batch.
        """
        if is_tpu_runtime():
            if forward_batch.forward_mode.is_extend():
                token_to_kv_pool.set_kv_buffer(
                    layer_id, forward_batch.out_cache_loc, k, v, is_decode=False
                )
            else:
                token_to_kv_pool.set_kv_buffer(
                    layer_id, forward_batch.out_cache_loc, k, v, is_decode=True
                )
            # Use fused layer directly from pool; derive K/V views without extra merge
            fused_layer = token_to_kv_pool.get_fused_kv_buffer(layer_id)
            k = fused_layer.at[:, ::2, :].get(out_sharding=kv_sharding)
            v = fused_layer.at[:, 1::2, :].get(out_sharding=kv_sharding)
            fused_return = fused_layer
        else:
            updated_layer = token_to_kv_pool.set_kv_buffer_legacy(
                layer_id, forward_batch.out_cache_loc, k, v
            )
            # Functional style: treat updated_layer as authoritative fused buffer for this layer in this step
            # Derive K/V views for attention computation from fused buffer directly
            k = updated_layer.at[:, ::2, :].get(out_sharding=kv_sharding)
            v = updated_layer.at[:, 1::2, :].get(out_sharding=kv_sharding)
            # Return fused buffer directly for persistence outside JIT
            fused_return = updated_layer
        return k, v, fused_return

    @staticmethod
    def get_max_running_reqests(max_context_len: int, page_size: int) -> int:
        # native attention backend do not care the max running requests
        return 4096


# @partial(jax.jit, static_argnames=["num_heads", "num_kv_heads", "is_causal", "mode"])
def forward_attention(
    layer_id: int,
    q: jax.Array,
    k_cache: jax.Array,
    v_cache: jax.Array,
    seq_lengths: jax.Array,
    loc: jax.Array,
    extend_prefix_lens: jax.Array,
    extend_seq_lens: jax.Array,
    num_heads,
    num_kv_heads,
    scale=None,
    is_causal=True,
    mode=ForwardMode.DECODE,
    kv_sharding=None,
    xai_temperature_len: float | None = None,
):
    """
    Forward pass using native JAX implementation with block-diagonal attention.
    This avoids padding while maintaining efficient matrix operations.

    Args:
        q: input token in decode mode, shape(batch_size, hidden_size), each batch has one token
        k_cache: prefix cache of key, shape(seq_len, hidden_size)
        v_cache: prefix cache of value, shape(seq_len, hidden_size)
        seq_lengths: sequence lengths of each batch
        loc: location of the key/value cache
        extend_prefix_lens: prefix lengths of each batch in extend mode
        extend_seq_lens: sequence lengths of each batch in extend mode
        num_heads: number of query heads
        num_kv_heads: number of key/value heads
        scale: scale for the attention weights
        seq_mask: boolean mask of shape [batch_size, total_prefix_len]
        xai_temperature_len: length of the xai temperature

    Returns:
        Output tensor of shape[batch_size, hidden_size]
    """

    cache_size = k_cache.shape[0]
    safe_loc = jnp.where(loc > 0, loc, cache_size)
    k_cache = k_cache.at[safe_loc].get(out_sharding=kv_sharding, mode="fill", fill_value=0)
    v_cache = v_cache.at[safe_loc].get(out_sharding=kv_sharding, mode="fill", fill_value=0)

    # Handle both 2D and 3D input formats for q
    if len(q.shape) == 2:
        # Traditional format: [num_tokens, hidden_size]
        num_tokens, hidden_size = q.shape
        head_dim = hidden_size // num_heads
        q_heads = q.reshape(num_tokens, num_heads, head_dim)
    else:
        # Already in multi-head format: [num_tokens, num_heads, head_dim]
        num_tokens, num_heads_input, head_dim = q.shape
        assert num_heads_input == num_heads, f"Expected {num_heads} heads, got {num_heads_input}"
        hidden_size = num_heads * head_dim  # Calculate hidden_size for proper reshaping
        q_heads = q

    # KV cache from get_kv_buffer is already in multi-head format: [cache_size, num_kv_heads, head_dim]
    k_heads = k_cache
    v_heads = v_cache

    # Transpose for efficient matrix operations
    # q: shape of (num_heads, num_tokens, head_dim)
    # k, v: shape of (total_prefix_len, num_heads, head_dim)
    if num_kv_heads != num_heads:
        # For GQA attention, we need to copy k and v heads to match the number of query heads
        num_copies = num_heads // num_kv_heads
        # Use repeat to copy k and v heads
        # [total_prefix_len, num_kv_heads, head_dim] -> [total_prefix_len, num_heads, head_dim]
        k_heads = jnp.repeat(k_heads, num_copies, axis=1, out_sharding=kv_sharding)
        v_heads = jnp.repeat(v_heads, num_copies, axis=1, out_sharding=kv_sharding)

    # Transpose for matmul: [num_heads, num_tokens, head_dim]
    q_t = jnp.transpose(q_heads, (1, 0, 2))
    k_t = jnp.transpose(k_heads, (1, 0, 2))
    v_t = jnp.transpose(v_heads, (1, 0, 2))

    if _should_try_local_window(layer_id, mode):
        local_window_kernels = _get_local_window_attn_kernels()
        if local_window_kernels is not None:
            local_window_size = int(os.getenv("SGL_JAX_LOCAL_WINDOW_SIZE", "128"))
            local_window_block_size = int(os.getenv("SGL_JAX_LOCAL_WINDOW_BLOCK_SIZE", "16"))
            requested_effective_batch = int(
                os.getenv("SGL_JAX_LOCAL_WINDOW_EFFECTIVE_BATCH", "8")
            )
            local_window_impl = _resolve_local_window_impl()

            # Keep this path JIT-safe: avoid host-side int conversions on traced values.
            # Apply local window attention over the aligned prefix [0:q_len].
            q_len = q_t.shape[1]
            k_local = k_t[:, :q_len, :]
            v_local = v_t[:, :q_len, :]

            try:
                if local_window_impl == "baseline":
                    local_out = jax.vmap(local_window_kernels["baseline"], in_axes=(0, 0, 0, None))(
                        q_t,
                        k_local,
                        v_local,
                        local_window_size,
                    )
                else:
                    if local_window_size % local_window_block_size != 0:
                        raise ValueError(
                            f"window_size ({local_window_size}) must be divisible by block_size "
                            f"({local_window_block_size})"
                        )

                    # Pallas kernels are block-tiled. Pad token dimension to tile size, then trim back.
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
                return local_out.reshape(num_tokens, hidden_size)
            except Exception as e:
                logger.warning(
                    "Local-window %s kernel failed for layer %s, falling back to native attention: %s",
                    local_window_impl,
                    layer_id,
                    e,
                )

    if scale is None:
        scale = 1.0 / jnp.sqrt(head_dim)
    attn_logits = jnp.einsum("hqd,hkd->hqk", q_t, k_t) * scale
    neg_inf = jnp.asarray(jnp.finfo(attn_logits.dtype).min, attn_logits.dtype)
    is_valid = loc > 0
    attn_logits = jnp.where(is_valid[jnp.newaxis, jnp.newaxis, :], attn_logits, neg_inf)

    # ** NEW: Apply XAI temperature scaling if specified **
    if xai_temperature_len is not None and xai_temperature_len > 0:
        query_len = q_heads.shape[0]

        # Determine the sequence position of each query token
        if mode == ForwardMode.EXTEND:
            q_starts = jnp.cumsum(extend_seq_lens) - extend_seq_lens
            q_batch_indicators = jnp.zeros(query_len, dtype=jnp.int32).at[q_starts].set(1)
            q_batch_ids = jnp.cumsum(q_batch_indicators) - 1
            q_relative_pos = jnp.arange(query_len, dtype=jnp.int32) - q_starts[q_batch_ids]
            q_positions = extend_prefix_lens[q_batch_ids] + q_relative_pos
        else:  # mode == ForwardMode.DECODE
            q_positions = seq_lengths

        # Calculate and apply the scaling factor
        xai_scale = 1.0 / jnp.log2(float(xai_temperature_len))
        log_pos = jnp.log2(jnp.maximum(q_positions.astype(jnp.float32), 1.0))
        temp_factor = log_pos * xai_scale
        regulator = jnp.where(q_positions > xai_temperature_len, temp_factor, 1.0)

        # Broadcast regulator from [num_tokens] to [1, num_tokens, 1] to scale weights
        attn_logits = attn_logits * regulator[None, :, None]

    # Apply appropriate masking
    if mode == ForwardMode.EXTEND:
        attn_logits = _apply_extend_mask(
            attn_logits, seq_lengths, extend_prefix_lens, extend_seq_lens, is_causal
        )
    else:
        attn_logits = _apply_decode_mask(attn_logits, seq_lengths)

    # Softmax
    attn_weights = jax.nn.softmax(attn_logits, axis=-1)

    attn_output = jnp.matmul(attn_weights, v_t)
    attn_output = jnp.transpose(attn_output, (1, 0, 2))
    return attn_output.reshape(num_tokens, hidden_size)


def _apply_extend_mask(
    attn_weights: jax.Array,
    seq_lengths: jax.Array,
    extend_prefix_lens: jax.Array,
    extend_seq_lens: jax.Array,
    is_causal: bool = True,
):
    """
    Applies a block-diagonal and optionally a causal mask in a unified,
    efficient way, correctly handling padding.
    """
    _, query_len, key_len = attn_weights.shape

    # --- Create validity masks to handle padding ---
    q_valid_mask = jnp.arange(query_len) < jnp.sum(extend_seq_lens)
    k_valid_mask = jnp.arange(key_len) < jnp.sum(seq_lengths)

    # --- 1. Generate Batch IDs (Optimized) ---
    q_starts = jnp.cumsum(extend_seq_lens, dtype=jnp.int32) - extend_seq_lens
    q_batch_indicators = jnp.zeros(query_len, dtype=jnp.int32).at[q_starts].set(1)
    q_batch_ids = jnp.cumsum(q_batch_indicators, dtype=jnp.int32) - 1

    full_seq_lens = seq_lengths
    k_starts = jnp.cumsum(full_seq_lens, dtype=jnp.int32) - full_seq_lens
    k_batch_indicators = jnp.zeros(key_len, dtype=jnp.int32).at[k_starts].set(1)
    k_batch_ids = jnp.cumsum(k_batch_indicators, dtype=jnp.int32) - 1

    # --- 2. Create block-diagonal mask ---
    final_mask = q_batch_ids[:, None] == k_batch_ids[None, :]

    # --- 3. Optionally add causal mask ---
    if is_causal:
        q_starts_per_pos = q_starts[q_batch_ids]
        q_relative_positions = jnp.arange(query_len, dtype=jnp.int32) - q_starts_per_pos
        prefix_lens_per_pos = extend_prefix_lens[q_batch_ids]
        q_actual_positions = prefix_lens_per_pos + q_relative_positions

        k_starts_per_pos = k_starts[k_batch_ids]
        k_relative_positions = jnp.arange(key_len, dtype=jnp.int32) - k_starts_per_pos

        causal_mask = q_actual_positions[:, None] >= k_relative_positions[None, :]
        final_mask = final_mask & causal_mask

    # --- 4. Apply the final combined mask ---
    # Combine with validity masks to handle padding
    final_mask = final_mask & q_valid_mask[:, None] & k_valid_mask[None, :]

    mask_value = jnp.finfo(attn_weights.dtype).min
    final_mask = final_mask[None, :, :]
    return jnp.where(final_mask, attn_weights, mask_value)


def _apply_decode_mask(attn_weights: jax.Array, seq_lengths: jax.Array):
    """Create a sequence mask that ensures tokens only attend within their sequence."""
    _, query_len, key_len = attn_weights.shape
    num_seqs = len(seq_lengths)

    def create_decode_sequence_mask():
        total_prefix_len = key_len
        seq_starts = jnp.cumsum(jnp.concatenate([jnp.array([0]), seq_lengths[:-1]]))
        seq_ends = seq_starts + seq_lengths
        all_positions = jnp.arange(total_prefix_len)
        seq_mask = (all_positions[None, :] >= seq_starts[:, None]) & (
            all_positions[None, :] < seq_ends[:, None]
        )
        return seq_mask

    per_sequence_mask = create_decode_sequence_mask()
    final_mask = jnp.zeros((query_len, key_len), dtype=jnp.bool_)
    final_mask = final_mask.at[:num_seqs, :].set(per_sequence_mask)

    mask_value = jnp.finfo(attn_weights.dtype).min
    final_mask = final_mask[None, :, :]
    return jnp.where(final_mask, attn_weights, mask_value)
