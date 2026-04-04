#ifndef GOMLX_METAL_H
#define GOMLX_METAL_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// ─── Device management ──────────────────────────────────────────────────────

int metal_device_count(void);
int metal_init(const char* metallib_path);
const char* metal_device_name(void);
void metal_finalize(void);

// Batch many dispatches into one command buffer (metal_encode_begin … end_wait).
// barrier_wait finishes the current batch so CPU reads of shared buffers are safe.
void metal_encode_begin(void);
int metal_encode_barrier_wait(void);
int metal_encode_end_wait(void);

// ─── Buffer management (StorageModeShared — zero-copy on Apple Silicon) ─────

typedef void* MetalBuffer;

MetalBuffer metal_buffer_alloc(size_t bytes);
void metal_buffer_free(MetalBuffer buf);
void* metal_buffer_contents(MetalBuffer buf);
size_t metal_buffer_length(MetalBuffer buf);

// ─── Elementwise unary ops ──────────────────────────────────────────────────
// dtype: 0=float16, 1=float32, 2=float64, 3=uint32 (bitwise family)
// All operate in-place on dst (which may alias src for in-place ops).

int metal_unary_op(const char* op_name, MetalBuffer src, MetalBuffer dst,
                   uint32_t num_elements, int dtype);

// ─── Elementwise binary ops ─────────────────────────────────────────────────

int metal_binary_op(const char* op_name, MetalBuffer lhs, MetalBuffer rhs,
                    MetalBuffer dst, uint32_t num_elements, int dtype);

// ─── Reductions ─────────────────────────────────────────────────────────────
// Reduces src along a contiguous inner dimension.
// inner_size: number of elements in the reduced dimension
// outer_size: number of independent reductions

int metal_reduce_op(const char* op_name, MetalBuffer src, MetalBuffer dst,
                    uint32_t outer_size, uint32_t inner_size, int dtype);

// ─── DotGeneral (batched matmul) ────────────────────────────────────────────
// C[b, m, n] = A[b, m, k] @ B[b, k, n]

int metal_dot_general(MetalBuffer a, MetalBuffer b, MetalBuffer c,
                      uint32_t batch, uint32_t m, uint32_t k, uint32_t n,
                      int dtype);

// ─── Fused ops ──────────────────────────────────────────────────────────────

int metal_fused_softmax(MetalBuffer src, MetalBuffer dst,
                        uint32_t outer_size, uint32_t axis_size, int dtype);

int metal_fused_layernorm(MetalBuffer x, MetalBuffer gamma, MetalBuffer beta,
                          MetalBuffer dst, uint32_t batch_size, uint32_t norm_size,
                          float epsilon, int has_gamma, int has_beta, int dtype);

int metal_fused_rmsnorm(MetalBuffer x, MetalBuffer weight, MetalBuffer dst,
                        uint32_t batch_size, uint32_t norm_size,
                        float epsilon, int has_weight, int dtype);

int metal_fused_gelu(MetalBuffer src, MetalBuffer dst,
                     uint32_t num_elements, int exact, int dtype);

// FusedQuantizedDense on GPU. Buffers: x(f32), w, scales(f32), zp(f32, optional), bias(f32, optional), out(f32).
// config: uint32[8] = M, K, N, blockSize, numBlocks, hasBias, hasZp, activation (backends.ActivationType).
// kind: 0=nf4 byte/elem, 1=nf4 packed nibbles, 2=linear int8, 3=linear uint8, 4=linear int4 packed, 5=linear uint4 packed.
// zp/bias may be NULL when unused (dummy bind); hasZp/hasBias in config must be 0.
int metal_quantized_dense(MetalBuffer x, MetalBuffer w, MetalBuffer scales,
                          MetalBuffer zp, MetalBuffer bias, MetalBuffer out,
                          const uint32_t* config, uint32_t config_num_uints,
                          uint32_t grid_mn, int kind);

int metal_fused_rope(MetalBuffer x, MetalBuffer dst,
                     uint32_t batch, uint32_t seq_len, uint32_t num_heads,
                     uint32_t head_dim, uint32_t rot_dim, int dtype);

// Scaled dot-product attention: Q[b,h,s,d] K[b,hkv,t,d] V[b,hkv,t,d] -> O[b,h,s,d]
// mask_batch_stride / mask_head_stride: element strides for mask layout [batch, heads?, seq, kv];
// use 0 for broadcast dims (size-1), matching backends/simplego sdpaComputeMaskStrides.
int metal_fused_sdpa(MetalBuffer q, MetalBuffer k, MetalBuffer v,
                     MetalBuffer mask, MetalBuffer dst,
                     uint32_t batch, uint32_t num_heads, uint32_t num_kv_heads,
                     uint32_t seq_len, uint32_t kv_len, uint32_t head_dim,
                     float scale, int causal, int mask_type,
                     uint32_t mask_batch_stride, uint32_t mask_head_stride, int dtype);

// General permutation transpose. Config buffer: see transpose_perm_* in tensor_ops.metal
int metal_transpose_perm(MetalBuffer src, MetalBuffer dst, MetalBuffer config,
                         uint32_t total_elements, uint32_t config_size, uint32_t elem_size);

// Thin wrappers for cgo/callers; implementations forward to the functions above.
int gomlx_metal_transpose_perm(MetalBuffer src, MetalBuffer dst, MetalBuffer config,
                               uint32_t total_elements, uint32_t config_size, uint32_t elem_size);
int gomlx_metal_fused_sdpa(MetalBuffer q, MetalBuffer k, MetalBuffer v,
                           MetalBuffer mask, MetalBuffer dst,
                           uint32_t batch, uint32_t num_heads, uint32_t num_kv_heads,
                           uint32_t seq_len, uint32_t kv_len, uint32_t head_dim,
                           float scale, int causal, int mask_type,
                           uint32_t mask_batch_stride, uint32_t mask_head_stride, int dtype);

// ─── Tensor ops dispatched to GPU ───────────────────────────────────────────

// Broadcast: repeat src to fill dst_size elements. dst[i] = src[i % src_size]
int metal_broadcast(MetalBuffer src, MetalBuffer dst,
                    uint32_t src_size, uint32_t dst_size, uint32_t elem_size);

// Iota: fill with incrementing values along one axis.
// Layout: [batch_size, iota_size, repeat_size]
int metal_iota(MetalBuffer dst, uint32_t batch_size, uint32_t iota_size,
               uint32_t repeat_size, int dtype);

// ConvertDType: element-wise type conversion.
// src_dtype/dst_dtype: 0=f16, 1=f32, 2=f64, 3=i32, 4=i64
int metal_convert_dtype(MetalBuffer src, MetalBuffer dst,
                        uint32_t num_elements, int src_dtype, int dst_dtype);

// ArgMinMax: find index of min/max along reduce axis.
// Layout: [prefix_size, reduce_size, suffix_size]
int metal_argminmax(MetalBuffer src, MetalBuffer dst,
                    uint32_t prefix_size, uint32_t reduce_size,
                    uint32_t suffix_size, int is_min, int dtype);

// Reverse inner axis: reverse the innermost dimension.
int metal_reverse_inner(MetalBuffer src, MetalBuffer dst,
                        uint32_t outer_size, uint32_t inner_size, int dtype);

// ─── Config-buffer tensor ops (variable-rank via uint32 config buffer) ──────

// BroadcastInDim: config = [rank, out_strides[rank], op_strides[rank]]
int metal_broadcast_in_dim(MetalBuffer src, MetalBuffer dst,
                           MetalBuffer config, uint32_t total_elements,
                           uint32_t config_size, uint32_t elem_size);

// Concatenate: all inputs pre-packed into src mega-buffer.
// config = [num_inputs, inner_block_size, axis_sizes[N], base_offsets[N]]
int metal_concatenate(MetalBuffer src, MetalBuffer dst,
                      MetalBuffer config, uint32_t total_elements,
                      uint32_t config_size, uint32_t elem_size);

// Slice: config = [rank, starts[rank], strides[rank], out_dims[rank], in_strides[rank]]
int metal_slice(MetalBuffer src, MetalBuffer dst,
                MetalBuffer config, uint32_t total_elements,
                uint32_t config_size, uint32_t elem_size);

// Pad: config = [rank, pad_low[rank], pad_interior[rank], in_dims[rank],
//               out_dims[rank], in_strides[rank]]
int metal_pad(MetalBuffer src, MetalBuffer pad_value, MetalBuffer dst,
              MetalBuffer config, uint32_t total_elements,
              uint32_t config_size, uint32_t elem_size);

// Reverse: config = [rank, dims[rank], strides[rank], reverse_flags[rank]]
int metal_reverse(MetalBuffer src, MetalBuffer dst,
                  MetalBuffer config, uint32_t total_elements,
                  uint32_t config_size, uint32_t elem_size);

// Where (ternary select): pred same dtype as values (nonzero = true)
int metal_where(MetalBuffer pred, MetalBuffer on_true, MetalBuffer on_false,
                MetalBuffer dst, uint32_t num_elements, int dtype);

// Where with Bool/uchar predicate (0/1); value_kind matches wherePredValueKind in metal.go.
int metal_where_bool_pred(MetalBuffer pred, MetalBuffer on_true, MetalBuffer on_false,
                          MetalBuffer dst, uint32_t num_elements, int value_kind);

int metal_cast_i32_to_i64(MetalBuffer src, MetalBuffer dst, uint32_t num_elements);

// Bool mask (uchar) to float mask; dtype 0=f16, 1=f32.
int metal_bool_mask_to_float(MetalBuffer src, MetalBuffer dst, uint32_t num_elements, int dtype);

// Sort: permute one fiber along an axis (element offsets base_off, axis_stride; elem_size in bytes).
int metal_gather_axis_row_bytes(MetalBuffer src, MetalBuffer row, uint32_t base_off,
                                uint32_t axis_stride, uint32_t axis_size, uint32_t elem_size);
int metal_scatter_axis_row_perm_bytes(MetalBuffer row, MetalBuffer dst, MetalBuffer indices,
                                      uint32_t base_off, uint32_t axis_stride,
                                      uint32_t axis_size, uint32_t elem_size);

// Sort bitonic (unstable, power-of-2 axis): load pair into comparator args; swap indices from predicate.
int metal_sort_load_pair_bytes(MetalBuffer flat, MetalBuffer lhs, MetalBuffer rhs, MetalBuffer idx,
                               uint32_t base_elem, uint32_t axis_stride, uint32_t elem_size,
                               uint32_t sort_i, uint32_t sort_j);
int metal_sort_bitonic_swap_idx(MetalBuffer idx, MetalBuffer pred, uint32_t step_k, uint32_t step_j,
                                uint32_t n);
int metal_sort_adjacent_swap_idx(MetalBuffer idx, MetalBuffer pred, uint32_t pair_i, uint32_t n,
                                 uint32_t swap_when_pred_nonzero);

// Gather: config packs all XLA-style gather params; elem_size is DType.Size() bytes per element.
int metal_gather(MetalBuffer operand, MetalBuffer indices, MetalBuffer dst,
                 MetalBuffer config, uint32_t total_elements,
                 uint32_t config_size, uint32_t elem_size);

// Scatter: scatter_elem 0=f32 1=i32 2=u32 3=i64 4=u64.
// i64/u64: large updates (>= ~2048) use LSD digit passes (parallel bitonic on (digit<<24)|i), segmented scan, parallel apply; smaller n uses a serial kernel (fast path needs padded n <= 2^24).
int metal_scatter_sum(MetalBuffer operand, MetalBuffer indices,
                      MetalBuffer updates, MetalBuffer dst,
                      MetalBuffer config, uint32_t total_elements,
                      uint32_t config_size, int scatter_elem);

int metal_scatter_max(MetalBuffer operand, MetalBuffer indices,
                      MetalBuffer updates, MetalBuffer dst,
                      MetalBuffer config, uint32_t total_elements,
                      uint32_t config_size, int scatter_elem);

int metal_scatter_min(MetalBuffer operand, MetalBuffer indices,
                      MetalBuffer updates, MetalBuffer dst,
                      MetalBuffer config, uint32_t total_elements,
                      uint32_t config_size, int scatter_elem);

// ConvGeneral: config packs spatial_rank, channels, kernel dims, strides, etc.
int metal_conv_general(MetalBuffer input, MetalBuffer kernel_buf, MetalBuffer dst,
                       MetalBuffer config, uint32_t total_elements,
                       uint32_t config_size, int dtype);

// ReduceWindow: config packs rank, reduce_type, dims, window, strides, etc.
int metal_reduce_window(MetalBuffer src, MetalBuffer dst,
                        MetalBuffer config, uint32_t total_elements,
                        uint32_t config_size, int dtype);

// RNG: PCG DXSM matching Go math/rand/v2 (state is 3x uint64; [0],[1] are PCG128 state).
int metal_rng_pcg_fill(MetalBuffer state_in, MetalBuffer state_out, MetalBuffer dst,
                       uint32_t num_bytes);

typedef struct {
    uint32_t channels;
    uint32_t inner;
    uint32_t outer;
    uint32_t numel;
} MetalBnGeom;

// elem_dtype: dtypeToMetal operand (0=float16, 1=float32).
int metal_bn_training_forward(
    MetalBuffer x, MetalBuffer scale, MetalBuffer offset,
    MetalBuffer out_norm, MetalBuffer out_mean, MetalBuffer out_var,
    MetalBnGeom geom, float epsilon, int elem_dtype);

int metal_bn_gradient(
    MetalBuffer x, MetalBuffer scale, MetalBuffer mean, MetalBuffer var,
    MetalBuffer dy, MetalBuffer dx, MetalBuffer dgamma, MetalBuffer dbeta,
    MetalBnGeom geom, float epsilon, int elem_dtype);

// ─── Optimizer steps ────────────────────────────────────────────────────────

int metal_adamw_step(MetalBuffer param, MetalBuffer grad,
                     MetalBuffer m, MetalBuffer v, MetalBuffer dst,
                     uint32_t num_elements, float lr, float beta1, float beta2,
                     float epsilon, float weight_decay, int step, int dtype);

#ifdef __cplusplus
}
#endif

#endif // GOMLX_METAL_H
