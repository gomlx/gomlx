#include <metal_stdlib>
using namespace metal;

// ─── Configuration packing convention ───────────────────────────────────────
// Complex ops pack their variable-length config into a uint32 buffer.
// The kernel reads rank, strides, dims, etc. from this buffer.
// This avoids the "Metal can't take variable-length args" problem.

// ─── Broadcast (repeat operand to fill target shape) ────────────────────────
// dst[i] = src[i % src_size]

kernel void broadcast_bytes(device const uchar* src [[buffer(0)]],
                            device uchar* dst       [[buffer(1)]],
                            constant uint& src_size [[buffer(2)]],
                            constant uint& elem_size [[buffer(3)]],
                            uint id [[thread_position_in_grid]]) {
    uint src_idx = id % src_size;
    uint src_off = src_idx * elem_size;
    uint dst_off = id * elem_size;
    for (uint b = 0; b < elem_size; b++) {
        dst[dst_off + b] = src[src_off + b];
    }
}

kernel void broadcast_f32(device const float* src [[buffer(0)]],
                          device float* dst       [[buffer(1)]],
                          constant uint& src_size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    dst[id] = src[id % src_size];
}

kernel void broadcast_f16(device const half* src [[buffer(0)]],
                          device half* dst       [[buffer(1)]],
                          constant uint& src_size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    dst[id] = src[id % src_size];
}

kernel void broadcast_i32(device const int* src [[buffer(0)]],
                          device int* dst       [[buffer(1)]],
                          constant uint& src_size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    dst[id] = src[id % src_size];
}

kernel void broadcast_u32(device const uint* src [[buffer(0)]],
                          device uint* dst       [[buffer(1)]],
                          constant uint& src_size [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    dst[id] = src[id % src_size];
}

// GoMLX bool storage (1 byte). Kernel name "broadcast" matches metal.m dtype 4 (no suffix).
kernel void broadcast(device const uchar* src [[buffer(0)]],
                      device uchar* dst [[buffer(1)]],
                      constant uint& src_size [[buffer(2)]],
                      uint id [[thread_position_in_grid]]) {
    dst[id] = src[id % src_size];
}

// ─── BroadcastInDim ─────────────────────────────────────────────────────────
// Config buffer layout:
//   [0]       = rank
//   [1..rank] = output_strides[rank]
//   [rank+1..2*rank] = operand_strides[rank]  (0 for broadcast dims with size 1)
// For each output flat index, decompose via output_strides, then recompose
// via operand_strides to get the source index.

kernel void broadcast_in_dim_bytes(
    device const uchar* src       [[buffer(0)]],
    device uchar* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    constant uint& elem_size      [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* out_strides = config + 1;
    device const uint* op_strides  = config + 1 + rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * op_strides[d];
    }

    uint src_off = src_idx * elem_size;
    uint dst_off = id * elem_size;
    for (uint b = 0; b < elem_size; b++) {
        dst[dst_off + b] = src[src_off + b];
    }
}

kernel void broadcast_in_dim_f32(
    device const float* src       [[buffer(0)]],
    device float* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* out_strides = config + 1;
    device const uint* op_strides  = config + 1 + rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * op_strides[d];
    }
    dst[id] = src[src_idx];
}

kernel void broadcast_in_dim_f16(
    device const half* src        [[buffer(0)]],
    device half* dst              [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* out_strides = config + 1;
    device const uint* op_strides  = config + 1 + rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * op_strides[d];
    }
    dst[id] = src[src_idx];
}

kernel void broadcast_in_dim_i32(
    device const int* src         [[buffer(0)]],
    device int* dst               [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* out_strides = config + 1;
    device const uint* op_strides  = config + 1 + rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * op_strides[d];
    }
    dst[id] = src[src_idx];
}

kernel void broadcast_in_dim_u32(
    device const uint* src        [[buffer(0)]],
    device uint* dst              [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* out_strides = config + 1;
    device const uint* op_strides  = config + 1 + rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * op_strides[d];
    }
    dst[id] = src[src_idx];
}

kernel void broadcast_in_dim(
    device const uchar* src       [[buffer(0)]],
    device uchar* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* out_strides = config + 1;
    device const uint* op_strides  = config + 1 + rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * op_strides[d];
    }
    dst[id] = src[src_idx];
}

// ─── Concatenate ────────────────────────────────────────────────────────────
// Config buffer layout:
//   [0] = num_inputs
//   [1] = inner_block_size (product of dims after concat axis)
//   [2..2+num_inputs] = per-input concat_axis_size
//   [2+num_inputs..2+2*num_inputs] = per-input base offset into src mega-buffer
// All inputs are pre-packed into a single contiguous src buffer by the Go side.
// Each thread computes one output element.

kernel void concatenate_bytes(
    device const uchar* src       [[buffer(0)]],
    device uchar* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    constant uint& elem_size      [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint num_inputs      = config[0];
    uint inner_block_size = config[1];
    device const uint* axis_sizes  = config + 2;
    device const uint* base_offsets = config + 2 + num_inputs;

    // Decompose id into (outer, concat_pos, inner)
    uint total_concat_size = 0;
    for (uint i = 0; i < num_inputs; i++) total_concat_size += axis_sizes[i];

    uint inner = id % inner_block_size;
    uint tmp = id / inner_block_size;
    uint concat_pos = tmp % total_concat_size;
    uint outer = tmp / total_concat_size;

    // Find which input this concat_pos falls into
    uint accum = 0;
    uint input_idx = 0;
    for (uint i = 0; i < num_inputs; i++) {
        if (concat_pos < accum + axis_sizes[i]) {
            input_idx = i;
            break;
        }
        accum += axis_sizes[i];
    }
    uint local_pos = concat_pos - accum;
    uint src_idx = base_offsets[input_idx] + outer * axis_sizes[input_idx] * inner_block_size
                   + local_pos * inner_block_size + inner;
    uint src_off = src_idx * elem_size;
    uint dst_off = id * elem_size;
    for (uint b = 0; b < elem_size; b++) {
        dst[dst_off + b] = src[src_off + b];
    }
}

kernel void concatenate_f32(
    device const float* src       [[buffer(0)]],
    device float* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint num_inputs      = config[0];
    uint inner_block_size = config[1];
    device const uint* axis_sizes  = config + 2;
    device const uint* base_offsets = config + 2 + num_inputs;

    uint total_concat_size = 0;
    for (uint i = 0; i < num_inputs; i++) total_concat_size += axis_sizes[i];

    uint inner = id % inner_block_size;
    uint tmp = id / inner_block_size;
    uint concat_pos = tmp % total_concat_size;
    uint outer = tmp / total_concat_size;

    uint accum = 0;
    uint input_idx = 0;
    for (uint i = 0; i < num_inputs; i++) {
        if (concat_pos < accum + axis_sizes[i]) { input_idx = i; break; }
        accum += axis_sizes[i];
    }
    uint local_pos = concat_pos - accum;
    uint src_idx = base_offsets[input_idx] + outer * axis_sizes[input_idx] * inner_block_size
                   + local_pos * inner_block_size + inner;
    dst[id] = src[src_idx];
}

// ─── Slice ──────────────────────────────────────────────────────────────────
// Config buffer layout:
//   [0] = rank
//   [1..rank]        = starts[rank]
//   [rank+1..2*rank] = strides[rank]
//   [2*rank+1..3*rank] = output_dims[rank]
//   [3*rank+1..4*rank] = input_strides[rank] (flat strides of input)

kernel void slice_f32(
    device const float* src       [[buffer(0)]],
    device float* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* starts       = config + 1;
    device const uint* slice_strides = config + 1 + rank;
    device const uint* out_dims     = config + 1 + 2 * rank;
    device const uint* in_strides   = config + 1 + 3 * rank;

    // Decompose output flat index into per-axis coords
    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        // Compute output stride for this axis
        uint out_stride = 1;
        for (uint dd = d + 1; dd < rank; dd++) out_stride *= out_dims[dd];

        uint coord = remaining / out_stride;
        remaining %= out_stride;

        uint src_coord = starts[d] + coord * slice_strides[d];
        src_idx += src_coord * in_strides[d];
    }
    dst[id] = src[src_idx];
}

kernel void slice_f16(
    device const half* src        [[buffer(0)]],
    device half* dst              [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* starts       = config + 1;
    device const uint* slice_strides = config + 1 + rank;
    device const uint* out_dims     = config + 1 + 2 * rank;
    device const uint* in_strides   = config + 1 + 3 * rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint out_stride = 1;
        for (uint dd = d + 1; dd < rank; dd++) out_stride *= out_dims[dd];
        uint coord = remaining / out_stride;
        remaining %= out_stride;
        uint src_coord = starts[d] + coord * slice_strides[d];
        src_idx += src_coord * in_strides[d];
    }
    dst[id] = src[src_idx];
}

kernel void slice_bytes(
    device const uchar* src       [[buffer(0)]],
    device uchar* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    constant uint& elem_size      [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* starts        = config + 1;
    device const uint* slice_strides = config + 1 + rank;
    device const uint* out_dims      = config + 1 + 2 * rank;
    device const uint* in_strides    = config + 1 + 3 * rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint out_stride = 1;
        for (uint dd = d + 1; dd < rank; dd++) out_stride *= out_dims[dd];

        uint coord = remaining / out_stride;
        remaining %= out_stride;

        uint src_coord = starts[d] + coord * slice_strides[d];
        src_idx += src_coord * in_strides[d];
    }

    uint src_off = src_idx * elem_size;
    uint dst_off = id * elem_size;
    for (uint b = 0; b < elem_size; b++) {
        dst[dst_off + b] = src[src_off + b];
    }
}

// ─── Pad ────────────────────────────────────────────────────────────────────
// For each output element, determine if it maps to a valid input element or
// should be filled with the padding value.
// Config buffer layout:
//   [0] = rank
//   [1..rank]          = pad_low[rank]
//   [rank+1..2*rank]   = pad_interior[rank]  (0 = no interior padding)
//   [2*rank+1..3*rank] = input_dims[rank]
//   [3*rank+1..4*rank] = output_dims[rank]
//   [4*rank+1..5*rank] = input_strides[rank]
// Padding value is passed as the first element of a separate buffer.

kernel void pad_f32(
    device const float* src        [[buffer(0)]],
    device float* dst              [[buffer(1)]],
    device const uint* config      [[buffer(2)]],
    device const float* pad_value  [[buffer(3)]],
    constant uint& total_elements  [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* pad_low      = config + 1;
    device const uint* pad_interior = config + 1 + rank;
    device const uint* in_dims      = config + 1 + 2 * rank;
    device const uint* out_dims     = config + 1 + 3 * rank;
    device const uint* in_strides   = config + 1 + 4 * rank;

    float fill = pad_value[0];

    // Decompose output index into coords
    uint remaining = id;
    uint src_idx = 0;
    bool valid = true;

    for (uint d = 0; d < rank; d++) {
        uint out_stride = 1;
        for (uint dd = d + 1; dd < rank; dd++) out_stride *= out_dims[dd];
        uint coord = remaining / out_stride;
        remaining %= out_stride;

        // Subtract low padding
        if (coord < pad_low[d]) { valid = false; break; }
        uint shifted = coord - pad_low[d];

        // Check interior padding alignment
        uint interior = pad_interior[d];
        if (interior > 0) {
            if (shifted % (interior + 1) != 0) { valid = false; break; }
            shifted /= (interior + 1);
        }

        // Check within input bounds
        if (shifted >= in_dims[d]) { valid = false; break; }
        src_idx += shifted * in_strides[d];
    }

    dst[id] = valid ? src[src_idx] : fill;
}

kernel void pad_f16(
    device const half* src        [[buffer(0)]],
    device half* dst              [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    device const half* pad_value  [[buffer(3)]],
    constant uint& total_elements [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* pad_low      = config + 1;
    device const uint* pad_interior = config + 1 + rank;
    device const uint* in_dims      = config + 1 + 2 * rank;
    device const uint* out_dims     = config + 1 + 3 * rank;
    device const uint* in_strides   = config + 1 + 4 * rank;

    half fill = pad_value[0];
    uint remaining = id;
    uint src_idx = 0;
    bool valid = true;

    for (uint d = 0; d < rank; d++) {
        uint out_stride = 1;
        for (uint dd = d + 1; dd < rank; dd++) out_stride *= out_dims[dd];
        uint coord = remaining / out_stride;
        remaining %= out_stride;
        if (coord < pad_low[d]) { valid = false; break; }
        uint shifted = coord - pad_low[d];
        uint interior = pad_interior[d];
        if (interior > 0) {
            if (shifted % (interior + 1) != 0) { valid = false; break; }
            shifted /= (interior + 1);
        }
        if (shifted >= in_dims[d]) { valid = false; break; }
        src_idx += shifted * in_strides[d];
    }
    dst[id] = valid ? src[src_idx] : fill;
}

kernel void pad_bytes(
    device const uchar* src        [[buffer(0)]],
    device uchar* dst              [[buffer(1)]],
    device const uint* config      [[buffer(2)]],
    device const uchar* pad_value  [[buffer(3)]],
    constant uint& total_elements  [[buffer(4)]],
    constant uint& elem_size       [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* pad_low      = config + 1;
    device const uint* pad_interior = config + 1 + rank;
    device const uint* in_dims      = config + 1 + 2 * rank;
    device const uint* out_dims     = config + 1 + 3 * rank;
    device const uint* in_strides   = config + 1 + 4 * rank;

    uint remaining = id;
    uint src_idx = 0;
    bool valid = true;

    for (uint d = 0; d < rank; d++) {
        uint out_stride = 1;
        for (uint dd = d + 1; dd < rank; dd++) out_stride *= out_dims[dd];
        uint coord = remaining / out_stride;
        remaining %= out_stride;

        if (coord < pad_low[d]) { valid = false; break; }
        uint shifted = coord - pad_low[d];
        uint interior = pad_interior[d];
        if (interior > 0) {
            if (shifted % (interior + 1) != 0) { valid = false; break; }
            shifted /= (interior + 1);
        }
        if (shifted >= in_dims[d]) { valid = false; break; }
        src_idx += shifted * in_strides[d];
    }

    uint dst_off = id * elem_size;
    if (valid) {
        uint src_off = src_idx * elem_size;
        for (uint b = 0; b < elem_size; b++) {
            dst[dst_off + b] = src[src_off + b];
        }
    } else {
        for (uint b = 0; b < elem_size; b++) {
            dst[dst_off + b] = pad_value[b];
        }
    }
}

// ─── Reverse ────────────────────────────────────────────────────────────────
// Config buffer layout:
//   [0] = rank
//   [1..rank]          = dims[rank]
//   [rank+1..2*rank]   = strides[rank]
//   [2*rank+1..3*rank] = reverse_flag[rank] (1 if axis is reversed, 0 otherwise)

kernel void reverse_f32(
    device const float* src       [[buffer(0)]],
    device float* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* dims    = config + 1;
    device const uint* strides = config + 1 + rank;
    device const uint* rev     = config + 1 + 2 * rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / strides[d];
        remaining %= strides[d];
        uint src_coord = rev[d] ? (dims[d] - 1 - coord) : coord;
        src_idx += src_coord * strides[d];
    }
    dst[id] = src[src_idx];
}

kernel void reverse_f16(
    device const half* src        [[buffer(0)]],
    device half* dst              [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* dims    = config + 1;
    device const uint* strides = config + 1 + rank;
    device const uint* rev     = config + 1 + 2 * rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / strides[d];
        remaining %= strides[d];
        uint src_coord = rev[d] ? (dims[d] - 1 - coord) : coord;
        src_idx += src_coord * strides[d];
    }
    dst[id] = src[src_idx];
}

kernel void reverse_bytes(
    device const uchar* src       [[buffer(0)]],
    device uchar* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    constant uint& elem_size      [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    device const uint* dims    = config + 1;
    device const uint* strides = config + 1 + rank;
    device const uint* rev     = config + 1 + 2 * rank;

    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / strides[d];
        remaining %= strides[d];
        uint src_coord = rev[d] ? (dims[d] - 1 - coord) : coord;
        src_idx += src_coord * strides[d];
    }

    uint src_off = src_idx * elem_size;
    uint dst_off = id * elem_size;
    for (uint b = 0; b < elem_size; b++) {
        dst[dst_off + b] = src[src_off + b];
    }
}

// ─── Gather ─────────────────────────────────────────────────────────────────
// XLA-style gather. Each thread produces one output element.
// Config buffer layout:
//   [0] = operand_rank
//   [1] = indices_rank
//   [2] = output_rank
//   [3] = index_vector_axis
//   [4] = num_offset_axes
//   [5] = num_collapsed_axes
//   [6] = num_start_index_map
//   [7..7+operand_rank]                         = operand_dims
//   [7+operand_rank..7+operand_rank+operand_rank] = operand_strides
//   [+indices_rank]                              = indices_dims
//   [+output_rank]                               = output_strides
//   [+num_offset_axes]                           = offset_output_axes
//   [+num_collapsed_axes]                        = collapsed_slice_axes
//   [+num_start_index_map]                       = start_index_map
//   [+operand_rank]                              = slice_sizes
//
// This is complex enough that we define a helper struct.

// Due to the complexity of XLA Gather semantics (variable config arrays,
// collapsed axes, offset axes, index clamping), this kernel uses a
// straightforward per-element approach that reads config from a buffer.

kernel void gather_f32(
    device const float* operand    [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device float* output           [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& total_elements  [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;

    uint operand_rank      = config[0];
    uint indices_rank      = config[1];
    uint output_rank       = config[2];
    uint index_vector_axis = config[3];
    uint num_offset_axes   = config[4];
    uint num_collapsed     = config[5];
    uint num_start_idx_map = config[6];

    uint offset = 7;
    device const uint* operand_dims    = config + offset; offset += operand_rank;
    device const uint* operand_strides = config + offset; offset += operand_rank;
    offset += indices_rank; // indices_dims (unused)
    device const uint* indices_strides = config + offset; offset += indices_rank;
    offset += output_rank; // output_dims (unused)
    device const uint* output_strides  = config + offset; offset += output_rank;
    device const uint* offset_axes     = config + offset; offset += num_offset_axes;
    device const uint* collapsed_axes  = config + offset; offset += num_collapsed;
    device const uint* start_idx_map   = config + offset; offset += num_start_idx_map;
    device const uint* slice_sizes     = config + offset;

    // Decompose output index into coords
    uint out_coords[16]; // max rank 16
    uint remaining = id;
    for (uint d = 0; d < output_rank; d++) {
        out_coords[d] = remaining / output_strides[d];
        remaining %= output_strides[d];
    }

    // Split output coords into batch coords and offset coords.
    // Batch coords = output coords NOT in offset_output_axes.
    // Offset coords = output coords in offset_output_axes.

    // Build indices coord from batch coords
    uint indices_coord[16];
    uint batch_idx = 0;
    for (uint d = 0; d < output_rank; d++) {
        bool is_offset = false;
        for (uint j = 0; j < num_offset_axes; j++) {
            if (offset_axes[j] == d) { is_offset = true; break; }
        }
        if (!is_offset) {
            if (batch_idx < indices_rank) {
                // Skip index_vector_axis in indices
                uint target = batch_idx;
                if (target >= index_vector_axis) target++; // shift past index_vector_axis
                if (target < indices_rank) indices_coord[target] = out_coords[d];
            }
            batch_idx++;
        }
    }

    // Read start indices from the indices tensor
    uint operand_coord[16];
    for (uint d = 0; d < operand_rank; d++) operand_coord[d] = 0;

    for (uint i = 0; i < num_start_idx_map; i++) {
        indices_coord[index_vector_axis] = i;
        // Compute flat index into indices tensor
        uint idx_flat = 0;
        for (uint d = 0; d < indices_rank; d++) {
            idx_flat += indices_coord[d] * indices_strides[d];
        }
        int32_t start_val = indices[idx_flat];
        uint target_axis = start_idx_map[i];
        // Clamp to [0, operand_dims[target_axis] - slice_sizes[target_axis]]
        int32_t max_val = int32_t(operand_dims[target_axis] - slice_sizes[target_axis]);
        if (start_val < 0) start_val = 0;
        if (start_val > max_val) start_val = max_val;
        operand_coord[target_axis] = uint(start_val);
    }

    // Add offset coords (expanding collapsed axes)
    uint offset_idx = 0;
    for (uint d = 0; d < operand_rank; d++) {
        bool is_collapsed = false;
        for (uint j = 0; j < num_collapsed; j++) {
            if (collapsed_axes[j] == d) { is_collapsed = true; break; }
        }
        if (!is_collapsed) {
            // Find the corresponding offset output axis
            if (offset_idx < num_offset_axes) {
                operand_coord[d] += out_coords[offset_axes[offset_idx]];
            }
            offset_idx++;
        }
    }

    // Compute operand flat index
    uint src_idx = 0;
    for (uint d = 0; d < operand_rank; d++) {
        src_idx += operand_coord[d] * operand_strides[d];
    }

    output[id] = operand[src_idx];
}

// Type-agnostic gather: elem_size in bytes (buffer(5)). Index math identical to gather_f32.
kernel void gather_bytes(
    device const uchar* operand    [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device uchar* output           [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& total_elements  [[buffer(4)]],
    constant uint& elem_size       [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    if (elem_size == 0 || elem_size > 16) return;

    uint operand_rank      = config[0];
    uint indices_rank      = config[1];
    uint output_rank       = config[2];
    uint index_vector_axis = config[3];
    uint num_offset_axes   = config[4];
    uint num_collapsed     = config[5];
    uint num_start_idx_map = config[6];

    uint offset = 7;
    device const uint* operand_dims    = config + offset; offset += operand_rank;
    device const uint* operand_strides = config + offset; offset += operand_rank;
    offset += indices_rank;
    device const uint* indices_strides = config + offset; offset += indices_rank;
    offset += output_rank;
    device const uint* output_strides  = config + offset; offset += output_rank;
    device const uint* offset_axes     = config + offset; offset += num_offset_axes;
    device const uint* collapsed_axes  = config + offset; offset += num_collapsed;
    device const uint* start_idx_map   = config + offset; offset += num_start_idx_map;
    device const uint* slice_sizes     = config + offset;

    uint out_coords[16];
    uint remaining = id;
    for (uint d = 0; d < output_rank; d++) {
        out_coords[d] = remaining / output_strides[d];
        remaining %= output_strides[d];
    }

    uint indices_coord[16];
    uint batch_idx = 0;
    for (uint d = 0; d < output_rank; d++) {
        bool is_offset = false;
        for (uint j = 0; j < num_offset_axes; j++) {
            if (offset_axes[j] == d) { is_offset = true; break; }
        }
        if (!is_offset) {
            if (batch_idx < indices_rank) {
                uint target = batch_idx;
                if (target >= index_vector_axis) target++;
                if (target < indices_rank) indices_coord[target] = out_coords[d];
            }
            batch_idx++;
        }
    }

    uint operand_coord[16];
    for (uint d = 0; d < operand_rank; d++) operand_coord[d] = 0;

    for (uint i = 0; i < num_start_idx_map; i++) {
        indices_coord[index_vector_axis] = i;
        uint idx_flat = 0;
        for (uint d = 0; d < indices_rank; d++) {
            idx_flat += indices_coord[d] * indices_strides[d];
        }
        int32_t start_val = indices[idx_flat];
        uint target_axis = start_idx_map[i];
        int32_t max_val = int32_t(operand_dims[target_axis] - slice_sizes[target_axis]);
        if (start_val < 0) start_val = 0;
        if (start_val > max_val) start_val = max_val;
        operand_coord[target_axis] = uint(start_val);
    }

    uint offset_idx = 0;
    for (uint d = 0; d < operand_rank; d++) {
        bool is_collapsed = false;
        for (uint j = 0; j < num_collapsed; j++) {
            if (collapsed_axes[j] == d) { is_collapsed = true; break; }
        }
        if (!is_collapsed) {
            if (offset_idx < num_offset_axes) {
                operand_coord[d] += out_coords[offset_axes[offset_idx]];
            }
            offset_idx++;
        }
    }

    uint src_idx = 0;
    for (uint d = 0; d < operand_rank; d++) {
        src_idx += operand_coord[d] * operand_strides[d];
    }

    uint src_byte = src_idx * elem_size;
    uint dst_byte = id * elem_size;
    for (uint b = 0; b < elem_size; b++) {
        output[dst_byte + b] = operand[src_byte + b];
    }
}

// ─── Scatter (Sum/Max/Min) ──────────────────────────────────────────────────
// Scatter is the inverse of gather: updates are scattered INTO the operand.
// Because multiple updates can map to the same output element, we need atomics.
// Config is similar to gather.
// For ScatterSum we use atomic_fetch_add; for Max/Min we use compare-and-swap loops.

// ScatterSum with atomic float add
kernel void scatter_sum_f32(
    device float* output           [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const float* updates    [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= num_updates) return;

    uint operand_rank      = config[0];
    uint indices_rank      = config[1];
    uint updates_rank      = config[2];
    uint index_vector_axis = config[3];
    uint num_update_window = config[4];
    uint num_inserted      = config[5];
    uint num_scatter_map   = config[6];

    uint off = 7;
    device const uint* operand_dims    = config + off; off += operand_rank;
    device const uint* operand_strides = config + off; off += operand_rank;
    off += indices_rank; // indices_dims (unused)
    device const uint* indices_strides = config + off; off += indices_rank;
    off += updates_rank; // updates_dims (unused)
    device const uint* updates_strides = config + off; off += updates_rank;
    device const uint* update_window_axes     = config + off; off += num_update_window;
    device const uint* inserted_window_axes   = config + off; off += num_inserted;
    device const uint* scatter_to_operand_map = config + off; off += num_scatter_map;

    // Decompose update index
    uint update_coords[16];
    uint remaining = id;
    for (uint d = 0; d < updates_rank; d++) {
        update_coords[d] = remaining / updates_strides[d];
        remaining %= updates_strides[d];
    }

    // Extract batch coords (non-window dims of updates)
    uint indices_coord[16];
    uint batch_idx = 0;
    for (uint d = 0; d < updates_rank; d++) {
        bool is_window = false;
        for (uint j = 0; j < num_update_window; j++) {
            if (update_window_axes[j] == d) { is_window = true; break; }
        }
        if (!is_window) {
            uint target = batch_idx;
            if (target >= index_vector_axis) target++;
            if (target < indices_rank) indices_coord[target] = update_coords[d];
            batch_idx++;
        }
    }

    // Read scatter indices
    uint operand_coord[16];
    for (uint d = 0; d < operand_rank; d++) operand_coord[d] = 0;

    for (uint i = 0; i < num_scatter_map; i++) {
        indices_coord[index_vector_axis] = i;
        uint idx_flat = 0;
        for (uint d = 0; d < indices_rank; d++) {
            idx_flat += indices_coord[d] * indices_strides[d];
        }
        int32_t start_val = indices[idx_flat];
        uint target_axis = scatter_to_operand_map[i];
        int32_t max_val = int32_t(operand_dims[target_axis]) - 1;
        if (start_val < 0) start_val = 0;
        if (start_val > max_val) start_val = max_val;
        operand_coord[target_axis] = uint(start_val);
    }

    // Add window offsets
    uint window_idx = 0;
    for (uint d = 0; d < operand_rank; d++) {
        bool is_inserted = false;
        for (uint j = 0; j < num_inserted; j++) {
            if (inserted_window_axes[j] == d) { is_inserted = true; break; }
        }
        if (!is_inserted) {
            if (window_idx < num_update_window) {
                operand_coord[d] += update_coords[update_window_axes[window_idx]];
            }
            window_idx++;
        }
    }

    // Bounds check
    bool in_bounds = true;
    uint dst_idx = 0;
    for (uint d = 0; d < operand_rank; d++) {
        if (operand_coord[d] >= operand_dims[d]) { in_bounds = false; break; }
        dst_idx += operand_coord[d] * operand_strides[d];
    }

    if (in_bounds) {
        // Atomic add for scatter_sum
        device atomic_float* addr = (device atomic_float*)(output + dst_idx);
        atomic_fetch_add_explicit(addr, updates[id], memory_order_relaxed);
    }
}

#define GOMLX_SCATTER_INDEX_BODY                                                 \
    if (id >= num_updates) return;                                               \
    uint operand_rank      = config[0];                                         \
    uint indices_rank      = config[1];                                         \
    uint updates_rank      = config[2];                                         \
    uint index_vector_axis = config[3];                                         \
    uint num_update_window = config[4];                                         \
    uint num_inserted      = config[5];                                         \
    uint num_scatter_map   = config[6];                                         \
    uint off = 7;                                                               \
    device const uint* operand_dims    = config + off; off += operand_rank;   \
    device const uint* operand_strides = config + off; off += operand_rank;     \
    off += indices_rank;                                                        \
    device const uint* indices_strides = config + off; off += indices_rank;      \
    off += updates_rank;                                                        \
    device const uint* updates_strides = config + off; off += updates_rank;      \
    device const uint* update_window_axes     = config + off; off += num_update_window; \
    device const uint* inserted_window_axes   = config + off; off += num_inserted; \
    device const uint* scatter_to_operand_map = config + off; off += num_scatter_map; \
    uint update_coords[16];                                                      \
    uint remaining = id;                                                         \
    for (uint d = 0; d < updates_rank; d++) {                                    \
        update_coords[d] = remaining / updates_strides[d];                      \
        remaining %= updates_strides[d];                                         \
    }                                                                            \
    uint indices_coord[16];                                                      \
    uint batch_idx = 0;                                                          \
    for (uint d = 0; d < updates_rank; d++) {                                    \
        bool is_window = false;                                                  \
        for (uint j = 0; j < num_update_window; j++) {                         \
            if (update_window_axes[j] == d) { is_window = true; break; }        \
        }                                                                        \
        if (!is_window) {                                                        \
            uint target = batch_idx;                                             \
            if (target >= index_vector_axis) target++;                           \
            if (target < indices_rank) indices_coord[target] = update_coords[d]; \
            batch_idx++;                                                         \
        }                                                                        \
    }                                                                            \
    uint operand_coord[16];                                                      \
    for (uint d = 0; d < operand_rank; d++) operand_coord[d] = 0;              \
    for (uint i = 0; i < num_scatter_map; i++) {                                 \
        indices_coord[index_vector_axis] = i;                                   \
        uint idx_flat = 0;                                                       \
        for (uint d = 0; d < indices_rank; d++) {                               \
            idx_flat += indices_coord[d] * indices_strides[d];                   \
        }                                                                        \
        int32_t start_val = indices[idx_flat];                                   \
        uint target_axis = scatter_to_operand_map[i];                            \
        int32_t max_val = int32_t(operand_dims[target_axis]) - 1;                \
        if (start_val < 0) start_val = 0;                                       \
        if (start_val > max_val) start_val = max_val;                            \
        operand_coord[target_axis] = uint(start_val);                            \
    }                                                                            \
    uint window_idx = 0;                                                         \
    for (uint d = 0; d < operand_rank; d++) {                                    \
        bool is_inserted = false;                                                \
        for (uint j = 0; j < num_inserted; j++) {                               \
            if (inserted_window_axes[j] == d) { is_inserted = true; break; }    \
        }                                                                        \
        if (!is_inserted) {                                                      \
            if (window_idx < num_update_window) {                               \
                operand_coord[d] += update_coords[update_window_axes[window_idx]]; \
            }                                                                    \
            window_idx++;                                                        \
        }                                                                        \
    }                                                                            \
    bool in_bounds = true;                                                       \
    uint dst_idx = 0;                                                            \
    for (uint d = 0; d < operand_rank; d++) {                                    \
        if (operand_coord[d] >= operand_dims[d]) { in_bounds = false; break; }  \
        dst_idx += operand_coord[d] * operand_strides[d];                        \
    }                                                                            \
    if (!in_bounds) return;

kernel void scatter_sum_i32(
    device int* output            [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const int* updates     [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    GOMLX_SCATTER_INDEX_BODY
    device atomic_int* addr = (device atomic_int*)(output + dst_idx);
    atomic_fetch_add_explicit(addr, updates[id], memory_order_relaxed);
}

kernel void scatter_sum_u32(
    device uint* output           [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const uint* updates     [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    GOMLX_SCATTER_INDEX_BODY
    device atomic_uint* addr = (device atomic_uint*)(output + dst_idx);
    atomic_fetch_add_explicit(addr, updates[id], memory_order_relaxed);
}

#undef GOMLX_SCATTER_INDEX_BODY

// ScatterMax: CAS loop on uint bits (IEEE float)
kernel void scatter_max_f32(
    device float* output           [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const float* updates    [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= num_updates) return;

    uint operand_rank      = config[0];
    uint indices_rank      = config[1];
    uint updates_rank      = config[2];
    uint index_vector_axis = config[3];
    uint num_update_window = config[4];
    uint num_inserted      = config[5];
    uint num_scatter_map   = config[6];

    uint off = 7;
    device const uint* operand_dims    = config + off; off += operand_rank;
    device const uint* operand_strides = config + off; off += operand_rank;
    off += indices_rank;
    device const uint* indices_strides = config + off; off += indices_rank;
    off += updates_rank;
    device const uint* updates_strides = config + off; off += updates_rank;
    device const uint* update_window_axes     = config + off; off += num_update_window;
    device const uint* inserted_window_axes   = config + off; off += num_inserted;
    device const uint* scatter_to_operand_map = config + off; off += num_scatter_map;

    uint update_coords[16];
    uint remaining = id;
    for (uint d = 0; d < updates_rank; d++) {
        update_coords[d] = remaining / updates_strides[d];
        remaining %= updates_strides[d];
    }

    uint indices_coord[16];
    uint batch_idx = 0;
    for (uint d = 0; d < updates_rank; d++) {
        bool is_window = false;
        for (uint j = 0; j < num_update_window; j++) {
            if (update_window_axes[j] == d) { is_window = true; break; }
        }
        if (!is_window) {
            uint target = batch_idx;
            if (target >= index_vector_axis) target++;
            if (target < indices_rank) indices_coord[target] = update_coords[d];
            batch_idx++;
        }
    }

    uint operand_coord[16];
    for (uint d = 0; d < operand_rank; d++) operand_coord[d] = 0;

    for (uint i = 0; i < num_scatter_map; i++) {
        indices_coord[index_vector_axis] = i;
        uint idx_flat = 0;
        for (uint d = 0; d < indices_rank; d++) {
            idx_flat += indices_coord[d] * indices_strides[d];
        }
        int32_t start_val = indices[idx_flat];
        uint target_axis = scatter_to_operand_map[i];
        int32_t max_val = int32_t(operand_dims[target_axis]) - 1;
        if (start_val < 0) start_val = 0;
        if (start_val > max_val) start_val = max_val;
        operand_coord[target_axis] = uint(start_val);
    }

    uint window_idx = 0;
    for (uint d = 0; d < operand_rank; d++) {
        bool is_inserted = false;
        for (uint j = 0; j < num_inserted; j++) {
            if (inserted_window_axes[j] == d) { is_inserted = true; break; }
        }
        if (!is_inserted) {
            if (window_idx < num_update_window) {
                operand_coord[d] += update_coords[update_window_axes[window_idx]];
            }
            window_idx++;
        }
    }

    bool in_bounds = true;
    uint dst_idx = 0;
    for (uint d = 0; d < operand_rank; d++) {
        if (operand_coord[d] >= operand_dims[d]) { in_bounds = false; break; }
        dst_idx += operand_coord[d] * operand_strides[d];
    }

    if (in_bounds) {
        float new_val = updates[id];
        device atomic_uint* atom = (device atomic_uint*)(output + dst_idx);
        uint old_bits = atomic_load_explicit(atom, memory_order_relaxed);
        while (true) {
            float oldf = as_type<float>(old_bits);
            float merged = fmax(oldf, new_val);
            uint new_bits = as_type<uint>(merged);
            uint expected = old_bits;
            if (atomic_compare_exchange_weak_explicit(
                    atom, &expected, new_bits, memory_order_relaxed, memory_order_relaxed)) {
                break;
            }
            old_bits = expected;
        }
    }
}

kernel void scatter_min_f32(
    device float* output           [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const float* updates    [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= num_updates) return;

    uint operand_rank      = config[0];
    uint indices_rank      = config[1];
    uint updates_rank      = config[2];
    uint index_vector_axis = config[3];
    uint num_update_window = config[4];
    uint num_inserted      = config[5];
    uint num_scatter_map   = config[6];

    uint off = 7;
    device const uint* operand_dims    = config + off; off += operand_rank;
    device const uint* operand_strides = config + off; off += operand_rank;
    off += indices_rank;
    device const uint* indices_strides = config + off; off += indices_rank;
    off += updates_rank;
    device const uint* updates_strides = config + off; off += updates_rank;
    device const uint* update_window_axes     = config + off; off += num_update_window;
    device const uint* inserted_window_axes   = config + off; off += num_inserted;
    device const uint* scatter_to_operand_map = config + off; off += num_scatter_map;

    uint update_coords[16];
    uint remaining = id;
    for (uint d = 0; d < updates_rank; d++) {
        update_coords[d] = remaining / updates_strides[d];
        remaining %= updates_strides[d];
    }

    uint indices_coord[16];
    uint batch_idx = 0;
    for (uint d = 0; d < updates_rank; d++) {
        bool is_window = false;
        for (uint j = 0; j < num_update_window; j++) {
            if (update_window_axes[j] == d) { is_window = true; break; }
        }
        if (!is_window) {
            uint target = batch_idx;
            if (target >= index_vector_axis) target++;
            if (target < indices_rank) indices_coord[target] = update_coords[d];
            batch_idx++;
        }
    }

    uint operand_coord[16];
    for (uint d = 0; d < operand_rank; d++) operand_coord[d] = 0;

    for (uint i = 0; i < num_scatter_map; i++) {
        indices_coord[index_vector_axis] = i;
        uint idx_flat = 0;
        for (uint d = 0; d < indices_rank; d++) {
            idx_flat += indices_coord[d] * indices_strides[d];
        }
        int32_t start_val = indices[idx_flat];
        uint target_axis = scatter_to_operand_map[i];
        int32_t max_val = int32_t(operand_dims[target_axis]) - 1;
        if (start_val < 0) start_val = 0;
        if (start_val > max_val) start_val = max_val;
        operand_coord[target_axis] = uint(start_val);
    }

    uint window_idx = 0;
    for (uint d = 0; d < operand_rank; d++) {
        bool is_inserted = false;
        for (uint j = 0; j < num_inserted; j++) {
            if (inserted_window_axes[j] == d) { is_inserted = true; break; }
        }
        if (!is_inserted) {
            if (window_idx < num_update_window) {
                operand_coord[d] += update_coords[update_window_axes[window_idx]];
            }
            window_idx++;
        }
    }

    bool in_bounds = true;
    uint dst_idx = 0;
    for (uint d = 0; d < operand_rank; d++) {
        if (operand_coord[d] >= operand_dims[d]) { in_bounds = false; break; }
        dst_idx += operand_coord[d] * operand_strides[d];
    }

    if (in_bounds) {
        float new_val = updates[id];
        device atomic_uint* atom = (device atomic_uint*)(output + dst_idx);
        uint old_bits = atomic_load_explicit(atom, memory_order_relaxed);
        while (true) {
            float oldf = as_type<float>(old_bits);
            float merged = fmin(oldf, new_val);
            uint new_bits = as_type<uint>(merged);
            uint expected = old_bits;
            if (atomic_compare_exchange_weak_explicit(
                    atom, &expected, new_bits, memory_order_relaxed, memory_order_relaxed)) {
                break;
            }
            old_bits = expected;
        }
    }
}

#define GOMLX_SCATTER_INDEX_BODY2                                                \
    if (id >= num_updates) return;                                               \
    uint operand_rank      = config[0];                                         \
    uint indices_rank      = config[1];                                         \
    uint updates_rank      = config[2];                                         \
    uint index_vector_axis = config[3];                                         \
    uint num_update_window = config[4];                                         \
    uint num_inserted      = config[5];                                         \
    uint num_scatter_map   = config[6];                                         \
    uint off = 7;                                                               \
    device const uint* operand_dims    = config + off; off += operand_rank;   \
    device const uint* operand_strides = config + off; off += operand_rank;     \
    off += indices_rank;                                                        \
    device const uint* indices_strides = config + off; off += indices_rank;      \
    off += updates_rank;                                                        \
    device const uint* updates_strides = config + off; off += updates_rank;      \
    device const uint* update_window_axes     = config + off; off += num_update_window; \
    device const uint* inserted_window_axes   = config + off; off += num_inserted; \
    device const uint* scatter_to_operand_map = config + off; off += num_scatter_map; \
    uint update_coords[16];                                                      \
    uint remaining = id;                                                         \
    for (uint d = 0; d < updates_rank; d++) {                                    \
        update_coords[d] = remaining / updates_strides[d];                      \
        remaining %= updates_strides[d];                                         \
    }                                                                            \
    uint indices_coord[16];                                                      \
    uint batch_idx = 0;                                                          \
    for (uint d = 0; d < updates_rank; d++) {                                    \
        bool is_window = false;                                                  \
        for (uint j = 0; j < num_update_window; j++) {                         \
            if (update_window_axes[j] == d) { is_window = true; break; }        \
        }                                                                        \
        if (!is_window) {                                                        \
            uint target = batch_idx;                                             \
            if (target >= index_vector_axis) target++;                           \
            if (target < indices_rank) indices_coord[target] = update_coords[d]; \
            batch_idx++;                                                         \
        }                                                                        \
    }                                                                            \
    uint operand_coord[16];                                                      \
    for (uint d = 0; d < operand_rank; d++) operand_coord[d] = 0;              \
    for (uint i = 0; i < num_scatter_map; i++) {                                 \
        indices_coord[index_vector_axis] = i;                                   \
        uint idx_flat = 0;                                                       \
        for (uint d = 0; d < indices_rank; d++) {                               \
            idx_flat += indices_coord[d] * indices_strides[d];                   \
        }                                                                        \
        int32_t start_val = indices[idx_flat];                                   \
        uint target_axis = scatter_to_operand_map[i];                            \
        int32_t max_val = int32_t(operand_dims[target_axis]) - 1;                \
        if (start_val < 0) start_val = 0;                                       \
        if (start_val > max_val) start_val = max_val;                            \
        operand_coord[target_axis] = uint(start_val);                            \
    }                                                                            \
    uint window_idx = 0;                                                         \
    for (uint d = 0; d < operand_rank; d++) {                                    \
        bool is_inserted = false;                                                \
        for (uint j = 0; j < num_inserted; j++) {                               \
            if (inserted_window_axes[j] == d) { is_inserted = true; break; }    \
        }                                                                        \
        if (!is_inserted) {                                                      \
            if (window_idx < num_update_window) {                               \
                operand_coord[d] += update_coords[update_window_axes[window_idx]]; \
            }                                                                    \
            window_idx++;                                                        \
        }                                                                        \
    }                                                                            \
    bool in_bounds = true;                                                       \
    uint dst_idx = 0;                                                            \
    for (uint d = 0; d < operand_rank; d++) {                                    \
        if (operand_coord[d] >= operand_dims[d]) { in_bounds = false; break; }  \
        dst_idx += operand_coord[d] * operand_strides[d];                        \
    }                                                                            \
    if (!in_bounds) return;

// Like GOMLX_SCATTER_INDEX_BODY2 but for "for (id ...)" loops (no early return on OOB).
#define GOMLX_SCATTER_INDEX_BODY2_LOOP                                           \
    uint operand_rank      = config[0];                                         \
    uint indices_rank      = config[1];                                         \
    uint updates_rank      = config[2];                                         \
    uint index_vector_axis = config[3];                                         \
    uint num_update_window = config[4];                                         \
    uint num_inserted      = config[5];                                         \
    uint num_scatter_map   = config[6];                                         \
    uint off = 7;                                                               \
    device const uint* operand_dims    = config + off; off += operand_rank;   \
    device const uint* operand_strides = config + off; off += operand_rank;     \
    off += indices_rank;                                                        \
    device const uint* indices_strides = config + off; off += indices_rank;      \
    off += updates_rank;                                                        \
    device const uint* updates_strides = config + off; off += updates_rank;      \
    device const uint* update_window_axes     = config + off; off += num_update_window; \
    device const uint* inserted_window_axes   = config + off; off += num_inserted; \
    device const uint* scatter_to_operand_map = config + off; off += num_scatter_map; \
    uint update_coords[16];                                                      \
    uint remaining = id;                                                         \
    for (uint d = 0; d < updates_rank; d++) {                                    \
        update_coords[d] = remaining / updates_strides[d];                      \
        remaining %= updates_strides[d];                                         \
    }                                                                            \
    uint indices_coord[16];                                                      \
    uint batch_idx = 0;                                                          \
    for (uint d = 0; d < updates_rank; d++) {                                    \
        bool is_window = false;                                                  \
        for (uint j = 0; j < num_update_window; j++) {                         \
            if (update_window_axes[j] == d) { is_window = true; break; }        \
        }                                                                        \
        if (!is_window) {                                                        \
            uint target = batch_idx;                                             \
            if (target >= index_vector_axis) target++;                           \
            if (target < indices_rank) indices_coord[target] = update_coords[d]; \
            batch_idx++;                                                         \
        }                                                                        \
    }                                                                            \
    uint operand_coord[16];                                                      \
    for (uint d = 0; d < operand_rank; d++) operand_coord[d] = 0;              \
    for (uint i = 0; i < num_scatter_map; i++) {                                 \
        indices_coord[index_vector_axis] = i;                                   \
        uint idx_flat = 0;                                                       \
        for (uint d = 0; d < indices_rank; d++) {                               \
            idx_flat += indices_coord[d] * indices_strides[d];                   \
        }                                                                        \
        int32_t start_val = indices[idx_flat];                                   \
        uint target_axis = scatter_to_operand_map[i];                            \
        int32_t max_val = int32_t(operand_dims[target_axis]) - 1;                \
        if (start_val < 0) start_val = 0;                                       \
        if (start_val > max_val) start_val = max_val;                            \
        operand_coord[target_axis] = uint(start_val);                            \
    }                                                                            \
    uint window_idx = 0;                                                         \
    for (uint d = 0; d < operand_rank; d++) {                                    \
        bool is_inserted = false;                                                \
        for (uint j = 0; j < num_inserted; j++) {                               \
            if (inserted_window_axes[j] == d) { is_inserted = true; break; }    \
        }                                                                        \
        if (!is_inserted) {                                                      \
            if (window_idx < num_update_window) {                               \
                operand_coord[d] += update_coords[update_window_axes[window_idx]]; \
            }                                                                    \
            window_idx++;                                                        \
        }                                                                        \
    }                                                                            \
    bool in_bounds = true;                                                       \
    uint dst_idx = 0;                                                            \
    for (uint d = 0; d < operand_rank; d++) {                                    \
        if (operand_coord[d] >= operand_dims[d]) { in_bounds = false; break; }  \
        dst_idx += operand_coord[d] * operand_strides[d];                        \
    }                                                                            \
    if (!in_bounds) continue;

// Project one update slot: OOB writes sentinel key 0xFFFFFFFF (skipped at finalize).
#define GOMLX_SCATTER_INDEX_BODY2_PROJ                                         \
    uint operand_rank      = config[0];                                         \
    uint indices_rank      = config[1];                                         \
    uint updates_rank      = config[2];                                         \
    uint index_vector_axis = config[3];                                         \
    uint num_update_window = config[4];                                         \
    uint num_inserted      = config[5];                                         \
    uint num_scatter_map   = config[6];                                         \
    uint off = 7;                                                               \
    device const uint* operand_dims    = config + off; off += operand_rank;   \
    device const uint* operand_strides = config + off; off += operand_rank;     \
    off += indices_rank;                                                        \
    device const uint* indices_strides = config + off; off += indices_rank;      \
    off += updates_rank;                                                        \
    device const uint* updates_strides = config + off; off += updates_rank;      \
    device const uint* update_window_axes     = config + off; off += num_update_window; \
    device const uint* inserted_window_axes   = config + off; off += num_inserted; \
    device const uint* scatter_to_operand_map = config + off; off += num_scatter_map; \
    uint update_coords[16];                                                      \
    uint remaining = id;                                                         \
    for (uint d = 0; d < updates_rank; d++) {                                    \
        update_coords[d] = remaining / updates_strides[d];                      \
        remaining %= updates_strides[d];                                         \
    }                                                                            \
    uint indices_coord[16];                                                      \
    uint batch_idx = 0;                                                          \
    for (uint d = 0; d < updates_rank; d++) {                                    \
        bool is_window = false;                                                  \
        for (uint j = 0; j < num_update_window; j++) {                         \
            if (update_window_axes[j] == d) { is_window = true; break; }        \
        }                                                                        \
        if (!is_window) {                                                        \
            uint target = batch_idx;                                             \
            if (target >= index_vector_axis) target++;                           \
            if (target < indices_rank) indices_coord[target] = update_coords[d]; \
            batch_idx++;                                                         \
        }                                                                        \
    }                                                                            \
    uint operand_coord[16];                                                      \
    for (uint d = 0; d < operand_rank; d++) operand_coord[d] = 0;              \
    for (uint i = 0; i < num_scatter_map; i++) {                                 \
        indices_coord[index_vector_axis] = i;                                   \
        uint idx_flat = 0;                                                       \
        for (uint d = 0; d < indices_rank; d++) {                               \
            idx_flat += indices_coord[d] * indices_strides[d];                   \
        }                                                                        \
        int32_t start_val = indices[idx_flat];                                   \
        uint target_axis = scatter_to_operand_map[i];                            \
        int32_t max_val = int32_t(operand_dims[target_axis]) - 1;                \
        if (start_val < 0) start_val = 0;                                       \
        if (start_val > max_val) start_val = max_val;                            \
        operand_coord[target_axis] = uint(start_val);                            \
    }                                                                            \
    uint window_idx = 0;                                                         \
    for (uint d = 0; d < operand_rank; d++) {                                    \
        bool is_inserted = false;                                                \
        for (uint j = 0; j < num_inserted; j++) {                               \
            if (inserted_window_axes[j] == d) { is_inserted = true; break; }    \
        }                                                                        \
        if (!is_inserted) {                                                      \
            if (window_idx < num_update_window) {                               \
                operand_coord[d] += update_coords[update_window_axes[window_idx]]; \
            }                                                                    \
            window_idx++;                                                        \
        }                                                                        \
    }                                                                            \
    bool in_bounds = true;                                                       \
    uint dst_idx = 0;                                                            \
    for (uint d = 0; d < operand_rank; d++) {                                    \
        if (operand_coord[d] >= operand_dims[d]) { in_bounds = false; break; }  \
        dst_idx += operand_coord[d] * operand_strides[d];                        \
    }                                                                            \
    if (!in_bounds) {                                                            \
        keys[id] = 0xFFFFFFFFu;                                                    \
        vals[id] = 0;                                                            \
        return;                                                                  \
    }

kernel void scatter_max_i32(
    device int* output            [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const int* updates     [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    GOMLX_SCATTER_INDEX_BODY2
    int nv = updates[id];
    device atomic_int* atom = (device atomic_int*)(output + dst_idx);
    int ov = atomic_load_explicit(atom, memory_order_relaxed);
    while (nv > ov) {
        int expected = ov;
        if (atomic_compare_exchange_weak_explicit(
                atom, &expected, nv, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
        ov = expected;
    }
}

kernel void scatter_min_i32(
    device int* output            [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const int* updates     [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    GOMLX_SCATTER_INDEX_BODY2
    int nv = updates[id];
    device atomic_int* atom = (device atomic_int*)(output + dst_idx);
    int ov = atomic_load_explicit(atom, memory_order_relaxed);
    while (nv < ov) {
        int expected = ov;
        if (atomic_compare_exchange_weak_explicit(
                atom, &expected, nv, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
        ov = expected;
    }
}

kernel void scatter_max_u32(
    device uint* output           [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const uint* updates     [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    GOMLX_SCATTER_INDEX_BODY2
    uint nv = updates[id];
    device atomic_uint* atom = (device atomic_uint*)(output + dst_idx);
    uint ov = atomic_load_explicit(atom, memory_order_relaxed);
    while (nv > ov) {
        uint expected = ov;
        if (atomic_compare_exchange_weak_explicit(
                atom, &expected, nv, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
        ov = expected;
    }
}

kernel void scatter_min_u32(
    device uint* output           [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const uint* updates     [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    GOMLX_SCATTER_INDEX_BODY2
    uint nv = updates[id];
    device atomic_uint* atom = (device atomic_uint*)(output + dst_idx);
    uint ov = atomic_load_explicit(atom, memory_order_relaxed);
    while (nv < ov) {
        uint expected = ov;
        if (atomic_compare_exchange_weak_explicit(
                atom, &expected, nv, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
        ov = expected;
    }
}

// int64 / uint64 slow path: single GPU thread (small n).
kernel void scatter_sum_i64_serial(
    device long* output            [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const long* updates     [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    for (uint id = 0; id < num_updates; id++) {
        GOMLX_SCATTER_INDEX_BODY2_LOOP
        output[dst_idx] += updates[id];
    }
}

kernel void scatter_max_i64_serial(
    device long* output            [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const long* updates     [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    for (uint id = 0; id < num_updates; id++) {
        GOMLX_SCATTER_INDEX_BODY2_LOOP
        long nv = updates[id];
        long ov = output[dst_idx];
        if (nv > ov)
            output[dst_idx] = nv;
    }
}

kernel void scatter_min_i64_serial(
    device long* output            [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const long* updates     [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    for (uint id = 0; id < num_updates; id++) {
        GOMLX_SCATTER_INDEX_BODY2_LOOP
        long nv = updates[id];
        long ov = output[dst_idx];
        if (nv < ov)
            output[dst_idx] = nv;
    }
}

kernel void scatter_sum_u64_serial(
    device ulong* output           [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const ulong* updates    [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    for (uint id = 0; id < num_updates; id++) {
        GOMLX_SCATTER_INDEX_BODY2_LOOP
        output[dst_idx] += updates[id];
    }
}

kernel void scatter_max_u64_serial(
    device ulong* output           [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const ulong* updates    [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    for (uint id = 0; id < num_updates; id++) {
        GOMLX_SCATTER_INDEX_BODY2_LOOP
        ulong nv = updates[id];
        ulong ov = output[dst_idx];
        if (nv > ov)
            output[dst_idx] = nv;
    }
}

kernel void scatter_min_u64_serial(
    device ulong* output           [[buffer(0)]],
    device const int32_t* indices  [[buffer(1)]],
    device const ulong* updates    [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& num_updates     [[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid != 0) return;
    for (uint id = 0; id < num_updates; id++) {
        GOMLX_SCATTER_INDEX_BODY2_LOOP
        ulong nv = updates[id];
        ulong ov = output[dst_idx];
        if (nv < ov)
            output[dst_idx] = nv;
    }
}

// ─── int64/uint64 fast scatter: radix sort + segmented scan + parallel writes ─
kernel void scatter_i64_proj_keys(
    device uint* keys              [[buffer(0)]],
    device long* vals              [[buffer(1)]],
    device const int32_t* indices  [[buffer(2)]],
    device const long* updates     [[buffer(3)]],
    device const uint* config      [[buffer(4)]],
    constant uint& num_updates     [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= num_updates) return;
    GOMLX_SCATTER_INDEX_BODY2_PROJ
    keys[id] = dst_idx;
    vals[id] = updates[id];
}

kernel void scatter_u64_proj_keys(
    device uint* keys              [[buffer(0)]],
    device ulong* vals             [[buffer(1)]],
    device const int32_t* indices  [[buffer(2)]],
    device const ulong* updates    [[buffer(3)]],
    device const uint* config      [[buffer(4)]],
    constant uint& num_updates     [[buffer(5)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= num_updates) return;
    GOMLX_SCATTER_INDEX_BODY2_PROJ
    keys[id] = dst_idx;
    vals[id] = updates[id];
}

kernel void scatter_pad_dummy_keys_i64(
    device uint* keys              [[buffer(0)]],
    device long* vals              [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& npad          [[buffer(3)]],
    constant uint& pad_key_base  [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n || i >= npad) return;
    keys[i] = pad_key_base + (i - n);
    vals[i] = 0;
}

kernel void scatter_pad_dummy_keys_u64(
    device uint* keys              [[buffer(0)]],
    device ulong* vals             [[buffer(1)]],
    constant uint& n               [[buffer(2)]],
    constant uint& npad          [[buffer(3)]],
    constant uint& pad_key_base  [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i < n || i >= npad) return;
    keys[i] = pad_key_base + (i - n);
    vals[i] = 0ul;
}

// LSD radix permutation: parallel stable order for one byte of ki via bitonic sort on unique keys
// (digit << 24) | i (requires n <= 2^24). npad must be a power of 2.
kernel void scatter_radix_digit_sort_prepare(
    device const uint* kin         [[buffer(0)]],
    device uint* sort_key          [[buffer(1)]],
    device uint* perm              [[buffer(2)]],
    constant uint& shift           [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    uint d = (kin[i] >> shift) & 255u;
    sort_key[i] = (d << 24) | (i & 0xFFFFFFu);
    perm[i] = i;
}

kernel void scatter_bitonic_step_u32(
    device uint* keys              [[buffer(0)]],
    device uint* perm              [[buffer(1)]],
    constant uint& step_k          [[buffer(2)]],
    constant uint& step_j          [[buffer(3)]],
    constant uint& n               [[buffer(4)]],
    uint gid [[thread_position_in_grid]])
{
    if (gid >= n) return;
    uint ix = gid ^ step_j;
    if (ix > gid) {
        uint a = keys[gid];
        uint b = keys[ix];
        bool up = ((gid & step_k) == 0);
        bool do_swap = up ? (a > b) : (a < b);
        if (do_swap) {
            keys[gid] = b;
            keys[ix] = a;
            uint pa = perm[gid];
            uint pb = perm[ix];
            perm[gid] = pb;
            perm[ix] = pa;
        }
    }
}

kernel void scatter_gather_pair_i64(
    device const uint* kin         [[buffer(0)]],
    device const long* vin         [[buffer(1)]],
    device const uint* perm        [[buffer(2)]],
    device uint* kout              [[buffer(3)]],
    device long* vout              [[buffer(4)]],
    constant uint& n               [[buffer(5)]],
    uint p [[thread_position_in_grid]])
{
    if (p >= n) return;
    uint s = perm[p];
    kout[p] = kin[s];
    vout[p] = vin[s];
}

kernel void scatter_gather_pair_u64(
    device const uint* kin         [[buffer(0)]],
    device const ulong* vin        [[buffer(1)]],
    device const uint* perm        [[buffer(2)]],
    device uint* kout              [[buffer(3)]],
    device ulong* vout             [[buffer(4)]],
    constant uint& n               [[buffer(5)]],
    uint p [[thread_position_in_grid]])
{
    if (p >= n) return;
    uint s = perm[p];
    kout[p] = kin[s];
    vout[p] = vin[s];
}

kernel void scatter_seg_scan_i64_add(
    device const uint* keys        [[buffer(0)]],
    device const long* ain         [[buffer(1)]],
    device long* aout              [[buffer(2)]],
    constant uint& stride          [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (i >= stride && keys[i] == keys[i - stride])
        aout[i] = ain[i] + ain[i - stride];
    else
        aout[i] = ain[i];
}

kernel void scatter_seg_scan_u64_add(
    device const uint* keys        [[buffer(0)]],
    device const ulong* ain        [[buffer(1)]],
    device ulong* aout             [[buffer(2)]],
    constant uint& stride          [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (i >= stride && keys[i] == keys[i - stride])
        aout[i] = ain[i] + ain[i - stride];
    else
        aout[i] = ain[i];
}

kernel void scatter_seg_scan_i64_max(
    device const uint* keys        [[buffer(0)]],
    device const long* ain         [[buffer(1)]],
    device long* aout              [[buffer(2)]],
    constant uint& stride          [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (i >= stride && keys[i] == keys[i - stride])
        aout[i] = max(ain[i], ain[i - stride]);
    else
        aout[i] = ain[i];
}

kernel void scatter_seg_scan_i64_min(
    device const uint* keys        [[buffer(0)]],
    device const long* ain         [[buffer(1)]],
    device long* aout              [[buffer(2)]],
    constant uint& stride          [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (i >= stride && keys[i] == keys[i - stride])
        aout[i] = min(ain[i], ain[i - stride]);
    else
        aout[i] = ain[i];
}

kernel void scatter_seg_scan_u64_max(
    device const uint* keys        [[buffer(0)]],
    device const ulong* ain        [[buffer(1)]],
    device ulong* aout             [[buffer(2)]],
    constant uint& stride          [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (i >= stride && keys[i] == keys[i - stride])
        aout[i] = max(ain[i], ain[i - stride]);
    else
        aout[i] = ain[i];
}

kernel void scatter_seg_scan_u64_min(
    device const uint* keys        [[buffer(0)]],
    device const ulong* ain        [[buffer(1)]],
    device ulong* aout             [[buffer(2)]],
    constant uint& stride          [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (i >= stride && keys[i] == keys[i - stride])
        aout[i] = min(ain[i], ain[i - stride]);
    else
        aout[i] = ain[i];
}

kernel void scatter_i64_apply_sum(
    device long* output            [[buffer(0)]],
    device const uint* keys        [[buffer(1)]],
    device const long* seg_vals    [[buffer(2)]],
    constant uint& numel           [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (keys[i] >= numel || keys[i] == 0xFFFFFFFFu) return;
    bool tail = (i + 1 >= n) || (keys[i] != keys[i + 1]);
    if (!tail) return;
    output[keys[i]] += seg_vals[i];
}

kernel void scatter_i64_apply_max(
    device long* output            [[buffer(0)]],
    device const uint* keys        [[buffer(1)]],
    device const long* seg_vals    [[buffer(2)]],
    constant uint& numel           [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (keys[i] >= numel || keys[i] == 0xFFFFFFFFu) return;
    bool tail = (i + 1 >= n) || (keys[i] != keys[i + 1]);
    if (!tail) return;
    long ov = output[keys[i]];
    output[keys[i]] = max(ov, seg_vals[i]);
}

kernel void scatter_i64_apply_min(
    device long* output            [[buffer(0)]],
    device const uint* keys        [[buffer(1)]],
    device const long* seg_vals    [[buffer(2)]],
    constant uint& numel           [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (keys[i] >= numel || keys[i] == 0xFFFFFFFFu) return;
    bool tail = (i + 1 >= n) || (keys[i] != keys[i + 1]);
    if (!tail) return;
    long ov = output[keys[i]];
    output[keys[i]] = min(ov, seg_vals[i]);
}

kernel void scatter_u64_apply_sum(
    device ulong* output           [[buffer(0)]],
    device const uint* keys        [[buffer(1)]],
    device const ulong* seg_vals   [[buffer(2)]],
    constant uint& numel           [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (keys[i] >= numel || keys[i] == 0xFFFFFFFFu) return;
    bool tail = (i + 1 >= n) || (keys[i] != keys[i + 1]);
    if (!tail) return;
    output[keys[i]] += seg_vals[i];
}

kernel void scatter_u64_apply_max(
    device ulong* output           [[buffer(0)]],
    device const uint* keys        [[buffer(1)]],
    device const ulong* seg_vals   [[buffer(2)]],
    constant uint& numel           [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (keys[i] >= numel || keys[i] == 0xFFFFFFFFu) return;
    bool tail = (i + 1 >= n) || (keys[i] != keys[i + 1]);
    if (!tail) return;
    ulong ov = output[keys[i]];
    output[keys[i]] = max(ov, seg_vals[i]);
}

kernel void scatter_u64_apply_min(
    device ulong* output           [[buffer(0)]],
    device const uint* keys        [[buffer(1)]],
    device const ulong* seg_vals   [[buffer(2)]],
    constant uint& numel           [[buffer(3)]],
    constant uint& n              [[buffer(4)]],
    uint i [[thread_position_in_grid]])
{
    if (i >= n) return;
    if (keys[i] >= numel || keys[i] == 0xFFFFFFFFu) return;
    bool tail = (i + 1 >= n) || (keys[i] != keys[i + 1]);
    if (!tail) return;
    ulong ov = output[keys[i]];
    output[keys[i]] = min(ov, seg_vals[i]);
}

#undef GOMLX_SCATTER_INDEX_BODY2_PROJ
#undef GOMLX_SCATTER_INDEX_BODY2_LOOP
#undef GOMLX_SCATTER_INDEX_BODY2

// ─── ConvGeneral ────────────────────────────────────────────────────────────
// N-D convolution. Each thread computes one output element.
// Config buffer layout:
//   [0] = spatial_rank (number of spatial dimensions)
//   [1] = batch_size
//   [2] = in_channels
//   [3] = out_channels
//   [4..4+spatial_rank]                 = input_spatial_dims
//   [+spatial_rank]                     = kernel_spatial_dims
//   [+spatial_rank]                     = output_spatial_dims
//   [+spatial_rank]                     = strides
//   [+spatial_rank]                     = input_dilations
//   [+spatial_rank]                     = kernel_dilations
//   [+2*spatial_rank]                   = paddings (low, high per axis = 2*spatial_rank)
//   [+1]                               = channel_group_count

kernel void conv_general_f32(
    device const float* input      [[buffer(0)]],
    device const float* kernel_w   [[buffer(1)]],
    device float* output           [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& total_elements  [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;

    uint spatial_rank = config[0];
    uint batch_size   = config[1];
    uint in_channels  = config[2];
    uint out_channels = config[3];

    uint off = 4;
    device const uint* in_spatial  = config + off; off += spatial_rank;
    device const uint* k_spatial   = config + off; off += spatial_rank;
    device const uint* out_spatial = config + off; off += spatial_rank;
    device const uint* conv_stride = config + off; off += spatial_rank;
    device const uint* in_dilation = config + off; off += spatial_rank;
    device const uint* k_dilation  = config + off; off += spatial_rank;
    device const uint* pad_low     = config + off; off += spatial_rank;
    // pad_high not needed for forward computation
    off += spatial_rank;
    uint group_count = config[off];

    // Decompose output index: [batch, out_channels, out_spatial...]
    uint out_spatial_size = 1;
    for (uint d = 0; d < spatial_rank; d++) out_spatial_size *= out_spatial[d];

    uint spatial_idx = id % out_spatial_size;
    uint tmp = id / out_spatial_size;
    uint oc = tmp % out_channels;
    uint batch = tmp / out_channels;
    if (batch >= batch_size) return;

    // Decompose spatial index
    uint out_coords[8]; // max 8 spatial dims
    uint rem = spatial_idx;
    for (uint d = spatial_rank; d > 0; d--) {
        out_coords[d-1] = rem % out_spatial[d-1];
        rem /= out_spatial[d-1];
    }

    // Channel grouping
    uint channels_per_group = in_channels / group_count;
    uint group = oc / (out_channels / group_count);
    uint ic_start = group * channels_per_group;

    // Accumulate convolution
    float acc = 0.0f;

    // Iterate over input channels in this group
    for (uint ic_offset = 0; ic_offset < channels_per_group; ic_offset++) {
        uint ic = ic_start + ic_offset;

        // Iterate over kernel spatial positions
        uint k_total = 1;
        for (uint d = 0; d < spatial_rank; d++) k_total *= k_spatial[d];

        for (uint ki = 0; ki < k_total; ki++) {
            // Decompose kernel index
            uint k_coords[8];
            uint k_rem = ki;
            for (uint d = spatial_rank; d > 0; d--) {
                k_coords[d-1] = k_rem % k_spatial[d-1];
                k_rem /= k_spatial[d-1];
            }

            // Map to input position
            bool in_bounds = true;
            uint in_coords[8];
            for (uint d = 0; d < spatial_rank; d++) {
                int pos = int(out_coords[d] * conv_stride[d]) + int(k_coords[d] * k_dilation[d]) - int(pad_low[d]);
                if (pos < 0 || uint(pos) >= in_spatial[d] * in_dilation[d]) { in_bounds = false; break; }
                if (in_dilation[d] > 1) {
                    if (uint(pos) % in_dilation[d] != 0) { in_bounds = false; break; }
                    pos /= int(in_dilation[d]);
                }
                if (uint(pos) >= in_spatial[d]) { in_bounds = false; break; }
                in_coords[d] = uint(pos);
            }

            if (!in_bounds) continue;

            // Compute flat indices
            // Input layout: [batch, in_channels, in_spatial...]
            uint in_flat = batch;
            in_flat = in_flat * in_channels + ic;
            for (uint d = 0; d < spatial_rank; d++) {
                in_flat = in_flat * in_spatial[d] + in_coords[d];
            }

            // Kernel layout: [out_channels, channels_per_group, k_spatial...]
            uint k_flat = oc;
            k_flat = k_flat * channels_per_group + ic_offset;
            for (uint d = 0; d < spatial_rank; d++) {
                k_flat = k_flat * k_spatial[d] + k_coords[d];
            }

            acc += input[in_flat] * kernel_w[k_flat];
        }
    }

    output[id] = acc;
}

kernel void conv_general_f16(
    device const half* input      [[buffer(0)]],
    device const half* kernel_w   [[buffer(1)]],
    device half* output           [[buffer(2)]],
    device const uint* config      [[buffer(3)]],
    constant uint& total_elements  [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;

    uint spatial_rank = config[0];
    uint batch_size   = config[1];
    uint in_channels  = config[2];
    uint out_channels = config[3];

    uint off = 4;
    device const uint* in_spatial  = config + off; off += spatial_rank;
    device const uint* k_spatial   = config + off; off += spatial_rank;
    device const uint* out_spatial = config + off; off += spatial_rank;
    device const uint* conv_stride = config + off; off += spatial_rank;
    device const uint* in_dilation = config + off; off += spatial_rank;
    device const uint* k_dilation  = config + off; off += spatial_rank;
    device const uint* pad_low     = config + off; off += spatial_rank;
    off += spatial_rank;
    uint group_count = config[off];

    uint out_spatial_size = 1;
    for (uint d = 0; d < spatial_rank; d++) out_spatial_size *= out_spatial[d];

    uint spatial_idx = id % out_spatial_size;
    uint tmp = id / out_spatial_size;
    uint oc = tmp % out_channels;
    uint batch = tmp / out_channels;
    if (batch >= batch_size) return;

    uint out_coords[8];
    uint rem = spatial_idx;
    for (uint d = spatial_rank; d > 0; d--) {
        out_coords[d-1] = rem % out_spatial[d-1];
        rem /= out_spatial[d-1];
    }

    uint channels_per_group = in_channels / group_count;
    uint group = oc / (out_channels / group_count);
    uint ic_start = group * channels_per_group;

    float acc = 0.0f;

    for (uint ic_offset = 0; ic_offset < channels_per_group; ic_offset++) {
        uint ic = ic_start + ic_offset;

        uint k_total = 1;
        for (uint d = 0; d < spatial_rank; d++) k_total *= k_spatial[d];

        for (uint ki = 0; ki < k_total; ki++) {
            uint k_coords[8];
            uint k_rem = ki;
            for (uint d = spatial_rank; d > 0; d--) {
                k_coords[d-1] = k_rem % k_spatial[d-1];
                k_rem /= k_spatial[d-1];
            }

            bool in_bounds = true;
            uint in_coords[8];
            for (uint d = 0; d < spatial_rank; d++) {
                int pos = int(out_coords[d] * conv_stride[d]) + int(k_coords[d] * k_dilation[d]) - int(pad_low[d]);
                if (pos < 0 || uint(pos) >= in_spatial[d] * in_dilation[d]) { in_bounds = false; break; }
                if (in_dilation[d] > 1) {
                    if (uint(pos) % in_dilation[d] != 0) { in_bounds = false; break; }
                    pos /= int(in_dilation[d]);
                }
                if (uint(pos) >= in_spatial[d]) { in_bounds = false; break; }
                in_coords[d] = uint(pos);
            }

            if (!in_bounds) continue;

            uint in_flat = batch;
            in_flat = in_flat * in_channels + ic;
            for (uint d = 0; d < spatial_rank; d++) {
                in_flat = in_flat * in_spatial[d] + in_coords[d];
            }

            uint k_flat = oc;
            k_flat = k_flat * channels_per_group + ic_offset;
            for (uint d = 0; d < spatial_rank; d++) {
                k_flat = k_flat * k_spatial[d] + k_coords[d];
            }

            acc += float(input[in_flat]) * float(kernel_w[k_flat]);
        }
    }

    output[id] = half(acc);
}

// ─── ReduceWindow ───────────────────────────────────────────────────────────
// Pooling-style reduction over a sliding window.
// Config buffer layout:
//   [0] = rank
//   [1] = reduce_type (0=sum, 1=max, 2=min, 3=product)
//   [2..2+rank]        = input_dims
//   [+rank]            = output_dims
//   [+rank]            = window_dims
//   [+rank]            = strides
//   [+rank]            = base_dilations
//   [+rank]            = window_dilations
//   [+2*rank]          = paddings (low, high per axis)
//   [+rank]            = input_strides (flat)

kernel void reduce_window_f32(
    device const float* src       [[buffer(0)]],
    device float* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;

    uint rank         = config[0];
    uint reduce_type  = config[1];

    uint off = 2;
    device const uint* in_dims      = config + off; off += rank;
    device const uint* out_dims     = config + off; off += rank;
    device const uint* win_dims     = config + off; off += rank;
    device const uint* win_strides  = config + off; off += rank;
    device const uint* base_dil     = config + off; off += rank;
    device const uint* win_dil      = config + off; off += rank;
    device const uint* pad_low      = config + off; off += rank;
    off += rank; // skip pad_high
    device const uint* in_strides   = config + off;

    // Initialize accumulator based on reduction type
    float acc;
    switch (reduce_type) {
        case 0: acc = 0.0f; break;        // sum
        case 1: acc = -HUGE_VALF; break;   // max
        case 2: acc = HUGE_VALF; break;    // min
        case 3: acc = 1.0f; break;         // product
        default: acc = 0.0f;
    }

    // Decompose output index into coords
    uint out_coords[16];
    uint remaining = id;
    for (uint d = 0; d < rank; d++) {
        uint stride = 1;
        for (uint dd = d + 1; dd < rank; dd++) stride *= out_dims[dd];
        out_coords[d] = remaining / stride;
        remaining %= stride;
    }

    // Iterate over window
    uint win_total = 1;
    for (uint d = 0; d < rank; d++) win_total *= win_dims[d];

    for (uint wi = 0; wi < win_total; wi++) {
        // Decompose window index
        uint w_coords[16];
        uint w_rem = wi;
        for (uint d = rank; d > 0; d--) {
            w_coords[d-1] = w_rem % win_dims[d-1];
            w_rem /= win_dims[d-1];
        }

        // Map to input position
        bool valid = true;
        uint in_flat = 0;
        for (uint d = 0; d < rank; d++) {
            int pos = int(out_coords[d] * win_strides[d]) + int(w_coords[d] * win_dil[d]) - int(pad_low[d]);
            if (pos < 0) { valid = false; break; }
            uint upos = uint(pos);
            if (base_dil[d] > 1) {
                if (upos % base_dil[d] != 0) { valid = false; break; }
                upos /= base_dil[d];
            }
            if (upos >= in_dims[d]) { valid = false; break; }
            in_flat += upos * in_strides[d];
        }

        if (!valid) continue;

        float val = src[in_flat];
        switch (reduce_type) {
            case 0: acc += val; break;
            case 1: acc = max(acc, val); break;
            case 2: acc = min(acc, val); break;
            case 3: acc *= val; break;
        }
    }

    dst[id] = acc;
}

kernel void reduce_window_f16(
    device const half* src       [[buffer(0)]],
    device half* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;

    uint rank         = config[0];
    uint reduce_type  = config[1];

    uint off = 2;
    device const uint* in_dims      = config + off; off += rank;
    device const uint* out_dims     = config + off; off += rank;
    device const uint* win_dims     = config + off; off += rank;
    device const uint* win_strides  = config + off; off += rank;
    device const uint* base_dil     = config + off; off += rank;
    device const uint* win_dil      = config + off; off += rank;
    device const uint* pad_low      = config + off; off += rank;
    off += rank;
    device const uint* in_strides   = config + off;

    float acc;
    switch (reduce_type) {
        case 0: acc = 0.0f; break;
        case 1: acc = -HUGE_VALF; break;
        case 2: acc = HUGE_VALF; break;
        case 3: acc = 1.0f; break;
        default: acc = 0.0f;
    }

    uint out_coords[16];
    uint remaining = id;
    for (uint d = 0; d < rank; d++) {
        uint stride = 1;
        for (uint dd = d + 1; dd < rank; dd++) stride *= out_dims[dd];
        out_coords[d] = remaining / stride;
        remaining %= stride;
    }

    uint win_total = 1;
    for (uint d = 0; d < rank; d++) win_total *= win_dims[d];

    for (uint wi = 0; wi < win_total; wi++) {
        uint w_coords[16];
        uint w_rem = wi;
        for (uint d = rank; d > 0; d--) {
            w_coords[d-1] = w_rem % win_dims[d-1];
            w_rem /= win_dims[d-1];
        }

        bool valid = true;
        uint in_flat = 0;
        for (uint d = 0; d < rank; d++) {
            int pos = int(out_coords[d] * win_strides[d]) + int(w_coords[d] * win_dil[d]) - int(pad_low[d]);
            if (pos < 0) { valid = false; break; }
            uint upos = uint(pos);
            if (base_dil[d] > 1) {
                if (upos % base_dil[d] != 0) { valid = false; break; }
                upos /= base_dil[d];
            }
            if (upos >= in_dims[d]) { valid = false; break; }
            in_flat += upos * in_strides[d];
        }

        if (!valid) continue;

        float val = float(src[in_flat]);
        switch (reduce_type) {
            case 0: acc += val; break;
            case 1: acc = max(acc, val); break;
            case 2: acc = min(acc, val); break;
            case 3: acc *= val; break;
        }
    }

    dst[id] = half(acc);
}

// ─── Iota ───────────────────────────────────────────────────────────────────

kernel void iota_f32(device float* dst          [[buffer(0)]],
                     constant uint& batch_size  [[buffer(1)]],
                     constant uint& iota_size   [[buffer(2)]],
                     constant uint& repeat_size [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    uint total = batch_size * iota_size * repeat_size;
    if (id >= total) return;
    dst[id] = float((id / repeat_size) % iota_size);
}

kernel void iota_f16(device half* dst           [[buffer(0)]],
                     constant uint& batch_size  [[buffer(1)]],
                     constant uint& iota_size   [[buffer(2)]],
                     constant uint& repeat_size [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    uint total = batch_size * iota_size * repeat_size;
    if (id >= total) return;
    dst[id] = half((id / repeat_size) % iota_size);
}

kernel void iota_i32(device int32_t* dst        [[buffer(0)]],
                     constant uint& batch_size  [[buffer(1)]],
                     constant uint& iota_size   [[buffer(2)]],
                     constant uint& repeat_size [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    uint total = batch_size * iota_size * repeat_size;
    if (id >= total) return;
    dst[id] = int32_t((id / repeat_size) % iota_size);
}

kernel void iota_i64(device int64_t* dst        [[buffer(0)]],
                     constant uint& batch_size  [[buffer(1)]],
                     constant uint& iota_size   [[buffer(2)]],
                     constant uint& repeat_size [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    uint total = batch_size * iota_size * repeat_size;
    if (id >= total) return;
    dst[id] = int64_t((id / repeat_size) % iota_size);
}

kernel void iota_u32(device uint32_t* dst       [[buffer(0)]],
                     constant uint& batch_size  [[buffer(1)]],
                     constant uint& iota_size   [[buffer(2)]],
                     constant uint& repeat_size [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    uint total = batch_size * iota_size * repeat_size;
    if (id >= total) return;
    dst[id] = uint32_t((id / repeat_size) % iota_size);
}

kernel void iota_u64(device uint64_t* dst       [[buffer(0)]],
                     constant uint& batch_size  [[buffer(1)]],
                     constant uint& iota_size   [[buffer(2)]],
                     constant uint& repeat_size [[buffer(3)]],
                     uint id [[thread_position_in_grid]]) {
    uint total = batch_size * iota_size * repeat_size;
    if (id >= total) return;
    dst[id] = uint64_t((id / repeat_size) % iota_size);
}

// ─── ConvertDType ───────────────────────────────────────────────────────────

// Convert kind codes must match dtypeToMetalExt in executable.go.
#define LOAD_SCALAR(T, base, off) (*((device const T*)((base) + (off))))
#define STORE_SCALAR(T, base, off, value) (*((device T*)((base) + (off))) = (value))

static inline uint convert_kind_size(uint kind) {
    switch (kind) {
        case 0: return sizeof(half);
        case 1: return sizeof(float);
        case 3: return sizeof(int32_t);
        case 4: return sizeof(int64_t);
        case 5: return sizeof(uint32_t);
        case 6: return sizeof(uint64_t);
        case 7: return sizeof(uchar);
        case 8: return sizeof(int8_t);
        case 9: return sizeof(int16_t);
        case 10: return sizeof(uint8_t);
        case 11: return sizeof(uint16_t);
        default: return 0;
    }
}

static inline bool convert_load_bool(device const uchar* src, uint off, uint kind) {
    switch (kind) {
        case 0: return LOAD_SCALAR(half, src, off) != half(0.0h);
        case 1: return LOAD_SCALAR(float, src, off) != 0.0f;
        case 3: return LOAD_SCALAR(int32_t, src, off) != 0;
        case 4: return LOAD_SCALAR(int64_t, src, off) != 0;
        case 5: return LOAD_SCALAR(uint32_t, src, off) != 0u;
        case 6: return LOAD_SCALAR(uint64_t, src, off) != 0ul;
        case 7: return LOAD_SCALAR(uchar, src, off) != uchar(0);
        case 8: return LOAD_SCALAR(int8_t, src, off) != 0;
        case 9: return LOAD_SCALAR(int16_t, src, off) != 0;
        case 10: return LOAD_SCALAR(uint8_t, src, off) != 0;
        case 11: return LOAD_SCALAR(uint16_t, src, off) != 0;
        default: return false;
    }
}

static inline long convert_load_i64(device const uchar* src, uint off, uint kind) {
    switch (kind) {
        case 0: return long(float(LOAD_SCALAR(half, src, off)));
        case 1: return long(LOAD_SCALAR(float, src, off));
        case 3: return long(LOAD_SCALAR(int32_t, src, off));
        case 4: return long(LOAD_SCALAR(int64_t, src, off));
        case 5: return long(LOAD_SCALAR(uint32_t, src, off));
        case 6: return long(LOAD_SCALAR(uint64_t, src, off));
        case 7: return convert_load_bool(src, off, kind) ? 1l : 0l;
        case 8: return long(LOAD_SCALAR(int8_t, src, off));
        case 9: return long(LOAD_SCALAR(int16_t, src, off));
        case 10: return long(LOAD_SCALAR(uint8_t, src, off));
        case 11: return long(LOAD_SCALAR(uint16_t, src, off));
        default: return 0l;
    }
}

static inline ulong convert_load_u64(device const uchar* src, uint off, uint kind) {
    switch (kind) {
        case 0: return ulong(float(LOAD_SCALAR(half, src, off)));
        case 1: return ulong(LOAD_SCALAR(float, src, off));
        case 3: return ulong(LOAD_SCALAR(int32_t, src, off));
        case 4: return ulong(LOAD_SCALAR(int64_t, src, off));
        case 5: return ulong(LOAD_SCALAR(uint32_t, src, off));
        case 6: return ulong(LOAD_SCALAR(uint64_t, src, off));
        case 7: return convert_load_bool(src, off, kind) ? 1ul : 0ul;
        case 8: return ulong(LOAD_SCALAR(int8_t, src, off));
        case 9: return ulong(LOAD_SCALAR(int16_t, src, off));
        case 10: return ulong(LOAD_SCALAR(uint8_t, src, off));
        case 11: return ulong(LOAD_SCALAR(uint16_t, src, off));
        default: return 0ul;
    }
}

static inline float convert_load_f32(device const uchar* src, uint off, uint kind) {
    switch (kind) {
        case 0: return float(LOAD_SCALAR(half, src, off));
        case 1: return LOAD_SCALAR(float, src, off);
        case 3: return float(LOAD_SCALAR(int32_t, src, off));
        case 4: return float(LOAD_SCALAR(int64_t, src, off));
        case 5: return float(LOAD_SCALAR(uint32_t, src, off));
        case 6: return float(LOAD_SCALAR(uint64_t, src, off));
        case 7: return convert_load_bool(src, off, kind) ? 1.0f : 0.0f;
        case 8: return float(LOAD_SCALAR(int8_t, src, off));
        case 9: return float(LOAD_SCALAR(int16_t, src, off));
        case 10: return float(LOAD_SCALAR(uint8_t, src, off));
        case 11: return float(LOAD_SCALAR(uint16_t, src, off));
        default: return 0.0f;
    }
}

kernel void convert_dtype(device const uchar* src [[buffer(0)]],
                          device uchar* dst       [[buffer(1)]],
                          constant uint& src_kind [[buffer(2)]],
                          constant uint& dst_kind [[buffer(3)]],
                          uint id [[thread_position_in_grid]]) {
    uint src_es = convert_kind_size(src_kind);
    uint dst_es = convert_kind_size(dst_kind);
    if (src_es == 0 || dst_es == 0) return;

    uint src_off = id * src_es;
    uint dst_off = id * dst_es;

    switch (dst_kind) {
        case 0:
            STORE_SCALAR(half, dst, dst_off, half(convert_load_f32(src, src_off, src_kind)));
            break;
        case 1:
            STORE_SCALAR(float, dst, dst_off, convert_load_f32(src, src_off, src_kind));
            break;
        case 3:
            STORE_SCALAR(int32_t, dst, dst_off, int32_t(convert_load_i64(src, src_off, src_kind)));
            break;
        case 4:
            STORE_SCALAR(int64_t, dst, dst_off, int64_t(convert_load_i64(src, src_off, src_kind)));
            break;
        case 5:
            STORE_SCALAR(uint32_t, dst, dst_off, uint32_t(convert_load_u64(src, src_off, src_kind)));
            break;
        case 6:
            STORE_SCALAR(uint64_t, dst, dst_off, uint64_t(convert_load_u64(src, src_off, src_kind)));
            break;
        case 7:
            STORE_SCALAR(uchar, dst, dst_off, convert_load_bool(src, src_off, src_kind) ? uchar(1) : uchar(0));
            break;
        case 8:
            STORE_SCALAR(int8_t, dst, dst_off, int8_t(convert_load_i64(src, src_off, src_kind)));
            break;
        case 9:
            STORE_SCALAR(int16_t, dst, dst_off, int16_t(convert_load_i64(src, src_off, src_kind)));
            break;
        case 10:
            STORE_SCALAR(uint8_t, dst, dst_off, uint8_t(convert_load_u64(src, src_off, src_kind)));
            break;
        case 11:
            STORE_SCALAR(uint16_t, dst, dst_off, uint16_t(convert_load_u64(src, src_off, src_kind)));
            break;
        default:
            break;
    }
}

kernel void convert_f16_to_f32(device const half* src [[buffer(0)]],
                               device float* dst      [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = float(src[id]);
}

kernel void convert_f32_to_f16(device const float* src [[buffer(0)]],
                               device half* dst        [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = half(src[id]);
}

kernel void convert_i32_to_f32(device const int32_t* src [[buffer(0)]],
                               device float* dst         [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = float(src[id]);
}

kernel void convert_f32_to_i32(device const float* src [[buffer(0)]],
                               device int32_t* dst     [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = int32_t(src[id]);
}

kernel void convert_i64_to_f32(device const int64_t* src [[buffer(0)]],
                               device float* dst         [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = float(src[id]);
}

kernel void convert_f32_to_i64(device const float* src [[buffer(0)]],
                               device int64_t* dst     [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = int64_t(src[id]);
}

kernel void convert_f16_to_i32(device const half* src [[buffer(0)]],
                               device int32_t* dst    [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = int32_t(float(src[id]));
}

kernel void convert_i32_to_f16(device const int32_t* src [[buffer(0)]],
                               device half* dst          [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = half(float(src[id]));
}

// ─── ArgMinMax ──────────────────────────────────────────────────────────────

kernel void argmin_f32(device const float* src     [[buffer(0)]],
                       device int32_t* dst         [[buffer(1)]],
                       constant uint& prefix_size  [[buffer(2)]],
                       constant uint& reduce_size  [[buffer(3)]],
                       constant uint& suffix_size  [[buffer(4)]],
                       uint id [[thread_position_in_grid]]) {
    uint total = prefix_size * suffix_size;
    if (id >= total) return;
    uint p = id / suffix_size;
    uint s = id % suffix_size;
    uint base = p * reduce_size * suffix_size + s;
    float best = src[base];
    int32_t best_idx = 0;
    for (uint r = 1; r < reduce_size; r++) {
        float val = src[base + r * suffix_size];
        if (val < best || isnan(val)) { best = val; best_idx = int32_t(r); }
    }
    dst[id] = best_idx;
}

kernel void argmax_f32(device const float* src     [[buffer(0)]],
                       device int32_t* dst         [[buffer(1)]],
                       constant uint& prefix_size  [[buffer(2)]],
                       constant uint& reduce_size  [[buffer(3)]],
                       constant uint& suffix_size  [[buffer(4)]],
                       uint id [[thread_position_in_grid]]) {
    uint total = prefix_size * suffix_size;
    if (id >= total) return;
    uint p = id / suffix_size;
    uint s = id % suffix_size;
    uint base = p * reduce_size * suffix_size + s;
    float best = src[base];
    int32_t best_idx = 0;
    for (uint r = 1; r < reduce_size; r++) {
        float val = src[base + r * suffix_size];
        if (val > best || isnan(val)) { best = val; best_idx = int32_t(r); }
    }
    dst[id] = best_idx;
}

kernel void argmin_f16(device const half* src      [[buffer(0)]],
                       device int32_t* dst         [[buffer(1)]],
                       constant uint& prefix_size  [[buffer(2)]],
                       constant uint& reduce_size  [[buffer(3)]],
                       constant uint& suffix_size  [[buffer(4)]],
                       uint id [[thread_position_in_grid]]) {
    uint total = prefix_size * suffix_size;
    if (id >= total) return;
    uint p = id / suffix_size;
    uint s = id % suffix_size;
    uint base = p * reduce_size * suffix_size + s;
    float best = float(src[base]);
    int32_t best_idx = 0;
    for (uint r = 1; r < reduce_size; r++) {
        float val = float(src[base + r * suffix_size]);
        if (val < best || isnan(val)) { best = val; best_idx = int32_t(r); }
    }
    dst[id] = best_idx;
}

kernel void argmax_f16(device const half* src      [[buffer(0)]],
                       device int32_t* dst         [[buffer(1)]],
                       constant uint& prefix_size  [[buffer(2)]],
                       constant uint& reduce_size  [[buffer(3)]],
                       constant uint& suffix_size  [[buffer(4)]],
                       uint id [[thread_position_in_grid]]) {
    uint total = prefix_size * suffix_size;
    if (id >= total) return;
    uint p = id / suffix_size;
    uint s = id % suffix_size;
    uint base = p * reduce_size * suffix_size + s;
    float best = float(src[base]);
    int32_t best_idx = 0;
    for (uint r = 1; r < reduce_size; r++) {
        float val = float(src[base + r * suffix_size]);
        if (val > best || isnan(val)) { best = val; best_idx = int32_t(r); }
    }
    dst[id] = best_idx;
}

// ─── Transpose (general permutation) ────────────────────────────────────────
// Config (uint32): [rank, unused_total, src_dims[rank], perm[rank], src_strides[rank]]
// Row-major output with out_dims[i] = src_dims[perm[i]]. perm[i] is the input axis
// that becomes output axis i (matches backends.Transpose).

kernel void transpose_perm_bytes(
    device const uchar* src       [[buffer(0)]],
    device uchar* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    constant uint& elem_size      [[buffer(4)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    if (rank == 0 || rank > 16) return;
    device const uint* src_dims = config + 2;
    device const uint* perm = src_dims + rank;
    device const uint* src_strides = perm + rank;

    uint out_strides[16];
    uint stride = 1;
    for (int i = int(rank) - 1; i >= 0; i--) {
        uint od = src_dims[perm[i]];
        out_strides[i] = stride;
        stride *= od;
    }
    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * src_strides[perm[d]];
    }

    uint src_off = src_idx * elem_size;
    uint dst_off = id * elem_size;
    for (uint b = 0; b < elem_size; b++) {
        dst[dst_off + b] = src[src_off + b];
    }
}

kernel void transpose_perm_f32(
    device const float* src       [[buffer(0)]],
    device float* dst             [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    if (rank == 0 || rank > 16) return;
    device const uint* src_dims = config + 2;
    device const uint* perm = src_dims + rank;
    device const uint* src_strides = perm + rank;

    uint out_strides[16];
    uint stride = 1;
    for (int i = int(rank) - 1; i >= 0; i--) {
        uint od = src_dims[perm[i]];
        out_strides[i] = stride;
        stride *= od;
    }
    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * src_strides[perm[d]];
    }
    dst[id] = src[src_idx];
}

kernel void transpose_perm_f16(
    device const half* src        [[buffer(0)]],
    device half* dst              [[buffer(1)]],
    device const uint* config     [[buffer(2)]],
    constant uint& total_elements [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (id >= total_elements) return;
    uint rank = config[0];
    if (rank == 0 || rank > 16) return;
    device const uint* src_dims = config + 2;
    device const uint* perm = src_dims + rank;
    device const uint* src_strides = perm + rank;

    uint out_strides[16];
    uint stride = 1;
    for (int i = int(rank) - 1; i >= 0; i--) {
        uint od = src_dims[perm[i]];
        out_strides[i] = stride;
        stride *= od;
    }
    uint remaining = id;
    uint src_idx = 0;
    for (uint d = 0; d < rank; d++) {
        uint coord = remaining / out_strides[d];
        remaining %= out_strides[d];
        src_idx += coord * src_strides[perm[d]];
    }
    dst[id] = src[src_idx];
}

// ─── Sort: apply permutation along one axis (byte-wise; any element size) ─────
// base_off, axis_stride: element offsets; elem_size: bytes per element; axis_size: len(indices)

kernel void gather_axis_row_bytes(device const uchar* src [[buffer(0)]],
                                  device uchar* row [[buffer(1)]],
                                  constant uint4& p [[buffer(2)]],
                                  uint id [[thread_position_in_grid]]) {
    uint base = p.x, stride = p.y, n = p.z, es = p.w;
    if (id >= n) return;
    uint src_off = (base + id * stride) * es;
    uint dst_off = id * es;
    for (uint b = 0; b < es; b++) {
        row[dst_off + b] = src[src_off + b];
    }
}

kernel void scatter_axis_row_perm_bytes(device const uchar* row [[buffer(0)]],
                                        device uchar* dst [[buffer(1)]],
                                        device const int* indices [[buffer(2)]],
                                        constant uint4& p [[buffer(3)]],
                                        uint id [[thread_position_in_grid]]) {
    uint base = p.x, stride = p.y, n = p.z, es = p.w;
    if (id >= n) return;
    int idx = indices[id];
    uint dst_off = (base + uint(id) * stride) * es;
    uint src_off = uint(idx) * es;
    for (uint b = 0; b < es; b++) {
        dst[dst_off + b] = row[src_off + b];
    }
}

// Comparator prep for host-free Sort bitonic: copy scalar elements at idx[sort_i] / idx[sort_j] into lhs/rhs scratch.
kernel void sort_load_pair_bytes(device const uchar* flat [[buffer(0)]],
                                 device uchar* lhs [[buffer(1)]],
                                 device uchar* rhs [[buffer(2)]],
                                 device const int* idx [[buffer(3)]],
                                 constant uint4& base_stride_es_pad [[buffer(4)]],
                                 constant uint2& sort_ij [[buffer(5)]],
                                 uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint base = base_stride_es_pad.x;
    uint axis_stride = base_stride_es_pad.y;
    uint es = base_stride_es_pad.z;
    uint sort_i = sort_ij.x;
    uint sort_j = sort_ij.y;
    int ii = idx[sort_i];
    int jj = idx[sort_j];
    uint e0 = (base + uint(ii) * axis_stride) * es;
    uint e1 = (base + uint(jj) * axis_stride) * es;
    for (uint b = 0; b < es; b++) {
        lhs[b] = flat[e0 + b];
        rhs[b] = flat[e1 + b];
    }
}

// Odd-even: swap_when_pred: 1 = swap if pred (stable with comp(lhs,rhs)=less(keys at pair_i+1, keys at pair_i));
// 0 = swap if !pred (unstable with comp(keys_i, keys_{i+1})).
kernel void sort_adjacent_swap_idx(device int* idx_buf [[buffer(0)]],
                                    device const uchar* pred [[buffer(1)]],
                                    constant uint& pair_i [[buffer(2)]],
                                    constant uint& n [[buffer(3)]],
                                    constant uint& swap_when_pred [[buffer(4)]],
                                    uint tid [[thread_position_in_grid]]) {
    if (tid != 0) return;
    uint i = pair_i;
    if (i + 1 >= n) return;
    bool lt = pred[0] != 0;
    bool do_swap = (swap_when_pred != 0) ? lt : !lt;
    if (do_swap) {
        int t = idx_buf[i];
        idx_buf[i] = idx_buf[i + 1];
        idx_buf[i + 1] = t;
    }
}

// Bitonic exchange on index array; pred[0] is less(sort_min, sort_max) for the two sort positions (min,max) = (gid, gid^j) if gid < (gid^j).
kernel void sort_bitonic_swap_idx(device int* idx_buf [[buffer(0)]],
                                  device const uchar* pred [[buffer(1)]],
                                  constant uint& step_k [[buffer(2)]],
                                  constant uint& step_j [[buffer(3)]],
                                  constant uint& n [[buffer(4)]],
                                  uint gid [[thread_position_in_grid]]) {
    if (gid >= n) return;
    uint ix = gid ^ step_j;
    if (ix <= gid) return;
    bool lt = pred[0] != 0;
    bool up = ((gid & step_k) == 0);
    bool do_swap = up ? !lt : lt;
    if (do_swap) {
        int t = idx_buf[gid];
        idx_buf[gid] = idx_buf[ix];
        idx_buf[ix] = t;
    }
}

// RNG (PCG DXSM matching Go math/rand/v2) lives in rng_pcg.metal.
