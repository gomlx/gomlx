#include "gomlx_erf.h"

// ─── Unary kernels ──────────────────────────────────────────────────────────

#define UNARY_KERNEL(NAME, EXPR)                                               \
kernel void NAME##_f32(device const float* src [[buffer(0)]],                  \
                       device float* dst       [[buffer(1)]],                  \
                       uint id [[thread_position_in_grid]]) {                  \
    float x = src[id];                                                         \
    dst[id] = EXPR;                                                            \
}                                                                              \
kernel void NAME##_f16(device const half* src [[buffer(0)]],                   \
                       device half* dst       [[buffer(1)]],                   \
                       uint id [[thread_position_in_grid]]) {                  \
    half x = src[id];                                                          \
    dst[id] = half(EXPR);                                                      \
}

UNARY_KERNEL(op_abs,        abs(x))
UNARY_KERNEL(op_neg,        -x)
UNARY_KERNEL(op_ceil,       ceil(x))
UNARY_KERNEL(op_floor,      floor(x))
UNARY_KERNEL(op_round,      rint(x))
UNARY_KERNEL(op_sign,       sign(x))
UNARY_KERNEL(op_sqrt,       sqrt(x))
UNARY_KERNEL(op_rsqrt,      rsqrt(x))
UNARY_KERNEL(op_exp,        exp(x))
UNARY_KERNEL(op_expm1,      exp(x) - 1.0)
UNARY_KERNEL(op_log,        log(x))
UNARY_KERNEL(op_log1p,      log(1.0 + x))
UNARY_KERNEL(op_sin,        sin(x))
UNARY_KERNEL(op_cos,        cos(x))
UNARY_KERNEL(op_tanh,       tanh(x))
UNARY_KERNEL(op_erf,        gomlx_erf(float(x)))
UNARY_KERNEL(op_logistic,   1.0 / (1.0 + exp(-x)))
// Predicate outputs (uint8 / Go bool layout: 0 or 1 per element)
kernel void op_is_finite_pred_f32(device const float* src [[buffer(0)]],
                                  device uchar* dst       [[buffer(1)]],
                                  uint id [[thread_position_in_grid]]) {
    dst[id] = isfinite(src[id]) ? 1 : 0;
}
kernel void op_is_finite_pred_f16(device const half* src [[buffer(0)]],
                                  device uchar* dst      [[buffer(1)]],
                                  uint id [[thread_position_in_grid]]) {
    dst[id] = isfinite(src[id]) ? 1 : 0;
}
kernel void op_is_nan_pred_f32(device const float* src [[buffer(0)]],
                               device uchar* dst       [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = isnan(src[id]) ? 1 : 0;
}
kernel void op_is_nan_pred_f16(device const half* src [[buffer(0)]],
                               device uchar* dst      [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = isnan(src[id]) ? 1 : 0;
}

// Bitwise unary on uint32
kernel void op_bitwise_not_u32(device const uint32_t* src [[buffer(0)]],
                               device uint32_t* dst       [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = ~src[id];
}

kernel void op_clz_u32(device const uint32_t* src [[buffer(0)]],
                       device uint32_t* dst       [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    dst[id] = clz(src[id]);
}

kernel void op_bitcount_u32(device const uint32_t* src [[buffer(0)]],
                            device uint32_t* dst       [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    dst[id] = popcount(src[id]);
}

// ─── Binary kernels ─────────────────────────────────────────────────────────

#define BINARY_KERNEL(NAME, EXPR)                                              \
kernel void NAME##_f32(device const float* lhs [[buffer(0)]],                  \
                       device const float* rhs [[buffer(1)]],                  \
                       device float* dst       [[buffer(2)]],                  \
                       uint id [[thread_position_in_grid]]) {                  \
    float a = lhs[id], b = rhs[id];                                            \
    dst[id] = EXPR;                                                            \
}                                                                              \
kernel void NAME##_f16(device const half* lhs [[buffer(0)]],                   \
                       device const half* rhs [[buffer(1)]],                   \
                       device half* dst       [[buffer(2)]],                   \
                       uint id [[thread_position_in_grid]]) {                  \
    half a = lhs[id], b = rhs[id];                                             \
    dst[id] = half(EXPR);                                                      \
}

BINARY_KERNEL(op_add,   a + b)
BINARY_KERNEL(op_sub,   a - b)
BINARY_KERNEL(op_mul,   a * b)
BINARY_KERNEL(op_div,   a / b)
BINARY_KERNEL(op_pow,   pow(a, b))
BINARY_KERNEL(op_max,   max(a, b))
BINARY_KERNEL(op_min,   min(a, b))
BINARY_KERNEL(op_atan2, atan2(a, b))
BINARY_KERNEL(op_rem,   a - b * floor(a / b))

// Comparison kernels (output as float: 1.0 = true, 0.0 = false)
#define CMP_KERNEL(NAME, OP)                                                   \
kernel void NAME##_f32(device const float* lhs [[buffer(0)]],                  \
                       device const float* rhs [[buffer(1)]],                  \
                       device float* dst       [[buffer(2)]],                  \
                       uint id [[thread_position_in_grid]]) {                  \
    dst[id] = (lhs[id] OP rhs[id]) ? 1.0f : 0.0f;                            \
}                                                                              \
kernel void NAME##_f16(device const half* lhs [[buffer(0)]],                   \
                       device const half* rhs [[buffer(1)]],                   \
                       device half* dst       [[buffer(2)]],                   \
                       uint id [[thread_position_in_grid]]) {                  \
    dst[id] = (lhs[id] OP rhs[id]) ? half(1.0) : half(0.0);                  \
}

CMP_KERNEL(op_equal,            ==)
CMP_KERNEL(op_not_equal,        !=)
CMP_KERNEL(op_less,             <)
CMP_KERNEL(op_less_or_equal,    <=)
CMP_KERNEL(op_greater,          >)
CMP_KERNEL(op_greater_or_equal, >=)

// Comparison kernels with uint8 / bool output (dtype suffix still _f16/_f32 on inputs)
#define CMP_PRED_KERNEL(NAME, OP)                                              \
kernel void NAME##_f32(device const float* lhs [[buffer(0)]],                   \
                       device const float* rhs [[buffer(1)]],                   \
                       device uchar* dst       [[buffer(2)]],                  \
                       uint id [[thread_position_in_grid]]) {                  \
    dst[id] = (lhs[id] OP rhs[id]) ? 1 : 0;                                     \
}                                                                              \
kernel void NAME##_f16(device const half* lhs [[buffer(0)]],                   \
                       device const half* rhs [[buffer(1)]],                   \
                       device uchar* dst       [[buffer(2)]],                  \
                       uint id [[thread_position_in_grid]]) {                  \
    dst[id] = (lhs[id] OP rhs[id]) ? 1 : 0;                                    \
}

CMP_PRED_KERNEL(op_equal_pred, ==)
CMP_PRED_KERNEL(op_not_equal_pred, !=)
CMP_PRED_KERNEL(op_less_pred, <)
CMP_PRED_KERNEL(op_less_or_equal_pred, <=)
CMP_PRED_KERNEL(op_greater_pred, >)
CMP_PRED_KERNEL(op_greater_or_equal_pred, >=)

// Int32 elementwise (Go int32 / backends.BinaryOp on integers; dtype code 5 in metal.m)
#define I32_BIN(NAME, EXPR)                                                       \
kernel void NAME(device const int* lhs [[buffer(0)]],                             \
                 device const int* rhs [[buffer(1)]],                             \
                 device int* dst       [[buffer(2)]],                             \
                 uint id [[thread_position_in_grid]]) {                            \
    int a = lhs[id], b = rhs[id];                                                 \
    dst[id] = EXPR;                                                               \
}

I32_BIN(op_add_i32, a + b)
I32_BIN(op_sub_i32, a - b)
I32_BIN(op_mul_i32, a * b)
I32_BIN(op_div_i32, (b != 0) ? (a / b) : 0)
I32_BIN(op_rem_i32, (b != 0) ? (a % b) : 0)
I32_BIN(op_max_i32, max(a, b))
I32_BIN(op_min_i32, min(a, b))
I32_BIN(op_pow_i32, int(pow(float(a), float(b))))
I32_BIN(op_atan2_i32, int(rint(atan2(float(a), float(b)))))
I32_BIN(op_bitwise_and_i32, a & b)
I32_BIN(op_bitwise_or_i32, a | b)
I32_BIN(op_bitwise_xor_i32, a ^ b)

#define CMP_PRED_KERNEL_I32(NAME, OP)                                             \
kernel void NAME##_i32(device const int* lhs [[buffer(0)]],                        \
                       device const int* rhs [[buffer(1)]],                        \
                       device uchar* dst       [[buffer(2)]],                       \
                       uint id [[thread_position_in_grid]]) {                      \
    dst[id] = (lhs[id] OP rhs[id]) ? 1 : 0;                                        \
}

CMP_PRED_KERNEL_I32(op_equal_pred, ==)
CMP_PRED_KERNEL_I32(op_not_equal_pred, !=)
CMP_PRED_KERNEL_I32(op_less_pred, <)
CMP_PRED_KERNEL_I32(op_less_or_equal_pred, <=)
CMP_PRED_KERNEL_I32(op_greater_pred, >)
CMP_PRED_KERNEL_I32(op_greater_or_equal_pred, >=)

kernel void op_neg_i32(device const int* src [[buffer(0)]],
                       device int* dst       [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    dst[id] = -src[id];
}
kernel void op_abs_i32(device const int* src [[buffer(0)]],
                       device int* dst       [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    int x = src[id];
    dst[id] = x < 0 ? -x : x;
}
kernel void op_sign_i32(device const int* src [[buffer(0)]],
                        device int* dst       [[buffer(1)]],
                        uint id [[thread_position_in_grid]]) {
    int x = src[id];
    dst[id] = (x > 0) - (x < 0);
}
kernel void op_bitwise_not_i32(device const int* src [[buffer(0)]],
                               device int* dst       [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = ~src[id];
}
kernel void op_clz_i32(device const int* src [[buffer(0)]],
                       device int* dst       [[buffer(1)]],
                       uint id [[thread_position_in_grid]]) {
    dst[id] = int(clz(as_type<uint32_t>(src[id])));
}
kernel void op_bitcount_i32(device const int* src [[buffer(0)]],
                            device int* dst       [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    dst[id] = int(popcount(as_type<uint32_t>(src[id])));
}

// Bitwise binary on uint32
#define BITWISE_BINARY_KERNEL(NAME, OP)                                        \
kernel void NAME##_u32(device const uint32_t* lhs [[buffer(0)]],               \
                       device const uint32_t* rhs [[buffer(1)]],               \
                       device uint32_t* dst       [[buffer(2)]],               \
                       uint id [[thread_position_in_grid]]) {                  \
    dst[id] = lhs[id] OP rhs[id];                                             \
}

BITWISE_BINARY_KERNEL(op_bitwise_and, &)
BITWISE_BINARY_KERNEL(op_bitwise_or,  |)
BITWISE_BINARY_KERNEL(op_bitwise_xor, ^)

// Logical binary (on float, treating nonzero as true)
#define LOGICAL_KERNEL(NAME, OP)                                               \
kernel void NAME##_f32(device const float* lhs [[buffer(0)]],                  \
                       device const float* rhs [[buffer(1)]],                  \
                       device float* dst       [[buffer(2)]],                  \
                       uint id [[thread_position_in_grid]]) {                  \
    bool a = (lhs[id] != 0.0f), b = (rhs[id] != 0.0f);                       \
    dst[id] = (a OP b) ? 1.0f : 0.0f;                                         \
}

LOGICAL_KERNEL(op_logical_and, &&)
LOGICAL_KERNEL(op_logical_or,  ||)
LOGICAL_KERNEL(op_logical_xor, !=)

kernel void op_logical_not_f32(device const float* src [[buffer(0)]],
                               device float* dst       [[buffer(1)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = (src[id] == 0.0f) ? 1.0f : 0.0f;
}

// Logical ops on Go bool (uchar 0/1 layout). Kernel names have no type suffix; see metal.m dtype_suffix(4).
kernel void op_logical_not(device const uchar* src [[buffer(0)]],
                          device uchar* dst       [[buffer(1)]],
                          uint id [[thread_position_in_grid]]) {
    dst[id] = (src[id] == 0) ? 1 : 0;
}
kernel void op_logical_and(device const uchar* lhs [[buffer(0)]],
                          device const uchar* rhs [[buffer(1)]],
                          device uchar* dst       [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    bool a = lhs[id] != 0, b = rhs[id] != 0;
    dst[id] = (a && b) ? 1 : 0;
}
kernel void op_logical_or(device const uchar* lhs [[buffer(0)]],
                         device const uchar* rhs [[buffer(1)]],
                         device uchar* dst       [[buffer(2)]],
                         uint id [[thread_position_in_grid]]) {
    bool a = lhs[id] != 0, b = rhs[id] != 0;
    dst[id] = (a || b) ? 1 : 0;
}
kernel void op_logical_xor(device const uchar* lhs [[buffer(0)]],
                          device const uchar* rhs [[buffer(1)]],
                          device uchar* dst       [[buffer(2)]],
                          uint id [[thread_position_in_grid]]) {
    bool a = lhs[id] != 0, b = rhs[id] != 0;
    dst[id] = (a != b) ? 1 : 0;
}

// ─── Where (ternary select) ─────────────────────────────────────────────────

kernel void op_where_f32(device const float* pred [[buffer(0)]],
                         device const float* on_true [[buffer(1)]],
                         device const float* on_false [[buffer(2)]],
                         device float* dst [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0.0f) ? on_true[id] : on_false[id];
}

kernel void op_where_f16(device const half* pred [[buffer(0)]],
                         device const half* on_true [[buffer(1)]],
                         device const half* on_false [[buffer(2)]],
                         device half* dst [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != half(0.0)) ? on_true[id] : on_false[id];
}

// Bool predicate (uchar 0/1) with float payloads — keeps Where on-GPU for compare → where chains.
kernel void op_where_pred_f32(device const uchar* pred [[buffer(0)]],
                              device const float* on_true [[buffer(1)]],
                              device const float* on_false [[buffer(2)]],
                              device float* dst [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

kernel void op_where_pred_f16(device const uchar* pred [[buffer(0)]],
                               device const half* on_true [[buffer(1)]],
                               device const half* on_false [[buffer(2)]],
                               device half* dst [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

kernel void op_where_pred_i8(device const uchar* pred [[buffer(0)]],
                             device const char* on_true [[buffer(1)]],
                             device const char* on_false [[buffer(2)]],
                             device char* dst [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

kernel void op_where_pred_i16(device const uchar* pred [[buffer(0)]],
                              device const short* on_true [[buffer(1)]],
                              device const short* on_false [[buffer(2)]],
                              device short* dst [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

kernel void op_where_pred_i32(device const uchar* pred [[buffer(0)]],
                              device const int* on_true [[buffer(1)]],
                              device const int* on_false [[buffer(2)]],
                              device int* dst [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

kernel void op_where_pred_i64(device const uchar* pred [[buffer(0)]],
                              device const long* on_true [[buffer(1)]],
                              device const long* on_false [[buffer(2)]],
                              device long* dst [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

kernel void op_where_pred_u8(device const uchar* pred [[buffer(0)]],
                             device const uchar* on_true [[buffer(1)]],
                             device const uchar* on_false [[buffer(2)]],
                             device uchar* dst [[buffer(3)]],
                             uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

kernel void op_where_pred_u16(device const uchar* pred [[buffer(0)]],
                              device const ushort* on_true [[buffer(1)]],
                              device const ushort* on_false [[buffer(2)]],
                              device ushort* dst [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

kernel void op_where_pred_u32(device const uchar* pred [[buffer(0)]],
                              device const uint* on_true [[buffer(1)]],
                              device const uint* on_false [[buffer(2)]],
                              device uint* dst [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

kernel void op_where_pred_u64(device const uchar* pred [[buffer(0)]],
                              device const ulong* on_true [[buffer(1)]],
                              device const ulong* on_false [[buffer(2)]],
                              device ulong* dst [[buffer(3)]],
                              uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

kernel void op_where_pred_bool(device const uchar* pred [[buffer(0)]],
                               device const uchar* on_true [[buffer(1)]],
                               device const uchar* on_false [[buffer(2)]],
                               device uchar* dst [[buffer(3)]],
                               uint id [[thread_position_in_grid]]) {
    dst[id] = (pred[id] != 0) ? on_true[id] : on_false[id];
}

// Bool mask → float (SDPA additive mask materialization)
kernel void op_bool_to_f16(device const uchar* src [[buffer(0)]],
                           device half* dst [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    dst[id] = src[id] != 0 ? half(1.0) : half(0.0);
}

kernel void op_bool_to_f32(device const uchar* src [[buffer(0)]],
                           device float* dst [[buffer(1)]],
                           uint id [[thread_position_in_grid]]) {
    dst[id] = src[id] != 0 ? 1.0f : 0.0f;
}

// ArgMinMax int64 output: device path from int32 reduction
kernel void cast_i32_to_i64(device const int* src [[buffer(0)]],
                            device long* dst [[buffer(1)]],
                            uint id [[thread_position_in_grid]]) {
    dst[id] = (long)src[id];
}

// ─── IEEE-754 totalOrder predicates (float16/float32; sortable integer key) ──

static inline uint32_t gomlx_f32_total_order_key(float f) {
    uint32_t u = as_type<uint32_t>(f);
    return (u & 0x80000000u) ? (u ^ 0xffffffffu) : (u ^ 0x80000000u);
}

static inline uint32_t gomlx_f16_total_order_key(half h) {
    uint16_t u = as_type<uint16_t>(h);
    uint16_t k = (u & 0x8000u) ? (uint16_t)(u ^ 0xffffu) : (uint16_t)(u ^ 0x8000u);
    return (uint32_t)k;
}

#define TOTAL_ORDER_PRED(NAME, CMP)                                               \
kernel void NAME##_f32(device const float* lhs [[buffer(0)]],                    \
                       device const float* rhs [[buffer(1)]],                    \
                       device uchar* dst       [[buffer(2)]],                     \
                       uint id [[thread_position_in_grid]]) {                      \
    uint32_t ka = gomlx_f32_total_order_key(lhs[id]);                              \
    uint32_t kb = gomlx_f32_total_order_key(rhs[id]);                              \
    dst[id] = (ka CMP kb) ? 1 : 0;                                                \
}                                                                                 \
kernel void NAME##_f16(device const half* lhs [[buffer(0)]],                     \
                       device const half* rhs [[buffer(1)]],                     \
                       device uchar* dst       [[buffer(2)]],                     \
                       uint id [[thread_position_in_grid]]) {                     \
    uint32_t ka = gomlx_f16_total_order_key(lhs[id]);                              \
    uint32_t kb = gomlx_f16_total_order_key(rhs[id]);                              \
    dst[id] = (ka CMP kb) ? 1 : 0;                                                \
}

TOTAL_ORDER_PRED(op_equal_total_order_pred, ==)
TOTAL_ORDER_PRED(op_not_equal_total_order_pred, !=)
TOTAL_ORDER_PRED(op_less_total_order_pred, <)
TOTAL_ORDER_PRED(op_less_or_equal_total_order_pred, <=)
TOTAL_ORDER_PRED(op_greater_total_order_pred, >)
TOTAL_ORDER_PRED(op_greater_or_equal_total_order_pred, >=)
