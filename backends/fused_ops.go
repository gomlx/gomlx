// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backends

// FusedOpType identifies optional high-level operations that backends
// may implement natively. If supported, GoMLX uses the fused op instead
// of decomposing into primitives.
type FusedOpType string

const (
	// FusedOpSoftmax computes softmax along a single axis.
	// Decomposition: ReduceMax → Sub → Exp → ReduceSum → Div
	FusedOpSoftmax FusedOpType = "Softmax"

	// FusedOpLayerNorm applies layer normalization.
	// Decomposition: ReduceMean → Sub → Square → ReduceMean → Add → Sqrt → Div → Mul → Add
	FusedOpLayerNorm FusedOpType = "LayerNorm"

	// FusedOpEinsum performs Einstein summation.
	// Decomposition: DotGeneral → Transpose (varies by equation)
	FusedOpEinsum FusedOpType = "Einsum"

	// FusedOpGelu computes Gaussian Error Linear Unit.
	// Decomposition: x * 0.5 * (1 + erf(x / sqrt(2)))
	FusedOpGelu FusedOpType = "Gelu"

	// FusedOpLinear performs fused matmul + bias: y = x @ W^T + b
	// Decomposition: DotGeneral → Add
	FusedOpLinear FusedOpType = "Linear"

	// FusedOpLinearActivation performs Linear + activation in one op.
	// Decomposition: DotGeneral → Add → Activation
	FusedOpLinearActivation FusedOpType = "LinearActivation"

	// FusedOpScaledDotProductAttention computes attention scores.
	// Implements: softmax(Q @ K^T / sqrt(d)) @ V
	FusedOpScaledDotProductAttention FusedOpType = "ScaledDotProductAttention"
)

// ActivationType specifies the activation function for fused operations.
type ActivationType int

const (
	ActivationNone ActivationType = iota
	ActivationGelu
	ActivationRelu
	ActivationSilu
	ActivationTanh
)

// String returns the name of the activation type.
func (a ActivationType) String() string {
	switch a {
	case ActivationNone:
		return "none"
	case ActivationGelu:
		return "gelu"
	case ActivationRelu:
		return "relu"
	case ActivationSilu:
		return "silu"
	case ActivationTanh:
		return "tanh"
	default:
		return "unknown"
	}
}

// FusedOps defines optional high-level operations that backends may implement.
// These methods are only called if the backend declares support via Capabilities.FusedOperations.
// Backends that don't support a fused op need not implement this interface;
// the graph-level code will fall back to decomposition.
//
// A backend's Function should implement this interface to opt in to fused operations.
// The graph-level code will type-assert to FusedOps when the capability is declared.
type FusedOps interface {
	// Softmax computes softmax along the specified axis.
	// axis: single axis to compute softmax over (negative indexing supported).
	Softmax(x Value, axis int) (Value, error)

	// LayerNorm applies layer normalization over specified axes.
	// gamma, beta can be nil if no learned scale/offset.
	// epsilon: numerical stability constant (typically 1e-5).
	LayerNorm(x Value, axes []int, epsilon float64, gamma, beta Value) (Value, error)

	// Einsum performs Einstein summation with the given equation.
	// Returns error for unsupported patterns (caller should fall back to decomposition).
	Einsum(equation string, operands ...Value) (Value, error)

	// Gelu computes Gaussian Error Linear Unit activation.
	// mode: "exact" or "tanh_approximation".
	Gelu(x Value, mode string) (Value, error)

	// Linear performs fused matmul + bias: y = x @ weight^T + bias.
	// bias can be nil for no bias addition.
	Linear(x, weight, bias Value) (Value, error)

	// LinearActivation performs Linear followed by activation in one op.
	LinearActivation(x, weight, bias Value, activation ActivationType) (Value, error)

	// ScaledDotProductAttention computes: softmax(Q @ K^T / sqrt(d)) @ V.
	// q: [batch, heads, seq_q, d_k]
	// k: [batch, heads, seq_k, d_k]
	// v: [batch, heads, seq_k, d_v]
	// mask: optional attention mask (nil for no mask).
	// scale: scaling factor (typically 1/sqrt(d_k)).
	// Returns: [batch, heads, seq_q, d_v].
	ScaledDotProductAttention(q, k, v, mask Value, scale float64) (Value, error)
}
