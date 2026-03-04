// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package activations implements several common activations, and includes a generic Apply method to apply an
// activation by its type.
//
// There is also FromName to convert an activation name (string) to its type, and ApplyFromContext that applies
// an activation based on the hyperparameter ParamActivation defined in a context.
package activations

import (
	"math"

	"github.com/gomlx/gomlx/backends"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
)

const (
	// ParamActivation context hyperparameter defines the activation to use, for models using ApplyFromContext.
	// Available values are: `none`, `relu`, `leaky_relu`, `sigmoid`, `tanh`, `swish` (same as `silu`), `hard_swish`,
	// `selu`, `gelu` or `gelu_approx`.
	// The default is `relu`.
	// See activations.TypeValues for complete list.
	ParamActivation = "activation"
)

// Type is an enum for the supported activation functions.
//
// It is converted to snake-format strings (e.g.: TypeLeakyRelu -> "leaky_relu"), and can be converted
// from string by using
type Type int

const (
	TypeNone Type = iota
	TypeRelu
	TypeSigmoid
	TypeLeakyRelu
	TypeSelu
	TypeSwish
	TypeHardSwish

	// TypeSilu is an alias to TypeSwish
	TypeSilu

	TypeTanh

	TypeGelu
	TypeGeluApprox
)

// ToBackend converts an activations.Type to the corresponding backends.ActivationType.
// Unsupported activation types map to backends.ActivationNone.
func (t Type) ToBackend() backends.ActivationType {
	switch t {
	case TypeNone:
		return backends.ActivationNone
	case TypeGelu, TypeGeluApprox:
		return backends.ActivationGelu
	case TypeRelu:
		return backends.ActivationRelu
	case TypeSwish, TypeSilu:
		return backends.ActivationSilu
	case TypeHardSwish:
		return backends.ActivationHardSwish
	case TypeTanh:
		return backends.ActivationTanh
	default:
		return backends.ActivationNone
	}
}

//go:generate go tool enumer -type Type -trimprefix=Type -output=gen_type_enumer.go activations.go

// ApplyFromContext picks an activation function from the context using [ParamActivation] parameter,
// and applies it to x.
//
// It defaults to "relu".
func ApplyFromContext(ctx *context.Context, x *Node) *Node {
	activationName := context.GetParamOr(ctx, ParamActivation, "relu")
	return Apply(FromName(activationName), x)
}

// Apply the given activation type.
// The TypeNone activation is a no-op.
//
// See TypeValues for valid values.
func Apply(activation Type, x *Node) *Node {
	switch activation {
	case TypeNone:
		return x
	case TypeRelu:
		return Relu(x)
	case TypeLeakyRelu:
		return LeakyRelu(x)
	case TypeSigmoid:
		return Sigmoid(x)
	case TypeTanh:
		return Tanh(x)
	case TypeSwish, TypeSilu:
		return Swish(x)
	case TypeHardSwish:
		return HardSwish(x)
	case TypeSelu:
		return Selu(x)
	case TypeGelu:
		return Gelu(x)
	case TypeGeluApprox:
		return GeluApproximate(x)
	default:
		Panicf("Apply got invalid activation value %q: options are %v", activation, TypeValues())
	}
	return nil
}

// FromName converts the name of an activation to its type.
// It panics with a helpful message if name is invalid.
//
// And empty string is converted to TypeNone.
func FromName(activationName string) Type {
	if activationName == "" {
		return TypeNone
	}
	activation, err := TypeString(activationName)
	if err != nil {
		Panicf("invalid activation name %q: options are %v", activationName, TypeValues())
	}
	return activation
}

// Relu activation function. It returns Max(x, 0), and is commonly used as an activation function in neural networks.
func Relu(x *Node) *Node {
	return Max(x, ZerosLike(x))
}

// LeakyRelu activation function. It allows a small gradient when the unit is not active (x < 0).
// The `alpha` parameter is fixed at 0.3.
//
// It returns `x if x >= 0; alpha*x if x < 0`.
func LeakyRelu(x *Node) *Node {
	return LeakyReluWithAlpha(x, 0.3)
}

// LeakyReluWithAlpha activation function. It allows a small gradient when the unit is not active (x < 0).
//
// It returns `x if x >= 0; alpha*x if x < 0`.
func LeakyReluWithAlpha(x *Node, alpha float64) *Node {
	g := x.Graph()
	return Where(
		GreaterOrEqual(x, ScalarZero(g, x.DType())),
		x,
		MulScalar(x, alpha))
}

// Swish activation (or SiLU) returns `x * Sigmoid(x)`.
//
// The SiLU activation function was introduced in "Gaussian Error Linear Units
// (GELUs)" [Hendrycks et al. 2016](https://arxiv.org/abs/1606.08415) and
// "Sigmoid-Weighted Linear Units for Neural Network Function Approximation in
// Reinforcement Learning"
// [Elfwing et al. 2017](https://arxiv.org/abs/1702.03118) and was independently
// discovered (and called swish) in "Searching for Apply Functions"
// [Ramachandran et al. 2017](https://arxiv.org/abs/1710.05941)
//
// Here the beta parameter is fixed at 1.0.
func Swish(x *Node) *Node {
	return Mul(x, Sigmoid(x))
}

// HardSwish activation function.
//
// It returns x·ReLU6(x+3)/6.
//
// See [1] for details and analysis.
//
// [1]: "Evaluating Model Performance with Hard-Swish Activation Function Adjustments", https://arxiv.org/abs/2410.06879
func HardSwish(x *Node) *Node {
	g := x.Graph()
	scale := Scalar(g, x.DType(), 1.0/6.0)
	bias := Scalar(g, x.DType(), 0.5)
	return ParameterizedHardSwish(x, scale, bias)
}

// ParameterizedHardSwish activation function, it allows more flexibility in setting the HardSwish bias
// and scale parameters.
//
// It returns x·Clip((x·scale+bias), 0, 1).
//
// - x: operand to be "activated"
// - scale: scale parameter, scalar.
// - bias: bias parameter, scalar.
//
// See HardSwish for the default parameters and [1] for details and analysis.
//
// [1]: "Evaluating Model Performance with Hard-Swish Activation Function Adjustments", https://arxiv.org/abs/2410.06879
func ParameterizedHardSwish(x, scale, bias *Node) *Node {
	return Mul(x,
		ClipScalar(Add(Mul(x, scale), bias), 0, 1)) // (x·scale+bias) cliped to [0, 1].
}

const (
	SeluAlpha = 1.67326324
	SeluScale = 1.05070098
)

// Selu stands for Scaled Exponential Linear Unit (SELU) activation function is defined as:
// . $SeluScale * x$ if $x > 0$
// . $SeluScale * SeluAlpha * (e^x - 1)$ if $x < 0$
//
// Ideally, it should be matched with a "LecunNormal initializer" and the dropout variant called "AlphaDropout"
// -- TODO, neither are implemented yet.
func Selu(x *Node) *Node {
	x = Where(GreaterThan(x, ScalarZero(x.Graph(), x.DType())),
		x,
		MulScalar(MinusOne(Exp(x)), SeluAlpha),
	)
	return MulScalar(x, SeluScale)
}

// Gelu activation function, the original Gelu function.
//
// It is defined as Gelu(x) = x * 0.5 * (1 + Erf(x / √2)).
//
// The GELU activation function was introduced in "Gaussian Error Linear Units
// (GELUs)" [Hendrycks et al. 2016](https://arxiv.org/abs/1606.08415).
//
// The exact version is slower in TPUs due to the "Erf" function, but some argue it is more stable. See discussion in:
// https://github.com/jax-ml/jax/issues/4428
func Gelu(x *Node) *Node {
	return InternalFusedOpCaller(
		func() *Node { return BackendFusedGelu(x, true) },
		func() *Node {
			// Φ(x) = 0.5 * (1 + Erf(x / √2))
			cdf := MulScalar(AddScalar(Erf(DivScalar(x, math.Sqrt2)), 1), 0.5)
			return Mul(x, cdf)
		},
	)
}

// GeluApproximate is a close approximation to the original Gelu function.
//
// It is defined as Gelu(x) = x * 0.5 * (1 + Tanh(Sqrt(2/Pi) * (x+0.044715*x^3))).
//
// The GELU activation function was introduced in "Gaussian Error Linear Units
// (GELUs)" [Hendrycks et al. 2016](https://arxiv.org/abs/1606.08415).
//
// The exact version is slower in TPUs, some argue it is more stable. See discussion in:
// https://github.com/jax-ml/jax/issues/4428
func GeluApproximate(x *Node) *Node {
	cdfApprox := Add(x, MulScalar(PowScalar(x, 3), 0.044715))
	sqrt2ByPi := math.Sqrt(2.0 / math.Pi)
	cdfApprox = Tanh(MulScalar(cdfApprox, sqrt2ByPi))
	cdfApprox = MulScalar(OnePlus(cdfApprox), 0.5)
	return Mul(x, cdfApprox)
}
