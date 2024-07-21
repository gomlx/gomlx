package layers

import (
	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
)

const (
	// ParamActivation context hyperparameter defines the activation to use, for models using ActivationFromContext.
	// Available values are: `none`, `relu`, `leaky_relu`, `sigmoid`, `tanh` or `swish` (same as `silu`).
	// The default is `relu`.
	ParamActivation = "activation"
)

// ActivationFromContext picks an activation function from the context using [ParamActivation] parameter,
// and applies it to `x`.
func ActivationFromContext(ctx *context.Context, x *Node) *Node {
	activation := context.GetParamOr(ctx, ParamActivation, "relu")
	return Activation(activation, x)
}

// Activation allows a configurable activation.
// Currently supported activations are "relu", "sigmoid", "leaky_relu", "swish" (== "silu"), "tanh".
func Activation(activation string, x *Node) *Node {
	switch activation {
	case "none":
		return x
	case "relu":
		return Relu(x)
	case "leaky_relu":
		return LeakyRelu(x)
	case "sigmoid":
		return Sigmoid(x)
	case "tanh":
		return Tanh(x)
	case "swish":
		return Swish(x)
	case "silu":
		return Swish(x)
	case "selu":
		return Selu(x)
	default:
		Panicf("invalid activation type %q, valid types are: \"relu\", \"sigmoid\", \"leaky_relu\", \"selu\", \"silu\", \"swish\", \"tanh\"", activation)
	}
	return nil
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
// discovered (and called swish) in "Searching for Activation Functions"
// [Ramachandran et al. 2017](https://arxiv.org/abs/1710.05941)
//
// Here the beta parameter is fixed at 1.0.
func Swish(x *Node) *Node {
	return Mul(x, Sigmoid(x))
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
