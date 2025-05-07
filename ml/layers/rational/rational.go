// Package rational implements "learnable rational functions".
//
// They can be used for activations or simply as a univariate learnable function -- they are used for
// KANs (Kolmogorov-Arnold Network) in the KAT (Kolmogorov-Arnold Transformer) paper [1].
//
// Rational functions take the form of f(x) = w*P(x)/Q(x), where P(x) and Q(x) are polynomial functions
// on x of order m/n, or for short, degree m/n
//
// See details in New.
//
// Several sources of inspiration for this implementation:
//
// [1] "Kolmogorov-Arnold Transformer" by Xingyi Yang and Xinchao Wang, https://arxiv.org/abs/2409.10594
// [2] https://github.com/ml-research/rational_activations/
// [3] "Padé Activation Units: End-to-end Learning of Flexible Activation Functions in Deep Networks" by
//
//	Alejandro Molina, Patrick Schramowski, Kristian Kersting, https://arxiv.org/abs/1907.06732
//
// [4] "Rational neural networks" by Nicolas Boullé, Yuji Nakatsukasa, Alex Townsend, https://arxiv.org/abs/2004.01902
package rational

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"math"
)

// Config holds the configuration for a learnable rational function.
//
// Call its several methods to configure, and Done when configuration is finished to apply the rational function.
type Config struct {
	ctx   *context.Context
	input *Node

	// numInputGroups must be a divisor of the feature (last) dimension of input, and allows multiple inputs to share the same learnable rational function.
	// The default is one function per feature dimension.
	numInputGroups int

	numOutputsPerInput int

	// Polynomial orders
	numeratorDegree, denominatorDegree int
	version                            string
	initApproximation                  string

	useMultiplier        bool
	wInitializerVariance float64

	noiseDeviation float64

	numeratorInit, denominatorInit *tensors.Tensor
}

// New creates the configuration for a "learnable rational function".
//
// They can be used for activations or simply as a univariate learnable function -- they are used for
// KANs (Kolmogorov-Arnold Network) in the KAT (Kolmogorov-Arnold Transformer) paper [1].
//
// It comes with sane defaults, but it can be further configured using the various configuration methods.
// Once configuration is finished, call Config.Done and it will return the application of the resulting rational function.
//
// By default, it is configured to create one learnable function per input feature (the last dimension of x).
// So if x has shape [batch_size=64, feature_dim=32], it will create 32 functions initialized the same, but which will
// learn separately. See Config.WithInputGroups and Config.NumOutputs for other options.
//
// New returns a Config object that can be further configured. Once finished, call Config.Done and it will
// return the result of "rational(x)" with the same as x.Shape(), except if configured with a Config.WithMultipleOutputs,
// in which case there is an extra output axis with dimension equal to the number of the outputs.
func New(ctx *context.Context, x *Node) *Config {
	if x.IsScalar() {
		x = Reshape(x, 1)
	}
	return (&Config{ctx: ctx, input: x}).
		WithInputGroups(0).
		WithMultipleOutputs(1).
		WithDegrees(5, 4).
		Version("B").
		Approximate(IdentityApproximation).
		WithNoise(0.1)
}

// WithInputGroups allows multiple inputs to share the same learnable rational function.
// The numInputGroups must be a divisor of the features dimension, that is, the last dimension of the input x.
//
// So if x is shaped [64, 32], and numInputGroups == 2, the inputs x[:, 0:16] uses one learnable function, and
// x[:, 16:] uses the second.
//
// If numInputGroups is 0, the default, numInputGroups is set to x.Shape().Dim(-1), that is, one function per
// input feature.
func (c *Config) WithInputGroups(numInputGroups int) *Config {
	if numInputGroups == 0 {
		if c.input.IsScalar() {
			numInputGroups = 1
		} else {
			numInputGroups = c.input.Shape().Dim(-1)
		}
	}
	c.numInputGroups = numInputGroups
	return c
}

// WithMultipleOutputs allows the layer to generate multiple outputs per input feature (last dimension of the input x).
// This can be useful for instance for KANs (see KAT [1]) using rational functions, where each input feature is used
// (with its own learnable rational function) in calculating each of the outputs.
//
// The default is 1, so the output is the same size and shape as the input.
//
// If the value is different from 1, the output of Config.Done will have one extra axis appended to the end
// with dimension equal to numOutputsPerInput.
func (c *Config) WithMultipleOutputs(numOutputsPerInput int) *Config {
	c.numOutputsPerInput = numOutputsPerInput
	return c
}

// WithDegrees configures the degree of the rational functions.
// It defaults to 5,4 (numerator is 5, denominator is 4).
func (c *Config) WithDegrees(numerator, denominator int) *Config {
	c.numeratorDegree = numerator
	c.denominatorDegree = denominator
	return c
}

// Version of Rational to use. Rational(x) = w*P(x)/Q(x), where
//
//		P(x) = (a_0 + a_1 * x + a_2 * x^2 + ... + a_n * x^n) and
//
//	  - "A": Q(x) = (1 + |b_0 * x| + | b_1 * x^2| + ... +  | b_m * x^{m+1}|)
//	  - "B": Q(x) = (1 + |b_0 * x + b_1 * x^2 + ... + b_m * x^{m + 1}|)
//	  - "C": Q(x) = (0.1 + |b_0 + b_1 * x + b_2 * x^2 + ... + b_m * x^m|)
//	  - "D": like `B` with noised coefficients a_i and b_i. See WithRandomDeviation to set the amount of noise.
//	    Noise only applied during training. No noise during inference.
//
// Based on https://github.com/ml-research/rational_activations/blob/master/rational/keras/rationals.py, using the
// same version notation for compatibility.
//
// Default is version "B".
func (c *Config) Version(version string) *Config {
	c.version = version
	return c
}

// WithMultiplier if set adds a learnable multiplier weight w that multiplies the rational function P(x)/Q(x).
// Default is 0, that means, no multiplier term.
func (c *Config) WithMultiplier(useMultiplier bool) *Config {
	c.useMultiplier = useMultiplier
	return c
}

// WithMultiplierInitVariance defines the variance of the normal distribution used to initialize the values of w,
// the multiplier.
//
// Set initVariance to 0 to have a default initializer variance be selected for you, based on the
// Config.Approximate function chosen, in order to keep the variance constant layer over layer.
//
// See KAT/GR-KAN paper [1] for details.
func (c *Config) WithMultiplierInitVariance(initializerVariance float64) *Config {
	c.wInitializerVariance = initializerVariance
	return c
}

// WithNoise sets that amount of uniform noise (around 0.0) to add to the coefficients in version "D".
// This only has an impact if version "D" is selected.
//
// Default is 0.1.
func (c *Config) WithNoise(randomDeviation float64) *Config {
	c.noiseDeviation = randomDeviation
	return c
}

// Approximate takes as input an activation function name (see package activations for valid names), and it will
// initialize the parameters of the rational function such that it approximates the given function.
//
// The rational package contains a table of various (version, degrees, activation) initial values, and if the combination
// is not there, it will fail when Done is called. But it's easy to generate new values for new combinations, see
// notebook https://github.com/gomlx/gomlx/blob/main/ml/layers/rational/rational.ipynb . In this notebook
// you enter the univariate function you want to learn (the `target` function), and it approximates it
// and generates a cache entry, that can be added to the `cache.go` file, or you can enter the values
// manually with Config.WithInitialValues and Config.WithMultiplierInitVariance.
//
// The default is "identity" (IdentityApproximation), and its alias "".
//
// See also Config.WithInitialValues.
func (c *Config) Approximate(activation string) *Config {
	c.initApproximation = activation
	return c
}

// IdentityApproximation is a value to use in Config.Approximate to initialize the rational function with an identity function.
const IdentityApproximation = "identity"

// WithInitialValues takes the given tensors as inputs for the numerator and denominators learnable coefficients.
//
// The shape of the numerator should be 1+degree(numerator), and the denominator should be degree(denominator) --
// there is one less parameter in the denominator.
//
// If set, this supersedes Config.Approximate.
//
// By default, this is unset (nil).
func (c *Config) WithInitialValues(numeratorInit, denominatorInit *tensors.Tensor) *Config {
	c.numeratorInit = numeratorInit
	c.denominatorInit = denominatorInit
	return c
}

// Done creates and applies the learnable rational function configured, returning the result of Rational(x).
//
// The returned shape is the same as x.Shape(), except if configured with a Config.WithMultipleOutputs, in which
// case there is an extra output axis with dimension equal to the number of the outputs.
func (c *Config) Done() *Node {
	if c.numeratorDegree <= 0 || c.denominatorDegree <= 0 {
		exceptions.Panicf("rational functions requires degrees >= 1, got numerator/denominator degrees = %d/%d",
			c.numeratorDegree, c.denominatorDegree)
	}

	// Aliases.
	ctx := c.ctx
	g := c.input.Graph()
	dtype := c.input.DType()
	x := c.input

	// Make x shaped [batchSize, inputDim]
	if x.IsScalar() {
		x = Reshape(x, 1, 1)
	}
	inputDim := x.Shape().Dim(-1)
	if x.Rank() != 2 {
		x = Reshape(x, -1, inputDim)
	}
	batchSize := x.Shape().Dim(0)

	// Aggregate input into groups that share the same kernel (learnable rational function)
	// x will be shaped [batchSize, numInputGroups, inputGroupSize]
	numInputGroups := c.numInputGroups
	if numInputGroups <= 0 {
		// default to no input groups.
		numInputGroups = inputDim
	}
	if c.numInputGroups <= 0 || inputDim%c.numInputGroups != 0 || c.numInputGroups > inputDim {
		exceptions.Panicf("rational input is shaped %s (features dimension=%d): it cannot be organized in %d input groups -- it must be a divisor.",
			c.input.Shape(), inputDim, c.numInputGroups)
	}
	inputGroupSize := inputDim / numInputGroups
	x = Reshape(x, batchSize, numInputGroups, inputGroupSize)

	// Fetch initialization values from cache, if one is provided.
	numeratorInit, denominatorInit, gain := c.getInitTensorsAndGain()

	// Scalar multiplier w shaped [numInputGroups, numOutputsPerInput].
	outputDim := c.numOutputsPerInput
	var w *Node
	if c.useMultiplier {
		wInitializerVariance := c.wInitializerVariance
		if wInitializerVariance <= 0 {
			if gain >= 0 {
				wInitializerVariance = gain / float64(inputDim)
			} else {
				// Default variance for unknown approximations:
				wInitializerVariance = 1.0
			}
		}
		wInitializerStddev := math.Sqrt(wInitializerVariance)
		w = ctx.WithInitializer(initializers.RandomNormalFn(ctx, wInitializerStddev)).
			VariableWithShape("w", shapes.Make(dtype, outputDim, numInputGroups)).ValueGraph(g)
	}

	// Numerator/Denominator coefficients shaped [outputDim, numInputGroups, <degree>].
	if c.numeratorInit != nil {
		numeratorInit = c.numeratorInit
	}
	if c.denominatorInit != nil {
		denominatorInit = c.denominatorInit
	}
	if numeratorInit == nil || denominatorInit == nil {
		exceptions.Panicf("rational function approximation %q for algorithm %q and degrees %d/%d not known, "+
			"see documentation in method rational.Config.WithApproximate on how to find the initial values and either set them "+
			"with rational.Config.WithInitialValues() or add them to the cache table for future use",
			c.initApproximation, c.version, c.numeratorDegree, c.denominatorDegree)
	}
	numeratorCoeffs := ctx.WithInitializer(initializers.BroadcastTensorToShape(numeratorInit)).
		VariableWithShape("numeratorCoeffs", shapes.Make(dtype, outputDim, numInputGroups, c.numeratorDegree+1)).
		ValueGraph(g)
	denominatorCoeffs := ctx.WithInitializer(initializers.BroadcastTensorToShape(denominatorInit)).
		VariableWithShape("denominatorCoeffs", shapes.Make(dtype, outputDim, numInputGroups, c.denominatorDegree)).
		ValueGraph(g)

	// Version "D" adds noise to coefficients.
	if c.version == "D" && ctx.IsTraining(g) {
		// In version "D", if training, apply noise to the coefficients.
		for _, vRef := range []**Node{&numeratorCoeffs, &denominatorCoeffs} {
			noise := ctx.RandomUniform(g, (*vRef).Shape()) // Uniform [0, 1]
			noise = AddScalar(MulScalar(noise, 2), -1)     // Uniform [-1,1]
			noise = MulScalar(noise, c.noiseDeviation)     // Uniform [-noiseDeviation, +noiseDeviation]
			noise = AddScalar(noise, 1.0)
			*vRef = Mul(*vRef, noise)
		}
	}

	// Creating powers of x:
	maxDegree := max(c.numeratorDegree, c.denominatorDegree)
	expandedX := InsertAxes(x, -1)
	powersOfX := []*Node{OnesLike(expandedX), expandedX}
	powerOfX := expandedX
	for range maxDegree - 1 {
		powerOfX = Mul(powerOfX, expandedX)
		powersOfX = append(powersOfX, powerOfX)
	}
	// numerator includes x^0 (1.0)
	numPowersOfX := Concatenate(powersOfX[0:c.numeratorDegree+1], -1)
	denPowersOfX := Concatenate(powersOfX[1:c.denominatorDegree+1], -1)

	// Numerator:
	//   Versions 'A', 'B', 'C': P(x) = a_0 + a_1*x + a_2*x^2 + ... + a_n*x^n
	//   Version 'D': P(X) = noised(a_0) + noised(a_1) * x + noised(a_2) * x^2 + ... + noised(a_n) * x^n
	//   (noise only applied if training).
	//
	// Einsum axes:
	//   B -> batchSize
	//   O -> numOutputsPerInput
	//   I -> numInputGroups
	//   G -> inputGroupSize
	//   N -> # numerator coefficients == #numPowersOfX
	Px := Einsum("BIGN,OIN->BOIG", numPowersOfX, numeratorCoeffs)

	// Denominator: depends on the version:
	// - "A": Q(x) = (1 + |b_0 * x| + | b_1 * x^2| + ... +  | b_m * x^{m+1}|)
	// - "B": Q(x) = (1 + |b_0 * x + b_1 * x^2 + ... + b_m * x^{m + 1}|)
	// - "C": Q(x) = (0.1 + |b_0 + b_1 * x + b_2 * x^2 + ... + b_m * x^m|)
	// - "D": like `B` with noised coefficients b_i
	//
	// Einsum axes:
	//   B -> aggregated batch size of the input
	//   O -> numOutputsPerInput
	//   I -> numInputGroups
	//   G -> inputGroupSize
	//   N -> # numerator coefficients == #denPowersOfX
	var Qx *Node
	switch c.version {
	case "A":
		Qx = Einsum("BIGN,OIN->BOIGN", denPowersOfX, denominatorCoeffs)
		Qx = ReduceSum(Abs(Qx), -1)
		Qx = OnePlus(Qx)
	case "B", "D":
		Qx = Einsum("BIGN,OIN->BOIG", denPowersOfX, denominatorCoeffs)
		Qx = OnePlus(Abs(Qx))
	case "C":
		Qx = Einsum("BIGN,OIN->BOIG", denPowersOfX, denominatorCoeffs)
		Qx = AddScalar(Abs(Qx), 0.1)
	default:
		exceptions.Panicf("rational functions: unknown version %s", c.version)
	}

	// Output: w.P(x)/Q(x), shaped [batchSize, outputDim, numInputGroups, inputGroupSize]
	output := Div(Px, Qx)
	if w != nil {
		// Multiply by learnable scalar.
		output = Einsum("BOIG,OI->BOIG", output, w)
	}
	output.AssertDims(batchSize, outputDim, numInputGroups, inputGroupSize)

	// Regroup input groups and squeeze output, if numOutputsPerInput == 1.
	if c.numOutputsPerInput == 1 {
		output = ReshapeWithShape(output, c.input.Shape())
	} else {
		newDims := make([]int, c.input.Rank()+1)
		copy(newDims, c.input.Shape().Dimensions)
		newDims[c.input.Rank()-1] = outputDim
		newDims[c.input.Rank()] = inputDim // Presumably, the caller will want to reduce over the inputDim.
		output = Reshape(output, newDims...)
	}
	return output
}
