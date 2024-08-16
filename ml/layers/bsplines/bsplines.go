// Package bsplines provide a GoMLX version of github.com/gomlx/bsplines: it provides evaluation of bsplines
// curves, that can be used as layers.
// Currently, it's best used through `gomlx/ml/layers/kan` package.
//
// It returns a computation graph that can be combined/used on other computations, e.g., to implement
// calibration layers for input of neural-networks, or for "KAN - Kolmogorov-Arnold Networks" [1]
//
// It is meant to work for batches of inputs, each example with multiple inputs and outputs, pay special
// attention to the shapes of the control points, inputs and outputs. They are documented in the [Evaluator]
// function.
//
// [1] https://arxiv.org/pdf/2404.19756
package bsplines

import (
	"github.com/gomlx/bsplines"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// Evaluate creates the computation graph to evaluate the B-splines defined by b (it's used only for the knots) and
// the controlPoints at the inputs values. Notice b and controlPoints defined multiple B-splines, see description below.
//
// Parameters:
//   - b: bsplines.BSpline with the specification of the B-spline. The control points in b is ignored, instead this
//     uses the explicitly passed controlPoints.
//   - inputs: tensor (graph.Node) with shape `[batchSize, numInputs]`: the B-spline functions are evaluated on each of the
//     examples (batchSize is the number of examples). Per example there are numInputs inputs, each one gets its
//     own B-spline, each has its own control points.
//     If inputs is a scalar value, it is automatically expanded to shape `[batchSize=1, numInputs=1]`.
//     Notice the dtype of inputs must match the dtype of controlPoints.
//   - controlPoints: tensor (graph.Node) with shape `[numInputs, numOutputs, numControlPoints]`.
//     There are effectively numInputs*numOutputs B-splines defined, each of these takes numControlPoints.
//     And `numControlPoints` must match `b.NumControlPoints()`.
//     If controlPoints is rank 1, it is expanded to shape `[numInputs=1, numOutputs=1, numControlPoints]`.
//     Any other rank is assumed to be an error.
//     Notice the dtype of controlPoints must match the dtype of inputs.
//
// The returned tensor (graph.Node) is shaped `[batchSize, numOutputs, numInputs]`. Each example in the batch
// is evaluated by numOutputs*numInputs B-spline functions.
//
// If the inputs tensor was a scalar, and numInputs==1 and numOutput==1, it returns a scalar
// as well -- for individual points inference, useful for testing.
func Evaluate(b *bsplines.BSpline, inputs, controlPoints *Node) *Node {
	// Sanity checks.
	if inputs.DType() != controlPoints.DType() {
		exceptions.Panicf("bsplines.gomlx.Evaluate() requires the inputs.dtype=%s and controlPoints.dtype=%s to be the same",
			inputs.DType(), controlPoints.DType())
	}
	if controlPoints.Rank() == 1 {
		controlPoints = ExpandDims(controlPoints, 0, 0)
	}
	if controlPoints.Rank() != 3 {
		exceptions.Panicf("bsplines.gomlx.Evaluate() requires control points to have rank 3, shape [numInputs, numOutputs, numControlPoints], instead got shape %s",
			controlPoints.Shape())
	}
	numInputs := controlPoints.Shape().Dimensions[0]
	numOutputs := controlPoints.Shape().Dimensions[1]
	numControlPoints := controlPoints.Shape().Dimensions[2]
	if numControlPoints != b.NumControlPoints() {
		exceptions.Panicf("bsplines.gomlx.Evaluate() the controlPoints (shape=%s) last dimension doesn't match the B-spline b's required control points %d",
			controlPoints.Shape(), b.NumControlPoints())
	}
	inputIsScalar := inputs.Shape().IsScalar()
	if inputIsScalar {
		inputs = Reshape(inputs, 1, 1) // `[batchSize, numInputs]`
		if numInputs != 1 {
			exceptions.Panicf("bsplines.gomlx.Evaluate() the controlPoints has shape=%s (numInputs=%d), but inputs given is a scalar, shapes don't match",
				controlPoints.Shape(), numInputs)
		}
	} else if inputs.Rank() == 2 { // `[batchSize, numInputs]`
		if inputs.Shape().Dimensions[1] != numInputs {
			exceptions.Panicf("bsplines.gomlx.Evaluate() the controlPoints (shape=%s) numInputs=%d doesn't match the inputs (%s) numInputs=%d",
				controlPoints.Shape(), numInputs, inputs.Shape(), inputs.Shape().Dimensions[1])
		}
	} else {
		exceptions.Panicf("bsplines.gomlx.Evaluate() expects inputs to be of rank=2 or a scalar, got inputs.shape=%s",
			inputs.Shape())
	}

	// Create knots constant.
	knots := ConstAsDType(inputs.Graph(), inputs.DType(), b.ExpandedKnots())
	numKnots := knots.Shape().Dimensions[0]
	knots = ExpandDims(knots, 0) // shape [1, numKnots]

	out := (&evalData{
		bspline:          b,
		graph:            inputs.Graph(),
		dtype:            inputs.DType(),
		batchSize:        inputs.Shape().Dimensions[0],
		numInputs:        numInputs,
		numOutputs:       numOutputs,
		numControlPoints: numControlPoints,
		numKnots:         numKnots,
		inputs:           inputs,
		controlPoints:    controlPoints,
		knots:            knots,
		flatInputs:       Reshape(inputs, -1, 1), // shape [batchSize*numInputs, 1]
	}).Eval()
	if numOutputs == 1 && inputIsScalar {
		out = Reshape(out) // reshape to scalar
	}
	return out
}

// evalData holds all parameters for building an B-Splines evaluation graph, after all inputs have been checked.
type evalData struct {
	bspline                                                      *bsplines.BSpline
	graph                                                        *Graph
	dtype                                                        dtypes.DType
	batchSize, numInputs, numOutputs, numControlPoints, numKnots int // dimensions
	inputs, controlPoints, knots, flatInputs                     *Node
}

func (e *evalData) Eval() *Node {
	//e.flatInputs.SetLogged("x")
	basisFlat := e.basisFunction(e.bspline.Degree())                                 // shaped [batchSize*numInputs, numKnots]
	basis := Reshape(basisFlat, e.batchSize, e.numInputs, e.numKnots)                // shaped [batchSize, numInputs, numKnots]
	basis = Slice(basis, AxisRange(), AxisRange(), AxisRange(0, e.numControlPoints)) // shaped [batchSize, numInputs, numControlPoints]
	//basis.SetLogged(fmt.Sprintf("basis[%d]", e.bspline.Degree()))

	// Carefully set up Einsum:
	// - i: batchSize, preserve
	// - j: numInputs, matched
	// - k: numControlPoints, sum reduced.
	// - l: numOutputs
	// Result: [batchSize, numOutputs, numInputs]
	output := Einsum("ijk,jlk->ilj", basis, e.controlPoints)
	if e.bspline.Extrapolation() != bsplines.ExtrapolateZero {
		// Default extrapolated values are already zero, so extrapolation only needed if != ExtrapolateZero.
		where, extrapolation := e.Extrapolate()
		output = Where(where, extrapolation, output)
	}
	return output
}

// basisFunction will return the basisFunction weights for each of the flatInputs, for each knot.
// The returned value is shaped `[batchSize*numInputs, numKnots]`.
func (e *evalData) basisFunction(degree int) *Node {
	if degree == 0 {
		// flatInputs >= knots[i] && flatInputs < knots[i+1]
		cond := And(
			GreaterOrEqual(e.flatInputs, e.knots),
			ShiftLeft(LessThan(e.flatInputs, e.knots), 1, 0.0))
		p0 := ConvertDType(cond, e.dtype) // true -> 1.0, false -> 0.0
		// after broadcasting p0 is shaped [batchSize*numInputs, numKnots]
		//p0.SetLogged("basis(0)")
		return p0
	}

	recursiveBasis := e.basisFunction(degree - 1)

	// Find knotsDelta `degree` steps ahead: replace zeros with ones for numeric safety.
	knotsDelta := Sub(Shift(e.knots, -1, ShiftDirLeft, degree), e.knots)
	//knotsDelta.SetLogged(fmt.Sprintf("knotsDelta(%d)", degree))
	knotsDeltaIsZero := Equal(knotsDelta, ZerosLike(knotsDelta))
	knotsDelta = Where(knotsDeltaIsZero, OnesLike(knotsDelta), knotsDelta)
	zeros := ZerosLike(recursiveBasis)
	broadcastToBasis := func(x *Node) *Node { return BroadcastToDims(x, zeros.Shape().Dimensions...) }
	//knotsDeltaIsZero.SetLogged(fmt.Sprintf("knotsDeltaIsZero(%d)", degree))

	weightsLeft := Div(
		Sub(e.flatInputs, e.knots),
		knotsDelta)
	weightsLeft = Where(broadcastToBasis(knotsDeltaIsZero), zeros, weightsLeft)
	left := Mul(weightsLeft, recursiveBasis)
	//left.SetLogged(fmt.Sprintf("left(%d)", degree))

	weightsRight := Sub(Shift(e.knots, -1, ShiftDirLeft, degree+1), e.flatInputs)
	weightsRight = Div(weightsRight, Shift(knotsDelta, -1, ShiftDirLeft, 1))
	weightsRight = Where(
		broadcastToBasis(Shift(knotsDeltaIsZero, -1, ShiftDirLeft, 1)),
		zeros, weightsRight)
	right := Mul(weightsRight, Shift(recursiveBasis, -1, ShiftDirLeft, 1))
	//right.SetLogged(fmt.Sprintf("right(%d)", degree))
	return Add(left, right)
}

// last returns the last element of a slice.
func last[E any](s []E) E {
	return s[len(s)-1]
}

// Extrapolate returns a boolean tensor of which values should be replaced by extrapolation, and
// the extrapolated values. Both are shaped `[batchSize, numOutput, numInput]`.
func (e *evalData) Extrapolate() (where, value *Node) {
	staticKnots := e.bspline.Knots()
	kFirst := Scalar(e.graph, e.dtype, staticKnots[0])
	kLast := Scalar(e.graph, e.dtype, last(staticKnots))

	// broadcastInputs from shape [batchSize, numInputs] to [batchSize, numOutputs, numInputs]
	broadcastInputs := func(x *Node) *Node {
		return ExpandAndBroadcast(x, []int{e.batchSize, e.numOutputs, e.numInputs}, []int{1})
	}
	expandedInputs := broadcastInputs(e.inputs)
	tooLow := LessThan(expandedInputs, kFirst)
	where = Or(
		tooLow,
		GreaterOrEqual(expandedInputs, kLast))

	// From shape [numInputs, numOutputs, 1] to [batchSize, numOutputs, numInputs].
	transposeAndBroadcastControlPoints := func(control *Node) *Node {
		// Input shape: [numInputs, numOutputs, 1]
		// Output shape: [batchSize, numOutputs, numInputs]
		control = TransposeAllDims(control, 2, 1, 0)
		control = BroadcastToDims(control, e.batchSize, e.numOutputs, e.numInputs)
		return control
	}

	switch e.bspline.Extrapolation() {
	case bsplines.ExtrapolateZero:
		// Not necessary, since values will already be zero outsize of the knots range.
		value = Zeros(e.graph, shapes.Make(e.dtype, e.batchSize, e.numOutputs, e.numInputs))

	case bsplines.ExtrapolateConstant:
		controlFirst := Slice(e.controlPoints, AxisRange(), AxisRange(), AxisElem(0))
		controlFirst = transposeAndBroadcastControlPoints(controlFirst)
		controlLast := Slice(e.controlPoints, AxisRange(), AxisRange(), AxisElem(-1))
		controlLast = transposeAndBroadcastControlPoints(controlLast)
		value = Where(tooLow, controlFirst, controlLast)

	case bsplines.ExtrapolateLinear:
		// Low -> for values below the first knot.
		// High -> for values above the last knot.

		// Shapes: [numInputs, numOutputs, 1]
		lowKnotRatio, highKnotRatio := e.bspline.LinearExtrapolationKnotRatios()
		lowStart := Slice(e.controlPoints /*numInputs*/, AxisRange() /*numOutputs*/, AxisRange(), AxisElem(0))
		lowLinearCoefficient := Sub(
			Slice(e.controlPoints /*numInputs*/, AxisRange() /*numOutputs*/, AxisRange(), AxisElem(1)),
			lowStart)
		lowLinearCoefficient = MulScalar(lowLinearCoefficient, lowKnotRatio)
		highStart := Slice(e.controlPoints /*numInputs*/, AxisRange() /*numOutputs*/, AxisRange(), AxisElem(-1))
		highLinearCoef := Sub(
			highStart,
			Slice(e.controlPoints /*numInputs*/, AxisRange() /*numOutputs*/, AxisRange(), AxisElem(-2)))
		highLinearCoef = MulScalar(highLinearCoef, highKnotRatio)

		// Shapes: [batchSize, numInputs]
		lowDelta := AddScalar(e.inputs, -staticKnots[0])     // x - knots[0], a negative number if x < knots[0]
		highDelta := AddScalar(e.inputs, -last(staticKnots)) // x - knots[-1]

		// Broadcast everything to [batchSize, numOutputs, numInputs]
		lowLinearCoefficient = transposeAndBroadcastControlPoints(lowLinearCoefficient)
		lowStart = transposeAndBroadcastControlPoints(lowStart)
		highLinearCoef = transposeAndBroadcastControlPoints(highLinearCoef)
		highStart = transposeAndBroadcastControlPoints(highStart)
		lowDelta = broadcastInputs(lowDelta)
		highDelta = broadcastInputs(highDelta)

		// Calculate linear extrapolations:
		lowExtrapolation := Add(
			Mul(lowDelta, lowLinearCoefficient),
			lowStart)
		highExtrapolation := Add(
			Mul(highDelta, highLinearCoef),
			highStart)
		value = Where(tooLow, lowExtrapolation, highExtrapolation)
	}
	return
}
