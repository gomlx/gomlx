package graph

import (
	"github.com/gomlx/gomlx/types/shapes"
	"golang.org/x/exp/constraints"
)

// File with image manipulation tools

func scalarMin[T constraints.Ordered](a, b T) T {
	if b > a {
		return a
	}
	return b
}

// InterpolationConfig is created with Interpolate and then actually executed
// with a call to Done.
//
// Between the its construction and execution one can set the various parameters for
// the interpolation.
type InterpolationConfig struct {
	g                                         *Graph
	input                                     *Node
	alignCorner, isBilinear, halfPixelCenters bool
	outputSizes                               []int
}

// Interpolate will interpolate the tensor to the given output sizes. The outputSizes should have the same
// rank as the image, and can have values set to NoInterpolation (`== -1`) for dimensions that shouldn't be changed.
//
// Example:
//
//	image.AssertDims(100, 72, 72, 3)  // Shape `[batch_size, height=72, width=72, depth]`
//	image = Interpolate(image, -1, 64, 64, -1).Bilinear().Done()
//	image.AssertDims(100, 64, 64, 3)
//
// Interpolate will return an InterpolationConfig that can be configured. When Done() is called it builds the
// graph for the interpolation and returns the interpolated tensor. The default set up is using Bilinear
// interpolation and HalfPixelCenters set to true.
//
// This can be used for images (2D) but also for anything volumetric (3D or higher) or also for time-series (1D).
//
// The implementation is based on the Tensorflow `tf2xla` one.
func Interpolate(input *Node, outputSizes ...int) *InterpolationConfig {
	return &InterpolationConfig{
		g:                input.Graph(),
		input:            input,
		isBilinear:       true,
		alignCorner:      false,
		halfPixelCenters: true,
		outputSizes:      outputSizes,
	}
}

// Bilinear configures the interpolation to be bilinear (as opposed to nearest). Default is Bilinear.
// See also Nearest.
//
// It returns the InterpolationConfig passed, to allow cascaded method calls.
func (c *InterpolationConfig) Bilinear() *InterpolationConfig {
	c.isBilinear = true
	return c
}

// Nearest configures the interpolation to be bilinear (as opposed to nearest). Default is Bilinear.
// See also Bilinear.
//
// It returns the InterpolationConfig passed, to allow cascaded method calls.
func (c *InterpolationConfig) Nearest() *InterpolationConfig {
	c.isBilinear = false
	return c
}

// AlignCorner configures the interpolation to be value of "align corner": if set to true, the input and output
// tensors corner pixels are aligned at the center points of their corner pixels, preserving the values at the
// corner pixels. If set to false, the input and output tensors are aligned by the corner points of their
// corner pixels, and the interpolation uses edge value padding for out-of-boundary values. Default is false.
//
// One cannot select both, HalfPixelCenters(true) and AlignCorner(true).
//
// Default is true.
//
// It returns the InterpolationConfig passed, to allow cascaded method calls.
func (c *InterpolationConfig) AlignCorner(alignCorner bool) *InterpolationConfig {
	c.alignCorner = alignCorner
	return c
}

// HalfPixelCenters is used, if set. Defaults to true.
//
// One cannot select both, HalfPixelCenters(true) and AlignCorner(true).
//
// Default is false.
//
// It returns the InterpolationConfig passed, to allow cascaded method calls.
func (c *InterpolationConfig) HalfPixelCenters(halfPixelCenters bool) *InterpolationConfig {
	c.halfPixelCenters = halfPixelCenters
	return c
}

// NoInterpolation can be used for the outputSizes of the Interpolation call.
const NoInterpolation = int(-1)

// Done finishes the configuration of the interpolation and creates the computation graph that resizes
// the input to the given output sizes. It returns the resized input.
//
// Any errors are returned in the graph.
func (c *InterpolationConfig) Done() (output *Node) {
	g := c.g
	output = g.InvalidNode()
	if !g.Ok() {
		return
	}

	if c.alignCorner && c.halfPixelCenters {
		g.SetErrorf("invalid Interpolate configuration, one cannot set alignCorner and halfPixelCenters " +
			"to true at the same time")
		return
	}

	// Find axes that are going to be interpolated and the output shape.
	input := c.input
	inputShape := input.Shape()
	dtype := input.DType()
	outputSizes := c.outputSizes
	if len(outputSizes) != inputShape.Rank() {
		g.SetErrorf("Output sizes for interpolation has a different rank (%v) than the input (%s)",
			outputSizes, input.Shape())
		return
	}

	// Find axisToInterpolate, and their target dimensions in interpolationDims:
	axisToInterpolate := make([]int, 0, input.Rank())
	outputShape := inputShape.Copy()
	interpolationDims := make([]int, 0, input.Rank()+1)
	for axis, s := range outputSizes {
		if s != NoInterpolation && s <= 0 {
			g.SetErrorf("Output sizes set to invalid value (%d <= 0): %v", s, outputSizes)
			return
		}
		if s != NoInterpolation && s != inputShape.Dimensions[axis] {
			axisToInterpolate = append(axisToInterpolate, axis)
			outputShape.Dimensions[axis] = s
			interpolationDims = append(interpolationDims, s)
		}
	}

	// Find slices and weights for each interpolation axis:
	spanStarts := make([]*Node, 0, len(axisToInterpolate))
	spanSizes := make([]int, 0, len(axisToInterpolate))
	weights := make([]*Node, 0, len(axisToInterpolate))
	for axisIdx, axis := range axisToInterpolate {
		inSize := inputShape.Dimensions[axis]
		outSize := outputShape.Dimensions[axis]
		var scale float64
		if c.alignCorner && outSize > 1 {
			// If alignCorner, then range is one less the size, since the half corner at the start and end
			// of the axis will remain fixed (or "unscaled").
			scale = float64(inSize-1) / float64(outSize-1)
		} else {
			scale = float64(inSize) / float64(outSize)
		}

		// Find the range start in the input from where to interpolate.
		spanStart := IotaFull(g, shapes.Make(dtype, outSize))
		if c.halfPixelCenters {
			spanStart = AddScalar(spanStart, 0.5)
		}
		sampleFraction := MulScalar(spanStart, scale)
		if c.isBilinear {
			spanStart = AddScalar(sampleFraction, -1)
			if c.halfPixelCenters {
				spanStart = AddScalar(spanStart, -0.5)
			}
			spanStart = Ceil(spanStart)
		} else {
			if c.alignCorner {
				spanStart = Round(sampleFraction)
			} else {
				spanStart = Floor(sampleFraction)
			}
		}

		// Find the range size in the input that needs interpolation.
		spanSize := 1
		if c.isBilinear {
			spanSize = scalarMin(inSize, 3)
		}
		upperBound := MulScalar(OnesLike(spanStart), float64(inSize-spanSize))
		if !c.isBilinear && !c.halfPixelCenters {
			spanStart = Min(spanStart, upperBound)
		} else {
			spanStart = Clip(spanStart, ZerosLike(spanStart), upperBound)
		}
		spanSizes = append(spanSizes, spanSize)

		// Broadcast spanStart to common shape so it can be combined with other interpolation axes.
		// The final shape will be [interpolationDims..., 1].
		broadcastSpanStart := ConvertType(spanStart, shapes.I64)
		for ii := 0; ii < axisIdx; ii++ {
			broadcastSpanStart = ExpandDims(broadcastSpanStart, 0)
		}
		for ii := axisIdx + 1; ii < len(axisToInterpolate); ii++ {
			broadcastSpanStart = ExpandDims(broadcastSpanStart, -1)
		}
		broadcastSpanStart = BroadcastToDims(broadcastSpanStart, interpolationDims...)
		broadcastSpanStart = ExpandDims(broadcastSpanStart, -1) // One last index of size 1, we will concatenate on this new axis.
		spanStarts = append(spanStarts, broadcastSpanStart)

		// Weights
		var weight *Node
		if c.isBilinear {
			weight = Sub(spanStart, sampleFraction)
			weight = BroadcastToDims(ExpandDims(weight, -1), outSize, spanSize)
			offset := Iota(g, shapes.Make(dtype, 1, spanSize), 1)
			weight = Add(weight, offset)
			if c.halfPixelCenters {
				weight = AddScalar(weight, 0.5)
			}
			weight = MaxScalar(OneMinus(Abs(weight)), 0)
		} else {
			weight = Ones(g, shapes.Make(dtype, outSize, spanSize))
		}
		// Normalize (probably not needed for bilinear).
		normalization := ReduceAndKeep(weight, ReduceSum, 1)
		weight = Div(weight, normalization)
		weights = append(weights, weight)
	}

	// gatheredElements will be shaped [interpolationDims..., spanSizes...]
	spanStart := Concatenate(spanStarts, -1)
	gatheredElements := GatherSlices(input, axisToInterpolate, spanStart, spanSizes)
	_ = gatheredElements

	// weightsCrosses of all the weights
	var weightsCrosses *Node
	for _, w := range weights {
		if weightsCrosses == nil {
			weightsCrosses = w
			continue
		}
		weightsCrosses = EinsumAxes(weightsCrosses, w, nil, nil)
	}
	numAxisToInterpolate := len(axisToInterpolate)
	weightsCrosses.AssertRank(2 * numAxisToInterpolate)

	// broadcastWeights transpose weightsCrosses and expand the dimensions to match the same as gatheredElements.
	// First we need to transpose all the interpolated dimensions to the the front.
	transposeDims := make([]int, 0, 2*numAxisToInterpolate)
	for ii := 0; ii < numAxisToInterpolate; ii++ {
		// Append the dimension of the interpolated axis.
		transposeDims = append(transposeDims, 2*ii)
	}
	for ii := 0; ii < numAxisToInterpolate; ii++ {
		// Append the dimension of the interpolation span size
		transposeDims = append(transposeDims, 2*ii+1)
	}
	broadcastWeights := TransposeAllDims(weightsCrosses, transposeDims...)
	for ii := 0; ii < inputShape.Rank(); ii++ {
		if inputShape.Dimensions[ii] != outputShape.Dimensions[ii] {
			// Interpolated dimensions already included in broadcastWeight, skip.
			continue
		}
		// Add a new axis in the corresponding position in the interpolated weights.
		broadcastWeights = ExpandDims(broadcastWeights, numAxisToInterpolate+ii)
	}
	if !g.Ok() {
		return
	}
	broadcastWeights = BroadcastToShape(broadcastWeights, gatheredElements.Shape())

	// Compute the interpolation by the product of the gathered value and their weights, contracted
	// across their slices.
	contractingAxes := make([][2]int, 0, numAxisToInterpolate)
	batchAxes := make([][2]int, 0, gatheredElements.Rank()-numAxisToInterpolate)
	for axis := 0; axis < gatheredElements.Rank(); axis++ {
		if axis > numAxisToInterpolate {
			inputAxis := axis - numAxisToInterpolate
			if inputShape.Dimensions[inputAxis] != outputShape.Dimensions[inputAxis] {
				// One of the axis being interpolated, that we want to contract.
				contractingAxes = append(contractingAxes, [2]int{axis, axis})
				continue
			}
		}
		batchAxes = append(batchAxes, [2]int{axis, axis})
	}
	output = EinsumAxes(gatheredElements, broadcastWeights, contractingAxes, batchAxes)
	output.AssertRank(input.Rank()) // Same rank, but interpolated and transposed.

	// Now we need to transpose the interpolated dimensions, that are at the start, back to their locations.
	transposeDims = make([]int, 0, input.Rank())
	interpolatedAxis, unchangedAxis := 0, numAxisToInterpolate
	for ii := 0; ii < input.Rank(); ii++ {
		if inputShape.Dimensions[ii] != outputShape.Dimensions[ii] {
			transposeDims = append(transposeDims, interpolatedAxis)
			interpolatedAxis++
		} else {
			transposeDims = append(transposeDims, unchangedAxis)
			unchangedAxis++
		}
	}
	output = TransposeAllDims(output, transposeDims...)
	return
}
