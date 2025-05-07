package graph

import (
	"cmp"

	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// File with image manipulation tools

func scalarMin[T cmp.Ordered](a, b T) T {
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
	input.AssertValid()
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
//
// Note: there is a bug in that makes the Bilinear gradient fail if the input dimensions to interpolate <= 3.
// Use Nearest instead for now.
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
	if c.alignCorner && c.halfPixelCenters {
		Panicf("invalid Interpolate configuration, one cannot set alignCorner and halfPixelCenters " +
			"to true at the same time")
	}

	// Find axes that are going to be interpolated and the output shapes.
	input := c.input
	inputShape := input.Shape()
	dtype := input.DType()
	outputSizes := c.outputSizes
	if len(outputSizes) != inputShape.Rank() {
		Panicf("Output sizes for interpolation has a different rank (%v) than the input (%s)",
			outputSizes, input.Shape())
	}

	// Find axisToInterpolateList, and their target dimensions in interpolationDims:
	axisToInterpolateList := make([]int, 0, input.Rank())
	axisToInterpolateMap := make([]bool, input.Rank()) // Set to true if the axis is marked for interpolation.
	shape := inputShape.Clone()
	interpolationDims := make([]int, 0, input.Rank()+1)
	for axis, s := range outputSizes {
		if s != NoInterpolation && s <= 0 {
			Panicf("Output sizes set to invalid value (%d <= 0): %v", s, outputSizes)
		}
		if s == NoInterpolation || s == inputShape.Dimensions[axis] {
			continue
		}

		// Axis marked for interpolation:
		axisToInterpolateList = append(axisToInterpolateList, axis)
		axisToInterpolateMap[axis] = true
		shape.Dimensions[axis] = s
		interpolationDims = append(interpolationDims, s)
	}
	numAxesToInterpolate := len(axisToInterpolateList)
	if numAxesToInterpolate == 0 {
		// Nothing to do actually, the output shapes is exactly the same as the input. Silently return
		// the input.
		output = input
		return
	}

	// Find slices and weights for each interpolation axis:
	spanStarts := make([]*Node, 0, len(axisToInterpolateList))
	spanSizes := make([]int, 0, len(axisToInterpolateList))
	weights := make([]*Node, 0, len(axisToInterpolateList))
	for axisIdx, axis := range axisToInterpolateList {
		inSize := inputShape.Dimensions[axis]
		outSize := shape.Dimensions[axis]
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
		upperBound := Scalar(g, dtype, float64(inSize-spanSize))
		if !c.isBilinear && !c.halfPixelCenters {
			spanStart = Min(spanStart, upperBound)
		} else {
			spanStart = Clip(spanStart, ZerosLike(spanStart), upperBound)
		}
		spanSizes = append(spanSizes, spanSize)

		// Broadcast spanStart to common shapes so it can be combined with other interpolation axes.
		// The final shapes will be [interpolationDims..., 1].
		broadcastSpanStart := ConvertDType(spanStart, dtypes.Int32)
		{
			spanExpandAxes := make([]int, 0, numAxesToInterpolate)
			for axis := 0; axis < axisIdx; axis++ {
				spanExpandAxes = append(spanExpandAxes, axis)
			}
			for axis := axisIdx + 1; axis < numAxesToInterpolate; axis++ {
				spanExpandAxes = append(spanExpandAxes, axis)
			}
			spanExpandAxes = append(spanExpandAxes, numAxesToInterpolate) // Last dimension with the start index.
			broadcastSpanStart = ExpandAndBroadcast(broadcastSpanStart, append(interpolationDims, 1), spanExpandAxes)
		}
		spanStarts = append(spanStarts, broadcastSpanStart)

		// Weights
		var weight *Node
		if c.isBilinear {
			weight = Sub(spanStart, sampleFraction)
			weight = ExpandAndBroadcast(weight, []int{outSize, spanSize}, []int{-1})
			offset := Iota(g, shapes.Make(dtype, spanSize), 0)
			weight = Add(weight, ExpandAndBroadcast(offset, []int{outSize, spanSize}, []int{0}))
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
	gatheredElements := GatherSlices(input, axisToInterpolateList, spanStart, spanSizes, true)

	// weightsCrosses of all the weights
	var weightsCrosses *Node
	for _, w := range weights {
		if weightsCrosses == nil {
			weightsCrosses = w
			continue
		}
		weightsCrosses = EinsumAxes(weightsCrosses, w, nil, nil)
	}
	// weightCrosses will be shaped [I_0_dim, I_0_span_size, I_1_dim, I_1_span_size ...]: two
	// axes per interpolation axis the first one with the target dimension of the interpolation
	// and the second with the size of the span used for interpolating.
	weightsCrosses.AssertRank(2 * numAxesToInterpolate)

	// Now we need an Einsum with the gatheredElements, where we contract the gathered spans and
	// we "batch" (meaning they are matched) the interpolated sizes.
	contractingAxes := make([][2]int, 0, numAxesToInterpolate)
	batchAxes := make([][2]int, 0, numAxesToInterpolate)
	nextWeightsFullAxis := 0
	nextWeightsSpanAxis := 1
	for axis := 0; axis < gatheredElements.Rank(); axis++ {
		if axis < numAxesToInterpolate {
			// Matching target interpolated dimensions.
			batchAxes = append(batchAxes, [2]int{nextWeightsFullAxis, axis})
			nextWeightsFullAxis += 2
		} else {
			axisInInput := axis - numAxesToInterpolate
			if axisToInterpolateMap[axisInInput] {
				// This axis is just the span gathered / weights, which we need to contract (reduce_sum):
				contractingAxes = append(contractingAxes, [2]int{nextWeightsSpanAxis, axis})
				nextWeightsSpanAxis += 2
			}
		}
	}
	output = EinsumAxes(weightsCrosses, gatheredElements, contractingAxes, batchAxes)
	output.AssertRank(input.Rank()) // Same rank, but interpolated and wrongly transposed.

	// Now we need to transpose the interpolated dimensions, that are upfront (at the leading dimensions),
	// back to their original locations.
	transposeDims := make([]int, 0, input.Rank())
	interpolatedAxis, unchangedAxis := 0, numAxesToInterpolate // numAxesToInterpolate
	for ii := 0; ii < input.Rank(); ii++ {
		if axisToInterpolateMap[ii] {
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
