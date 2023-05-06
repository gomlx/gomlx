package graph

// File with image manipulation tools

// InterpolationConfig is created with Interpolate and then actually executed
// with a call to Done.
//
// Between the its construction and execution one can set the various parameters for
// the interpolation.
type InterpolationConfig struct {
	g                       *Graph
	input                   *Node
	alignCorner, isBilinear bool
	outputSizes             []int
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
// graph for the interpolation and returns the interpolated tensor.
//
// This can be used for images (2D) but also for anything volumetric (3D or higher) or also for time-series (1D).
//
// The implementation is based on the Tensorflow `tf2xla` one.
func Interpolate(input *Node, outputSizes ...int) *InterpolationConfig {
	return &InterpolationConfig{
		g:           input.Graph(),
		input:       input,
		alignCorner: true,
		outputSizes: outputSizes,
	}
}

// Bilinear configures the interpolation to be bilinear (as opposed to nearest). Default is Nearest.
// See also Nearest.
//
// It returns the InterpolationConfig passed, to allow cascaded method calls.
func (c *InterpolationConfig) Bilinear() *InterpolationConfig {
	c.isBilinear = true
	return c
}

// Nearest configures the interpolation to be bilinear (as opposed to nearest). Default is Nearest.
// See also Bilinear.
//
// It returns the InterpolationConfig passed, to allow cascaded method calls.
func (c *InterpolationConfig) Nearest() *InterpolationConfig {
	c.isBilinear = true
	return c
}

// AlignCorner configures the interpolation to be value of "align corner": if set to true, the input and output
// tensors corner pixels are aligned at the center points of their corner pixels, preserving the values at the
// corner pixels. If set to false, the input and output tensors are aligned by the corner points of their
// corner pixels, and the interpolation uses edge value padding for out-of-boundary values.
//
// Default is true.
//
// See also Bilinear.
//
// It returns the InterpolationConfig passed, to allow cascaded method calls.
func (c *InterpolationConfig) AlignCorner(alignCorner bool) *InterpolationConfig {
	c.alignCorner = alignCorner
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

	// Find axis that are going to be interpolated and the output shape.
	input := c.input
	inputShape := input.Shape()
	outputSizes := c.outputSizes
	if len(outputSizes) != inputShape.Rank() {
		g.SetErrorf("Output sizes for interpolation has a different rank (%v) than the input (%s)",
			outputSizes, input.Shape())
		return
	}
	axisToInterpolate := make([]int, 0, input.Rank())
	outputShape := inputShape.Copy()
	for axis, s := range outputSizes {
		if s != NoInterpolation && s <= 0 {
			g.SetErrorf("Output sizes set to invalid value (%d <= 0): %v", s, outputSizes)
			return
		}
		if s != NoInterpolation && s != inputShape.Dimensions[axis] {
			axisToInterpolate = append(axisToInterpolate, axis)
			outputShape.Dimensions[axis] = s
		}
	}

	output = Zeros(g, outputShape)
	return
}
