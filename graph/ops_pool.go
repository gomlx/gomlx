/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package graph

import (
	"slices"

	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
)

// This file contains all parts of the {Max|Sum|Prod}Pool implementation.

type ReduceOpType = backends.ReduceOpType

// PoolBuilder is a helper to build a pool computation.
// Create it with {Max|Sum|Mean|Prod}Pool, set the desired parameters and
// when set, call `IsNil()`.
type PoolBuilder struct {
	graph                *Graph
	x                    *Node
	reductionType        ReduceOpType
	numSpatialDims       int
	channelsAxisConfig   images.ChannelsAxisConfig
	isFullShape          bool
	spatialAxes          []int // Indices of spatial axes.
	channelsAxis         int
	windowSizes, strides []int
	paddings             [][2]int
	padSame              bool
	isMean, isConcat     bool // Divide by number of elements later if mean.
}

// MaxPool prepares a max pooling on x with the given kernel for arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.). It returns the max value
// for the selected window, on given strides.
//
// It is very flexible and to ease configuring of its parameters it returns a
// PoolBuilder for configuration. Once it is set up
// call `PoolBuilder.Done` and it will return the pooled
// x. Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerAxis.
//
// The shapes of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.ChannelsFirst)`, the shapes should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as depth or feature axis.
//
// Note: `images` refers to package `github.com/gomlx/gomlx/types/tensors/images`.
//
// The shapes of kernel should be `[<spatial_dimensions...>, input_channels, output_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.Channels)`, the shapes should be
// `[input_channels, <spatial_dimensions...>, output_channels]` instead.
func MaxPool(x *Node) *PoolBuilder {
	return makePoolBuilder(x, backends.ReduceOpMax)
}

// MinPool prepares a min pooling on x with the given kernel for arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.). It returns the min value
// for the selected window, on given strides.
//
// It is very flexible and to ease configuring of its parameters it returns a
// PoolBuilder for configuration. Once it is set up
// call `PoolBuilder.Done` and it will return the pooled
// x. Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerAxis.
//
// The shapes of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.ChannelsFirst)`, the shapes should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as depth or feature axis.
//
// Note: `images` refers to package `github.com/gomlx/gomlx/types/tensors/images`.
//
// The shapes of kernel should be `[<spatial_dimensions...>, input_channels, output_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.Channels)`, the shapes should be
// `[input_channels, <spatial_dimensions...>, output_channels]` instead.
func MinPool(x *Node) *PoolBuilder {
	return makePoolBuilder(x, backends.ReduceOpMin)
}

// SumPool prepares a sum pooling on x with the given kernel for arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.). It returns the sum value
// for the selected window, on given strides.
//
// It is very flexible and to ease configuring of its parameters it returns a
// PoolBuilder for configuration. Once it is set up
// call `PoolBuilder.Done` and it will return the pooled
// x. Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerAxis.
//
// The shapes of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.ChannelsFirst)`, the shapes should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as depth or feature axis.
//
// Note: `images` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// The shapes of kernel should be `[<spatial_dimensions...>, input_channels, output_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.Channels)`, the shapes should be
// `[input_channels, <spatial_dimensions...>, output_channels]` instead.
func SumPool(x *Node) *PoolBuilder {
	return makePoolBuilder(x, backends.ReduceOpSum)
}

// MeanPool prepares a mean pooling on x with the given kernel for arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.). It returns the mean value
// for the selected window, on given strides.
//
// It is very flexible and to ease configuring of its parameters it returns a
// PoolBuilder for configuration. Once it is set up
// call `PoolBuilder.Done` and it will return the pooled
// x. Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerAxis.
//
// The shapes of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.ChannelsFirst)`, the shapes should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as depth or feature axis.
//
// Note: `images` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// The shapes of kernel should be `[<spatial_dimensions...>, input_channels, output_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.Channels)`, the shapes should be
// `[input_channels, <spatial_dimensions...>, output_channels]` instead.
func MeanPool(x *Node) *PoolBuilder {
	pool := makePoolBuilder(x, backends.ReduceOpSum)
	pool.isMean = true
	return pool
}

// ConcatPool pool on the spatial dimensions by increasing the channels dimensions, across the window.
//
// Example: x.shape=[batch_size, height, width, 3] and Window(3): the output depth will be 9x3=27,
// with the concatenation of the channels of all the pixels around.
//
// The implementation actually uses a convolution with a fixed kernel, but it can be seen as a concatenating
// pool operation.
func ConcatPool(x *Node) *PoolBuilder {
	pool := makePoolBuilder(x, backends.ReduceOpUndefined)
	pool.isConcat = true
	return pool
}

func makePoolBuilder(x *Node, reductionType ReduceOpType) *PoolBuilder {
	g := validateBuildingGraphFromInputs(x)
	pool := &PoolBuilder{
		graph:         g,
		x:             x,
		reductionType: reductionType,
	}
	return pool.ChannelsAxis(images.ChannelsLast).NoPadding()
}

// ChannelsAxis configures the axis for the channels (aka. "depth" or "features") dimension.
// The default is `images.ChannelsLast`, meaning the "channels" dimension comes last.
//
// Note: `images` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// If you don't want to exclude the batch size and channels from the pooling, use FullShape instead.
//
// It returns the modified Config object, so calls can be cascaded.
func (pool *PoolBuilder) ChannelsAxis(channelsAxisConfig images.ChannelsAxisConfig) *PoolBuilder {
	pool.channelsAxisConfig = channelsAxisConfig
	pool.channelsAxis = images.GetChannelsAxis(pool.x, channelsAxisConfig)
	pool.spatialAxes = images.GetSpatialAxes(pool.x, channelsAxisConfig)
	pool.numSpatialDims = pool.x.Rank() - 2
	pool.isFullShape = false
	return pool
}

// FullShape configures the pooling operation to consider all its axes as part of the pooling, with no special
// considerations for the batch or channel axes.
//
// See ChannelsAxis to handle batch and channels specially.
//
// The default is configured with ChannelsAxis(images.ChannelsLast).
func (pool *PoolBuilder) FullShape() *PoolBuilder {
	pool.numSpatialDims = pool.x.Rank()
	pool.isFullShape = true
	pool.spatialAxes = xslices.Iota(0, pool.x.Rank())
	return pool
}

// Window sets the pooling window size for all spatial dimensions to the same windowSize.
//
// There is no default, and this must be set either with Window or WindowPerAxis.
func (pool *PoolBuilder) Window(windowSize int) *PoolBuilder {
	windowSizes := make([]int, pool.numSpatialDims)
	for ii := range windowSizes {
		windowSizes[ii] = windowSize
	}
	return pool.WindowPerAxis(windowSizes...)
}

// WindowPerAxis sets the pooling window size for each spatial dimension.
//
// There is no default, and this must be set either with Window or WindowPerAxis.
func (pool *PoolBuilder) WindowPerAxis(sizes ...int) *PoolBuilder {
	if len(sizes) != pool.numSpatialDims {
		Panicf("received %d window sizes in WindowPerAxis, but x has %d spatial dimensions",
			len(sizes), pool.numSpatialDims)
	}
	pool.windowSizes = sizes
	return pool
}

// Strides sets the strides of the pooling. It sets the same value for every spatial dimension.
//
// The default is the same value as the window size (set with Window or WindowPerAxis) -- except if one
// uses PadSame, then the default changes to 1.
//
// The stride is how many steps to move after the pooling of a window. A value of 2 will halve the
// input size, since the pooling will be done at every other position, and so on. It can be defined
// separately per dimension with StridePerAxis.
//
// One cannot use strides and dilation at the same time.
func (pool *PoolBuilder) Strides(strides int) *PoolBuilder {
	stridesPerAxis := xslices.SliceWithValue(pool.numSpatialDims, strides)
	return pool.StridePerAxis(stridesPerAxis...)
}

// StridePerAxis sets the strides for each spatial dimension of the pooling.
//
// The default is the same value as the window size (set with Window or WindowPerAxis) -- except if one
// uses PadSame, then the default changes to 1.
//
// The stride is how many steps to move after a pooling. A value of 2 will half the input
// size, since a pooling will be done at every other position, and so on. It can be defined
// separately per dimension.
//
// One cannot use strides and dilation at the same time.
func (pool *PoolBuilder) StridePerAxis(strides ...int) *PoolBuilder {
	if len(strides) != pool.numSpatialDims {
		Panicf("received %d strides in StridePerAxis, but x has %d spatial dimensions",
			len(strides), pool.numSpatialDims)
	}
	pool.strides = strides
	return pool
}

// PadSame adds paddings on the edges of x such that in the end the output
// of the convolution has the same shapes as the input.
//
// This changes the default value of Strides to 1, if it is not set -- if set to something else, PadSame won't really
// output the same shapes as the input.
//
// The default is NoPadding.
func (pool *PoolBuilder) PadSame() *PoolBuilder {
	pool.paddings = nil
	pool.padSame = true
	return pool
}

// NoPadding removes any paddings, so if the kernel spatial dimensions > 1,
// the output shapes will be reduced on the edges.
// This is the default.
func (pool *PoolBuilder) NoPadding() *PoolBuilder {
	pool.paddings = nil
	pool.padSame = false
	return pool
}

// PaddingPerDim specifies the paddings at the start and at the end to use per spatial dimension,
// that means one pair ([2]int) per spatial dimension.
// The default is PadSame.
func (pool *PoolBuilder) PaddingPerDim(paddings [][2]int) *PoolBuilder {
	if len(paddings) != pool.numSpatialDims {
		Panicf("received %d paddings in PaddingPerDim, but x has %d spatial dimensions",
			len(paddings), pool.numSpatialDims)
	}
	pool.paddings = paddings
	pool.padSame = false
	return pool
}

// Done indicates that the convolve operation is finished being configured and
// it updates the computation graph with convolution, and returns the resulting
// Node.
func (pool *PoolBuilder) Done() *Node {
	rank := pool.x.Rank()
	if pool.numSpatialDims <= 0 {
		Panicf("Input x must have rank >= 3, shaped by default as [batch, <spatial_dimensions...>, channels] "+
			"but x rank is %d -- alternatively configure the spatial dimensions with ChannelsAxis or FullShape", rank)
	}

	// Closure to create slice with value for every axis, using a default value
	// and the corresponding spatial values.
	makeSlice := func(name string, defaultValue int, valuesForSpatialDims []int) []int {
		s := xslices.SliceWithValue(rank, defaultValue)
		if len(valuesForSpatialDims) > 0 {
			if len(valuesForSpatialDims) != pool.numSpatialDims {
				Panicf("%s requires %d values (one per spatial dimension), but %d were given -- configure spatial "+
					"dimensions with FullShape or ChannelsAxis, and configure them before calling Window, Padding or Strides",
					name, pool.numSpatialDims, len(valuesForSpatialDims))
			}
			for ii, axis := range pool.spatialAxes {
				s[axis] = valuesForSpatialDims[ii]
			}
		}
		return s
	}

	// windowSizes is obligatory.
	if len(pool.windowSizes) == 0 {
		Panicf("window sizes required but not configured -- use .Window() or .WindowPerAxis()")
	}
	windowDimensions := makeSlice("Window", 1, pool.windowSizes)

	if pool.isConcat {
		return pool.doConcat()
	}

	// strides default to pooling window sizes.
	var strides []int
	if len(pool.strides) > 0 {
		strides = makeSlice("Stride", 1, pool.strides)
	} else {
		if pool.padSame {
			// if PadSame(), then the strides default to 1, to preserve the image size.
			strides = makeSlice("Stride", 1, nil)
		} else {
			// strides default to the window size.
			strides = makeSlice("Stride", 1, pool.windowSizes)
		}
	}

	// spatialPaddings can only be calculated after we are sure about the channels positioning.
	spatialPaddings := pool.paddings
	if spatialPaddings == nil && pool.padSame {
		// Pad such that the output is shaped the same as the input.
		spatialPaddings = make([][2]int, pool.numSpatialDims)
		for dim := range spatialPaddings {
			windowSize := pool.windowSizes[dim]            // for this dimension.
			spatialPaddings[dim][0] = (windowSize - 1) / 2 // For even sized kernels, the padding is asymmetric.
			spatialPaddings[dim][1] = windowSize / 2
		}
	}
	var paddings [][2]int
	if len(spatialPaddings) > 0 {
		paddings = make([][2]int, rank)
		if len(spatialPaddings) != pool.numSpatialDims {
			Panicf("Paddings require %d values (one per spatial dimension), but %d were given -- configure spatial "+
				"dimensions with FullShape or ChannelsAxis, and configure them before calling Window, Padding or Strides",
				pool.numSpatialDims, len(spatialPaddings))
		}
		for ii, axis := range pool.spatialAxes {
			paddings[axis] = spatialPaddings[ii]
		}
	}

	pooled := checkedReduceWindow(pool.x, pool.reductionType,
		windowDimensions, strides, nil, nil, paddings)

	// Take the mean.
	if pool.isMean && pool.reductionType == backends.ReduceOpSum {
		if len(paddings) == 0 {
			// If no padding, the number of elements to take the mean is fixed:
			totalWindowSize := 1
			for _, s := range pool.windowSizes {
				totalWindowSize *= s
			}
			pooled = DivScalar(pooled, float64(totalWindowSize))
		} else {
			pooled = takeMeanOfContributions(pool.x, pooled, pool.channelsAxis, windowDimensions, strides, paddings)
		}
	}

	return pooled
}

// takeMeanOfContributions divides the pooled sum by the number of contributions at each position.
func takeMeanOfContributions(x, pooledSum *Node, channelsAxis int, windowDimensions, strides []int, paddings [][2]int) *Node {
	// We need to normalize the sum by the number of elements that were actually used, ignoring the padding.
	// We use a similar checkedReduceWindow configuration, but as input a tensor with 1s and dropping the
	// batch and channels axes, since they will are the same.
	shapeNoBatchOrChannels := x.Shape().Clone()
	shapeNoBatchOrChannels.Dimensions[0] = 1
	shapeNoBatchOrChannels.Dimensions[channelsAxis] = 1
	ones := Ones(x.graph, shapeNoBatchOrChannels)
	pooledOnes := checkedReduceWindow(ones, backends.ReduceOpSum,
		windowDimensions, strides, nil, nil, paddings)
	pooledOnes = StopGradient(pooledOnes)
	return Div(pooledSum, pooledOnes)
}

// checkedReduceWindow is a checked version of backendReduceWindow.
func checkedReduceWindow(x *Node, reductionType ReduceOpType, windowDimensions, strides, baseDilations, windowDilations []int, paddings [][2]int) *Node {
	_ = validateBuildingGraphFromInputs(x)
	rank := x.Rank()
	if len(windowDimensions) != rank {
		Panicf("windowDimensions (length %d) must have the same length as the rank of x (rank %d)", len(windowDimensions), rank)
	}
	if len(strides) != 0 && len(strides) != rank {
		Panicf("strides (length %d) if gieven must have the same length as the rank of x (rank %d)", len(strides), rank)
	}
	if len(baseDilations) > 0 && len(baseDilations) != rank {
		Panicf("baseDilations (length %d) if given must have the same length as the rank of x (rank %d)", len(paddings), rank)
	}
	if len(windowDilations) > 0 && len(windowDilations) != rank {
		Panicf("windowDilations (length %d) if given must have the same length as the rank of x (rank %d)", len(paddings), rank)
	}
	if len(paddings) > 0 && len(paddings) != rank {
		Panicf("paddings (length %d) if given must have the same length as the rank of x (rank %d)", len(paddings), rank)
	}
	return backendReduceWindow(x, reductionType, windowDimensions, strides, baseDilations, windowDilations, paddings)
}

// checkedSelectAndScatter selects (largest) the element (according to reduceOp) from a window and scatter to those positions
// the value from source. It's used to calculate the gradient of a MaxPool.
func checkedSelectAndScatter(x, source *Node, reduceOp ReduceOpType, windowDimensions, strides []int, paddings [][2]int) *Node {
	_ = validateBuildingGraphFromInputs(x, source)
	rank := x.Rank()
	if len(windowDimensions) != rank {
		Panicf("windowSizes (length %d) must have same length as rank of input x (rank %d)", len(windowDimensions), rank)
	}
	if len(strides) != rank {
		Panicf("strides (length %d) must have same length as rank of input x (rank %d)", len(strides), rank)
	}
	if len(paddings) > 0 && len(paddings) != rank {
		Panicf("paddings (length %d) if given must have same length as rank of input x (rank %d)", len(paddings), rank)
	}

	switch reduceOp {
	case backends.ReduceOpMax:
		return backendSelectAndScatterMax(x, source, windowDimensions, strides, paddings)
	case backends.ReduceOpMin:
		return backendSelectAndScatterMin(x, source, windowDimensions, strides, paddings)
	case backends.ReduceOpSum:
		return backendSelectAndScatterSum(x, source, windowDimensions, strides, paddings)
	default:
		Panicf("SelectAndScatter not defined for original reduce operation %s", reduceOp)
		panic(nil) // Disable lint warning.
	}
}

var vjpReductionTypes = types.SetWith(backends.ReduceOpMax, backends.ReduceOpMin, backends.ReduceOpSum)

// reduceWindowVJP calculates v*d(reduceWindow(x))/{dx, d_kernel).
func reduceWindowVJP(node, v *Node, _ shapes.Shape) []*Node {
	// Recover parameters from serialized node.
	params := node.inputs.(*nodeInputsReduceWindow)
	if !vjpReductionTypes.Has(params.reductionType) {
		Panicf("ReduceWindow gradient only defined for %q operations, instead got %q", vjpReductionTypes, params.reductionType)
	}

	if len(params.baseDilations) > 0 || len(params.windowDilations) > 0 {
		Panicf("gradient of ReduceWindow with base or window dilations is not defined")
	}

	//fmt.Printf("Grad(reduceWindow(%s):\n", params.reductionType)
	//fmt.Printf("\tx.shape=%s\n", params.x.Shape())
	//fmt.Printf("\tnode.shape=%s\n", node.Shape())
	//fmt.Printf("\tv.shape=%s\n", v.Shape())
	//fmt.Printf("\twindowDimensions=%v\n", params.windowDimensions)
	//fmt.Printf("\tstrides=%v\n", params.strides)
	//fmt.Printf("\tpaddings=%v\n", params.paddings)

	var vjpX *Node
	if params.reductionType == backends.ReduceOpMax || params.reductionType == backends.ReduceOpMin {
		vjpX = checkedSelectAndScatter(params.x, v, params.reductionType, params.windowDimensions, params.strides, params.paddings)
	} else {
		// params.reductionType == backends.ReduceOpSum
		vjpX = dilateConvolveToMatchSumPooling(params.x, v, params.windowDimensions, params.strides, params.paddings)
	}
	return []*Node{vjpX}
}

// dilateConvolveToMatchSumPooling convolves the `backProp` to match `x`.
//
// Since the convolution would be with a kernel of 1s we instead use `graph.Pad` and checkedReduceWindow instead.
func dilateConvolveToMatchSumPooling(x, backProp *Node, windowDimensions, strides []int, paddings [][2]int) *Node {
	g := validateBuildingGraphFromInputs(x, backProp)
	dtype := x.DType()
	rank := x.Rank()
	//dims := x.Shape().Dimensions

	if len(windowDimensions) != rank {
		Panicf("windowSizes (length %d) must have the same length as the rank of x (rank %d)", len(windowDimensions), rank)
	}
	if len(strides) != rank {
		Panicf("strides (length %d) must have the same length as the rank of x (rank %d)", len(strides), rank)
	}
	if len(paddings) > 0 && len(paddings) != rank {
		Panicf("paddings (length %d) if given must have the same length as the rank of x (rank %d)", len(paddings), rank)
	}

	// Configure the padding needed to expand the backprop back to its original size.
	padConfig := make([]PadAxis, rank)
	for axis := range padConfig {
		conf := &padConfig[axis]

		// pad due to strides and window sizes: this reconstructs what was used from
		// the original dimension -- some may be padding ... but some padding (not used) may
		// not be included.
		conf.Interior = strides[axis] - 1
		conf.Start = windowDimensions[axis] / 2 // Notice that we switch the long/short half from start/end.
		conf.End = (windowDimensions[axis] - 1) / 2
	}
	// expanded should have the same spatial dimensions as the original input,
	// but with zeros filled in between.
	zero := ScalarZero(g, dtype)
	expanded := Pad(backProp, zero, padConfig...)

	// For each position on expanded, just sum the values that reach it.
	padSame := make([][2]int, rank) // pad to preserve expanded shapes.
	for axis := range padSame {
		windowSize := windowDimensions[axis]    // for this axis.
		padSame[axis][0] = (windowSize - 1) / 2 // For even sized kernels, the padding is asymmetric.
		padSame[axis][1] = windowSize / 2
	}
	stridesOne := xslices.SliceWithValue(rank, 1)
	expanded = checkedReduceWindow(expanded, backends.ReduceOpSum, windowDimensions, stridesOne, nil, nil, padSame)

	// We may need to trim the extra padding that was used originally, since we don't want to re-generate
	// the original padding. We many also need to pad extra, in case not everything from the input tensor
	// was used in the output.
	//
	// Care has to be taken, because we only want to take out padding that was effectively used,
	// and not padding that didn't have any effect.
	// E.g: input=[1, 3, 1], window=3, padding=[0, 500], stride=1000 -> in this case none of the padding was used.
	var requiresAdjustment bool
	for axis := range padConfig {
		conf := &padConfig[axis]
		*conf = PadAxis{}

		amountToAdjust := x.Shape().Dimensions[axis] - expanded.Shape().Dimensions[axis] // May be negative.
		paddingStart := 0
		if len(paddings) > 0 {
			paddingStart = paddings[axis][0]
		}
		conf.Start = -paddingStart             // Padding on the start is always used, and needs trimming.
		conf.End = amountToAdjust - conf.Start // Reminder of adjustment goes to the end of the spatial axis.
		if conf.Start != 0 || conf.End != 0 {
			requiresAdjustment = true
		}
	}
	var grad *Node
	if requiresAdjustment {
		grad = Pad(expanded, zero, padConfig...)
	} else {
		grad = expanded
	}
	return grad
}

func (pool *PoolBuilder) doConcat() *Node {
	x := pool.x
	g := x.Graph()
	dtype := x.DType()
	shape := x.Shape()
	inputChannelsSize := shape.Dimensions[pool.channelsAxis]
	kernelDims := slices.Clone(pool.windowSizes)
	outputChannelsSize := 1
	for _, size := range kernelDims {
		outputChannelsSize *= size
	}
	outputChannelsSize *= inputChannelsSize
	kernel := Iota(g, shapes.Make(dtypes.Int32, outputChannelsSize), 0)

	// Kernel order depends on the channels axes position.
	if pool.channelsAxisConfig == images.ChannelsLast {
		// Kernel so far shaped [<spatial_dims...>, inputChannelsSize],
		kernelDims = append(kernelDims, inputChannelsSize)
	} else {
		// Kernel so far shaped [inputChannelsSize, <spatial_dims...>],
		kernelDims = append([]int{inputChannelsSize}, kernelDims...)
	}
	kernel = Reshape(kernel, kernelDims...)

	// Add the last axis to kernel of outputChannelsSize, a one-hot encoding:
	kernel = OneHot(kernel, outputChannelsSize, dtype)

	// strides default to pooling window sizes.
	strides := pool.strides
	if len(strides) == 0 {
		if pool.padSame {
			// if PadSame(), then the strides default to 1, to preserve the image size.
			strides = xslices.SliceWithValue(len(pool.windowSizes), 1)
		} else {
			// strides default to the window size.
			strides = slices.Clone(pool.windowSizes)
		}
	}

	// Convolve with given kernel.
	convConfig := Convolve(x, kernel).ChannelsAxis(pool.channelsAxisConfig).StridePerDim(strides...)
	if pool.paddings != nil {
		convConfig.PaddingPerDim(pool.paddings)
	}
	if pool.padSame {
		convConfig.PadSame()
	}
	return convConfig.Done()
}
