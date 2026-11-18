// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/internal/graphimpl"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// This file contains all parts of the {Max|Sum|Prod}Pool implementation.

type ReduceOpType = compute.ReduceOpType

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
	windowDilations      []int
	paddings             [][2]int
	padSame              bool
	isMean, isConcat     bool // Divide by the number of elements later if mean.
}

// MaxPool prepares a max pooling on x for an arbitrary number of spatial dimensions (1D, 2D, 3D, etc.).
// It returns the max value for the selected window, on given strides.
//
// It is very flexible. To ease the configuring of its parameters, it returns a PoolBuilder for configuration.
// Once it is set up, call `PoolBuilder.Done` and it will return the pooled x.
// Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerAxis.
//
// The shapes of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.ChannelsFirst)`, the shapes should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as the depth or feature axis.
//
// Note: `images` refers to the package `github.com/gomlx/gomlx/core/tensors/images`.
func MaxPool(x *Node) *PoolBuilder {
	return makePoolBuilder(x, compute.ReduceOpMax)
}

// MinPool prepares a min pooling on x for an arbitrary number of spatial dimensions (1D, 2D, 3D, etc.).
// It returns the max value for the selected window, on given strides.
//
// It is very flexible. To ease the configuring of its parameters, it returns a PoolBuilder for configuration.
// Once it is set up, call `PoolBuilder.Done` and it will return the pooled x.
// Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerAxis.
//
// The shapes of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.ChannelsFirst)`, the shapes should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as the depth or feature axis.
//
// Note: `images` refers to the package `github.com/gomlx/gomlx/core/tensors/images`.
func MinPool(x *Node) *PoolBuilder {
	return makePoolBuilder(x, compute.ReduceOpMin)
}

// SumPool prepares a sum pooling on x for an arbitrary number of spatial dimensions (1D, 2D, 3D, etc.).
// It returns the max value for the selected window, on given strides.
//
// It is very flexible. To ease the configuring of its parameters, it returns a PoolBuilder for configuration.
// Once it is set up, call `PoolBuilder.Done` and it will return the pooled x.
// Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerAxis.
//
// The shapes of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.ChannelsFirst)`, the shapes should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as the depth or feature axis.
//
// Note: `images` refers to the package `github.com/gomlx/gomlx/core/tensors/images`.
func SumPool(x *Node) *PoolBuilder {
	return makePoolBuilder(x, compute.ReduceOpSum)
}

// MeanPool prepares a mean pooling on x for an arbitrary number of spatial dimensions (1D, 2D, 3D, etc.).
// It returns the max value for the selected window, on given strides.
//
// It is very flexible. To ease the configuring of its parameters, it returns a PoolBuilder for configuration.
// Once it is set up, call `PoolBuilder.Done` and it will return the pooled x.
// Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerAxis.
//
// The shapes of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(images.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(images.ChannelsFirst)`, the shapes should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as the depth or feature axis.
//
// Note: `images` refers to the package `github.com/gomlx/gomlx/core/tensors/images`.
func MeanPool(x *Node) *PoolBuilder {
	pool := makePoolBuilder(x, compute.ReduceOpSum)
	pool.isMean = true
	return pool
}

// ConcatPool pool on the spatial dimensions by increasing the channels' dimension, across the window.
//
// Example: x.shape=[batch_size, height, width, 3] and Window(3): the output depth will be 9x3=27,
// with the concatenation of the surrounding pixels' channels.
//
// The implementation actually uses a convolution with a fixed kernel, but it can be seen as a concatenating
// pool operation.
func ConcatPool(x *Node) *PoolBuilder {
	pool := makePoolBuilder(x, compute.ReduceOpUndefined)
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
// Note: `images` refers to package `github.com/gomlx/gomlx/core/tensors/images`.
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

// WindowDilations sets the window dilation factor to the same value across all spatial dimensions.
// A dilation of d inserts d-1 spacing positions between consecutive window elements (default is 1, no dilation).
func (pool *PoolBuilder) WindowDilations(dilation int) *PoolBuilder {
	dilationsPerAxis := xslices.SliceWithValue(pool.numSpatialDims, dilation)
	return pool.WindowDilationPerAxis(dilationsPerAxis...)
}

// WindowDilationPerAxis sets the window dilation factor for each spatial dimension.
// Each dilation factor must be >= 1 (default is 1, no dilation).
func (pool *PoolBuilder) WindowDilationPerAxis(dilations ...int) *PoolBuilder {
	if len(dilations) != pool.numSpatialDims {
		Panicf("received %d window dilations in WindowDilationPerAxis, but x has %d spatial dimensions",
			len(dilations), pool.numSpatialDims)
	}
	for i, d := range dilations {
		if d < 1 {
			Panicf("window dilation at spatial axis %d must be >= 1, got %d", i, d)
		}
	}
	pool.windowDilations = dilations
	return pool
}

// PadSame adds paddings on the edges of x such that in the end the output
// of the convolution has the same shapes as the input.
//
// This changes the default value of Strides to 1 if it is not set -- if set to something else, PadSame won't really
// output the same shapes as the input.
//
// The default is NoPadding.
func (pool *PoolBuilder) PadSame() *PoolBuilder {
	pool.paddings = nil
	pool.padSame = true
	return pool
}

// NoPadding removes any paddings, so if the pooling window > 1,
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

// Done finishes the PoolBuilder configuration and returns the pooled output.
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

	var windowDilations []int
	for _, d := range pool.windowDilations {
		if d > 1 {
			windowDilations = makeSlice("WindowDilation", 1, pool.windowDilations)
			break
		}
	}

	// spatialPaddings can only be calculated after we are sure about the channels positioning.
	spatialPaddings := pool.paddings
	if spatialPaddings == nil && pool.padSame {
		// Pad such that the output is shaped the same as the input.
		spatialPaddings = make([][2]int, pool.numSpatialDims)
		for dim := range spatialPaddings {
			dilation := 1
			if dim < len(pool.windowDilations) {
				dilation = pool.windowDilations[dim]
			}
			effectiveWindow := (pool.windowSizes[dim]-1)*dilation + 1
			spatialPaddings[dim][0] = (effectiveWindow - 1) / 2 // For an even-sized kernel, the padding is asymmetric.
			spatialPaddings[dim][1] = effectiveWindow / 2
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

	pooled := wrapNode(graphimpl.BackendReduceWindow(unwrapNode(pool.x), pool.reductionType,
		windowDimensions, strides, nil, windowDilations, paddings))

	// Take the mean.
	if pool.isMean && pool.reductionType == compute.ReduceOpSum {
		if len(paddings) == 0 {
			// If no padding, the number of elements to take the mean is fixed:
			totalWindowSize := 1
			for _, s := range pool.windowSizes {
				totalWindowSize *= s
			}
			pooled = DivScalar(pooled, float64(totalWindowSize))
		} else {
			pooled = takeMeanOfContributions(pool.x, pooled, pool.channelsAxis, windowDimensions, strides, windowDilations, paddings)
		}
	}

	return pooled
}

// takeMeanOfContributions divides the pooled sum by the number of contributions at each position.
func takeMeanOfContributions(x, pooledSum *Node, channelsAxis int, windowDimensions, strides, windowDilations []int, paddings [][2]int) *Node {
	// We need to normalize the sum by the number of elements that were actually used, ignoring the padding.
	// We use a similar BackendReduceWindow configuration, but as input a tensor with ones and dropping the
	// batch and channels axes, since they are the same.
	shapeNoBatchOrChannels := x.Shape().Clone()
	shapeNoBatchOrChannels.Dimensions[0] = 1
	shapeNoBatchOrChannels.Dimensions[channelsAxis] = 1
	if shapeNoBatchOrChannels.AxisNames != nil {
		shapeNoBatchOrChannels.AxisNames[0] = ""
		shapeNoBatchOrChannels.AxisNames[channelsAxis] = ""
	}
	ones := Ones(x.Graph(), shapeNoBatchOrChannels)
	pooledOnes := wrapNode(graphimpl.BackendReduceWindow(unwrapNode(ones), compute.ReduceOpSum,
		windowDimensions, strides, nil, windowDilations, paddings))
	pooledOnes = StopGradient(pooledOnes)
	return Div(pooledSum, pooledOnes)
}

func (pool *PoolBuilder) doConcat() *Node {
	x := pool.x
	rank := x.Rank()
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

	// Kernel order depends on the channels' position.
	if pool.channelsAxisConfig == images.ChannelsLast {
		// Kernel so far shaped [<spatial_dims...>, inputChannelsSize],
		kernelDims = append(kernelDims, inputChannelsSize)
	} else {
		// Kernel so far shaped [inputChannelsSize, <spatial_dims...>],
		kernelDims = append([]int{inputChannelsSize}, kernelDims...)
	}
	kernel = Reshape(kernel, kernelDims...)

	// Add the last axis to the kernel of outputChannelsSize, a one-hot encoding:
	kernel = OneHot(kernel, outputChannelsSize, dtype)
	if pool.channelsAxisConfig == images.ChannelsFirst {
		// Transposing from [in_channels, ... spatial .., out_channels] to [out_channels, in_channels, height, width]
		permutations := make([]int, rank)
		permutations[0] = rank - 1
		permutations[1] = 0
		for ii := 2; ii < rank; ii++ {
			permutations[ii] = ii - 1
		}
		kernel = TransposeAllAxes(kernel, permutations...)
	}

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

	// Convolve with the given kernel.
	convConfig := Convolve(x, kernel).ChannelsAxis(pool.channelsAxisConfig).StridePerAxis(strides...)
	if pool.paddings != nil {
		convConfig.PaddingPerDim(pool.paddings)
	}
	if pool.padSame {
		convConfig.PadSame()
	}
	return convConfig.Done()
}
