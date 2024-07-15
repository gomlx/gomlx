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
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	timage "github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gomlx/xla"
)

// This file contains all parts of the {Max|Sum|Prod}Pool implementation.

// PoolBuilder is a helper to build a pool computation.
// Create it with {Max|Sum|Mean|Prod}Pool, set the desired parameters and
// when set, call `IsNil()`.
type PoolBuilder struct {
	graph                *Graph
	x                    *Node
	reductionType        xla.NodeType
	numSpatialDims       int
	channelsAxisConfig   timage.ChannelsAxisConfig
	spatialAxes          []int // Indices of spatial axes.
	channelsAxis         int
	windowSizes, strides []int
	paddings             [][2]int
	padSame              bool
	isMean               bool // Divide by number of elements later if mean.
}

// MaxPool prepares a max pooling on x with the given kernel for arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.). It returns the max value
// for the selected window, on given strides.
//
// It is very flexible and to ease setting its parameters it returns a
// PoolBuilder for configuration. Once it is set up
// call `PoolBuilder.Done` and it will return the pooled
// x. Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerDim.
//
// The shape of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(timage.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(timage.ChannelsFirst)`, the shape should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as depth or feature axis.
//
// Note: `timage` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// The shape of kernel should be `[<spatial_dimensions...>, input_channels, output_channels]` if
// configured with `PoolBuilder.ChannelsAxis(timage.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(timage.Channels)`, the shape should be
// `[input_channels, <spatial_dimensions...>, output_channels]` instead.
func MaxPool(x *Node) *PoolBuilder {
	return makePoolBuilder(x, xla.ReduceMaxNode)
}

// SumPool prepares a sum pooling on x with the given kernel for arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.). It returns the sum value
// for the selected window, on given strides.
//
// It is very flexible and to ease setting its parameters it returns a
// PoolBuilder for configuration. Once it is set up
// call `PoolBuilder.Done` and it will return the pooled
// x. Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerDim.
//
// The shape of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(timage.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(timage.ChannelsFirst)`, the shape should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as depth or feature axis.
//
// Note: `timage` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// The shape of kernel should be `[<spatial_dimensions...>, input_channels, output_channels]` if
// configured with `PoolBuilder.ChannelsAxis(timage.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(timage.Channels)`, the shape should be
// `[input_channels, <spatial_dimensions...>, output_channels]` instead.
func SumPool(x *Node) *PoolBuilder {
	return makePoolBuilder(x, xla.ReduceSumNode)
}

// MeanPool prepares a mean pooling on x with the given kernel for arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.). It returns the mean value
// for the selected window, on given strides.
//
// It is very flexible and to ease setting its parameters it returns a
// PoolBuilder for configuration. Once it is set up
// call `PoolBuilder.Done` and it will return the pooled
// x. Browse through PoolBuilder to see the capabilities, and the defaults.
//
// The window sizes must be set with PoolBuilder.Window or PoolBuilder.WindowPerDim.
//
// The shape of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `PoolBuilder.ChannelsAxis(timage.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(timage.ChannelsFirst)`, the shape should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
//
// The "channels" axis is also known as depth or feature axis.
//
// Note: `timage` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// The shape of kernel should be `[<spatial_dimensions...>, input_channels, output_channels]` if
// configured with `PoolBuilder.ChannelsAxis(timage.ChannelsLast)`, the default. If one
// sets `PoolBuilder.ChannelsAxis(timage.Channels)`, the shape should be
// `[input_channels, <spatial_dimensions...>, output_channels]` instead.
func MeanPool(x *Node) *PoolBuilder {
	pool := makePoolBuilder(x, xla.ReduceSumNode)
	pool.isMean = true
	return pool
}

func makePoolBuilder(x *Node, reductionType xla.NodeType) *PoolBuilder {
	g := validateBuildingGraphFromInputs(x)
	pool := &PoolBuilder{
		graph:         g,
		x:             x,
		reductionType: reductionType,
	}
	pool.numSpatialDims = x.Rank() - 2
	if pool.numSpatialDims <= 0 {
		Panicf("Input x must have rank >= 3, shaped by default as [batch, <spatial_dimensions...>, channels] (alternatively channels come first), "+
			"but x rank is %d", x.Rank())
	}
	return pool.ChannelsAxis(timage.ChannelsLast).NoPadding()
}

// ChannelsAxis configures the axis for the channels (aka. "depth" or "features") dimension. The default is
// `timage.ChannelsLast`, meaning the "channels" dimension comes last.
//
// Note: `timage` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// It returns the modified Config object, so calls can be cascaded.
func (pool *PoolBuilder) ChannelsAxis(channelsAxisConfig timage.ChannelsAxisConfig) *PoolBuilder {
	pool.channelsAxisConfig = channelsAxisConfig
	pool.channelsAxis = timage.GetChannelsAxis(pool.x, channelsAxisConfig)
	pool.spatialAxes = timage.GetSpatialAxes(pool.x, channelsAxisConfig)
	return pool
}

// Window sets the pooling window size for all spatial dimensions to the same windowSize.
//
// There is no default, and this must be set either with Window or WindowPerDim.
func (pool *PoolBuilder) Window(windowSize int) *PoolBuilder {
	windowSizes := make([]int, pool.numSpatialDims)
	for ii := range windowSizes {
		windowSizes[ii] = windowSize
	}
	return pool.WindowPerDim(windowSizes...)
}

// WindowPerDim sets the pooling window size for each spatial dimension.
//
// There is no default, and this must be set either with Window or WindowPerDim.
func (pool *PoolBuilder) WindowPerDim(sizes ...int) *PoolBuilder {
	if len(sizes) != pool.numSpatialDims {
		Panicf("received %d window sizes in WindowPerDim, but x has %d spatial dimensions",
			len(sizes), pool.numSpatialDims)
	}
	pool.windowSizes = sizes
	return pool
}

// Strides sets the strides of the pooling. It sets the same value for every spatial dimension.
//
// The default is the same value as the window size (set with Window or WindowPerDim).
//
// The stride is how many steps to move after the pooling of a window. A value of 2 will halve the
// input size, since the pooling will be done at every other position, and so on. It can be defined
// separately per dimension with StridePerDim.
//
// One cannot use strides and dilation at the same time.
func (pool *PoolBuilder) Strides(strides int) *PoolBuilder {
	stridesPerAxis := make([]int, pool.numSpatialDims)
	for ii := range stridesPerAxis {
		stridesPerAxis[ii] = strides
	}
	return pool.StridePerDim(stridesPerAxis...)
}

// StridePerDim sets the strides for each spatial dimension of the pooling.
//
// The default is the same value as the window size (set with Window or WindowPerDim).
//
// The stride is how many steps to move after a pooling. A value of 2 will half the input
// size, since a pooling will be done at every other position, and so on. It can be defined
// separately per dimension.
//
// One cannot use strides and dilation at the same time.
func (pool *PoolBuilder) StridePerDim(strides ...int) *PoolBuilder {
	if len(strides) != pool.numSpatialDims {
		Panicf("received %d strides in StridePerDim, but x has %d spatial dimensions",
			len(strides), pool.numSpatialDims)
	}
	pool.strides = strides
	return pool
}

// PadSame adds paddings on the edges of x such that in the end the output
// of the convolution has the same shape as the input (assuming strides=1).
// The default is NoPadding.
func (pool *PoolBuilder) PadSame() *PoolBuilder {
	pool.paddings = nil
	pool.padSame = true
	return pool
}

// NoPadding removes any paddings, so if the kernel spatial dimensions > 1,
// the output shape will be reduced on the edges.
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

	// Closure to create slice with value for every axis, using a default value
	// and the corresponding spatial values.
	makeSlice := func(d int, valuesForSpatialDims []int) []int {
		s := make([]int, rank)
		for ii := range s {
			s[ii] = d
		}
		if len(valuesForSpatialDims) > 0 {
			for ii, axis := range pool.spatialAxes {
				s[axis] = valuesForSpatialDims[ii]
			}
		}
		return s
	}

	// windowSizes is obligatory.
	if len(pool.windowSizes) == 0 {
		Panicf("window sizes required but not configured -- use .Window() or .WindowPerDim()")
	}
	windowDimensions := makeSlice(1, pool.windowSizes)

	// strides default to pooling window sizes.
	var strides []int
	if len(pool.strides) > 0 {
		strides = makeSlice(1, pool.strides)
	} else {
		strides = makeSlice(1, pool.windowSizes)
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
		for ii, axis := range pool.spatialAxes {
			paddings[axis] = spatialPaddings[ii]
		}
	}
	pooled := reduceWindowXLA(pool.x, pool.reductionType,
		windowDimensions, strides, nil, nil, paddings)

	// Take the mean.
	if pool.isMean && pool.reductionType == xla.ReduceSumNode {
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
	// We use a similar reduceWindowXLA configuration, but as input a tensor with 1s and dropping the
	// batch and channels axes, since they will are the same.
	shapeNoBatchOrChannels := x.shape.Clone()
	shapeNoBatchOrChannels.Dimensions[0] = 1
	shapeNoBatchOrChannels.Dimensions[channelsAxis] = 1
	ones := Ones(x.graph, shapeNoBatchOrChannels)
	pooledOnes := reduceWindowXLA(ones, xla.ReduceSumNode,
		windowDimensions, strides, nil, nil, paddings)
	pooledOnes = StopGradient(pooledOnes)
	return Div(pooledSum, pooledOnes)
}

// reduceWindowXLA runs a reduction function, here limited to the value given by reductionType,
// it can be either ReduceMaxNode, ReduceSumNode or ReduceMultiplyNode.
//
// The parameter windowDimensions must be set and have a value for each axis. The parameters `strides`, `baseDilations`
// and `windowDilations` and `paddings` can be left as nil if not used.
func reduceWindowXLA(x *Node, reductionType xla.NodeType, windowDimensions, strides, baseDilations, windowDilations []int, paddings [][2]int) *Node {
	g := validateBuildingGraphFromInputs(x)
	dtype := x.DType()
	rank := x.Rank()
	if len(windowDimensions) != rank {
		Panicf("windowSizes (length %d) must have same length as rank of input x (rank %d)", len(windowDimensions), rank)
	}
	if len(strides) != 0 && len(strides) != rank {
		Panicf("strides (length %d) must have same length as rank of input x (rank %d)", len(strides), rank)
	}
	if len(paddings) > 0 && len(paddings) != rank {
		Panicf("paddings (length %d) if given must have same length as rank of input x (rank %d)", len(paddings), rank)
	}

	// Define the initial value for the reduction:
	var init *Node
	switch reductionType {
	case xla.ReduceMaxNode:
		init = lowestForDType(g, dtype)
	case xla.ReduceSumNode:
		init = ScalarZero(g, dtype)
	case xla.ReduceMultiplyNode:
		init = ScalarOne(g, dtype)
	default:
		Panicf("unsupported type of ReduceWindow: %s -- only %s, %s and %s are supported",
			reductionType, xla.ReduceMaxNode, xla.ReduceSumNode, xla.ReduceMultiplyNode)
	}

	// `strides` must always have rank elements. It defaults to 1.
	if strides == nil {
		strides = xslices.SliceWithValue[int](rank, 1)
	}

	// Encode parameters in ints.
	ints := make([]int, 0, 4+2*rank+len(baseDilations)+len(windowDilations)+2*len(paddings))
	encode := func(values ...int) {
		ints = append(ints, values...)
	}
	encode(rank, len(baseDilations), len(windowDilations), len(paddings))
	encode(windowDimensions...) // rank elements.
	encode(strides...)          // rank elements.
	encode(baseDilations...)
	encode(windowDilations...)
	for _, pair := range paddings {
		encode(pair[0], pair[1])
	}

	//fmt.Printf("ReduceWindow(%s):\n", reductionType)
	//fmt.Printf("\tx.shape=%s\n", x.Shape())
	//fmt.Printf("\twindowDimensions=%v\n", windowDimensions)
	//fmt.Printf("\tstrides=%v\n", strides)
	//fmt.Printf("\tpaddings=%v\n", paddings)

	// Create graph new node.
	output := newNode(g, &xla.SerializedNode{
		Type: xla.ReduceWindowNode,
		Int:  int(reductionType),
		Ints: ints,
	}, []*Node{x, init})

	//fmt.Printf("\toutput shape=%s\n", output.shape)
	//fmt.Println()
	return output
}

// selectAndScatterWithGeneralPaddingXLA selects (largest) element from a window and scatter to those positions
// the value from source. It's used to calculate the gradient of a MaxPool.
func selectAndScatterWithGeneralPaddingXLA(x, source *Node, windowDimensions, strides []int, paddings [][2]int) *Node {
	g := validateBuildingGraphFromInputs(x, source)
	dtype := x.DType()
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

	init := ScalarZero(g, dtype)
	ints := make([]int, 0, 2+2*rank+2*len(paddings))
	encode := func(values ...int) {
		ints = append(ints, values...)
	}
	encode(rank, len(paddings))
	encode(windowDimensions...)
	encode(strides...)
	for _, pair := range paddings {
		encode(pair[0], pair[1])
	}

	return newNode(g, &xla.SerializedNode{
		Type: xla.SelectAndScatterNode,
		Ints: ints,
	}, []*Node{x, source, init})
}

// reduceWindowVJP calculates v*d(reduceWindow(x))/{dx, d_kernel).
func reduceWindowVJP(node, v *Node, _ shapes.Shape) []*Node {
	// Recover parameters from serialized node.
	x := node.inputNodes[0]
	initValue := node.inputNodes[1]
	reductionType := xla.NodeType(node.serializedNode.Int)
	if reductionType != xla.ReduceMaxNode && reductionType != xla.ReduceSumNode {
		Panicf("ReduceWindow gradient only defined for ReduceMax or ReduceSum operation, instead got %s", reductionType)
	}

	packedPos := 0
	decode := func() int {
		i := node.serializedNode.Ints[packedPos]
		packedPos++
		return i
	}
	decodeN := func(n int) []int {
		slice := node.serializedNode.Ints[packedPos : packedPos+n]
		packedPos += n
		return slice
	}

	rank := decode()
	lenBaseDilations := decode()
	if lenBaseDilations > 0 {
		Panicf("reduceWindow(%s) does not define a gradient if using baseDilations", reductionType)
		return nil
	}
	lenWindowDilations := decode()
	if lenWindowDilations > 0 {
		Panicf("reduceWindow(%s) does not define a gradient if using windowDilations", reductionType)
		return nil
	}
	lenPaddings := decode()

	windowDimensions := decodeN(rank)
	strides := decodeN(rank)
	baseDilations := decodeN(lenBaseDilations)
	windowDilations := decodeN(lenWindowDilations)
	paddings := make([][2]int, lenPaddings)
	for ii := range paddings {
		paddings[ii][0] = decode()
		paddings[ii][1] = decode()
	}

	// Not used.
	_, _ = baseDilations, windowDilations

	if lenBaseDilations > 0 || lenWindowDilations > 0 {
		Panicf("gradient of ReduceWindow with base or window dilation not defined")
	}

	//fmt.Printf("Grad(reduceWindow(%s):\n", reductionType)
	//fmt.Printf("\tx.shape=%s\n", x.Shape())
	//fmt.Printf("\tnode.shape=%s\n", node.Shape())
	//fmt.Printf("\tv.shape=%s\n", v.Shape())
	//fmt.Printf("\twindowDimensions=%v\n", windowDimensions)
	//fmt.Printf("\tstrides=%v\n", strides)
	//fmt.Printf("\tpaddings=%v\n", paddings)

	var vjpX *Node
	if reductionType == xla.ReduceMaxNode {
		vjpX = selectAndScatterWithGeneralPaddingXLA(x, v, windowDimensions, strides, paddings)
	} else if reductionType == xla.ReduceSumNode {
		vjpX = dilateConvolveToMatchSumPooling(x, v, windowDimensions, strides, paddings)
	} else {
		Panicf("ReduceWindow gradient only defined for ReduceMax or ReduceSum operation, instead got %s", reductionType)
	}
	return []*Node{vjpX, ZerosLike(initValue)}
}

// dilateConvolveToMatchSumPooling convolves the `backProp` to match `x`.
//
// Since the convolution would be with a kernel of 1s we instead use `graph.Pad` and reduceWindowXLA instead.
func dilateConvolveToMatchSumPooling(x, backProp *Node, windowDimensions, strides []int, paddings [][2]int) *Node {
	g := validateBuildingGraphFromInputs(x, backProp)
	dtype := x.DType()
	rank := x.Rank()
	//dims := x.Shape().Dimensions

	if len(windowDimensions) != rank {
		Panicf("windowSizes (length %d) must have same length as rank of input x (rank %d)", len(windowDimensions), rank)
	}
	if len(strides) != rank {
		Panicf("strides (length %d) must have same length as rank of input x (rank %d)", len(strides), rank)
	}
	if len(paddings) > 0 && len(paddings) != rank {
		Panicf("paddings (length %d) if given must have same length as rank of input x (rank %d)", len(paddings), rank)
	}

	// Configure the padding needed to expand the backprop back to its original size.
	padConfig := make([]PadAxis, rank)
	for axis := range padConfig {
		conf := &padConfig[axis]

		// pad due to strides and window sizes: this reconstructs what was used from
		// the original dimension -- some may be padding ... but some padding (not used) may
		// not be included.
		conf.Interior = strides[axis] - 1
		conf.Start = (windowDimensions[axis]) / 2 // Notice that we switch the long/short half from start/end.
		conf.End = (windowDimensions[axis] - 1) / 2
	}
	// expanded should have the same spatial dimensions as the original input,
	// but with zeros filled in between.
	zero := ScalarZero(g, dtype)
	expanded := Pad(backProp, zero, padConfig...)

	// For each position on expanded, just sum the values that reach it.
	padSame := make([][2]int, rank) // pad to preserve expanded shape.
	for axis := range padSame {
		windowSize := windowDimensions[axis]    // for this axis.
		padSame[axis][0] = (windowSize - 1) / 2 // For even sized kernels, the padding is asymmetric.
		padSame[axis][1] = windowSize / 2
	}
	expanded = reduceWindowXLA(expanded, xla.ReduceSumNode, windowDimensions, nil, nil, nil, padSame)

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
		*conf = PadAxis{0, 0, 0}

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
