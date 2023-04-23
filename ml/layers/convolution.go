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

package layers

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
)

// This file contains all parts of the layers.Convolve implementation.

// ConvBuilder is a helper to build a convolution computation. Create it with Convolution, set the desired parameters
// and when all is set, call Done.
type ConvBuilder struct {
	ctx            *context.Context
	graph          *Graph
	x              *Node
	numSpatialDims int
	channelsFirst  bool
	filters        int
	kernelSize     []int
	bias           bool
	strides        []int
	padSame        bool
	dilations      []int
}

// Convolution prepares a convolution on x with the given kernel for arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.).
//
// It is very flexible and to ease setting its parameters it returns a ConvBuilder object for configuration. Once it is
// set up call `ConvBuilder.Done` and it will return the convolved x. Browse through ConvBuilder to see the
// capabilities, and the defaults.
//
// Two parameters need setting: Filters (or channels) and KernelSize. It will fail
// if they are not set.
//
// The shape of x should be `[batch, <spatial_dimensions...>, input_channels]` if
// configured with `ConvBuilder.ChannelsAfter()`, the default. If one
// sets `ConvBuilder.ChannelsFirst()`, the shape should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
func Convolution(ctx *context.Context, x *Node) *ConvBuilder {
	conv := &ConvBuilder{
		ctx:   ctx.In("conv"),
		graph: x.Graph(),
		x:     x,
	}
	conv.numSpatialDims = x.Rank() - 2
	if conv.numSpatialDims < 0 {
		conv.graph.SetErrorf("Input x must have rank >= 3, shaped by default as [batch, <spatial_dimensions...>, channels], "+
			"but x rank is %d", x.Rank())
	}
	return conv.ChannelsAfter().NoPadding().UseBias(true).Strides(1)
}

// Filters sets the number of filters -- specifies the number of output channels. There is no default
// and this number must be set, before Done is called.
func (conv *ConvBuilder) Filters(filters int) *ConvBuilder {
	conv.filters = filters
	if filters <= 0 {
		conv.graph.SetErrorf("number of filters must be > 0, it was set to %d", filters)
	}
	return conv
}

// KernelSize sets the kernel size for every axis. There is no default
// and this number must be set, before Done is called.
//
// You can also use KernelSizePerDim to set the kernel size per dimension (axis) individually.
func (conv *ConvBuilder) KernelSize(size int) *ConvBuilder {
	if !conv.graph.Ok() {
		return conv
	}
	perDim := slices.SliceWithValue(conv.numSpatialDims, size)
	return conv.KernelSizePerDim(perDim...)
}

// KernelSizePerDim sets the kernel size for each dimension(axis). There is no default
// and this number must be set, before Done is called.
//
// You can also use KernelSize to set the kernel size the same for all dimensions.
func (conv *ConvBuilder) KernelSizePerDim(sizes ...int) *ConvBuilder {
	if len(sizes) != conv.numSpatialDims {
		conv.graph.SetErrorf("received %d kernel sizes, but x has %d spatial dimensions",
			len(sizes), conv.numSpatialDims)
		return conv
	}
	conv.kernelSize = sizes
	return conv
}

// UseBias sets whether to add a trainable bias term to the convolution. Default is true.
func (conv *ConvBuilder) UseBias(useBias bool) *ConvBuilder {
	conv.bias = useBias
	return conv
}

// ChannelsFirst specify the order of the dimensions for x and kernel.
// The default is ChannelsAfter.
//
// If this is set x should be shaped `[batch, channels, <spatial_dimensions...>]`.
func (conv *ConvBuilder) ChannelsFirst() *ConvBuilder {
	conv.channelsFirst = true
	return conv
}

// ChannelsAfter specify the order of the dimensions for x and kernel.
// This is the default.
//
// If this is set x should be shaped `[batch, <spatial_dimensions...>, channels]`.
func (conv *ConvBuilder) ChannelsAfter() *ConvBuilder {
	conv.channelsFirst = false
	return conv
}

// PadSame adds paddings on the edges of x such that in the end the output
// of the convolution has the same shape as the input (assuming strides=1).
//
// The default is NoPadding.
func (conv *ConvBuilder) PadSame() *ConvBuilder {
	conv.padSame = true
	return conv
}

// NoPadding removes any paddings, so if the kernel spatial dimensions > 1,
// the output shape will be reduced on the edges.
//
// This is the default.
func (conv *ConvBuilder) NoPadding() *ConvBuilder {
	if !conv.graph.Ok() {
		return conv
	}
	conv.padSame = false
	return conv
}

// Strides sets the strides of the convolution. It sets the same value for every dimension.
// The default is 1.
//
// The stride is how many steps to move after a convolution. A value of 2 will half the input
// size, since a convolution will be done at every other position, and so on. It can be defined
// separately per dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvBuilder) Strides(strides int) *ConvBuilder {
	if !conv.graph.Ok() {
		return conv
	}
	perDim := slices.SliceWithValue(conv.numSpatialDims, strides)
	return conv.StridePerDim(perDim...)
}

// StridePerDim sets the strides for each spatial dimension of the convolution.
// The default is 1 for every dimension.
//
// The stride is how many steps to move after a convolution. A value of 2 will half the input
// size, since a convolution will be done at every other position, and so on. It can be defined
// separately per dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvBuilder) StridePerDim(strides ...int) *ConvBuilder {
	if !conv.graph.Ok() {
		return conv
	}
	if len(strides) != conv.numSpatialDims {
		conv.graph.SetErrorf("received %d strides in StridePerDim, but x has %d spatial dimensions",
			len(strides), conv.numSpatialDims)
		return conv
	}
	conv.strides = strides
	return conv
}

// Dilations sets the dilations of the convolution. It sets the same value for every dimension.
// The default is 1.
//
// Specifies the kernel up-sampling rate. In the literature, the same parameter
// is sometimes called input stride or dilation. The effective kernel size used for the convolution
// will be `kernel_shape + (kernel_shape - 1) * (dilation - 1)`, obtained by inserting (dilation-1) zeros
// between consecutive elements of the original filter in the spatial dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvBuilder) Dilations(dilation int) *ConvBuilder {
	if !conv.graph.Ok() {
		return conv
	}
	dilationsPerDim := slices.SliceWithValue(conv.numSpatialDims, dilation)
	return conv.DilationPerDim(dilationsPerDim...)
}

// DilationPerDim sets the kernel dilations for each spatial dimension of the convolution.
// The default is 1 for every dimension.
//
// Specifies the kernel up-sampling rate. In the literature, the same parameter
// is sometimes called input stride or dilation. The effective kernel size used for the convolution
// will be `kernel_shape + (kernel_shape - 1) * (dilation - 1)`, obtained by inserting (dilation-1) zeros
// between consecutive elements of the original filter in the spatial dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvBuilder) DilationPerDim(dilations ...int) *ConvBuilder {
	if !conv.graph.Ok() {
		return conv
	}
	if len(dilations) != conv.numSpatialDims {
		conv.graph.SetErrorf("received %d dilations in DilationPerDim, but x has %d spatial dimensions",
			len(dilations), conv.numSpatialDims)
		return conv
	}
	conv.dilations = dilations
	return conv
}

// Done indicates that the Convolution layer is finished being configured. It then
// creates the convolution and it's kernels (variables) and returns the resulting
// Node.
func (conv *ConvBuilder) Done() *Node {
	if !conv.graph.Ok() {
		return conv.graph.InvalidNode()
	}

	if len(conv.kernelSize) == 0 || conv.filters <= 0 {
		conv.graph.SetErrorf("layers.Convolution requires Filters and KernelSize to be set")
		return conv.graph.InvalidNode()
	}
	if conv.numSpatialDims <= 0 {
		conv.graph.SetErrorf("invalid x shape %s, can't figure spatial dimensions", conv.x.Shape())
		return conv.graph.InvalidNode()
	}

	// Check only one of strides / dilations are set.
	var dilationsSet, stridesSet bool
	if conv.strides != nil {
		for _, stride := range conv.strides {
			if stride != 1 {
				stridesSet = true
			}
		}
	}
	if conv.dilations != nil {
		for _, dilation := range conv.dilations {
			if dilation != 1 {
				dilationsSet = true
			}
		}
	}
	if dilationsSet && stridesSet {
		conv.graph.SetErrorf("both strides (%v) and dilations (%v) are set, but only one can be used at a time",
			conv.strides, conv.dilations)
		return conv.graph.InvalidNode()
	}

	// Create and apply kernel.
	xShape := conv.x.Shape()
	dtype := xShape.DType
	kernelShape := shapes.Make(dtype)
	kernelShape.Dimensions = make([]int, 0, conv.numSpatialDims+2)
	if conv.channelsFirst {
		inputChannels := xShape.Dimensions[1]
		kernelShape.Dimensions = append(kernelShape.Dimensions, inputChannels)
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.kernelSize...)
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.filters)
	} else {
		inputChannels := xShape.Dimensions[xShape.Rank()-1]
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.kernelSize...)
		kernelShape.Dimensions = append(kernelShape.Dimensions, inputChannels)
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.filters)
	}
	kernelVar := conv.ctx.VariableWithShape("weights", kernelShape)
	kernel := kernelVar.ValueGraph(conv.graph)
	convOpts := Convolve(conv.x, kernel).StridePerDim(conv.strides...)
	if conv.channelsFirst {
		convOpts.ChannelsFirst()
	} else {
		convOpts.ChannelsAfter()
	}
	if len(conv.dilations) > 0 {
		convOpts.DilationPerDim(conv.dilations...)
	}
	if conv.padSame {
		convOpts.PadSame()
	} else {
		convOpts.NoPadding()
	}
	output := convOpts.Done()
	if !conv.graph.Ok() {
		return output
	}

	// Create and apply bias.
	if conv.bias {
		biasVar := conv.ctx.VariableWithShape("biases", shapes.Make(dtype, conv.filters))
		bias := biasVar.ValueGraph(conv.graph)
		expandedDims := slices.SliceWithValue(output.Rank(), 1)
		if conv.channelsFirst {
			expandedDims[1] = conv.filters
		} else {
			expandedDims[output.Rank()-1] = conv.filters
		}
		bias = Reshape(bias, expandedDims...)
		output = Add(output, bias)
	}

	// Add regularization.
	if l2any, found := conv.ctx.GetParam(L2RegularizationKey); found {
		l2 := l2any.(float64)
		if l2 > 0 {
			l2Node := Const(conv.graph, shapes.CastAsDType(l2, dtype))
			AddL2Regularization(conv.ctx, l2Node, kernel)
		}
	}
	return output
}
