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
	timage "github.com/gomlx/gomlx/types/tensor/image"
)

// This file contains all parts of the layers.Convolve implementation.

// ConvBuilder is a helper to build a convolution computation. Create it with Convolution, set the desired parameters
// and when all is set, call Done.
type ConvBuilder struct {
	ctx                *context.Context
	graph              *Graph
	x                  *Node
	numSpatialDims     int
	channelsAxisConfig timage.ChannelsAxisConfig
	filters            int
	kernelSize         []int
	bias               bool
	strides            []int
	padSame            bool
	dilations          []int
	newScope           bool
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
// configured with `ConvBuilder.ChannelsAxis(timage.ChannelsLast)`, the default. If one
// sets `ConvBuilder.ChannelsAxis(timage.ChannelsFirst)`, the shape should be
// `[batch, input_channels, <spatial_dimensions...>]` instead.
func Convolution(ctx *context.Context, x *Node) *ConvBuilder {
	conv := &ConvBuilder{
		ctx:      ctx,
		graph:    x.Graph(),
		x:        x,
		newScope: true,
	}
	conv.numSpatialDims = x.Rank() - 2
	if conv.numSpatialDims < 0 {
		Panicf("Input x must have rank >= 3, shaped by default as [batch, <spatial_dimensions...>, channels], "+
			"but x rank is %d", x.Rank())
	}
	return conv.ChannelsAxis(timage.ChannelsLast).NoPadding().UseBias(true).Strides(1)
}

// Filters sets the number of filters -- specifies the number of output channels. There is no default
// and this number must be set, before Done is called.
func (conv *ConvBuilder) Filters(filters int) *ConvBuilder {
	conv.filters = filters
	if filters <= 0 {
		Panicf("number of filters must be > 0, it was set to %d", filters)
	}
	return conv
}

// KernelSize sets the kernel size for every axis. There is no default
// and this number must be set, before Done is called.
//
// You can also use KernelSizePerDim to set the kernel size per dimension (axis) individually.
func (conv *ConvBuilder) KernelSize(size int) *ConvBuilder {
	perDim := xslices.SliceWithValue(conv.numSpatialDims, size)
	return conv.KernelSizePerDim(perDim...)
}

// KernelSizePerDim sets the kernel size for each dimension(axis). There is no default
// and this number must be set, before Done is called.
//
// You can also use KernelSize to set the kernel size the same for all dimensions.
func (conv *ConvBuilder) KernelSizePerDim(sizes ...int) *ConvBuilder {
	if len(sizes) != conv.numSpatialDims {
		Panicf("received %d kernel sizes, but x has %d spatial dimensions",
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

// ChannelsAxis configures the axis for the channels (aka. "depth" or "features") dimension. The default is
// `timage.ChannelsLast`, meaning the "channels" dimension comes last.
//
// Note: `timage` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// It returns the modified Config object, so calls can be cascaded.
func (conv *ConvBuilder) ChannelsAxis(channelsAxisConfig timage.ChannelsAxisConfig) *ConvBuilder {
	conv.channelsAxisConfig = channelsAxisConfig
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
	perDim := xslices.SliceWithValue(conv.numSpatialDims, strides)
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
	if len(strides) != conv.numSpatialDims {
		Panicf("received %d strides in StridePerAxis, but x has %d spatial dimensions",
			len(strides), conv.numSpatialDims)
		return conv
	}
	conv.strides = strides
	return conv
}

// Dilations sets the dilations of the convolution. It sets the same value for every dimension.
// The default is 1.
//
// It specifies the kernel up-sampling rate. In the literature, the same parameter
// is sometimes called input stride or dilation. The effective kernel size used for the convolution
// will be `kernel_shape + (kernel_shape - 1) * (dilation - 1)`, obtained by inserting (dilation-1) zeros
// between consecutive elements of the original filter in the spatial dimension.
//
// One cannot use strides and dilation at the same time.
func (conv *ConvBuilder) Dilations(dilation int) *ConvBuilder {
	dilationsPerDim := xslices.SliceWithValue(conv.numSpatialDims, dilation)
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
	if len(dilations) != conv.numSpatialDims {
		Panicf("received %d dilations in DilationPerDim, but x has %d spatial dimensions",
			len(dilations), conv.numSpatialDims)
		return conv
	}
	conv.dilations = dilations
	return conv
}

// CurrentScope configures the convolution not to create a sub-scope for the kernel weights it needs,
// and instead use the current one provided in Convolution.
//
// By default, Convolution will create a sub-scope named "conv".
func (conv *ConvBuilder) CurrentScope() *ConvBuilder {
	conv.newScope = false
	return conv
}

// Done indicates that the Convolution layer is finished being configured. It then
// creates the convolution and it's kernels (variables) and returns the resulting
// Node.
func (conv *ConvBuilder) Done() *Node {
	// Default is to create a sub-scope for the convolution variables.
	ctxInScope := conv.ctx
	if conv.newScope {
		ctxInScope = ctxInScope.In("conv")
	}

	if len(conv.kernelSize) == 0 || conv.filters <= 0 {
		Panicf("layers.Convolution requires Filters and KernelSize to be set")
	}
	if conv.numSpatialDims <= 0 {
		Panicf("invalid x shape %s, can't figure spatial dimensions", conv.x.Shape())
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
		Panicf("both strides (%v) and dilations (%v) are set, but only one can be used at a time",
			conv.strides, conv.dilations)
	}

	// Create and apply kernel.
	xShape := conv.x.Shape()
	dtype := xShape.DType
	kernelShape := shapes.Make(dtype)
	kernelShape.Dimensions = make([]int, 0, conv.numSpatialDims+2)
	channelsAxis := timage.GetChannelsAxis(xShape, conv.channelsAxisConfig)
	inputChannels := xShape.Dimensions[channelsAxis]
	if conv.channelsAxisConfig == timage.ChannelsFirst {
		kernelShape.Dimensions = append(kernelShape.Dimensions, inputChannels)
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.kernelSize...)
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.filters)
	} else {
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.kernelSize...)
		kernelShape.Dimensions = append(kernelShape.Dimensions, inputChannels)
		kernelShape.Dimensions = append(kernelShape.Dimensions, conv.filters)
	}
	kernelVar := ctxInScope.VariableWithShape("weights", kernelShape)
	kernel := kernelVar.ValueGraph(conv.graph)
	convOpts := Convolve(conv.x, kernel).StridePerDim(conv.strides...).ChannelsAxis(conv.channelsAxisConfig)
	if len(conv.dilations) > 0 {
		convOpts.DilationPerDim(conv.dilations...)
	}
	if conv.padSame {
		convOpts.PadSame()
	} else {
		convOpts.NoPadding()
	}
	output := convOpts.Done()

	// Create and apply bias.
	if conv.bias {
		biasVar := ctxInScope.VariableWithShape("biases", shapes.Make(dtype, conv.filters))
		bias := biasVar.ValueGraph(conv.graph)
		expandedDims := xslices.SliceWithValue(output.Rank(), 1)
		outputChannelsAxis := timage.GetChannelsAxis(output, conv.channelsAxisConfig)
		expandedDims[outputChannelsAxis] = conv.filters
		bias = Reshape(bias, expandedDims...)
		output = Add(output, bias)
	}

	// Add regularization.
	if l2any, found := ctxInScope.GetParam(ParamL2Regularization); found {
		l2 := l2any.(float64)
		if l2 > 0 {
			l2Node := Const(conv.graph, shapes.CastAsDType(l2, dtype))
			AddL2Regularization(ctxInScope, l2Node, kernel)
		}
	}
	return output
}
