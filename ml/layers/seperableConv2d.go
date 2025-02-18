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
	"log"

	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gomlx/types/xslices"
)

// SeparableConvBuilder is a helper to build a separable convolution computation.
type SeparableConvBuilder struct {
	ctx                *context.Context
	graph              *Graph
	x                  *Node
	numSpatialDims     int
	channelsAxisConfig images.ChannelsAxisConfig
	filters            int
	kernelSize         []int
	depthMultiplier    int
	bias               bool
	strides            []int
	padSame            bool
	dilations          []int
	newScope           bool
	regularizer        regularizers.Regularizer
}

// SeparableConvolution prepares a separable convolution layer on x.
func SeparableConvolution(ctx *context.Context, x *Node) *SeparableConvBuilder {
	conv := &SeparableConvBuilder{
		ctx:             ctx,
		graph:           x.Graph(),
		x:               x,
		newScope:        true,
		regularizer:     regularizers.FromContext(ctx),
		depthMultiplier: 1,
	}
	conv.numSpatialDims = x.Rank() - 2
	if conv.numSpatialDims < 0 {
		Panicf("Input x must have rank >= 3, got %d", x.Rank())
	}
	return conv.ChannelsAxis(images.ChannelsLast).NoPadding().UseBias(true).Strides(1)
}

// DepthMultiplier sets the depth multiplier for depthwise convolution.
func (sep *SeparableConvBuilder) DepthMultiplier(multiplier int) *SeparableConvBuilder {
	sep.depthMultiplier = multiplier
	return sep
}

// Filters sets the number of output filters.
func (sep *SeparableConvBuilder) Filters(filters int) *SeparableConvBuilder {
	sep.filters = filters
	return sep
}

// KernelSize sets the kernel size for all spatial dimensions.
func (sep *SeparableConvBuilder) KernelSize(size int) *SeparableConvBuilder {
	sep.kernelSize = xslices.SliceWithValue(sep.numSpatialDims, size)
	return sep
}

// KernelSizePerDim sets the kernel size per dimension.
func (sep *SeparableConvBuilder) KernelSizePerDim(sizes ...int) *SeparableConvBuilder {
	if len(sizes) != sep.numSpatialDims {
		Panicf("expected %d kernel sizes, got %d", sep.numSpatialDims, len(sizes))
	}
	sep.kernelSize = sizes
	return sep
}

// UseBias enables/disables the bias term.
func (sep *SeparableConvBuilder) UseBias(useBias bool) *SeparableConvBuilder {
	sep.bias = useBias
	return sep
}

// ChannelsAxis sets the channels' position.
func (sep *SeparableConvBuilder) ChannelsAxis(config images.ChannelsAxisConfig) *SeparableConvBuilder {
	sep.channelsAxisConfig = config
	return sep
}

// PadSame enables same padding.
func (sep *SeparableConvBuilder) PadSame() *SeparableConvBuilder {
	sep.padSame = true
	return sep
}

// NoPadding disables padding.
func (sep *SeparableConvBuilder) NoPadding() *SeparableConvBuilder {
	sep.padSame = false
	return sep
}

// Strides sets the stride for all spatial dimensions.
func (sep *SeparableConvBuilder) Strides(stride int) *SeparableConvBuilder {
	sep.strides = xslices.SliceWithValue(sep.numSpatialDims, stride)
	return sep
}

// StridePerDim sets strides per dimension.
func (sep *SeparableConvBuilder) StridePerDim(strides ...int) *SeparableConvBuilder {
	log.Println("---------------------------------------")
	log.Println("\nStrides:", strides, "\nLen of strides:", len(strides), "\nConvolutionBuilder:", sep.numSpatialDims)
	if len(strides) != sep.numSpatialDims {
		Panicf("expected %d strides, got %d", sep.numSpatialDims, len(strides))
	}
	sep.strides = strides
	return sep
}

// Dilations sets dilation for all dimensions.
func (sep *SeparableConvBuilder) Dilations(dilation int) *SeparableConvBuilder {
	sep.dilations = xslices.SliceWithValue(sep.numSpatialDims, dilation)
	return sep
}

// DilationPerDim sets dilations per dimension.
func (sep *SeparableConvBuilder) DilationPerDim(dilations ...int) *SeparableConvBuilder {
	if len(dilations) != sep.numSpatialDims {
		Panicf("expected %d dilations, got %d", sep.numSpatialDims, len(dilations))
	}
	sep.dilations = dilations
	return sep
}

// CurrentScope uses the current scope for variables.
func (sep *SeparableConvBuilder) CurrentScope() *SeparableConvBuilder {
	sep.newScope = false
	return sep
}

// Regularizer sets the kernel regularizer.
func (sep *SeparableConvBuilder) Regularizer(regularizer regularizers.Regularizer) *SeparableConvBuilder {
	sep.regularizer = regularizer
	return sep
}

// Done constructs the separable convolution.
func (sep *SeparableConvBuilder) Done() *Node {
	if len(sep.kernelSize) == 0 || sep.filters <= 0 || sep.depthMultiplier <= 0 {
		Panicf("Filters, KernelSize, and DepthMultiplier must be set")
	}

	ctxInScope := sep.ctx
	if sep.newScope {
		ctxInScope = ctxInScope.In("separable_conv")
	}

	xShape := sep.x.Shape()
	dtype := xShape.DType
	channelsAxis := images.GetChannelsAxis(xShape, sep.channelsAxisConfig)
	inputChannels := xShape.Dimensions[channelsAxis]

	// Depthwise convolution kernel
	depthwiseKernelShape := shapes.Make(dtype)
	if sep.channelsAxisConfig == images.ChannelsFirst {
		depthwiseKernelShape.Dimensions = append([]int{inputChannels}, sep.kernelSize...)
		depthwiseKernelShape.Dimensions = append(depthwiseKernelShape.Dimensions, sep.depthMultiplier)
	} else {
		depthwiseKernelShape.Dimensions = append(sep.kernelSize, inputChannels)
		depthwiseKernelShape.Dimensions = append(depthwiseKernelShape.Dimensions, sep.depthMultiplier)
	}

	depthwiseKernelVar := ctxInScope.VariableWithShape("depthwise_weights", depthwiseKernelShape)
	if sep.regularizer != nil {
		sep.regularizer(ctxInScope, sep.graph, depthwiseKernelVar)
	}
	depthwiseKernel := depthwiseKernelVar.ValueGraph(sep.graph)

	// Ensure strides are set correctly for all spatial dimensions
	if len(sep.strides) != sep.numSpatialDims {
		Panicf("expected %d strides, got %d", sep.numSpatialDims, len(sep.strides))
	}

	log.Println("Stride before dwc", sep.strides, sep.numSpatialDims)
	convOpts := Convolve(sep.x, depthwiseKernel).
		StridePerDim(sep.strides...). // Pass the correct number of strides
		ChannelsAxis(sep.channelsAxisConfig)
	if len(sep.dilations) > 0 {
		convOpts.DilationPerDim(sep.dilations...)
	}
	if sep.padSame {
		convOpts.PadSame()
	} else {
		convOpts.NoPadding()
	}
	depthwiseOutput := convOpts.Done()

	// Pointwise convolution
	pointwiseKernelShape := shapes.Make(dtype)
	if sep.channelsAxisConfig == images.ChannelsFirst {
		pointwiseKernelShape.Dimensions = []int{inputChannels * sep.depthMultiplier, 1, 1, sep.filters}
	} else {
		pointwiseKernelShape.Dimensions = []int{1, 1, inputChannels * sep.depthMultiplier, sep.filters}
	}

	pointwiseKernelVar := ctxInScope.VariableWithShape("pointwise_weights", pointwiseKernelShape)
	if sep.regularizer != nil {
		sep.regularizer(ctxInScope, sep.graph, pointwiseKernelVar)
	}
	pointwiseKernel := pointwiseKernelVar.ValueGraph(sep.graph)

	log.Println("Stride before pwc", sep.strides, sep.numSpatialDims)
	pointwiseConvOpts := Convolve(depthwiseOutput, pointwiseKernel).
		StridePerDim(1, 1). // Stride is 1 for pointwise convolution ---> I changed from (1) to (1,1)
		ChannelsAxis(sep.channelsAxisConfig).
		NoPadding()
	pointwiseOutput := pointwiseConvOpts.Done()

	// Add bias
	if sep.bias {
		biasVar := ctxInScope.VariableWithShape("biases", shapes.Make(dtype, sep.filters))
		bias := biasVar.ValueGraph(sep.graph)
		expandedDims := xslices.SliceWithValue(pointwiseOutput.Rank(), 1)
		outputChannelsAxis := images.GetChannelsAxis(pointwiseOutput, sep.channelsAxisConfig)
		expandedDims[outputChannelsAxis] = sep.filters
		bias = Reshape(bias, expandedDims...)
		pointwiseOutput = Add(pointwiseOutput, bias)
	}

	// L2 regularization
	if l2any, found := ctxInScope.GetParam(ParamL2Regularization); found {
		l2 := l2any.(float64)
		if l2 > 0 {
			l2Node := Const(sep.graph, shapes.CastAsDType(l2, dtype))
			AddL2Regularization(ctxInScope, l2Node, depthwiseKernel)
			AddL2Regularization(ctxInScope, l2Node, pointwiseKernel)
		}
	}

	return pointwiseOutput
}
