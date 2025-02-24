/*
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

package layers

import (
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
    bias               bool
    strides            []int
    padSame            bool
    dilations          []int
    newScope           bool
    regularizer        regularizers.Regularizer
}

// SeparableConvolution prepares a separable convolution on x with the given kernel for arbitrary
// number of spatial dimensions (1D, 2D, 3D, etc.).
func SeparableConvolution(ctx *context.Context, x *Node) *SeparableConvBuilder {
    conv := &SeparableConvBuilder{
        ctx:         ctx,
        graph:       x.Graph(),
        x:           x,
        newScope:    true,
        regularizer: regularizers.FromContext(ctx),
    }
    conv.numSpatialDims = x.Rank() - 2
    if conv.numSpatialDims < 0 {
        Panicf("Input x must have rank >= 3, shaped by default as [batch, <spatial_dimensions...>, channels], "+
            "but x rank is %d", x.Rank())
    }
    return conv.ChannelsAxis(images.ChannelsLast).NoPadding().UseBias(true).Strides(1)
}

// Filters sets the number of filters -- specifies the number of output channels.
func (conv *SeparableConvBuilder) Filters(filters int) *SeparableConvBuilder {
    conv.filters = filters
    if filters <= 0 {
        Panicf("number of filters must be > 0, it was set to %d", filters)
    }
    return conv
}

// KernelSize sets the kernel size for every axis.
func (conv *SeparableConvBuilder) KernelSize(size int) *SeparableConvBuilder {
    perDim := xslices.SliceWithValue(conv.numSpatialDims, size)
    return conv.KernelSizePerDim(perDim...)
}

// KernelSizePerDim sets the kernel size for each dimension(axis).
func (conv *SeparableConvBuilder) KernelSizePerDim(sizes ...int) *SeparableConvBuilder {
    if len(sizes) != conv.numSpatialDims {
        Panicf("received %d kernel sizes, but x has %d spatial dimensions",
            len(sizes), conv.numSpatialDims)
        return conv
    }
    conv.kernelSize = sizes
    return conv
}

// UseBias sets whether to add a trainable bias term to the convolution.
func (conv *SeparableConvBuilder) UseBias(useBias bool) *SeparableConvBuilder {
    conv.bias = useBias
    return conv
}

// ChannelsAxis configures the axis for the channels dimension.
func (conv *SeparableConvBuilder) ChannelsAxis(channelsAxisConfig images.ChannelsAxisConfig) *SeparableConvBuilder {
    conv.channelsAxisConfig = channelsAxisConfig
    return conv
}

// PadSame adds paddings on the edges of x such that the output
// of the convolution has the same shape as the input.
func (conv *SeparableConvBuilder) PadSame() *SeparableConvBuilder {
    conv.padSame = true
    return conv
}

// NoPadding removes any paddings.
func (conv *SeparableConvBuilder) NoPadding() *SeparableConvBuilder {
    conv.padSame = false
    return conv
}

// Strides sets the strides of the convolution.
func (conv *SeparableConvBuilder) Strides(strides int) *SeparableConvBuilder {
    perDim := xslices.SliceWithValue(conv.numSpatialDims, strides)
    return conv.StridePerDim(perDim...)
}

// StridePerDim sets the strides for each spatial dimension of the convolution.
func (conv *SeparableConvBuilder) StridePerDim(strides ...int) *SeparableConvBuilder {
    if len(strides) != conv.numSpatialDims {
        Panicf("received %d strides in StridePerAxis, but x has %d spatial dimensions",
            len(strides), conv.numSpatialDims)
        return conv
    }
    conv.strides = strides
    return conv
}

// Dilations sets the dilations of the convolution.
func (conv *SeparableConvBuilder) Dilations(dilation int) *SeparableConvBuilder {
    dilationsPerDim := xslices.SliceWithValue(conv.numSpatialDims, dilation)
    return conv.DilationPerDim(dilationsPerDim...)
}

// DilationPerDim sets the kernel dilations for each spatial dimension of the convolution.
func (conv *SeparableConvBuilder) DilationPerDim(dilations ...int) *SeparableConvBuilder {
    if len(dilations) != conv.numSpatialDims {
        Panicf("received %d dilations in DilationPerDim, but x has %d spatial dimensions",
            len(dilations), conv.numSpatialDims)
        return conv
    }
    conv.dilations = dilations
    return conv
}

// CurrentScope configures the convolution not to create a sub-scope for the kernel weights it needs.
func (conv *SeparableConvBuilder) CurrentScope() *SeparableConvBuilder {
    conv.newScope = false
    return conv
}

// Regularizer to be applied to the learned weights.
func (conv *SeparableConvBuilder) Regularizer(regularizer regularizers.Regularizer) *SeparableConvBuilder {
    conv.regularizer = regularizer
    return conv
}

// Done indicates that the Separable Convolution layer is finished being configured.
func (conv *SeparableConvBuilder) Done() *Node {
    ctxInScope := conv.ctx
    if conv.newScope {
        ctxInScope = ctxInScope.In("separable_conv")
    }

    if len(conv.kernelSize) == 0 || conv.filters <= 0 {
        Panicf("layers.SeparableConvolution requires Filters and KernelSize to be set")
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

    // Create and apply depthwise kernel.
    xShape := conv.x.Shape()
    dtype := xShape.DType
    depthwiseKernelShape := shapes.Make(dtype)
    depthwiseKernelShape.Dimensions = make([]int, 0, conv.numSpatialDims+2)
    channelsAxis := images.GetChannelsAxis(xShape, conv.channelsAxisConfig)
    inputChannels := xShape.Dimensions[channelsAxis]
    if conv.channelsAxisConfig == images.ChannelsFirst {
        depthwiseKernelShape.Dimensions = append(depthwiseKernelShape.Dimensions, inputChannels)
        depthwiseKernelShape.Dimensions = append(depthwiseKernelShape.Dimensions, conv.kernelSize...)
        depthwiseKernelShape.Dimensions = append(depthwiseKernelShape.Dimensions, 1)
    } else {
        depthwiseKernelShape.Dimensions = append(depthwiseKernelShape.Dimensions, conv.kernelSize...)
        depthwiseKernelShape.Dimensions = append(depthwiseKernelShape.Dimensions, inputChannels)
        depthwiseKernelShape.Dimensions = append(depthwiseKernelShape.Dimensions, 1)
    }
    depthwiseKernelVar := ctxInScope.VariableWithShape("depthwise_weights", depthwiseKernelShape)
    if conv.regularizer != nil {
        conv.regularizer(ctxInScope, conv.graph, depthwiseKernelVar)
    }
    depthwiseKernel := depthwiseKernelVar.ValueGraph(conv.graph)
    depthwiseConvOpts := Convolve(conv.x, depthwiseKernel).StridePerDim(conv.strides...).ChannelsAxis(conv.channelsAxisConfig)
    if len(conv.dilations) > 0 {
        depthwiseConvOpts.DilationPerDim(conv.dilations...)
    }
    if conv.padSame {
        depthwiseConvOpts.PadSame()
    } else {
        depthwiseConvOpts.NoPadding()
    }
    depthwiseOutput := depthwiseConvOpts.Done()

    // Create and apply pointwise kernel.
    pointwiseKernelShape := shapes.Make(dtype, 1, 1, inputChannels, conv.filters)
    pointwiseKernelVar := ctxInScope.VariableWithShape("pointwise_weights", pointwiseKernelShape)
    if conv.regularizer != nil {
        conv.regularizer(ctxInScope, conv.graph, pointwiseKernelVar)
    }
    pointwiseKernel := pointwiseKernelVar.ValueGraph(conv.graph)
    pointwiseConvOpts := Convolve(depthwiseOutput, pointwiseKernel).ChannelsAxis(conv.channelsAxisConfig)
    pointwiseOutput := pointwiseConvOpts.Done()

    // Create and apply bias.
    if conv.bias {
        biasVar := ctxInScope.VariableWithShape("biases", shapes.Make(dtype, conv.filters))
        bias := biasVar.ValueGraph(conv.graph)
        expandedDims := xslices.SliceWithValue(pointwiseOutput.Rank(), 1)
        outputChannelsAxis := images.GetChannelsAxis(pointwiseOutput, conv.channelsAxisConfig)
        expandedDims[outputChannelsAxis] = conv.filters
        bias = Reshape(bias, expandedDims...)
        pointwiseOutput = Add(pointwiseOutput, bias)
    }

    // Add regularization.
    if l2any, found := ctxInScope.GetParam(ParamL2Regularization); found {
        l2 := l2any.(float64)
        if l2 > 0 {
            l2Node := Const(conv.graph, shapes.CastAsDType(l2, dtype))
            AddL2Regularization(ctxInScope, l2Node, depthwiseKernel)
            AddL2Regularization(ctxInScope, l2Node, pointwiseKernel)
        }
    }
    return pointwiseOutput
}
