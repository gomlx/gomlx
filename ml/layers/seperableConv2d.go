package layers

import (
	"fmt"

	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gomlx/types/xslices"
)

// SeparableConvBuilder builds a Separable Convolution layer.
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

// SeparableConvolution initializes the builder.
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
		Panicf("Input x must have rank >= 3, but rank is %d", x.Rank())
	}
	return conv.ChannelsAxis(images.ChannelsLast).NoPadding().UseBias(true).Strides(1)
}

func (conv *SeparableConvBuilder) Filters(filters int) *SeparableConvBuilder {
	conv.filters = filters
	if filters <= 0 {
		Panicf("number of filters must be > 0, set to %d", filters)
	}
	return conv
}

func (conv *SeparableConvBuilder) KernelSize(size int) *SeparableConvBuilder {
	perDim := xslices.SliceWithValue(conv.numSpatialDims, size)
	return conv.KernelSizePerDim(perDim...)
}

func (conv *SeparableConvBuilder) KernelSizePerDim(sizes ...int) *SeparableConvBuilder {
	if len(sizes) != conv.numSpatialDims {
		Panicf("received %d kernel sizes, but x has %d spatial dimensions", len(sizes), conv.numSpatialDims)
		return conv
	}
	conv.kernelSize = sizes
	return conv
}

func (conv *SeparableConvBuilder) UseBias(useBias bool) *SeparableConvBuilder {
	conv.bias = useBias
	return conv
}

func (conv *SeparableConvBuilder) ChannelsAxis(channelsAxisConfig images.ChannelsAxisConfig) *SeparableConvBuilder {
	conv.channelsAxisConfig = channelsAxisConfig
	return conv
}

func (conv *SeparableConvBuilder) PadSame() *SeparableConvBuilder {
	conv.padSame = true
	return conv
}

func (conv *SeparableConvBuilder) NoPadding() *SeparableConvBuilder {
	conv.padSame = false
	return conv
}

func (conv *SeparableConvBuilder) Strides(strides int) *SeparableConvBuilder {
	perDim := xslices.SliceWithValue(conv.numSpatialDims, strides)
	return conv.StridePerDim(perDim...)
}

func (conv *SeparableConvBuilder) StridePerDim(strides ...int) *SeparableConvBuilder {
	if len(strides) != conv.numSpatialDims {
		Panicf("received %d strides, but x has %d spatial dimensions", len(strides), conv.numSpatialDims)
		return conv
	}
	conv.strides = strides
	return conv
}

func (conv *SeparableConvBuilder) Dilations(dilation int) *SeparableConvBuilder {
	dilationsPerDim := xslices.SliceWithValue(conv.numSpatialDims, dilation)
	return conv.DilationPerDim(dilationsPerDim...)
}

func (conv *SeparableConvBuilder) DilationPerDim(dilations ...int) *SeparableConvBuilder {
	if len(dilations) != conv.numSpatialDims {
		Panicf("received %d dilations, but x has %d spatial dimensions", len(dilations), conv.numSpatialDims)
		return conv
	}
	conv.dilations = dilations
	return conv
}

func (conv *SeparableConvBuilder) Done() *Node {
	ctxInScope := conv.ctx
	if conv.newScope {
		ctxInScope = ctxInScope.In("separable_conv")
	}
	if len(conv.kernelSize) == 0 || conv.filters <= 0 {
		Panicf("SeparableConvolution requires Filters and KernelSize to be set")
	}

	xShape := conv.x.Shape()
	dtype := xShape.DType
	inputChannels := xShape.Dimensions[images.GetChannelsAxis(xShape, conv.channelsAxisConfig)]

	fmt.Printf("Input Shape: %v\n", xShape.Dimensions)
	fmt.Printf("Kernel Size: %v, Filters: %d, Input Channels: %d\n", conv.kernelSize, conv.filters, inputChannels)

	// Depthwise Convolution
	depthwiseKernelShape := shapes.Make(dtype, append(conv.kernelSize, 1, inputChannels)...) // Depthwise kernel
	depthwiseKernelVar := ctxInScope.VariableWithShape("depthwise_weights", depthwiseKernelShape)
	depthwiseKernel := depthwiseKernelVar.ValueGraph(conv.graph)

	convOpts := Convolve(conv.x, depthwiseKernel).StridePerDim(conv.strides...).ChannelsAxis(conv.channelsAxisConfig)
	if len(conv.dilations) > 0 {
		convOpts.DilationPerDim(conv.dilations...)
	}
	if conv.padSame {
		convOpts.PadSame()
	} else {
		convOpts.NoPadding()
	}
	depthwiseOutput := convOpts.Done()

	// Fix: Build Pointwise Kernel Shape as a Slice First
	pointwiseKernelShapeSlice := append(xslices.SliceWithValue(conv.numSpatialDims, 1), inputChannels, conv.filters)
	pointwiseKernelShape := shapes.Make(dtype, pointwiseKernelShapeSlice...) // Corrected

	pointwiseKernelVar := ctxInScope.VariableWithShape("pointwise_weights", pointwiseKernelShape)
	pointwiseKernel := pointwiseKernelVar.ValueGraph(conv.graph)

	output := Convolve(depthwiseOutput, pointwiseKernel).ChannelsAxis(conv.channelsAxisConfig).Done()

	// Bias Addition
	if conv.bias {
		biasVar := ctxInScope.VariableWithShape("biases", shapes.Make(dtype, conv.filters))
		bias := biasVar.ValueGraph(conv.graph)
		expandedDims := xslices.SliceWithValue(output.Rank(), 1)
		outputChannelsAxis := images.GetChannelsAxis(output, conv.channelsAxisConfig)
		expandedDims[outputChannelsAxis] = conv.filters
		bias = Reshape(bias, expandedDims...)
		output = Add(output, bias)
	}

	return output
}
