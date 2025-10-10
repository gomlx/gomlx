package layers

import (
	"slices"
	"sort"

	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

// RMSNormBuilder holds the configuration for RMSNorm.
// Once finished configuring, call RMSNormBuilder.Done()
type RMSNormBuilder struct {
	ctx               *context.Context
	operand           *Node
	useScale          bool
	normalizationAxes []int
	epsilon           float64
}

// RMSNorm starts the configuration of an RMS normalization operation,
// as described in https://arxiv.org/abs/1910.07467.
//
// It normalizes the input with a simple:
//
//		RMS(a) = Sqrt(1/n * \sum{a_i^2})
//	 RMSNorm(a) = a_i / RMS(a) * g_i
//
// Where g_i is a learnable gain (scale), enabled by default.
//
// Call RMSNormBuilder.Done when you finished the configuration.
func RMSNorm(ctx *context.Context, operand *Node) *RMSNormBuilder {
	return &RMSNormBuilder{
		ctx:               ctx,
		operand:           operand,
		useScale:          true,
		normalizationAxes: []int{-1},
		epsilon:           1e-6,
	}
}

// WithScale sets whether to use the gain parameter in the RMSNorm configuration and returns the updated builder.
func (rms *RMSNormBuilder) WithScale(useScale bool) *RMSNormBuilder {
	rms.useScale = useScale
	return rms
}

// WithEpsilon sets the epsilon value used in RMSNorm configuration and returns the updated builder.
// The default value is 1e-6.
func (rms *RMSNormBuilder) WithEpsilon(epsilon float64) *RMSNormBuilder {
	rms.epsilon = epsilon
	return rms
}

// WithNormalizationAxes sets the axes over which to normalize in the RMSNorm configuration and returns the updated builder.
// The default value is -1 (the last axis).
func (rms *RMSNormBuilder) WithNormalizationAxes(axes ...int) *RMSNormBuilder {
	rms.normalizationAxes = axes
	return rms
}

// Done uses the current configuration to perform RMSNorm.
// It returns the normalized operand.
func (rms *RMSNormBuilder) Done() *Node {
	ctx := rms.ctx.In("rms_norm")
	x := rms.operand
	g := x.Graph()
	shape := x.Shape()
	rank := x.Rank()
	dtype := x.DType()

	x2 := Square(x)
	rmsX := ReduceAndKeep(x2, ReduceMean, rms.normalizationAxes...)
	if rms.epsilon != 0 {
		rmsX = AddScalar(rmsX, rms.epsilon)
	}
	rmsX = Sqrt(rmsX)
	x = Div(x, rmsX)
	if rms.useScale {
		// Create scale variable: shaped with the dimensions of the sorted normalized axes.
		normAxes := xslices.Map(rms.normalizationAxes, func(axis int) int { return AdjustAxisToOperandRank(x, axis) })
		sort.Ints(normAxes)
		dims := xslices.Map(normAxes, func(axis int) int { return shape.Dim(axis) })
		scaleShape := shapes.Make(dtype, dims...)
		scaleVar := ctx.WithInitializer(initializers.One).VariableWithShape("scale", scaleShape)
		scale := scaleVar.ValueGraph(g)

		// Apply scale variable: broadcast to all axes not in normAxes.
		scaleToXShape := slices.Repeat([]int{1}, rank)
		for _, axis := range normAxes {
			scaleToXShape[axis] = shape.Dimensions[axis]
		}
		scale = Reshape(scale, scaleToXShape...)
		x = Mul(x, scale)
	}
	return x
}
