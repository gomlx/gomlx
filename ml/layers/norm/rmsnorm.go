// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package norm

import (
	"slices"
	"sort"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/support/exceptions"
)

// ParamRMSNormNormalizationDType is the hyperparameter name for the dtype to use when doing
// RMS normalization.
//
// Low precision dtypes (Float16, BFloat16) can overflow when squaring values or lose precision
// in reductions. The default (empty string) uses Float32 when the input dtype is Float16 or
// BFloat16, and the input dtype otherwise.
var ParamRMSNormNormalizationDType = "rms_norm_dtype"

// RMSNormBuilder holds the configuration for RMSNorm.
// Once finished configuring, call RMSNormBuilder.Done()
type RMSNormBuilder struct {
	scope              *model.Scope
	operand            *Node
	useScale           bool
	scaleOffset        float64
	normalizationAxes  []int
	epsilon            float64
	varDType           dtypes.DType
	normalizationDType dtypes.DType
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
func RMSNorm(scope *model.Scope, operand *Node) *RMSNormBuilder {
	return &RMSNormBuilder{
		scope:              scope,
		operand:            operand,
		useScale:           true,
		normalizationAxes:  []int{-1},
		epsilon:            1e-6,
		varDType:           dtypes.InvalidDType,
		normalizationDType: resolveNormalizationDType(scope, ParamRMSNormNormalizationDType, operand.DType()),
	}
}

// WithScale sets whether to use the gain parameter in the RMSNorm configuration and returns the updated builder.
func (rms *RMSNormBuilder) WithScale(useScale bool) *RMSNormBuilder {
	rms.useScale = useScale
	return rms
}

// WithScaleOffset sets a fixed value to be added to the learned scale. Some models like Gemma and Gemma-2 add 1.0 to the scale.
func (rms *RMSNormBuilder) WithScaleOffset(offset float64) *RMSNormBuilder {
	rms.scaleOffset = offset
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

// WithDType sets the dtype expected for the learned "scale" variable.
//
// By default (or when set to dtypes.InvalidDType), the variable uses the operand's dtype and any
// pre-existing variable in the store must match it (panicking on dtype mismatch).
// Setting a float dtype here explicitly allows variables with that dtype (e.g., Float32 scale
// with Float16 operands), converting it to the normalization dtype inside the graph while still
// panicking if a pre-existing variable in the store does not match the requested dtype.
func (rms *RMSNormBuilder) WithDType(dtype dtypes.DType) *RMSNormBuilder {
	if dtype != dtypes.InvalidDType && !dtype.IsFloat() {
		exceptions.Panicf("RMSNorm.WithDType requires a float dtype or InvalidDType, got %s", dtype)
	}
	rms.varDType = dtype
	return rms
}

// WithNormalizationDType sets the dtype used for computing the root mean square and scaling.
//
// By default, 16-bit floats (Float16, BFloat16) are promoted to Float32 for numerical stability,
// and other dtypes normalize in the operand's dtype. This can also be configured via
// [ParamRMSNormNormalizationDType].
func (rms *RMSNormBuilder) WithNormalizationDType(dtype dtypes.DType) *RMSNormBuilder {
	if !dtype.IsFloat() {
		exceptions.Panicf("RMSNorm.WithNormalizationDType requires a float dtype, got %s", dtype)
	}
	rms.normalizationDType = dtype
	return rms
}

// Done uses the current configuration to perform RMSNorm.
// It returns the normalized operand.
func (rms *RMSNormBuilder) Done() *Node {
	scope := rms.scope.In("rms_norm")
	origDType := rms.operand.DType()
	x := ConvertDType(rms.operand, rms.normalizationDType)
	g := x.Graph()
	shape := x.Shape()
	rank := x.Rank()

	x2 := Square(x)
	rmsX := ReduceAndKeep(x2, ReduceMean, rms.normalizationAxes...)
	if rms.epsilon != 0 {
		rmsX = AddScalar(rmsX, rms.epsilon)
	}
	rmsX = Sqrt(rmsX)
	x = Div(x, rmsX)
	if rms.useScale {
		// Create scale variable: shaped with the dimensions of the sorted normalized axes.
		normAxes := xslices.Map(rms.normalizationAxes, func(axis int) int { return MustAdjustAxis(axis, x) })
		sort.Ints(normAxes)
		dims := xslices.Map(normAxes, func(axis int) int { return shape.Dim(axis) })
		scaleShape := shapes.Make(resolveVariableDType(rms.varDType, origDType), dims...)
		scaleVar := scope.WithInitializer(initializer.One).VariableWithShape("scale", scaleShape)
		scale := ConvertDType(scaleVar.NodeValue(g), rms.normalizationDType)
		if rms.scaleOffset != 0 {
			scale = AddScalar(scale, rms.scaleOffset)
		}

		// Apply scale variable: broadcast to all axes not in normAxes.
		scaleToXShape := slices.Repeat([]int{1}, rank)
		for _, axis := range normAxes {
			scaleToXShape[axis] = shape.Dimensions[axis]
		}
		scale = Reshape(scale, scaleToXShape...)
		x = Mul(x, scale)
	}
	return ConvertDType(x, origDType)
}
