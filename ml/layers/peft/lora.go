// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package peft

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/ml/nn"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// Linear is a graph-building LoRA projection. Configure it with scope
// hyperparameters, then call Done.
type Linear struct {
	scope        *model.Scope
	input        *Node
	weight, bias *model.Variable
	config       config
}

// New configures a LoRA projection around an existing dense weight. The base
// projection remains unchanged when ParamRank is zero.
func New(scope *model.Scope, input *Node, weight, bias *model.Variable) *Linear {
	if scope == nil || input == nil || weight == nil {
		Panicf("peft.New requires non-nil scope, input, and weight")
	}
	config := configFromScope(scope)
	config.validate()
	shape := weight.Shape()
	if shape.Rank() != 2 || !supportedDType(shape.DType) {
		Panicf("peft.New requires a rank-2 F32/F16/BF16 weight, got %s", shape)
	}
	return &Linear{scope: scope, input: input, weight: weight, bias: bias, config: config}
}

// WithWeightLayout configures the layout of the existing base weight.
func (l *Linear) WithWeightLayout(layout compute.DenseLayout) *Linear {
	l.config.layout = layout
	l.config.validate()
	return l
}

// Done builds the frozen base projection plus its LoRA update. A and B live in
// scope/lora, use the Store RNG, and are included automatically in normal
// trainable-variable selection.
func (l *Linear) Done() *Node {
	in, out := l.dimensions()
	if l.bias != nil && (l.bias.Shape().Rank() != 1 || l.bias.Shape().Dimensions[0] != out) {
		Panicf("peft: bias shape must be [%d], got %s", out, l.bias.Shape())
	}
	if l.config.rank == 0 {
		return nn.Dense(l.input, l.weight.NodeValue(l.input), nodeValue(l.bias, l.input), l.config.layout)
	}
	l.weight.SetTrainable(false)
	if l.bias != nil {
		l.bias.SetTrainable(l.config.bias != BiasNone)
	}
	adapterScope := l.scope.In("lora")
	bound := 1 / math.Sqrt(float64(in))
	a := adapterScope.WithInitializer(initializer.RandomUniformFn(adapterScope, -bound, bound)).VariableWithShape("A", shapes.Make(l.weight.DType(), in, l.config.rank))
	b := adapterScope.WithInitializer(initializer.Zero).VariableWithShape("B", shapes.Make(l.weight.DType(), l.config.rank, out))
	base := nn.Dense(l.input, l.weight.NodeValue(l.input), nodeValue(l.bias, l.input), l.config.layout)
	update := nn.Dense(l.input, a.NodeValue(l.input), nil, compute.DenseLayoutInputOutputs)
	update = layers.DropoutStatic(l.scope, update, float64(l.config.dropout))
	update = nn.Dense(update, b.NodeValue(l.input), nil, compute.DenseLayoutInputOutputs)
	return Add(base, MulScalar(update, l.config.alpha/float32(l.config.rank)))
}

func (l *Linear) dimensions() (in, out int) {
	in, out = l.weight.Shape().Dimensions[0], l.weight.Shape().Dimensions[1]
	if l.config.layout == compute.DenseLayoutOutputsInput {
		in, out = out, in
	}
	return
}

// Apply is shorthand for New(scope, input, weight, bias).Done().
func Apply(scope *model.Scope, input *Node, weight, bias *model.Variable) *Node {
	return New(scope, input, weight, bias).Done()
}

func supportedDType(dtype dtypes.DType) bool {
	return dtype == dtypes.Float32 || dtype == dtypes.Float16 || dtype == dtypes.BFloat16
}
