package xla

import (
	"slices"

	"github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/go-xla/pkg/types/shardy"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Function implements backends.Function for XLA.
type Function struct {
	builder *Builder
	fn      *stablehlo.Function
	name    string

	parameterNames  []string
	parameterShapes []shapes.Shape
	parameterSpecs  []*backends.ShardingSpec

	// returned indicates Return() was called.
	returned bool

	// outputs stores the return values set by Return().
	outputs         []*Node
	outputShardings []*shardy.ShardingSpec
}

var _ backends.Function = (*Function)(nil)

// CheckValid returns an error if the builder or the function are not ok.
func (f *Function) CheckValid() error {
	if f == nil || f.fn == nil {
		return errors.Errorf("function is nil or undefined for %q", BackendName)
	}
	return f.builder.CheckValid()
}

// verifyAndCastValues sanity checks that the values (backends.Op) are valid and created with this builder.
// It returns the underlying *Node of the values.
func (f *Function) verifyAndCastValues(name string, values ...backends.Value) ([]*Node, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	nodes := make([]*Node, len(values))
	for i, input := range values {
		if input == nil {
			return nil, errors.Errorf("nil Op given as an input to %q", name)
		}
		node, ok := input.(*Node)
		if !ok {
			return nil, errors.Errorf(
				"nil or invalid Op (%T: %v) given as an input to %q, it must be an input created by the same "+
					"backend builder (%s:%s)", input, input, name, f.builder.backend.Name(), f.builder.name)
		}
		if node.builder != f.builder {
			return nil, errors.Errorf(
				"input given to parameter #%d (%q) was created with a different builder (%s) than the builder"+
					" (%s) it is being used in -- Ops cannot cross to different builders",
				i, name, node.builder.Name(), f.builder.Name())
		}
		nodes[i] = node
	}
	return nodes, nil
}

func (f *Function) newNode(value *stablehlo.Value) *Node {
	return &Node{
		value:   value,
		shape:   ShapeFromXLA(value.Shape()),
		builder: f.builder,
	}
}

// Parameter creates an input parameter for this function.
func (f *Function) Parameter(name string, shape shapes.Shape, sharding *backends.ShardingSpec) (
	backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	normalizedName := stablehlo.NormalizeIdentifier(name)
	if slices.Index(f.parameterNames, normalizedName) != -1 {
		if name == normalizedName {
			return nil, errors.Errorf("parameter named %q already exists", name)
		}
		return nil, errors.Errorf("parameter named %q (normalized to %q) already exists",
			name, normalizedName)
	}
	f.parameterNames = append(f.parameterNames, normalizedName)
	f.parameterShapes = append(f.parameterShapes, shape)
	f.parameterSpecs = append(f.parameterSpecs, sharding)
	var shardySpec *shardy.ShardingSpec
	if sharding != nil {
		var err error
		shardySpec, err = f.builder.shardingSpecToShardy(sharding)
		if err != nil {
			return nil, errors.WithMessagef(err, "while creating sharding spec for parameter %q", name)
		}
	}
	value, err := f.fn.NamedInputWithSharding(name, ShapeToXLA(shape), shardySpec)
	if err != nil {
		return nil, errors.WithMessagef(err, "while building parameter %q", name)
	}
	return f.newNode(value), nil
}

// Constant creates a constant in the function with the given flat values and the shape defined by the dimensions.
func (f *Function) Constant(flat any, dimensions ...int) (backends.Value, error) {
	if err := f.CheckValid(); err != nil {
		return nil, err
	}
	if flat == nil {
		return nil, errors.Errorf("nil value given to Constant")
	}
	if bf16Slice, ok := flat.([]bfloat16.BFloat16); ok {
		flat = any(BFloat16SliceToXLA(bf16Slice))
	}
	value, err := f.fn.ConstantFromFlatAndDimensions(flat, dimensions...)
	if err != nil {
		return nil, errors.WithMessagef(err, "while building op Constant()")
	}
	return f.newNode(value), nil
}

// Return marks the outputs of this function.
func (f *Function) Return(outputs []backends.Value, shardings []*backends.ShardingSpec) error {
	if err := f.CheckValid(); err != nil {
		return err
	}
	if f.returned {
		return errors.Errorf("Return() already called for function %q", f.name)
	}
	if len(outputs) == 0 {
		return errors.Errorf("Return() requires at least one output")
	}

	outputNodes, err := f.verifyAndCastValues("Return", outputs...)
	if err != nil {
		return err
	}

	// Convert shardings if provided
	var shardySpecs []*shardy.ShardingSpec
	if len(shardings) > 0 {
		shardySpecs = make([]*shardy.ShardingSpec, len(outputs))
		for i, spec := range shardings {
			shardySpecs[i], err = f.builder.shardingSpecToShardy(spec)
			if err != nil {
				return errors.WithMessagef(err, "failed to convert sharding spec for output #%d", i)
			}
		}
	}

	f.outputs = outputNodes
	f.outputShardings = shardySpecs
	f.returned = true

	// Call the stablehlo Return
	outputValues := make([]*stablehlo.Value, len(outputs))
	for i, node := range outputNodes {
		outputValues[i] = node.value
	}
	err = f.fn.ReturnWithShardingAndAttributes(outputValues, shardySpecs, nil)
	if err != nil {
		return errors.WithMessagef(err, "failed to set return for function %q", f.name)
	}

	return nil
}
