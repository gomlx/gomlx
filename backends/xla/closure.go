package xla

import (
	"reflect"

	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Ensure Builder implements ClosureBuilder
var _ backends.ClosureBuilder = (*Builder)(nil)

// NewClosure creates a new closure scope for building a sub-computation.
func (b *Builder) NewClosure() backends.Closure {
	if err := b.CheckValid(); err != nil {
		return &Closure{err: err}
	}
	return &Closure{
		parentBuilder: b,
		fn:            b.fn.Closure(),
	}
}

// Closure implements the backends.Closure interface for XLA.
type Closure struct {
	parentBuilder *Builder
	fn            *stablehlo.Function
	err           error
}

// Ensure Closure implements backends.Closure
var _ backends.Closure = (*Closure)(nil)

// checkError returns early if there was a previous error
func (c *Closure) checkError() error {
	if c.err != nil {
		return c.err
	}
	if c.fn == nil {
		return errors.New("closure is nil or already built")
	}
	return nil
}

// AddScalarInput adds a scalar input parameter to the closure.
func (c *Closure) AddScalarInput(name string, shape shapes.Shape) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	xlaShape := ShapeToXLA(shape)
	value, err := c.fn.NamedInput(name, xlaShape)
	if err != nil {
		return nil, errors.WithMessagef(err, "while adding input %q to closure", name)
	}
	return &ClosureNode{value: value, shape: shape}, nil
}

// Build finalizes the closure and returns the stablehlo.Function.
func (c *Closure) Build() (any, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	fn := c.fn
	c.fn = nil // Mark as built
	return fn, nil
}

// SetOutput sets the output(s) of the closure.
func (c *Closure) SetOutput(outputs ...backends.Op) error {
	if err := c.checkError(); err != nil {
		return err
	}
	if len(outputs) == 0 {
		return errors.New("SetOutput requires at least one output")
	}

	// For single output, just return it
	if len(outputs) == 1 {
		node, ok := outputs[0].(*ClosureNode)
		if !ok {
			return errors.Errorf("output must be a closure node, got %T", outputs[0])
		}
		return c.fn.Return(node.value)
	}

	// For multiple outputs, return them as variadic arguments
	values := make([]*stablehlo.Value, len(outputs))
	for i, output := range outputs {
		node, ok := output.(*ClosureNode)
		if !ok {
			return errors.Errorf("output %d must be a closure node, got %T", i, output)
		}
		values[i] = node.value
	}
	return c.fn.Return(values...)
}

// ClosureNode represents a value within a closure.
type ClosureNode struct {
	value *stablehlo.Value
	shape shapes.Shape
}

// helper to cast and extract values
func (c *Closure) getValues(ops ...backends.Op) ([]*stablehlo.Value, error) {
	values := make([]*stablehlo.Value, len(ops))
	for i, op := range ops {
		node, ok := op.(*ClosureNode)
		if !ok {
			return nil, errors.Errorf("operand %d must be a closure node, got %T", i, op)
		}
		values[i] = node.value
	}
	return values, nil
}

func (c *Closure) wrapResult(value *stablehlo.Value, err error) (backends.Op, error) {
	if err != nil {
		return nil, err
	}
	// Get the shape from the value
	xlaShape := value.Shape()
	shape := ShapeFromXLA(xlaShape)
	return &ClosureNode{value: value, shape: shape}, nil
}

// Comparison operations

func (c *Closure) LessThan(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode := lhs.(*ClosureNode)
	compareType := compareTypeForDType(lhsNode.shape.DType)
	result, err := stablehlo.Compare(values[0], values[1], stablehlotypes.CompareLT, compareType)
	return c.wrapResult(result, err)
}

func (c *Closure) LessOrEqual(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode := lhs.(*ClosureNode)
	compareType := compareTypeForDType(lhsNode.shape.DType)
	result, err := stablehlo.Compare(values[0], values[1], stablehlotypes.CompareLE, compareType)
	return c.wrapResult(result, err)
}

func (c *Closure) GreaterThan(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode := lhs.(*ClosureNode)
	compareType := compareTypeForDType(lhsNode.shape.DType)
	result, err := stablehlo.Compare(values[0], values[1], stablehlotypes.CompareGT, compareType)
	return c.wrapResult(result, err)
}

func (c *Closure) GreaterOrEqual(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode := lhs.(*ClosureNode)
	compareType := compareTypeForDType(lhsNode.shape.DType)
	result, err := stablehlo.Compare(values[0], values[1], stablehlotypes.CompareGE, compareType)
	return c.wrapResult(result, err)
}

func (c *Closure) Equal(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode := lhs.(*ClosureNode)
	compareType := compareTypeForDType(lhsNode.shape.DType)
	result, err := stablehlo.Compare(values[0], values[1], stablehlotypes.CompareEQ, compareType)
	return c.wrapResult(result, err)
}

func (c *Closure) NotEqual(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode := lhs.(*ClosureNode)
	compareType := compareTypeForDType(lhsNode.shape.DType)
	result, err := stablehlo.Compare(values[0], values[1], stablehlotypes.CompareNE, compareType)
	return c.wrapResult(result, err)
}

// Arithmetic operations

func (c *Closure) Add(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.Add(values[0], values[1])
	return c.wrapResult(result, err)
}

func (c *Closure) Sub(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.Subtract(values[0], values[1])
	return c.wrapResult(result, err)
}

func (c *Closure) Mul(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.Multiply(values[0], values[1])
	return c.wrapResult(result, err)
}

func (c *Closure) Div(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.Divide(values[0], values[1])
	return c.wrapResult(result, err)
}

// Logical operations

func (c *Closure) LogicalAnd(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.And(values[0], values[1])
	return c.wrapResult(result, err)
}

func (c *Closure) LogicalOr(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.Or(values[0], values[1])
	return c.wrapResult(result, err)
}

func (c *Closure) LogicalNot(x backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(x)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.Not(values[0])
	return c.wrapResult(result, err)
}

// Other operations

func (c *Closure) Neg(x backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(x)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.Negate(values[0])
	return c.wrapResult(result, err)
}

func (c *Closure) Abs(x backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(x)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.Abs(values[0])
	return c.wrapResult(result, err)
}

func (c *Closure) Min(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.Minimum(values[0], values[1])
	return c.wrapResult(result, err)
}

func (c *Closure) Max(lhs, rhs backends.Op) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	values, err := c.getValues(lhs, rhs)
	if err != nil {
		return nil, err
	}
	result, err := stablehlo.Maximum(values[0], values[1])
	return c.wrapResult(result, err)
}

func (c *Closure) Constant(flat any, dims ...int) (backends.Op, error) {
	if err := c.checkError(); err != nil {
		return nil, err
	}
	// Create a constant in the closure's function context
	var value *stablehlo.Value
	var err error
	if len(dims) == 0 {
		// Scalar constant
		value, err = c.fn.ConstantFromScalar(flat)
	} else {
		// Tensor constant
		value, err = c.fn.ConstantFromFlatAndDimensions(flat, dims...)
	}
	if err != nil {
		return nil, errors.WithMessage(err, "while creating constant in closure")
	}
	shape := shapeFromFlatAndDims(flat, dims)
	return &ClosureNode{value: value, shape: shape}, nil
}

// shapeFromFlatAndDims infers shape from flat data and dimensions
func shapeFromFlatAndDims(flat any, dims []int) shapes.Shape {
	dtype := dtypeFromFlat(flat)
	if len(dims) == 0 {
		// Scalar
		return shapes.Make(dtype)
	}
	return shapes.Make(dtype, dims...)
}

// dtypeFromFlat infers the DType from a flat value (scalar or slice)
func dtypeFromFlat(flat any) dtypes.DType {
	t := reflect.TypeOf(flat)
	if t.Kind() == reflect.Slice {
		t = t.Elem()
	}
	return dtypes.FromGoType(t)
}
