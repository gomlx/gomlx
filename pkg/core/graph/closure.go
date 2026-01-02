package graph

import (
	"reflect"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// Closure represents a sub-computation being built within a graph.
// It provides a subset of graph operations that can be used to build
// comparators for Sort or condition/body functions for While loops.
//
// Example usage for a sort comparator:
//
//	closure := g.NewClosure()
//	lhs := closure.AddScalarInput("lhs", shapes.Make(dtypes.Float32))
//	rhs := closure.AddScalarInput("rhs", shapes.Make(dtypes.Float32))
//	result := closure.LessThan(lhs, rhs)
//	closure.SetOutput(result)
//	comparator := closure.Build()
type Closure struct {
	graph   *Graph
	backend backends.Closure
	err     error
}

// ClosureNode represents a value within a closure.
// It wraps the backend's Op and provides type information.
type ClosureNode struct {
	op    backends.Op
	shape shapes.Shape
}

// Shape returns the shape of the closure node.
func (n *ClosureNode) Shape() shapes.Shape {
	return n.shape
}

// NewClosure creates a new closure for building sub-computations.
// The graph must have a builder that implements backends.ClosureBuilder.
func (g *Graph) NewClosure() *Closure {
	g.AssertBuilding()

	closureBuilder, ok := g.builder.(backends.ClosureBuilder)
	if !ok {
		exceptions.Panicf("backend %T does not support closure building", g.builder)
	}

	return &Closure{
		graph:   g,
		backend: closureBuilder.NewClosure(),
	}
}

// AddScalarInput adds a scalar input parameter to the closure.
// Returns a ClosureNode that can be used in closure operations.
func (c *Closure) AddScalarInput(name string, shape shapes.Shape) *ClosureNode {
	if c.err != nil {
		return nil
	}

	op, err := c.backend.AddScalarInput(name, shape)
	if err != nil {
		c.err = err
		return nil
	}

	return &ClosureNode{op: op, shape: shape}
}

// Build finalizes the closure and returns the backend-specific closure object.
// This can be passed to operations like Sort or While.
func (c *Closure) Build() any {
	if c.err != nil {
		exceptions.Panicf("closure build error: %v", c.err)
	}

	result, err := c.backend.Build()
	if err != nil {
		exceptions.Panicf("closure build error: %v", err)
	}

	return result
}

// SetOutput sets the output(s) of the closure.
func (c *Closure) SetOutput(outputs ...*ClosureNode) {
	if c.err != nil {
		return
	}

	ops := make([]backends.Op, len(outputs))
	for i, n := range outputs {
		ops[i] = n.op
	}

	if err := c.backend.SetOutput(ops...); err != nil {
		c.err = err
	}
}

// --- Comparison Operations ---

// LessThan returns a boolean node indicating lhs < rhs.
func (c *Closure) LessThan(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.LessThan, lhs, rhs)
}

// LessOrEqual returns a boolean node indicating lhs <= rhs.
func (c *Closure) LessOrEqual(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.LessOrEqual, lhs, rhs)
}

// GreaterThan returns a boolean node indicating lhs > rhs.
func (c *Closure) GreaterThan(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.GreaterThan, lhs, rhs)
}

// GreaterOrEqual returns a boolean node indicating lhs >= rhs.
func (c *Closure) GreaterOrEqual(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.GreaterOrEqual, lhs, rhs)
}

// Equal returns a boolean node indicating lhs == rhs.
func (c *Closure) Equal(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.Equal, lhs, rhs)
}

// NotEqual returns a boolean node indicating lhs != rhs.
func (c *Closure) NotEqual(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.NotEqual, lhs, rhs)
}

// --- Arithmetic Operations ---

// Add returns lhs + rhs.
func (c *Closure) Add(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.Add, lhs, rhs)
}

// Sub returns lhs - rhs.
func (c *Closure) Sub(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.Sub, lhs, rhs)
}

// Mul returns lhs * rhs.
func (c *Closure) Mul(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.Mul, lhs, rhs)
}

// Div returns lhs / rhs.
func (c *Closure) Div(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.Div, lhs, rhs)
}

// --- Logical Operations ---

// LogicalAnd returns lhs && rhs.
func (c *Closure) LogicalAnd(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.LogicalAnd, lhs, rhs)
}

// LogicalOr returns lhs || rhs.
func (c *Closure) LogicalOr(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.LogicalOr, lhs, rhs)
}

// LogicalNot returns !x.
func (c *Closure) LogicalNot(x *ClosureNode) *ClosureNode {
	return c.wrapUnaryOp(c.backend.LogicalNot, x)
}

// --- Other Operations ---

// Neg returns -x.
func (c *Closure) Neg(x *ClosureNode) *ClosureNode {
	return c.wrapUnaryOp(c.backend.Neg, x)
}

// Abs returns |x|.
func (c *Closure) Abs(x *ClosureNode) *ClosureNode {
	return c.wrapUnaryOp(c.backend.Abs, x)
}

// Min returns the element-wise minimum of lhs and rhs.
func (c *Closure) Min(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.Min, lhs, rhs)
}

// Max returns the element-wise maximum of lhs and rhs.
func (c *Closure) Max(lhs, rhs *ClosureNode) *ClosureNode {
	return c.wrapBinaryOp(c.backend.Max, lhs, rhs)
}

// Constant creates a constant value in the closure.
func (c *Closure) Constant(flat any, dims ...int) *ClosureNode {
	if c.err != nil {
		return nil
	}

	op, err := c.backend.Constant(flat, dims...)
	if err != nil {
		c.err = err
		return nil
	}

	// Infer shape from the constant
	shape := inferShapeFromConstant(flat, dims)
	return &ClosureNode{op: op, shape: shape}
}

// wrapBinaryOp is a helper for binary operations.
func (c *Closure) wrapBinaryOp(fn func(backends.Op, backends.Op) (backends.Op, error), lhs, rhs *ClosureNode) *ClosureNode {
	if c.err != nil {
		return nil
	}

	op, err := fn(lhs.op, rhs.op)
	if err != nil {
		c.err = err
		return nil
	}

	// Comparison ops return Bool, others preserve the input type
	return &ClosureNode{op: op, shape: lhs.shape}
}

// wrapUnaryOp is a helper for unary operations.
func (c *Closure) wrapUnaryOp(fn func(backends.Op) (backends.Op, error), x *ClosureNode) *ClosureNode {
	if c.err != nil {
		return nil
	}

	op, err := fn(x.op)
	if err != nil {
		c.err = err
		return nil
	}

	return &ClosureNode{op: op, shape: x.shape}
}

// inferShapeFromConstant infers the shape from a constant value.
func inferShapeFromConstant(flat any, dims []int) shapes.Shape {
	dtype := inferDTypeFromValue(flat)
	if len(dims) == 0 {
		return shapes.Make(dtype)
	}
	return shapes.Make(dtype, dims...)
}

// inferDTypeFromValue infers the DType from a value (scalar or slice).
func inferDTypeFromValue(flat any) dtypes.DType {
	t := reflect.TypeOf(flat)
	if t.Kind() == reflect.Slice {
		t = t.Elem()
	}
	return dtypes.FromGoType(t)
}
