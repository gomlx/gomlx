package backends

import "github.com/gomlx/gomlx/pkg/core/shapes"

// ClosureBuilder is an optional interface that backends can implement to support
// building closures (sub-computations) for operations like Sort and While.
//
// A closure is a sub-computation that can be passed to certain operations.
// For example, Sort requires a comparator closure, and While requires
// condition and body closures.
//
// Example usage:
//
//	closureBuilder, ok := builder.(backends.ClosureBuilder)
//	if !ok {
//	    panic("backend does not support closures")
//	}
//
//	// Create a comparator for Sort
//	closure := closureBuilder.NewClosure()
//	lhs := closure.AddScalarInput("lhs", dtypes.Float32)
//	rhs := closure.AddScalarInput("rhs", dtypes.Float32)
//	// Build comparison: lhs < rhs
//	result, _ := closure.LessThan(lhs, rhs)
//	closure.SetOutput(result)
//	comparator := closure.Build()
//
//	// Use the comparator with Sort
//	sorted, _ := builder.Sort(comparator, 0, true, input)
type ClosureBuilder interface {
	// NewClosure creates a new closure scope for building a sub-computation.
	NewClosure() Closure
}

// Closure represents a sub-computation being built.
// It provides methods for adding inputs, building operations, and finalizing the closure.
type Closure interface {
	// AddScalarInput adds a scalar input parameter to the closure.
	// Returns an Op that can be used in closure operations.
	AddScalarInput(name string, shape shapes.Shape) (Op, error)

	// Build finalizes the closure and returns the backend-specific closure object.
	// The returned value can be passed to operations like Sort or While.
	Build() (any, error)

	// ClosureOps provides the operations that can be used within a closure.
	ClosureOps
}

// ClosureOps defines the operations available within a closure.
// This is a subset of StandardOps focused on the operations commonly needed
// in comparators and loop conditions/bodies.
type ClosureOps interface {
	// SetOutput sets the output(s) of the closure.
	// For comparators, this should be a boolean scalar.
	// For While body, this should match the shape of initial states.
	SetOutput(outputs ...Op) error

	// Comparison operations
	LessThan(lhs, rhs Op) (Op, error)
	LessOrEqual(lhs, rhs Op) (Op, error)
	GreaterThan(lhs, rhs Op) (Op, error)
	GreaterOrEqual(lhs, rhs Op) (Op, error)
	Equal(lhs, rhs Op) (Op, error)
	NotEqual(lhs, rhs Op) (Op, error)

	// Arithmetic operations
	Add(lhs, rhs Op) (Op, error)
	Sub(lhs, rhs Op) (Op, error)
	Mul(lhs, rhs Op) (Op, error)
	Div(lhs, rhs Op) (Op, error)

	// Logical operations
	LogicalAnd(lhs, rhs Op) (Op, error)
	LogicalOr(lhs, rhs Op) (Op, error)
	LogicalNot(x Op) (Op, error)

	// Other useful operations
	Neg(x Op) (Op, error)
	Abs(x Op) (Op, error)
	Min(lhs, rhs Op) (Op, error)
	Max(lhs, rhs Op) (Op, error)

	// Constant creates a constant value in the closure.
	Constant(flat any, dims ...int) (Op, error)
}
