// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package backends

import "github.com/gomlx/gomlx/pkg/core/shapes"

// Function represents a computation function within a Builder.
//
// A Function contains operations (via StandardOps and CollectiveOps), constants,
// and parameters. Multiple functions can be composed within a Builder, with
// Main() being the entry point that gets compiled.
//
// Other top-level functions created via Builder.NewFunction() can be used for modular
// computation, while-loop bodies, conditional branches, reduce operations, etc.
//
// The typical lifecycle is:
//  1. Create parameters via Parameter()
//  2. Build computation using StandardOps/CollectiveOps methods
//  3. Mark outputs via Return()
//
// After all functions of a Builder are finished (and Return() has been called),
// one compiles the Builder with Builder.Compile().
type Function interface {
	// Name of the function. It will return "" for closures.
	Name() string

	// Parent returns the parent function of the current function.
	// This is only set for "closures" within another functions.
	// For top-level functions, like "main", or for backends that don't support fun this returns nil.
	Parent() Function

	// Closure returns a new local function, that can be used by certain operations like While, If, Sort.
	// Closure functions can access values from its parent function.
	Closure() (Function, error)

	// StandardOps includes all standard math/ML operations.
	StandardOps

	// CollectiveOps includes all collective (distributed cross-device) operations.
	CollectiveOps

	// Parameter creates an input parameter for this function.
	//
	// For the Main function, these become the computation's input parameters
	// that must be provided when executing the compiled computation.
	//
	// For sub-functions, these define the function's input signature.
	//
	// The sharding defines how the parameter will be sharded for distributed
	// operations. Set it to nil if not using distribution.
	Parameter(name string, shape shapes.Shape, sharding *ShardingSpec) (Value, error)

	// Constant creates a constant in the function with the given flat values
	// and the shape defined by the dimensions.
	//
	// The flat value must be a slice of a basic type supported (that can be
	// converted to a DType).
	//
	// The value is copied into the graph. It's recommended that for very large
	// tensors, even if constants, that they are passed as parameters instead.
	Constant(flat any, dims ...int) (Value, error)

	// Return marks the outputs of this function.
	// Once called, the function can no longer be futher modified.
	//
	// For the Main function, this defines what values will be returned when
	// the compiled computation is executed.
	//
	// For sub-functions, this defines what values are returned when the
	// function is called.
	//
	// The shardings parameter optionally specifies output sharding for
	// distributed computation with AutoSharding. Set to nil otherwise.
	//
	// Return must be called exactly once before Builder.Compile().
	Return(outputs []Value, shardings []*ShardingSpec) error
}
