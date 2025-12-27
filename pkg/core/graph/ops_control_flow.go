package graph

// This file contains control flow helper functions for GoMLX graphs.

import (
	"github.com/gomlx/go-xla/pkg/stablehlo"
	"github.com/gomlx/gomlx/backends/xla"
)

// StableHLOFunction returns the underlying stablehlo.Function for advanced use cases.
// This allows direct access to StableHLO features like While loop closures that don't yet have
// a high-level GoMLX API.
//
// Returns nil if the backend is not XLA.
//
// Example usage for While loops:
//
//	// Get the stablehlo function
//	fn := g.StableHLOFunction()
//	if fn == nil {
//	    panic("StableHLO backend required for While loops")
//	}
//
//	// Create condition closure: counter < 10
//	condFn := fn.Closure()
//	condInput, _ := condFn.Input(xla.ShapeToXLA(counter.Shape()))
//	limit, _ := condFn.ConstantFromScalar(int32(10))
//	cond, _ := stablehlo.Compare(condInput, limit, types.CompareLT, types.CompareSigned)
//	condFn.Return(cond)
//
//	// Create body closure: counter + 1
//	bodyFn := fn.Closure()
//	bodyInput, _ := bodyFn.Input(xla.ShapeToXLA(counter.Shape()))
//	one, _ := bodyFn.ConstantFromScalar(int32(1))
//	next, _ := stablehlo.Add(bodyInput, one)
//	bodyFn.Return(next)
//
//	// Execute While loop
//	results := While(condFn, bodyFn, counter)
func (g *Graph) StableHLOFunction() *stablehlo.Function {
	xlaBuilder, ok := g.builder.(*xla.Builder)
	if !ok {
		return nil
	}
	return xlaBuilder.StableHLOFunction()
}
