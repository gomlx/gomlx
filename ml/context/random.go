package context

import (
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
)

const (
	RngStateVariableName = "#rngState"
)

func (ctx *Context) getRngStateVar() *Variable {
	rngStateVar := ctx.InspectVariable(RootScope, RngStateVariableName)
	if rngStateVar == nil {
		rngStateVar = ctx.InAbsPath(RootScope).Checked(false).
			VariableWithValue(RngStateVariableName, graph.RngState()).SetTrainable(false)
	}
	return rngStateVar
}

// RngStateFromSeed initializes the default context random number generator (RNG) state with a static seed.
// It must be set before a graph is created from it, or the RNG state has been initialized otherwise.
//
// The random number generator (RNG) state is stored in a variable on the root scope
// of the context, called "#rngState" (RngStateVariableName).
func (ctx *Context) RngStateFromSeed(seed int64) {
	initialState := graph.RngStateFromSeed(seed)
	ctx.InAbsPath(RootScope).Checked(false).
		VariableWithValue(RngStateVariableName, initialState).SetTrainable(false)
}

// RandomNormal generates random numbers from a normal distribution, with mean 0.0
// and standard deviation 1.0.
// It generates values with the given shape, each value pseudo-randomly generated.
//
// If you need a different mean and standard deviation, just do something like the example below, where `mean`
// and `stddev` are the desired mean and standard deviation:
//
//	rngState, numbers = RandomNormal(rngState, myShape)
//	numbers = AddScalar(MulScalar(numbers, stddev), mean)
//
// The random number generator (RNG) state is stored in a variable on the root scope
// of the context, called "#rngState" (RngStateVariableName).
// The state is initialized with the nanosecond clock, the first time it is used -- so pretty random.
// But you can initialize it with a fixed seed before using any of the Random* methods.
//
// See details in graph.RandomNormal.
func (ctx *Context) RandomNormal(g *graph.Graph, shape shapes.Shape) (values *Node) {
	rngStateVar := ctx.getRngStateVar()
	rngState := rngStateVar.ValueGraph(g)
	rngState, values = graph.RandomNormal(rngState, shape)
	if g.Ok() {
		rngStateVar.SetValueGraph(rngState)
	}
	return
}

// RandomUniform generates random uniform values from 0.0 to 1.0 (half-open `[0.0, 1.0)`, so 1.0 is never returned)
// for float numbers in the given shape.
//
// The random number generator (RNG) state is stored in a variable on the root scope
// of the context, called "#rngState" (RngStateVariableName).
// The state is initialized with the nanosecond clock, the first time it is used -- so pretty random.
// But you can initialize it with a fixed seed before using any of the Random* methods.
//
// See details in graph.RandomNormal.
func (ctx *Context) RandomUniform(g *graph.Graph, shape shapes.Shape) (values *Node) {
	rngStateVar := ctx.getRngStateVar()
	rngState := rngStateVar.ValueGraph(g)
	rngState, values = graph.RandomUniform(rngState, shape)
	if g.Ok() {
		rngStateVar.SetValueGraph(rngState)
	}
	return
}
