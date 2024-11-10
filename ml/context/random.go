package context

import (
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"k8s.io/klog/v2"
)

const (
	RngStateVariableName = "#rngState"
)

func (ctx *Context) getRngStateVar() *Variable {
	rngStateVar := ctx.GetVariableByScopeAndName(RootScope, RngStateVariableName)
	if rngStateVar == nil {
		randomState := graph.RngState()
		rngStateVar = ctx.InAbsPath(RootScope).Checked(false).
			VariableWithValue(RngStateVariableName, randomState).SetTrainable(false)
	} else if rngStateVar.Trainable {
		klog.Warningf("Variable %q was trainable, marking it as non-trainable.", rngStateVar.ParameterName())
		rngStateVar.SetTrainable(false)
	}
	return rngStateVar
}

// RngStateReset resets the default context random number generator (RNG) to a random seed based on
// the nanosecond clock.
//
// This is done automatically for new contexts, but if the context was loaded from a checkpoint, and one
// wants to reset it (as opposed to continue the previous state), one can call this.
//
// The random number generator (RNG) state is stored in a variable on the root scope
// of the context, called "#rngState" (RngStateVariableName).
func (ctx *Context) RngStateReset() {
	v := ctx.getRngStateVar()
	v.SetValue(graph.RngState())
}

// RngStateFromSeed initializes the default context random number generator (RNG) state with a static seed.
// If the state has already been created, it is reset to a value based on the seed.
//
// The random number generator (RNG) state is stored in a variable on the root scope
// of the context, called "#rngState" (RngStateVariableName).
func (ctx *Context) RngStateFromSeed(seed int64) {
	initialState := graph.RngStateFromSeed(seed)
	v := ctx.getRngStateVar()
	v.SetValue(initialState)
}

// RandomNormal generates random numbers from a normal distribution, with mean 0.0
// and standard deviation 1.0.
// It generates values with the given shape, each value pseudo-randomly generated.
//
// If you need a different mean and standard deviation, just do something like the example below, where `mean`
// and `stddev` are the desired mean and standard deviation:
//
//	numbers = ctx.RandomNormal(g, myShape)
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
	rngStateVar.SetValueGraph(rngState)
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
	rngStateVar.SetValueGraph(rngState)
	return
}

// RandomIntN generates random numbers uniformly from 0 to N-1. It only works for integer types, see RandomUniform for
// float or complex data types. N can be given as a Node, or a static scalar integer value.
//
// Example:
//
//	D10 := ctx.RandomIntN(10, shapes.Make(dtypes.Int32))
//
// The random number generator (RNG) state is stored in a variable on the root scope
// of the context, called "#rngState" (RngStateVariableName).
// The state is initialized with the nanosecond clock, the first time it is used -- so pretty random.
// But you can initialize it with a fixed seed before using any of the Random* methods.
//
// See details in graph.RandomIntN.
func (ctx *Context) RandomIntN(g *graph.Graph, N any, shape shapes.Shape) (values *Node) {
	rngStateVar := ctx.getRngStateVar()
	rngState := rngStateVar.ValueGraph(g)
	switch n := N.(type) {
	case *Node:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case uint8:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case uint16:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case uint32:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case uint64:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case int8:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case int16:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case int32:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	case int64:
		rngState, values = graph.RandomIntN(rngState, n, shape)
	}
	rngStateVar.SetValueGraph(rngState)
	return
}
