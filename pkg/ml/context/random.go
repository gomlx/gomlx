package context

import (
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"k8s.io/klog/v2"
)

const (
	// RngStateVariableName is the name of a Context internal variable the holds the current
	// random number generator state.
	RngStateVariableName = "#rngState"
)

var (
	// ParamInitialSeed is the key for the hyperparameter to use for initial seed (int64). The default is 0,
	// which makes it non-deterministic. Set it to a value different from 0 for a deterministic (as long
	// as the model doesn't change) initialization.
	ParamInitialSeed = "initializers_seed"
)

// getRngStateVar panics if it fails to create the random state.
func (ctx *Context) getRngStateVar() *Variable {
	rngStateVar := ctx.GetVariableByScopeAndName(RootScope, RngStateVariableName)
	if rngStateVar != nil {
		return rngStateVar
	}
	rngStateVar = ctx.InAbsPath(RootScope).Checked(false).
		VariableWithShape(RngStateVariableName, graph.RngStateShape).SetTrainable(false)
	return rngStateVar
}

// mustGetRngStateVarWithValue panics if it fails to create the random state.
func (ctx *Context) mustGetRngStateVarWithValue() *Variable {
	v := ctx.getRngStateVar()
	if v == nil {
		exceptions.Panicf("random state variable not initialized for context -- see Context.RngStateReset()")
	}
	return v
}

// RngStateReset resets the default context random number generator (RNG) to a random seed based on
// the nanosecond clock.
//
// This is required if using the context RNG state variable.
// Variable initialization will call RngStateReset automatically, if it hasn't yet been called.
//
// If ParamInitialSeed is set, it will be used instead of cryptographically secure random seed.
//
// If the context is loaded from a checkpoint, and one wants to reset it (as opposed to continue
// with the previous state), one can call this.
//
// The random number generator (RNG) state is stored in a variable on the root scope
// of the context, called "#rngState" (RngStateVariableName).
func (ctx *Context) RngStateReset() error {
	v := ctx.getRngStateVar()
	var randomState *tensors.Tensor
	seedAny, found := ctx.GetParam(ParamInitialSeed)
	if !found {
		var err error
		randomState, err = graph.RngState()
		if err != nil {
			return err
		}
	} else {
		seed, ok := seedAny.(int64)
		if !ok {
			klog.Errorf("Seed in %q not an int64, using 0 instead", ParamInitialSeed)
		}
		randomState = graph.RngStateFromSeed(seed)
	}
	return v.SetValue(randomState)
}

// RngStateFromSeed initializes the default context random number generator (RNG) state with a static seed.
// If the state has already been created, it is reset to a value based on the seed.
//
// The random number generator (RNG) state is stored in a variable on the root scope
// of the context, called "#rngState" (RngStateVariableName).
//
// This overrides the seed used in ParamInitialSeed.
func (ctx *Context) RngStateFromSeed(seed int64) {
	initialState := graph.RngStateFromSeed(seed)
	v := ctx.getRngStateVar()
	v.MustSetValue(initialState)
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
	rngStateVar := ctx.mustGetRngStateVarWithValue()
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
	rngStateVar := ctx.mustGetRngStateVarWithValue()
	rngState := rngStateVar.ValueGraph(g)
	rngState, values = graph.RandomUniform(rngState, shape)
	rngStateVar.SetValueGraph(rngState)
	return
}

// RandomBernoulli generates 0s and 1s in the given shape (or True/False if shape dtype is Bool),
// with probability of 1s being prob.
//
// It uses a random number generation with precision equal to prob.DType().
//
// See Bernoulli Distribution article: https://en.wikipedia.org/wiki/Bernoulli_distribution
func (ctx *Context) RandomBernoulli(prob *Node, shape shapes.Shape) *Node {
	g := prob.Graph()
	maskShape := shape.Clone()
	maskShape.DType = prob.DType()
	mask := ctx.RandomUniform(g, maskShape)
	mask = graph.LessThan(mask, prob)
	return graph.ConvertDType(mask, shape.DType)
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
	rngStateVar := ctx.mustGetRngStateVarWithValue()
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
