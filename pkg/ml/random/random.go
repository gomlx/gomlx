package random

import (
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/polymorphicjson"
)

// Interface of for a generic random number generator.
//
// Use the concrete wrapper Random instead for the more convenient Uniform and Normal methods,
// as well as being serializable.
type Interface interface {
	polymorphicjson.JSONIdentifiable // To make it serializable when using the Random object.

	// Uniform returns a uniform random value from 0 to 1.
	Uniform(g *graph.Graph, shape shapes.Shape) (values *graph.Node)

	// Normal returns a random value from a normal distribution with mean 0 and standard deviation 1.
	Normal(g *graph.Graph, shape shapes.Shape) (values *graph.Node)

	// IntN returns a random integer uniformly from 0 to n-1.
	//
	// n is given statically (constant for the graph). Use Uniform()*n for dynamic values of n.
	IntN(g *graph.Graph, n int, shape shapes.Shape) (values *graph.Node)

	// SplitIface returns a new random number generator that is independent of this one.
	// It will be the same algorithm, but with different states.
	SplitIface(g *graph.Graph) Interface
}

const InterfaceName = "random.Interface"

// Random is a generic random number generator.
//
// It wraps a random.Interface and provides (de-)serialization and a couple of convenience methods.
type Random struct {
	polymorphicjson.Wrapper[Interface]
}

// JSONTags implements the polymorphicjson.JSONIdentifiable interface, used for (de-)serialization.
func (r *Random) JSONTags() (interfaceName, concreteType string) {
	return r.Wrapper.Value.JSONTags()
}

// Uniform returns a uniform random value from 0 to 1.
func (r *Random) Uniform(g *graph.Graph, shape shapes.Shape) (values *graph.Node) {
	return r.Wrapper.Value.Uniform(g, shape)
}

// Normal returns a random value from a normal distribution with mean 0 and standard deviation 1.
func (r *Random) Normal(g *graph.Graph, shape shapes.Shape) (values *graph.Node) {
	return r.Wrapper.Value.Normal(g, shape)
}

// IntN returns a random integer uniformly from 0 to n-1.
//
// n is given statically (constant for the graph). Use Uniform()*n for dynamic values of n.
func (r *Random) IntN(g *graph.Graph, n int, shape shapes.Shape) (values *graph.Node) {
	return r.Wrapper.Value.IntN(g, n, shape)
}

// SplitIface returns a new random number generator that is independent of this one.
// It will be the same algorithm, but with different states.
//
// See Split which returns a more convenient Random object.
func (r *Random) SplitIface(g *graph.Graph) Interface {
	return r.Wrapper.Value.SplitIface(g)
}

// Split returns a new random number generator that is independent of ("split from") this one.
// It will be the same algorithm, but with different states.
func (r *Random) Split(g *graph.Graph) *Random {
	return NewRandom(r.SplitIface(g))
}

// NewRandom from an Interface.
func NewRandom(r Interface) *Random {
	return &Random{Wrapper: polymorphicjson.Wrapper[Interface]{Value: r}}
}

// New creates a new Random object using the default algorithm (Philox).
func New() *Random {
	return NewRandom(NewPhilox())
}

// NewWithSeed creates a new Random object using the given seed and the default algorithm (Philox).
func NewWithSeed(seed int64) *Random {
	return NewRandom(NewPhiloxWithSeed(seed))
}
