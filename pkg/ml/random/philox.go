package random

import (
	"github.com/gomlx/gomlx/internal/must"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/model"
)

// Philox is the default implementation of the random.Interface.
type Philox struct {
	state *model.Variable
}

// JSONTags implements the polymorphicjson.JSONIdentifiable interface, used for (de-)serialization.
func (p *Philox) JSONTags() (interfaceName, concreteType string) {
	return InterfaceName, "Philox"
}

var _ Interface = &Philox{}

// NewPhilox returns a new Philox with a state initialized from the system clock or random generation engine.
func NewPhilox() *Philox {
	return &Philox{
		state: must.M1(model.VariableWithValue("rng_state", RngState())),
	}
}

// NewPhiloxWithSeed returns a new Philox with a state initialized from the given seed.
func NewPhiloxWithSeed(seed int64) *Philox {
	return &Philox{
		state: must.M1(model.VariableWithValue("rng_state", RngStateFromSeed(seed))),
	}
}

// Uniform returns a uniform random value from 0 to 1.
func (p *Philox) Uniform(g *Graph, shape shapes.Shape) (values *Node) {
	state := p.state.ValueGraph(g)
	state, values = RandomUniform(state, shape)
	p.state.SetValueGraph(state)
	return
}

// Normal returns a random value from a normal distribution with mean 0 and standard deviation 1.
func (p *Philox) Normal(g *Graph, shape shapes.Shape) (values *Node) {
	state := p.state.ValueGraph(g)
	state, values = RandomNormal(state, shape)
	p.state.SetValueGraph(state)
	return
}

// IntN returns a random integer uniformly from 0 to n-1.
//
// n is given statically (constant for the graph). Use Uniform()*n for dynamic values of n.
func (p *Philox) IntN(g *Graph, n int, shape shapes.Shape) (values *Node) {
	state := p.state.ValueGraph(g)
	state, values = RandomIntN(state, n, shape)
	p.state.SetValueGraph(state)
	return
}

// SplitIface returns a new random number generator that is independent of this one.
// It will be the same algorithm, but with different states.
func (p *Philox) SplitIface(g *Graph) Interface {
	state := p.state.ValueGraph(g)
	var splitState *Node
	state, splitState = RngStateSplit(state)
	p.state.SetValueGraph(state)

	newP := NewPhiloxWithSeed(0)
	newP.state.SetValueGraph(splitState)
	return newP
}
