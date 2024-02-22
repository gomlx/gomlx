package sampler

import (
	"fmt"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
)

// Rule defines one rule of the sampling strategy. It's created by the various methods
type Rule struct {
	sampler  *Sampler
	strategy *Strategy

	// name of the [Rule].
	name string

	// nodeTypeName of the nodes sampled by this rule.
	nodeTypeName string

	// sourceRule is the name of the [Rule] this rule uses as source, or empty if
	// this is a "Node" sampling rule (a root/seed sampling)
	sourceRule *Rule

	// dependents is the list of rules that depend on this one.
	dependents []*Rule

	// edgeTypeName used to sample from, if this is an "Edge" sampling rule, or empty.
	edgeTypeName string

	// count is the number of samples to create. It will define the last dimension of the tensor sampled.
	count int

	// shape of the sample for this rule.
	shape shapes.Shape

	// sourceSet is a set (a tensor) of indices that a "Node" rule is allowed to sample from.
	// E.g.: have separate sourceSet for train, test and validation datasets.
	sourceSet *tensor.Local
}

// IsNode returns whether this is a "Node" rule, it can also be seen as a root rule.
func (r *Rule) IsNode() bool {
	return r.sourceRule == nil
}

// String returns an informative description of the rule.
func (r *Rule) String() string {
	if r.IsNode() {
		var sourceSetDesc string
		if r.sourceSet != nil {
			sourceSetDesc = fmt.Sprintf(", sourceSet.size=%s", r.sourceSet.Shape().Size())
		}
		return fmt.Sprintf("Rule %q: type=Node, nodeType=%q, shape=%s%s", r.name, r.nodeTypeName, r.shape, sourceSetDesc)
	}
	return fmt.Sprintf("Rule %q: type=Edge, nodeType=%q, shape=%s, sourceRule=%q, edgeType=%q",
		r.name, r.nodeTypeName, r.shape, r.sourceRule.name, r.edgeTypeName)
}

// RandomNodes creates a rule (named `name`) to sample nodes randomly without replacement
// from the node type given by `nodeTypeName`.
//
// Nodes will be indices from 0 to the number of elements of the given node type.
//
// Node sampling (as opposed to Edges sampling) are typically the "root nodes" or "seed nodes" of a tree being
// sampled, that represent the sampled sub-graph.
//
// If this is used to sample the seed nodes, `count` in this case will be typically the batch size.
func (st *Strategy) RandomNodes(name, nodeTypeName string, count int) *Rule {
	if _, found := st.sampler.d.NodeTypesToCount[nodeTypeName]; !found {
		Panicf("unknown node type %q to for rule %q", nodeTypeName, name)
	}
	return &Rule{
		sampler:      st.sampler,
		strategy:     st,
		name:         name,
		nodeTypeName: nodeTypeName,
		count:        count,
	}
}
