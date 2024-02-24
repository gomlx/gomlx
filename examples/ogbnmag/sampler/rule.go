package sampler

import (
	"fmt"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
)

// Rule defines one rule of the sampling strategy. It's created by the various methods
type Rule struct {
	sampler  *Sampler
	strategy *Strategy

	// name of the [Rule].
	name string

	// nodeTypeName of the nodes sampled by this rule.
	nodeTypeName string

	// numNodes for nodeTypeName. Only used if nodeSet is not provided.
	numNodes int32

	// sourceRule is the name of the [Rule] this rule uses as source, or empty if
	// this is a "Node" sampling rule (a root/seed sampling)
	sourceRule *Rule

	// dependents is the list of rules that depend on this one.
	dependents []*Rule

	// edgeTypeName used to sample from, if this is an "Edge" sampling rule, or empty.
	edgeTypeName string

	// edgeType
	edgeType *edgeType

	// count is the number of samples to create. It will define the last dimension of the tensor sampled.
	count int

	// shape of the sample for this rule.
	shape shapes.Shape

	// nodeSet is a set of indices that a "Node" rule is allowed to sample from.
	// E.g.: have separate nodeSet for train, test and validation datasets.
	nodeSet []int32
}

// IsNode returns whether this is a "Node" rule, it can also be seen as a root rule.
func (r *Rule) IsNode() bool {
	return r.sourceRule == nil
}

// String returns an informative description of the rule.
func (r *Rule) String() string {
	if r.IsNode() {
		var sourceSetDesc string
		if r.nodeSet != nil {
			sourceSetDesc = fmt.Sprintf(", nodeSet.size=%d", len(r.nodeSet))
		}
		return fmt.Sprintf("Rule %q: type=Node, nodeType=%q, shape=%s (size=%d)%s", r.name, r.nodeTypeName, r.shape, r.shape.Size(), sourceSetDesc)
	}
	return fmt.Sprintf("Rule %q: type=Edge, nodeType=%q, shape=%s (size=%d), sourceRule=%q, edgeType=%q",
		r.name, r.nodeTypeName, r.shape, r.shape.Size(), r.sourceRule.name, r.edgeTypeName)
}

// FromEdges returns a [Rule] that samples nodes from the edges connecting the results of the current Rule `r`.
func (r *Rule) FromEdges(name, edgeTypeName string, count int) *Rule {
	strategy := r.strategy
	if strategy.frozen {
		Panicf("Strategy is frozen, that is, a dataset was already created and used with NewDataset() and hence can no longer be modified.")
	}
	if prevRule, found := strategy.rules[name]; found {
		Panicf("rule named %q already exists: %s", name, prevRule)
	}
	edgeDef, found := r.sampler.d.EdgeTypes[edgeTypeName]
	if !found {
		Panicf("edge type %q not found to sample from in rule %q", edgeTypeName, name)
	}
	if edgeDef.SourceNodeType != r.nodeTypeName {
		Panicf("edge type %q connects %q to %q: but you are using it on sampling rule %q, which is of node type %q",
			edgeTypeName, edgeDef.SourceNodeType, edgeDef.TargetNodeType, r.name, r.nodeTypeName)
	}
	newShape := r.shape.Copy()
	newShape.Dimensions = append(newShape.Dimensions, count)
	newRule := &Rule{
		sampler:      r.sampler,
		strategy:     strategy,
		name:         name,
		nodeTypeName: edgeDef.TargetNodeType,
		sourceRule:   r,
		edgeTypeName: edgeTypeName,
		edgeType:     edgeDef,
		count:        count,
		shape:        newShape,
	}
	r.dependents = append(r.dependents, newRule)
	strategy.rules[name] = newRule
	return newRule
}
