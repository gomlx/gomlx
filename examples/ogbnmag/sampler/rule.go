package sampler

import (
	"fmt"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
)

// Rule defines one rule of the sampling strategy.
// It's created by [Strategy.Nodes], [Strategy.NodesFromSet] and [Rule.FromEdges].
// Don't modity it directly.
type Rule struct {
	sampler  *Sampler
	strategy *Strategy

	// Name of the [Rule].
	Name string

	// NodeTypeName of the nodes sampled by this rule.
	NodeTypeName string

	// NumNodes for NodeTypeName. Only used if NodeSet is not provided.
	NumNodes int32

	// SourceRule is the Name of the [Rule] this rule uses as source, or empty if
	// this is a "Node" sampling rule (a root/seed sampling)
	SourceRule *Rule

	// Dependents is the list of Rules that depend on this one.
	Dependents []*Rule

	// EdgeTypeName used to sample from, if this is an "Edge" sampling rule, or empty.
	EdgeTypeName string

	// EdgeType
	EdgeType *edgeType

	// Count is the number of samples to create. It will define the last dimension of the tensor sampled.
	Count int

	// Shape of the sample for this rule.
	Shape shapes.Shape

	// NodeSet is a set of indices that a "Node" rule is allowed to sample from.
	// E.g.: have separate NodeSet for train, test and validation datasets.
	NodeSet []int32
}

// IsNode returns whether this is a "Node" rule, it can also be seen as a root rule.
func (r *Rule) IsNode() bool {
	return r.SourceRule == nil
}

// String returns an informative description of the rule.
func (r *Rule) String() string {
	if r.IsNode() {
		var sourceSetDesc string
		if r.NodeSet != nil {
			sourceSetDesc = fmt.Sprintf(", NodeSet.size=%d", len(r.NodeSet))
		}
		return fmt.Sprintf("Rule %q: type=Node, nodeType=%q, Shape=%s (size=%d)%s", r.Name, r.NodeTypeName, r.Shape, r.Shape.Size(), sourceSetDesc)
	}
	return fmt.Sprintf("Rule %q: type=Edge, nodeType=%q, Shape=%s (size=%d), SourceRule=%q, EdgeType=%q",
		r.Name, r.NodeTypeName, r.Shape, r.Shape.Size(), r.SourceRule.Name, r.EdgeTypeName)
}

// FromEdges returns a [Rule] that samples nodes from the edges connecting the results of the current Rule `r`.
func (r *Rule) FromEdges(name, edgeTypeName string, count int) *Rule {
	strategy := r.strategy
	if strategy.frozen {
		Panicf("Strategy is frozen, that is, a dataset was already created and used with NewDataset() and hence can no longer be modified.")
	}
	if prevRule, found := strategy.Rules[name]; found {
		Panicf("rule named %q already exists: %s", name, prevRule)
	}
	edgeDef, found := r.sampler.d.EdgeTypes[edgeTypeName]
	if !found {
		Panicf("edge type %q not found to sample from in rule %q", edgeTypeName, name)
	}
	if edgeDef.SourceNodeType != r.NodeTypeName {
		Panicf("edge type %q connects %q to %q: but you are using it on sampling rule %q, which is of node type %q",
			edgeTypeName, edgeDef.SourceNodeType, edgeDef.TargetNodeType, r.Name, r.NodeTypeName)
	}
	newShape := r.Shape.Copy()
	newShape.Dimensions = append(newShape.Dimensions, count)
	newRule := &Rule{
		sampler:      r.sampler,
		strategy:     strategy,
		Name:         name,
		NodeTypeName: edgeDef.TargetNodeType,
		SourceRule:   r,
		EdgeTypeName: edgeTypeName,
		EdgeType:     edgeDef,
		Count:        count,
		Shape:        newShape,
	}
	r.Dependents = append(r.Dependents, newRule)
	strategy.Rules[name] = newRule
	return newRule
}