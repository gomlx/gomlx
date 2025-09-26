package sampler

import (
	"fmt"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
)

// Rule defines one rule of the sampling strategy.
// It's created by [Strategy.Nodes], [Strategy.NodesFromSet] and [Rule.FromEdges].
// Don't modify it directly.
type Rule struct {
	Sampler  *Sampler
	Strategy *Strategy

	// Name of the [Rule].
	Name string

	// ConvKernelScopeName doesn't affect sampling, but can be used to uniquely identify
	// the scope used for the kernels in a GNN to do convolutions on this rule.
	// If two rules have the same ConvKernelScopeName, they will share weights.
	ConvKernelScopeName string

	// UpdateKernelScopeName doesn't affect sampling, but can be used to uniquely identify
	// the scope used for the kernels in a GNN to do convolutions on this rule.
	// If two rules have the same UpdateKernelScopeName, they will share weights.
	UpdateKernelScopeName string

	// NodeTypeName of the nodes sampled by this rule.
	NodeTypeName string

	// NumNodes for NodeTypeName. Only used if NodeSet is not provided.
	NumNodes int32

	// SourceRule is the Name of the [Rule] this rule uses as source, or empty if
	// this is a "Node" sampling rule (a root/seed sampling)
	SourceRule *Rule

	// Dependents is the list of Rules that depend on this one.
	// That is other rules that have this Rule as [SourceRule].
	// This is to keep track of the graph, and are not involved on the sampling of this rule.
	Dependents []*Rule

	// EdgeType that connects the [SourceRule] node type, to the node type ([NodeTypeName]) of this Rule.
	// This is only set if this is an edge sampling rule. A node sampling rule (for seeds) have this set to nil.
	EdgeType *EdgeType

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

// IsIdentitySubRule returns whether this is an identity sub-rule with a 1-to-1 mapping.
func (r *Rule) IsIdentitySubRule() bool {
	return r.SourceRule != nil && r.EdgeType == nil
}

// WithKernelScopeName will set both ConvKernelScopeName and UpdateKernelScopeName to `name`.
func (r *Rule) WithKernelScopeName(name string) *Rule {
	r.ConvKernelScopeName = name
	r.UpdateKernelScopeName = name
	return r
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
	if r.IsIdentitySubRule() {
		return fmt.Sprintf("Rule %q: type=Edge, nodeType=%q, Shape=%s (size=%d), SourceRule=%q, EdgeType=Identity",
			r.Name, r.NodeTypeName, r.Shape, r.Shape.Size(), r.SourceRule.Name)
	}
	return fmt.Sprintf("Rule %q: type=Edge, nodeType=%q, Shape=%s (size=%d), SourceRule=%q, EdgeType=%q",
		r.Name, r.NodeTypeName, r.Shape, r.Shape.Size(), r.SourceRule.Name, r.EdgeType.Name)
}

// FromEdges returns a [Rule] that samples nodes from the edges connecting the results of the current Rule `r`.
func (r *Rule) FromEdges(name, edgeTypeName string, count int) *Rule {
	strategy := r.Strategy
	if strategy.frozen {
		Panicf("Strategy is frozen, that is, a dataset was already created and used with NewDataset() and hence can no longer be modified.")
	}
	if prevRule, found := strategy.Rules[name]; found {
		Panicf("rule named %q already exists: %s", name, prevRule)
	}
	edgeDef, found := r.Sampler.EdgeTypes[edgeTypeName]
	if !found {
		Panicf("edge type %q not found to sample from in rule %q", edgeTypeName, name)
	}
	if edgeDef.SourceNodeType != r.NodeTypeName {
		Panicf("edge type %q connects %q to %q: but you are using it on sampling rule %q, which is of node type %q",
			edgeTypeName, edgeDef.SourceNodeType, edgeDef.TargetNodeType, r.Name, r.NodeTypeName)
	}
	newShape := r.Shape.Clone()
	newShape.Dimensions = append(newShape.Dimensions, count)
	numNodes, found := strategy.Sampler.NodeTypesToCount[edgeDef.TargetNodeType]
	if !found {
		Panicf("unknown target node type %q to for rule %q", edgeDef.TargetNodeType, name)
	}
	newRule := &Rule{
		Sampler:      r.Sampler,
		Strategy:     strategy,
		Name:         name,
		NodeTypeName: edgeDef.TargetNodeType,
		SourceRule:   r,
		EdgeType:     edgeDef,
		Count:        count,
		NumNodes:     numNodes,
		Shape:        newShape,
	}
	newRule = newRule.WithKernelScopeName("gnn:" + name)
	r.Dependents = append(r.Dependents, newRule)
	strategy.Rules[name] = newRule
	return newRule
}

// IdentitySubRule creates a sub-rule that copies over the current rule, adding one rank (but same size).
// This is useful when trying to split updates into different parts, with the "IdentitySubRule" taking a
// subset of the dependents.
func (r *Rule) IdentitySubRule(name string) *Rule {
	strategy := r.Strategy
	if strategy.frozen {
		Panicf("Strategy is frozen, that is, a dataset was already created and used with NewDataset() and hence can no longer be modified.")
	}
	if prevRule, found := strategy.Rules[name]; found {
		Panicf("rule named %q already exists: %s", name, prevRule)
	}
	newShape := r.Shape.Clone()
	newShape.Dimensions = append(newShape.Dimensions, 1)
	newRule := &Rule{
		Sampler:      r.Sampler,
		Strategy:     strategy,
		Name:         name,
		NodeTypeName: r.NodeTypeName,
		NumNodes:     r.NumNodes,
		SourceRule:   r,
		EdgeType:     nil, // This identifies this as an identity sub-rule.
		Count:        1,   // 1-to-1 mapping.
		Shape:        newShape,
	}
	newRule = newRule.WithKernelScopeName("gnn:" + name)
	r.Dependents = append(r.Dependents, newRule)
	strategy.Rules[name] = newRule
	return newRule

}
