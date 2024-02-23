package sampler

import (
	"fmt"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"strings"
)

// Strategy is created by [Sampler]. A [Sampler] can create multiple [Strategy]s, a typical
// example is creating one for training, one for validation and one for testing.
//
// After creation (see [Sampler.NewStrategy]), one defines what and how to sample a subgraph,
// by creating "rules" ([Rule]) that will translate to sampled nodes.
//
// Once the strategy is defined, it can be used to create one or more datasets -- and after datasets are created,
// the strategy can no longer be changed.
type Strategy struct {
	sampler *Sampler
	frozen  bool // If set to true, it can no longer be modified.

	rules map[string]*Rule
}

// String returns a multi-line informative description of the strategy.
func (strategy *Strategy) String() string {
	parts := make([]string, 0, 1+len(strategy.rules))
	var frozenDesc string
	if strategy.frozen {
		frozenDesc = ", Frozen"
	}
	parts = append(parts, fmt.Sprintf("Sampling strategy: (%d rules%s)", len(strategy.rules), frozenDesc))
	for _, rule := range strategy.rules {
		if !rule.IsNode() {
			// These will be included under their root nodes.
			continue
		}
		parts = appendRulesRecursively(parts, rule, 0)
	}
	return strings.Join(parts, "\n")
}

// appendRulesRecursively enumerates rule descriptions (strings) recursively.
func appendRulesRecursively(parts []string, rule *Rule, indent int) []string {
	parts = append(parts, fmt.Sprintf("%s> %s", strings.Repeat("  ", indent), rule))
	indent++
	for _, subRule := range rule.dependents {
		parts = appendRulesRecursively(parts, subRule, indent)
	}
	return parts
}

// Nodes creates a rule (named `name`) to sample nodes randomly without replacement
// from the node type given by `nodeTypeName`.
//
// Nodes will be indices from 0 to the number of elements of the given node type.
//
// Node sampling (as opposed to Edges sampling) are typically the "root nodes" or "seed nodes" of a tree being
// sampled, that represent the sampled sub-graph.
//
// If this is used to sample the seed nodes, `count` in this case will be typically the batch size.
func (strategy *Strategy) Nodes(name, nodeTypeName string, count int) *Rule {
	if strategy.frozen {
		Panicf("Strategy is frozen, that is, a dataset was already created and used with NewDataset() and hence can no longer be modified.")
	}
	if _, found := strategy.sampler.d.NodeTypesToCount[nodeTypeName]; !found {
		Panicf("unknown node type %q to for rule %q", nodeTypeName, name)
	}
	if prevRule, found := strategy.rules[name]; found {
		Panicf("rule named %q already exists: %s", name, prevRule)
	}
	r := &Rule{
		sampler:      strategy.sampler,
		strategy:     strategy,
		name:         name,
		nodeTypeName: nodeTypeName,
		count:        count,
		shape:        shapes.Make(shapes.Int32, count),
	}
	strategy.rules[name] = r
	return r
}

// NodesFromSet creates a rule (named `name`) to sample nodes randomly without replacement
// from the node type given by `nodeTypeName`, but selecting only from the given nodeSet.
//
// `nodeSet` is a list of valid node indices for the given node type from which to sample.
//
// Node sampling (as opposed to Edges sampling) are typically the "root nodes" or "seed nodes" of a tree being
// sampled, that represent the sampled sub-graph.
//
// If this is used to sample the seed nodes, `count` in this case will be typically the batch size.
func (strategy *Strategy) NodesFromSet(name, nodeTypeName string, count int, nodeSet []int32) *Rule {
	if strategy.frozen {
		Panicf("Strategy is frozen, that is, a dataset was already created and used with NewDataset() and hence can no longer be modified.")
	}
	if _, found := strategy.sampler.d.NodeTypesToCount[nodeTypeName]; !found {
		Panicf("unknown node type %q to for rule %q", nodeTypeName, name)
	}
	if prevRule, found := strategy.rules[name]; found {
		Panicf("rule named %q already exists: %s", name, prevRule)
	}
	r := &Rule{
		sampler:      strategy.sampler,
		strategy:     strategy,
		name:         name,
		nodeTypeName: nodeTypeName,
		count:        count,
		shape:        shapes.Make(shapes.Int32, count),
		nodeSet:      nodeSet,
	}
	strategy.rules[name] = r
	return r
}

// NewDataset creates a new [Dataset] from the configured [Strategy].
// One can create multiple datasets from the same [Strategy], but once a [Dataset] is created,
// the [Strategy] is considered frozen and can no longer be modified.
func (strategy *Strategy) NewDataset() any {
	strategy.frozen = true
	return nil
}
