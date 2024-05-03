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
// by creating "Rules" ([Rule]) that will translate to sampled nodes.
//
// Once the strategy is defined, it can be used to create one or more datasets -- and after datasets are created,
// the strategy can no longer be changed.
type Strategy struct {
	Sampler *Sampler
	frozen  bool // If set to true, it can no longer be modified.

	// KeepDegrees means the sampler should add a tensor for all edges with the degrees of source sampling nodes.
	KeepDegrees bool

	// Rules lists all the rules of a strategy.
	// It can be used for reading, but don't change it.
	Rules map[string]*Rule

	// Seeds lists all the rules that are seeds.
	// It can be used for reading, but don't change it.
	Seeds []*Rule
}

// String returns a multi-line informative description of the strategy.
func (strategy *Strategy) String() string {
	parts := make([]string, 0, 1+len(strategy.Rules))
	var frozenDesc string
	if strategy.frozen {
		frozenDesc = ", Frozen"
	}
	parts = append(parts, fmt.Sprintf("Sampling strategy: (%d Rules%s)", len(strategy.Rules), frozenDesc))
	for _, rule := range strategy.Rules {
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
	for _, subRule := range rule.Dependents {
		parts = appendRulesRecursively(parts, subRule, indent)
	}
	return parts
}

// Nodes creates a rule (named `Name`) to sample nodes randomly without replacement
// from the node type given by `NodeTypeName`.
//
// Nodes will be indices from 0 to the number of elements of the given node type.
//
// Node sampling (as opposed to Edges sampling) are typically the "root nodes" or "seed nodes" of a tree being
// sampled, that represent the sampled sub-graph.
//
// If this is used to sample the seed nodes, `Count` in this case will be typically the batch size.
func (strategy *Strategy) Nodes(name, nodeTypeName string, count int) *Rule {
	if strategy.frozen {
		Panicf("Strategy is frozen, that is, a dataset was already created and used with NewDataset() and hence can no longer be modified.")
	}
	numNodes, found := strategy.Sampler.NodeTypesToCount[nodeTypeName]
	if !found {
		Panicf("unknown node type %q to for rule %q", nodeTypeName, name)
	}
	if prevRule, found := strategy.Rules[name]; found {
		Panicf("rule named %q already exists: %s", name, prevRule)
	}
	r := &Rule{
		Sampler:      strategy.Sampler,
		Strategy:     strategy,
		Name:         name,
		NodeTypeName: nodeTypeName,
		NumNodes:     numNodes,
		Count:        count,
		Shape:        shapes.Make(shapes.Int32, count),
	}
	r = r.WithKernelScopeName("gnn:" + name)
	strategy.Rules[name] = r
	strategy.Seeds = append(strategy.Seeds, r)
	return r
}

// NodesFromSet creates a rule (named `Name`) to sample nodes randomly without replacement
// from the node type given by `NodeTypeName`, but selecting only from the given NodeSet.
//
// `NodeSet` is a list of valid node indices for the given node type from which to sample.
//
// Node sampling (as opposed to Edges sampling) are typically the "root nodes" or "seed nodes" of a tree being
// sampled, that represent the sampled sub-graph.
//
// If this is used to sample the seed nodes, `Count` in this case will be typically the batch size.
func (strategy *Strategy) NodesFromSet(name, nodeTypeName string, count int, nodeSet []int32) *Rule {
	if strategy.frozen {
		Panicf("Strategy is frozen, that is, a dataset was already created and used with NewDataset() and hence can no longer be modified.")
	}
	numNodes, found := strategy.Sampler.NodeTypesToCount[nodeTypeName]
	if !found {
		Panicf("unknown node type %q to for rule %q", nodeTypeName, name)
	}
	if prevRule, found := strategy.Rules[name]; found {
		Panicf("rule named %q already exists: %s", name, prevRule)
	}
	r := &Rule{
		Sampler:      strategy.Sampler,
		Strategy:     strategy,
		Name:         name,
		NodeTypeName: nodeTypeName,
		NumNodes:     numNodes,
		Count:        count,
		Shape:        shapes.Make(shapes.Int32, count),
		NodeSet:      nodeSet,
	}
	r = r.WithKernelScopeName("gnn:" + name)
	strategy.Rules[name] = r
	strategy.Seeds = append(strategy.Seeds, r)
	return r
}

// ValueMask contains a pair of [tensor.Tensor] or [*graph.Node] (Value, Mask).
type ValueMask[T any] struct {
	Value, Mask T
}

// MapInputsToStates convert inputs yielded by a [sampler.Dataset] to map of the Rules Name to the
// Value/Mask tensors with the samples for this example.
//
// It returns also the remaining not used inputs (or empty if all were consumed).
//
// Example 1: if using directly the outputs of a [sampler.Dataset] created by this Strategy:
//
//	spec, inputs, _, err := ds.Yield()
//	strategy := spec.(*Sampler.Strategy)
//	graphSample, _ := strategy.MapInputsToStates(inputs)
//	Seeds, mask := graphSample["Seeds"].Value, graphSample["Seeds"].Mask
//	...
//
// Example 2: usage in a model that is fed the output of a [sampler.Dataset]:
//
//	func MyModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
//		strategy := spec.(*Sampler.Strategy)
//		graphSample, _ := strategy.MapInputsToStates(inputs)
//		Seeds, mask := graphSample["Seeds"].Value, graphSample["Seeds"].Mask
//		...
//	}
func MapInputsToStates[T any](strategy *Strategy, inputs []T) (ruleToInput map[string]*ValueMask[T], remainingInputs []T) {
	mapNodes := make(map[string]*ValueMask[T], len(strategy.Rules))
	for _, seedRule := range strategy.Seeds {
		mapNodes[seedRule.Name] = &ValueMask[T]{inputs[0], inputs[1]}
		inputs = inputs[2:]
		inputs = recursivelyMapStateInputsToSubRules(inputs, seedRule, mapNodes)
	}
	return mapNodes, inputs
}

// recursivelyMapStateInputsToSubRules returns the remaining inputs and updates `mapNodes` with the sub-Rules
// dependent on `rule`.
func recursivelyMapStateInputsToSubRules[T any](inputs []T, rule *Rule, mapNodes map[string]*ValueMask[T]) []T {
	for _, subRule := range rule.Dependents {
		mapNodes[subRule.Name] = &ValueMask[T]{inputs[0], inputs[1]}
		inputs = inputs[2:]
		if rule.Strategy.KeepDegrees {
			// Extract degree tensor.
			mapNodes[NameForNodeDependentDegree(rule.Name, subRule.Name)] = &ValueMask[T]{Value: inputs[0], Mask: mapNodes[rule.Name].Mask}
			inputs = inputs[1:]
		}
		if len(subRule.Dependents) > 0 {
			inputs = recursivelyMapStateInputsToSubRules(inputs, subRule, mapNodes)
		}
	}
	return inputs
}

// EdgePair contains the source and target indices for the edges.
type EdgePair[T any] struct {
	SourceIndices, TargetIndices T
}

// MapInputsToEdges is similar to MapInputsToStates, but for the edges, when using LayerWiseInference.
func MapInputsToEdges[T any](strategy *Strategy, inputs []T) (ruleToEdgePair map[string]EdgePair[T], remainingInputs []T) {
	edges := make(map[string]EdgePair[T], len(strategy.Rules))
	for _, seedRule := range strategy.Seeds {
		// Seed nodes don't have "upwards" edges.
		inputs = recursivelyMapEdgeInputsToSubRules(inputs, seedRule, edges)
	}
	return edges, inputs
}

func recursivelyMapEdgeInputsToSubRules[T any](inputs []T, rule *Rule, edges map[string]EdgePair[T]) []T {
	for _, subRule := range rule.Dependents {
		edges[subRule.Name] = EdgePair[T]{SourceIndices: inputs[0], TargetIndices: inputs[1]}
		inputs = inputs[2:]
		if len(subRule.Dependents) > 0 {
			inputs = recursivelyMapEdgeInputsToSubRules(inputs, subRule, edges)
		}
	}
	return inputs
}
