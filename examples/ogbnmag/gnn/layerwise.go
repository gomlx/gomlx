package gnn

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"strings"
)

// LayerWiseConfig is a configuration object for [ComputeLayerWiseGNN].
// Once configured with its methods, call [LayerWiseConfig.Done] to actually run the layer-wise GNN.
type LayerWiseConfig struct {
	ctx                    *context.Context
	strategy               *sampler.Strategy
	sampler                *sampler.Sampler
	keepIntermediaryStates bool
	freeAcceleratorMemory  bool

	numGraphUpdates       int
	graphUpdateType       string
	dependentsUpdateFirst bool
}

// LayerWiseGNN can perform a GNN on a full layer (== node set convolution) at a time.
// It works only for [NodePrediction] GNNs.
//
// It returns a configuration option that can be furthered configured.
// It runs the layer-wise GNN once [Compute] is called.
func LayerWiseGNN(ctx *context.Context, strategy *sampler.Strategy) (*LayerWiseConfig, error) {
	lw := &LayerWiseConfig{
		ctx:                    ctx,
		strategy:               strategy,
		sampler:                strategy.Sampler,
		keepIntermediaryStates: true,
		freeAcceleratorMemory:  true,
		numGraphUpdates:        context.GetParamOr(ctx, ParamNumGraphUpdates, 2),
		graphUpdateType:        context.GetParamOr(ctx, ParamGraphUpdateType, "tree"),
	}
	lw.dependentsUpdateFirst = "tree" == lw.graphUpdateType
	if lw.graphUpdateType != "tree" && lw.graphUpdateType != "simultaneous" {
		return nil, errors.Errorf("unsupported graph update type: %s", lw.graphUpdateType)
	}
	if context.GetParamOr(ctx, ParamUsePathToRootStates, false) {
		return nil, errors.Errorf("layerwise inference doesn't work if using `%q=true`",
			ParamUsePathToRootStates)
	}
	return lw, nil
}

// KeepIntermediaryStates configures whether the GNN will return the final state of the seed nodes only, or if it also
// keeps the intermediary states.
//
// Default is true, it keeps intermediary states.
func (lw *LayerWiseConfig) KeepIntermediaryStates(keep bool) *LayerWiseConfig {
	lw.keepIntermediaryStates = keep
	return lw
}

// FreeAcceleratorMemory configures whether the layer wise GNN computation will free the tensors stored in accelerator
// (GPU) as soon as they are no longer used, or if they are left on GPU -- can be useful if it fits in the GPU
// and someone is going to use it to something else.
//
// The final seed node readouts are not freed from the accelerator, since we assume they will be further used.
//
// Default is true, it immediately frees space from the GPU when not in use anymore.
func (lw *LayerWiseConfig) FreeAcceleratorMemory(free bool) *LayerWiseConfig {
	lw.freeAcceleratorMemory = free
	return lw
}

// Compute computes the layer-wise GNN after and should be called once it's fully configured.
//
// `initialStates` should map node set names to its initial values, and be shaped `[num_nodes, initial_state_dim]`.
// Any preprocessing transformation (like normalizations) need to be calculated and materialized to tensors.
//
// It can be called more than once with different `initialStates`.
// Subsequent calls will by-step the JIT-compilation of the models.
func (lw *LayerWiseConfig) Compute(ctx *context.Context, graphStates map[string]*Node, edges map[string]sampler.EdgePair[*Node]) {
	for round := range lw.numGraphUpdates {
		for _, rule := range lw.strategy.Seeds {
			lw.recursivelyApplyGraphConvolution(ctxForGraphUpdateRound(ctx, round), rule, graphStates, edges)
		}
	}
}

func (lw *LayerWiseConfig) recursivelyApplyGraphConvolution(
	ctx *context.Context, rule *sampler.Rule, graphStates map[string]*Node, edges map[string]sampler.EdgePair[*Node]) {
	if rule.Name == "" || rule.ConvKernelScopeName == "" {
		Panicf("strategy's rule name=%q or kernel scope name=%q are empty, they both must be defined",
			rule.Name, rule.ConvKernelScopeName)
	}

	// Makes sure there is a state for the current dependent.
	state, found := graphStates[rule.Name]
	if !found {
		Panicf("state for rule %q not given in `graphStates`, states given for rules: %q", rule.Name, slices.Keys(graphStates))
	}
	if state == nil {
		Panicf("state for rule %q is set to nil in `graphStates` -- for LayerWise inference on needs to set the first dimension to the number of nodes in the node set for the rule, even if the last axis has dimension 0", rule.Name)
		panic(nil) // Remove lint error on state==nil not having been checked.
	}
	numNodes := state.Shape().Dimensions[0]

	// Leaf nodes are not updated.
	if len(rule.Dependents) == 0 {
		return
	}

	updateInputs := make([]*Node, 0, len(rule.Dependents)+1)
	if state.Shape().Size() > 0 { // state size is 0 for latent node types, at their initial state.
		updateInputs = append(updateInputs, state)
	}

	// Update dependents and calculate their convolved messages: it's a depth-first-search on dependents.
	for _, dependent := range rule.Dependents {
		dependentEdges, found := edges[rule.Name]
		if !found {
			Panicf("edges for rule %q not given in `edges`, edges given for rules: %q", rule.Name, slices.Keys(edges))
		}

		if lw.dependentsUpdateFirst {
			lw.recursivelyApplyGraphConvolution(ctx, dependent, graphStates, edges)
		}
		dependentState := graphStates[dependent.Name]
		convolveCtx := ctx.In(dependent.ConvKernelScopeName).In("conv")
		_ = convolveCtx
		if dependentState != nil {
			updateInputs = append(updateInputs,
				lw.sampledConvolveEdgeSet(convolveCtx, dependentState, dependentEdges.SourceIndices, dependentEdges.TargetIndices, numNodes))
		}
		if !lw.dependentsUpdateFirst {
			lw.recursivelyApplyGraphConvolution(ctx, dependent, graphStates, edges)
		}
	}
}

func (lw *LayerWiseConfig) sampledConvolveEdgeSet(ctx *context.Context, sourceState, edgesSource, edgesTarget *Node, numTargetNodes int) *Node {
	messages, _ := edgeMessageGraph(ctx.In("message"), sourceState, nil)
	return poolMessagesWithAdjacency(ctx, messages, edgesSource, edgesTarget, numTargetNodes, nil)
}

func (lw *LayerWiseConfig) LayerComputer(current map[string]tensor.Tensor, rule *sampler.Rule) {

}

type stateInfo struct {
	// Name is a combination of the round of the graph updates with either the name of the nodeset or the
	// edge set's kernel name.
	// E.g.: "1.papers" would be the state of the papers for round 1, "0.cites" would be
	// the cites convolution of the first message interation.
	// This is also the value used as key in the map of states.
	Name string

	// Exec is the accelerated program the computes either the update for the node state, if this refers to a node,
	// of the accelerated program that computes the convolution for an edge set.
	Exec context.Exec

	// State is the resulting state if it is calculated.
	// This is maintained while the result is needed, and freed after use, if [LayerWiseConfig.keepIntermediaryResults]
	// is false.
	State tensor.Tensor

	// Uses is the number of uses of this kernel in one graph update. Current is the number of times this has
	// been used. When Current reaches Count, and [LayerWiseConfig.keepIntermediaryResults] is false, the
	// [State] can be freed.
	Uses, Current int
}

// mapLayerWiseStateInfo creates the structure for the states of a layer-wise GNN evaluation.
func mapLayerWiseStateInfo(ctx *context.Context, strategy *sampler.Strategy) map[string]*stateInfo {
	states := make(map[string]*stateInfo)
	numGraphUpdates := context.GetParamOr(ctx, ParamNumGraphUpdates, 2)
	for round := range numGraphUpdates {
		ctxRound := ctxForGraphUpdateRound(ctx, round)
		for _, rule := range strategy.Seeds {
			recursivelyMapLayerWiseStateInfo(ctxRound, round, rule, states)
		}
	}
	return states
}

func recursivelyMapLayerWiseStateInfo(ctx *context.Context, round int, rule *sampler.Rule, states map[string]*stateInfo) {
	_ = ctx
	_ = round
	_ = rule
	_ = states
}

// computeLayerUpdate calculates the updated `targetState`, given the current state and
// states of the rule dependent nodes.
//
// The context `ctx` must be scoped into the message number (see [NodePrediction]), typically named `"graph_update_%d"`.
//
// The source states must be given in the same order as `rule.Dependents".
func recursivelyApplyLayeredGraphConvolution(ctx *context.Context, rule *sampler.Rule, currentStates map[string]*Node) (*tensor.Local, error) {
	if rule == nil || ctx == nil {
		return nil, errors.New("ctx and rule cannot be nil")
	}
	if rule.NodeSet != nil {
		return nil, errors.Errorf("ComputeLayerConvolution is only implemented to the full node set (all nodes), but rule includes a subset (NodeSet).")
	}
	countTarget, found := rule.Sampler.NodeTypesToCount[rule.NodeTypeName]
	if !found {
		return nil, errors.Errorf("rule %q node type %q is not known it sampler!?", rule, rule.NodeTypeName)
	}
	_ = countTarget
	return nil, nil
}

// layeredConvolveEdgeSet runs a convolution over an edge set, using the given state values for the source node set.
// This runs the convolution for one edge set: the target node set my have several incoming edge sets defined in its
// [sampler.Rule] -- it's defined in its dependants.
//
// This function should do the same as `sampledConvolveEdgeSet`, but layer-wise instead. They must be aligned.
func layeredConvolveEdgeSet(ctx *context.Context, dependantRule *sampler.Rule, sourceState *Node) *Node {
	if dependantRule.EdgeType == nil {
		Panicf("can only run edge convolution on edge type rules, got instead %s", dependantRule)
	}
	convolveCtx := ctx.In(dependantRule.ConvKernelScopeName).In("conv")
	edgeState := gatherToEdgesGraph(convolveCtx, sourceState, dependantRule.EdgeType)
	targetState := poolEdgesGraph(convolveCtx, edgeState, dependantRule.EdgeType)
	return targetState
}

var (
	SamplerStrategyScope = context.RootScope + "sampler_strategy"
	EdgeIndicesScope     = strings.Join([]string{SamplerStrategyScope, "edge_indices"}, context.ScopeSeparator)
)

// gatherToEdges takes the current state of a node set, and gather its state to each of the edges.
//
// Notice that the direction of sampling is the opposite of the direction of the convolution: if we sample from `A`
// to `B`, we want to convolve from `B` to `A` (generally).
//
// It stores the edge type indices tensors in the context, in scope "/sampler_strategy/edge_indices/"
func gatherToEdgesGraph(ctx *context.Context, state *Node, edgeType *sampler.EdgeType) (newState *Node) {
	// Create target (for sampling) indices, if not yet there.
	ctxEdgeIndices := ctx.InAbsPath(EdgeIndicesScope).Checked(false)
	numEdges := len(edgeType.EdgeTargets)

	sourceVarName := edgeType.Name + ".SourceIndices"
	sourceVar := ctxEdgeIndices.InspectVariableInScope(sourceVarName)
	if sourceVar == nil {
		sourceT := tensor.FromFlatDataAndDimensions(edgeType.EdgeTargets, numEdges, 1)
		sourceVar = ctxEdgeIndices.VariableWithValue(sourceVarName, sourceT)
	}

	// The target indices for sampling are the source for the convolution (directions are reversed), so we gather
	// from them.
	indices := sourceVar.ValueGraph(state.Graph())
	return Gather(state, indices)
}

// poolEdgesGraph takes the computed messages for each edge and poll (by scattering) them
// to the corresponding node in a current state of a node set, and gather its state to each of the edges.
//
// Notice that the direction of sampling is the opposite of the direction of the convolution: if we sample from `A`
// to `B`, we want to convolve from `B` to `A` (generally).
//
// It stores the edge type indices tensors in the context, in scope "/sampler_strategy/edge_indices/"
func poolEdgesGraph(ctx *context.Context, state *Node, edgeType *sampler.EdgeType) (newState *Node) {
	// Create target (for sampling) indices, if not yet there.
	ctxEdgeIndices := ctx.InAbsPath(EdgeIndicesScope).Checked(false)
	numEdges := edgeType.NumEdges()
	numTargetNodes := edgeType.NumTargetNodes()
	embeddingSize := state.Shape().Dimensions[state.Rank()-1]

	// Target indices for each edge, shape `Int32[numEdges, 1]`
	targetVarName := edgeType.Name + ".TargetIndices"
	targetVar := ctxEdgeIndices.InspectVariableInScope(targetVarName)
	if targetVar == nil {
		targetIndices := targetIndicesForEdgeType(edgeType)
		targetT := tensor.FromFlatDataAndDimensions(targetIndices, numEdges, 1)
		targetVar = ctxEdgeIndices.VariableWithValue(targetVarName, targetT)
	}
	targetIndices := targetVar.ValueGraph(state.Graph())

	// Find pool types.
	poolTypes := context.GetParamOr(ctx, ParamPoolingType, "mean|sum")
	poolTypesList := strings.Split(poolTypes, "|")
	parts := make([]*Node, len(poolTypesList))
	for ii, poolType := range poolTypesList {
		switch poolType {
		case "sum":
			parts[ii] = Scatter(targetIndices, state, shapes.Make(state.DType(), numTargetNodes, embeddingSize))
		case "mean":
			sum := Scatter(targetIndices, state, shapes.Make(state.DType(), numTargetNodes, embeddingSize))
			denominator := Scatter(targetIndices,
				Ones(state.Graph(), shapes.Make(state.DType(), numEdges, 1)),
				shapes.Make(state.DType(), numTargetNodes, 1))
			denominator = MaxScalar(denominator, 1.0)
			parts[ii] = Div(sum, denominator)
		case "max":
			Panicf("Pool type %q not supported for layer-wise GNN: ScatterMax is not implemented yet.", poolType)
		default:
			Panicf("unknown graph convolution pooling type (%q) given in context: value given %q (of %q) -- valid values are sum or mean, or a combination of them separated by '|'",
				ParamPoolingType, poolType, poolTypes)
		}
	}

	// Concatenate the different pool methods: `Concatenate` if there is only one part is a no-op.
	return Concatenate(parts, -1)
}

// targetIndicesForEdgeType returns a `[]int32` slice with the indices of the target node
// (for the convolution) of th edges.
func targetIndicesForEdgeType(edgeType *sampler.EdgeType) []int32 {
	numEdges := len(edgeType.EdgeTargets)
	numTargetNodes := len(edgeType.Starts)
	targetIndices := make([]int32, numEdges)
	currentTargetIdx := int32(0)
	startsIdx := 0
	for ii := range targetIndices {
		// Skip to the next target node that has edges pointing to it.
		for startsIdx < numTargetNodes && int(edgeType.Starts[startsIdx]) <= ii {
			currentTargetIdx++
			startsIdx++
		}
		targetIndices[ii] = currentTargetIdx
	}
	return targetIndices
}

func FastEval(ds train.Dataset, strategy *sampler.Strategy) {
}
