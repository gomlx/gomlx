// Package gnn implements a generic GNN modeling based on [TF-GNN MtAlbis].
//
// [TF-GNN MtAlbis]: https://github.com/tensorflow/gnn/tree/main/tensorflow_gnn/models/mt_albis
package gnn

import (
	"fmt"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/types/slices"
	"strings"
)

var (
	// ParamNumGraphUpdates is the context parameter that defines the number of messages to
	// send in the GNN tree of nodes.
	// The default is 2.
	ParamNumGraphUpdates = "gnn_num_messages"

	// ParamReadoutHiddenLayers context parameter that defines the number or hidden layers
	// connected to the readout of a GNN model.
	// Default to 0.
	ParamReadoutHiddenLayers = "gnn_readout_hidden_layers"

	// ParamMessageDim context hyperparameter defines the dimension of the messages calculated per node.
	// The default is 128.
	ParamMessageDim = "gnn_message_dim"

	// ParamStateDim context hyperparameter defines the dimension of the updated hidden states per node.
	// The default is 128.
	ParamStateDim = "gnn_node_state_dim"

	// ParamActivation context hyperparameter defines the activation to use, see [layers.ParamActivation]
	ParamActivation = layers.ParamActivation

	// ParamDropoutRate is applied to FNN kernels, as usual, randomly turning off individual values.
	// Default is 0.0, meaning no dropout.
	ParamDropoutRate = layers.ParamDropoutRate

	// ParamEdgeDropoutRate is applied to full edges, disabling the whole edge.
	// Default is 0.0, meaning no edge dropout.
	ParamEdgeDropoutRate = "gnn_edge_dropout_rate"

	// ParamL2Regularization is an alias for layers.ParamL2Regularization, and adds regularizations to the NN kernels.
	// The default is `0.0`.
	ParamL2Regularization = layers.ParamL2Regularization

	// ParamPoolingType context hyperparameter defines how incoming messages in the graph convolutions are pooled (that is, reduced
	// or aggregated).
	// It can take values `mean`, `sum` or `max` or a combination of them separated by `|`.
	// The default is `mean|sum`.
	ParamPoolingType = "gnn_pooling_type"

	// ParamNormalizationType context hyperparameter can take values `none` ( or `""`), `batch` or `layer`.
	// It's an alias to [layers.ParamNormalizationType].
	// The default is `layer`.
	ParamNormalizationType = layers.ParamNormalizationType

	// ParamUpdateStateType context hyperparameter can take values `residual` or `none`.
	// The default is `residual`.
	ParamUpdateStateType = "gnn_update_state_type"

	// ParamUsePathToRootStates context hyperparameter that if set allows each update state to see the states
	// of all nodes in its path to root.
	// Default is false.
	ParamUsePathToRootStates = "gnn_use_path_to_root"

	// ParamGraphUpdateType context hyperparameter can take values `tree` or `simultaneous`.
	// Graph updates in `tree` fashion will update from leaf all the way to the seeds (the roots of the trees),
	// for each message configured with [ParamNumGraphUpdates].
	// Graph updates in `simultaneous` fashion will update all states from it's dependents "simultaneously". In that
	// sense it will require [ParamNumGraphUpdates] to be at least equal to the depth of the sampling tree for the
	// influence of the leaf nodes to reach to the root nodes.
	// The default is `tree`.
	ParamGraphUpdateType = "gnn_graph_update_type"
)

// NodePrediction performs graph convolutions from leaf nodes to the seeds (the roots of the trees), this
// is called a "graph update".
//
// This process is repeated [ParamNumGraphUpdates] times (parameter set in `ctx` with key [ParamNumGraphUpdates]).
// After that the state of the seed nodes go through [ParamReadoutHiddenLayers] hidden layers,
// and these seed states (updated in `graphStates`) can be read out and converted to whatever is the output to match the
// task.
//
// The `strategy` describes which convolutions and their order.
//
// There are several hyperparameters that control the GNN model. They can be set as parameters
// in the context. If scoped in specific [Rule.KernelScopeName] (rules of the `strategy`), they
// can be different for different node sets (so different node sets can have different state
// dimensions, for instance). See variables `Param...`
//
// Example of a `ModelGraph` function, that describes a model:
//
//	func MyGnnModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
//		g := inputs[0].Graph()
//		optimizers.CosineAnnealingSchedule(ctx, g, shapes.F32)
//		ctx = ctx.WithInitializer(initializers.GlorotUniformFn(initializers.NoSeed))
//		strategy := spec.(*sampler.Strategy)
//		graphStates := MyFeaturePreprocessing(ctx, strategy, inputs)
//		NodePrediction(ctx, strategy, graphStates)
//		readoutState = graphStates["my_seed_nodes"]
//		logits := layers.DenseWithBias(ctx.In("readout"), readoutState.Value, numClasses)
//		return []*Node{logits}
//	}
func NodePrediction(ctx *context.Context, strategy *sampler.Strategy, graphStates map[string]*sampler.ValueMask[*Node]) {
	numGraphUpdates := context.GetParamOr(ctx, ParamNumGraphUpdates, 2)
	graphUpdateType := context.GetParamOr(ctx, ParamGraphUpdateType, "tree")
	for round := range numGraphUpdates {
		switch graphUpdateType {
		case "tree":
			TreeGraphStateUpdate(ctxForGraphUpdateRound(ctx, round), strategy, graphStates)
		case "simultaneous":
			SimultaneousGraphStateUpdate(ctxForGraphUpdateRound(ctx, round), strategy, graphStates)
		default:
			Panicf("invalid value for %q: valid values are \"tree\" or \"simultaneous\"", ParamGraphUpdateType)
		}
	}
	numHiddenLayers := context.GetParamOr(ctx, ParamReadoutHiddenLayers, 0)
	for _, rule := range strategy.Seeds {
		seedState := graphStates[rule.Name]
		for ii := range numHiddenLayers {
			ctxReadout := ctx.In(rule.KernelScopeName).In(fmt.Sprintf("readout_hidden_%d", ii))
			seedState.Value = updateState(ctxReadout, seedState.Value, seedState.Value, seedState.Mask)
		}
	}
}

// ctxForGraphUpdateRound returns the context with scope for the given round of graph update.
func ctxForGraphUpdateRound(ctx *context.Context, n int) *context.Context {
	return ctx.In(fmt.Sprintf("graph_update_%d", n))
}

// TreeGraphStateUpdate takes a `graphStates`, a map of name of node sets to their hidden states,
// and updates them by running graph convolutions on the reverse direction of the sampling
// rules in `strategy`, that is, from leaves back to the root of the trees -- trees rooted on the seed rules.
//
// All hyperparameters are read from the context `ctx`, and can be set to individual node sets,
// by setting them in the scope `"gnn:"+rule.Name`, where `rule.Name` is the name of the corresponding
// rule in the given `strategy` object.
//
// The object `strategy` defines the tree of node sets by its rules, and the convolutions
// run from the leaf nodes towards the root nodes (seeds).
//
// It updates all states in `graphStates` except the leaves. The masks are left unchanged.
func TreeGraphStateUpdate(ctx *context.Context, strategy *sampler.Strategy, graphStates map[string]*sampler.ValueMask[*Node]) {
	// Starting from the seed node sets, do updates recursively.
	for _, rule := range strategy.Seeds {
		recursivelyApplyGraphConvolution(ctx, rule, nil, graphStates, true)
	}
}

// SimultaneousGraphStateUpdate executes one step of state update on all node sets of the graph "simultaneously".
// If the graph has a tree like structure, one needs to call this function at least `N` times, where `N` is the depth
// of the tree, until the signal arrives from the leaf node to the root.
func SimultaneousGraphStateUpdate(ctx *context.Context, strategy *sampler.Strategy, graphStates map[string]*sampler.ValueMask[*Node]) {
	// Starting from the seed node sets, do updates recursively.
	for _, rule := range strategy.Seeds {
		recursivelyApplyGraphConvolution(ctx, rule, nil, graphStates, false)
	}
}

func recursivelyApplyGraphConvolution(ctx *context.Context, rule *sampler.Rule,
	pathToRootStates []*Node,
	graphStates map[string]*sampler.ValueMask[*Node],
	dependentsUpdateFirst bool) {
	if rule.Name == "" || rule.KernelScopeName == "" {
		Panicf("strategy's rule name=%q or kernel scope name=%q are empty, they both must be defined",
			rule.Name, rule.KernelScopeName)
	}

	// Makes sure there is a state for the current dependent.
	state, found := graphStates[rule.Name]
	if !found {
		Panicf("state for node %q not given in `graphStates`, states given: %v", rule.Name, slices.Keys(graphStates))
	}

	// Leaf nodes are not updated.
	if len(rule.Dependents) == 0 {
		return
	}

	var subPathToRootStates []*Node
	if context.GetParamOr(ctx, ParamUsePathToRootStates, false) {
		// subPathToRootStates: passed to the children rules. They need to be expanded at each level to get the correct
		// broadcasting (the broadcasting to the right shapes will happen automatically).
		subPathToRootStates = make([]*Node, 0, len(pathToRootStates)+1)
		for _, contextState := range pathToRootStates {
			// We need to expand, so it will be properly broadcast.
			// Noted the new axis is in between the "BatchSize" and following axes, and the last embedding
			// dimensions, which remains unchanged.
			subPathToRootStates = append(subPathToRootStates, ExpandDims(contextState, -2))
		}
		if state.Value != nil {
			// If the state of the current rule is not latent, include it as well.
			subPathToRootStates = append(subPathToRootStates, ExpandDims(state.Value, -2))
		}
	}

	// Collections of inputs to be updated to update current hidden state.
	var hasNewUpdateInputs bool
	updateInputs := make([]*Node, 0, len(rule.Dependents)+1+len(pathToRootStates))
	if state.Value != nil { // state == nil for latent node types, at their initial state.
		updateInputs = append(updateInputs, state.Value)
	}
	for _, contextState := range pathToRootStates {
		// Broadcast dimensions where needed.
		dims := make([]int, 0, rule.Shape.Rank()+1)
		dims = append(dims, rule.Shape.Dimensions...)
		dims = append(dims, contextState.Shape().Dimensions[contextState.Rank()-1])
		contextState = BroadcastToDims(contextState, dims...)
		updateInputs = append(updateInputs, contextState)
		hasNewUpdateInputs = true
	}

	// Update dependents and calculate their convolved messages: it's a depth-first-search on dependents.
	for _, dependent := range rule.Dependents {
		if dependentsUpdateFirst {
			recursivelyApplyGraphConvolution(ctx, dependent, subPathToRootStates, graphStates, dependentsUpdateFirst)
		}
		dependentState := graphStates[dependent.Name]
		dependentDegreePair := graphStates[sampler.NameForNodeDependentDegree(rule.Name, dependent.Name)]
		var dependentDegree *Node
		if dependentDegreePair != nil {
			dependentDegree = dependentDegreePair.Value
		}
		convolveCtx := ctx.In(dependent.KernelScopeName).In("conv")
		if dependentState.Value != nil {
			updateInputs = append(updateInputs, sampledConvolveEdgeSet(convolveCtx, dependentState.Value, dependentState.Mask, dependentDegree))
			hasNewUpdateInputs = true
		}
		if !dependentsUpdateFirst {
			recursivelyApplyGraphConvolution(ctx, dependent, subPathToRootStates, graphStates, dependentsUpdateFirst)
		}
	}

	// Update state of current rule: only update state if there was any new incoming input.
	if hasNewUpdateInputs {
		updateCtx := ctx.In(rule.KernelScopeName).In("update")
		state.Value = updateState(updateCtx, state.Value, Concatenate(updateInputs, -1), state.Mask)
	}
}

// sampledConvolveEdgeSet creates messages from a node set and aggregates them to the same prefix dimension of their target
// node sets. This runs a convolution over one of the edge sets that connects a Rule to its `SourceRule`.
//
// The context `ctx` must already have been properly scoped.
//
// This function should do the same as `layeredConvolveEdgeSet`, this function does it for sampled graphs.
// They must be aligned.
func sampledConvolveEdgeSet(ctx *context.Context, value, mask, degree *Node) *Node {
	messages, mask := edgeMessageGraph(ctx, value, mask)
	return poolMessages(ctx, messages, mask, degree)
}

// edgeMessageGraph calculates the graph for messages being sent across edges.
// It takes as input the source node state already gathered for the edge: their shape should
// look like: `[batch_size, ..., num_edges, source_node_state_dim]`.
func edgeMessageGraph(ctx *context.Context, gatheredStates, gatheredMask *Node) (messages, mask *Node) {
	messageDim := context.GetParamOr(ctx, ParamMessageDim, 128)
	messages = layers.DenseWithBias(ctx.In("message"), gatheredStates, messageDim)
	messages = layers.ActivationFromContext(ctx, messages)

	mask = gatheredMask
	edgeDropOutRate := context.GetParamOr(ctx, ParamEdgeDropoutRate, 0.0)
	if edgeDropOutRate > 0 {
		// We apply edge dropout to the mask: values disabled here will mask the whole edge.
		mask = layers.DropoutStatic(ctx, gatheredMask, edgeDropOutRate)
	}
	return
}

// poolMessages will pool according to [ParamPoolingType].
// Let's say `value` is shaped `[d_0, d_1, ..., d_{n-1}, d_{n}, e]`: we assume `e` is the embedding dimension of the
// tensor, and we want to reduce the axis `n`.
// So the returned shape will be `[d_0, d_1, ..., d_{n-1}, k*e]`, where `k` is the number of pooling types configured.
// E.g.: If the pooling types (see [ParamPoolingType]) are configured to `mean|sum`, then `k=2`.
//
// The parameter `degree` is optional, but if given, the `sum` pooling will scale the sum to the given degree.
// It's expected to be of shape `[d_0, d_1, ..., d_{n-1}, 1]`.
func poolMessages(ctx *context.Context, value, mask, degree *Node) *Node {
	poolTypes := context.GetParamOr(ctx, ParamPoolingType, "mean|sum")
	poolTypesList := strings.Split(poolTypes, "|")
	parts := make([]*Node, 0, len(poolTypesList))
	var pooled *Node
	for _, poolType := range poolTypesList {
		reduceAxis := value.Rank() - 2
		switch poolType {
		case "sum", "logsum":
			if degree == nil {
				pooled = MaskedReduceSum(value, mask, reduceAxis)
			} else {
				// Sum pondered by degree, that is, `mean(value)*degree`.
				pooled = MaskedReduceMean(value, mask, reduceAxis)
				pooled = Mul(pooled, ConvertType(degree, pooled.DType()))
			}
			if poolType == "logsum" {
				pooled = MirroredLog1p(pooled)
			}
		case "mean":
			pooled = MaskedReduceMean(value, mask, reduceAxis)
		case "max":
			pooled = MaskedReduceMax(value, mask, reduceAxis)
			// Makes it 0 where every element is masked out.
			pooledMask := ReduceMax(mask, -1)
			pooled = Where(pooledMask, pooled, ZerosLike(pooled))
		default:
			Panicf("unknown graph convolution pooling type (%q) given in context: value given %q (of %q) -- valid values are sum, mean and max, or a combination of them separated by '|'",
				ParamPoolingType, poolType, poolTypes)
		}
		parts = append(parts, pooled)
	}
	if len(parts) == 1 {
		return parts[0]
	}
	return Concatenate(parts, -1)
}

// updateState of a node set, given the `input` (should be a concatenation of previous
// state and all pooled messages) and its `mask`.
func updateState(ctx *context.Context, prevState, input, mask *Node) *Node {
	updateType := context.GetParamOr(ctx, ParamUpdateStateType, "residual")
	if updateType != "residual" && updateType != "none" {
		Panicf("invalid GNN update type %q (given by parameter %q) -- valid values are 'residual' and 'none'",
			updateType, ParamUpdateStateType)
	}

	// Inputs: both previous state and pooled messages passes through a dropout first.
	input = layers.DropoutFromContext(ctx, input)
	stateDim := context.GetParamOr(ctx, ParamStateDim, 128)
	state := layers.DenseWithBias(ctx.In("message"), input, stateDim)
	state = layers.ActivationFromContext(ctx, state)
	state = layers.DropoutFromContext(ctx, state)

	if updateType == "residual" && prevState.Shape().Eq(state.Shape()) {
		state = Add(state, prevState)
	}
	//state = layers.MaskedNormalizeFromContext(ctxRuleSpecific, state, mask)
	state = layers.MaskedNormalizeFromContext(ctx.In("normalization"), state, mask)
	return state
}
