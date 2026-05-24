// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package gnn implements a generic GNN modeling based on [TF-GNN MtAlbis].
//
// [TF-GNN MtAlbis]: https://github.com/tensorflow/gnn/tree/main/tensorflow_gnn/models/mt_albis
package gnn

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/nanlogger"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/layers/kan"
	"github.com/gomlx/gomlx/ml/model"
	. "github.com/gomlx/gomlx/support/exceptions"
)

var (
	// ParamNumGraphUpdates is the scope parameter that defines the number of messages to
	// send in the GNN tree of nodes.
	// The default is 2.
	ParamNumGraphUpdates = "gnn_num_messages"

	// ParamMessageDim scope hyperparameter defines the dimension of the messages calculated per node.
	// The default is 128.
	ParamMessageDim = "gnn_message_dim"

	// ParamStateDim scope hyperparameter defines the dimension of the updated hidden states per node.
	// The default is 128.
	ParamStateDim = "gnn_node_state_dim"

	// ParamEdgeDropoutRate is applied to full edges, disabling the whole edge.
	// Default is 0.0, meaning no edge dropout.
	ParamEdgeDropoutRate = "gnn_edge_dropout_rate"

	// ParamPoolingType scope hyperparameter defines how incoming messages in the graph convolutions are pooled (that is, reduced
	// or aggregated).
	// It can take values `mean`, `sum` or `max` or a combination of them separated by `|`.
	// The default is `mean|sum`.
	ParamPoolingType = "gnn_pooling_type"

	// ParamUpdateStateType scope hyperparameter can take values `residual` or `none`.
	// The default is `residual`.
	ParamUpdateStateType = "gnn_update_state_type"

	// ParamUpdateNumHiddenLayers scope hyperparameter that defines the number of hidden layers for the update kernel.
	ParamUpdateNumHiddenLayers = "gnn_update_num_hidden_layers"

	// ParamUsePathToRootStates scope hyperparameter that if set allows each update state to see the states
	// of all nodes in its path to root.
	// Default is false.
	ParamUsePathToRootStates = "gnn_use_path_to_root"

	// ParamUseRootAsScope scope hyperparameter that if set uses the root state as a scope state.
	ParamUseRootAsScope = "gnn_use_root_as_scope"

	// ParamGraphUpdateType scope hyperparameter can take values `tree` or `simultaneous`.
	// Graph updates in `tree` fashion will update from leaf all the way to the seeds (the roots of the trees),
	// for each message configured with [ParamNumGraphUpdates].
	// Graph updates in `simultaneous` fashion will update all states from its dependents "simultaneously". In that
	// sense it will require [ParamNumGraphUpdates] to be at least equal to the depth of the sampling tree for the
	// influence of the leaf nodes to reach to the root nodes.
	// The default is `tree`.
	ParamGraphUpdateType = "gnn_graph_update_type"

	// ParamNoKanForLayers is a list of layers (comma separated) for which it should not use KAN layers.
	ParamNoKanForLayers = "gnn_no_kan"
)

// NanLogger is used if not nil.
var NanLogger *nanlogger.NanLogger

// NodePrediction performs graph convolutions from leaf nodes to the seeds (the roots of the trees), this
// is called a "graph update".
//
// This process is repeated [ParamNumGraphUpdates] times (parameter set in `scope` with key [ParamNumGraphUpdates]).
// After that the state of the seed nodes go through [ParamReadoutHiddenLayers] hidden layers,
// and these seed states (updated in `graphStates`) can be read out and converted to whatever is the output to match the
// task.
//
// The `strategy` describes which convolutions and their order.
//
// There are several hyperparameters that control the GNN model. They can be set as parameters
// in the model. If scoped in specific [Rule.ConvKernelScopeName] (rules of the `strategy`), they
// can be different for different node sets (so different node sets can have different state
// dimensions, for instance). See variables `Param...`
//
// Example of a `ModelGraph` function, that describes a model:
//
//	func MyGnnModelGraph(scope *model.Scope, spec any, inputs []*Node) []*Node {
//		g := inputs[0].Graph()
//		optimizer.CosineAnnealingSchedule(scope, g, dtypes.Float32)
//		scope = scope.WithInitializer(initializer.GlorotUniformFn(initializer.NoSeed))
//		strategy := spec.(*sampler.Strategy)
//		graphStates := MyFeaturePreprocessing(scope, strategy, inputs)
//		NodePrediction(scope, strategy, graphStates)
//		readoutState = graphStates["my_seed_nodes"]
//		logits := layers.DenseWithBias(scope.In("readout"), readoutState.Value, numClasses)
//		return []*Node{logits}
//	}
func NodePrediction(scope *model.Scope, strategy *sampler.Strategy, graphStates map[string]*sampler.ValueMask[*Node]) {
	numGraphUpdates := model.GetParamOr(scope, ParamNumGraphUpdates, 2)
	graphUpdateType := model.GetParamOr(scope, ParamGraphUpdateType, "tree")
	for round := range numGraphUpdates {
		switch graphUpdateType {
		case "tree":
			TreeGraphStateUpdate(scopeForGraphUpdateRound(scope, round), strategy, graphStates)
		case "simultaneous":
			SimultaneousGraphStateUpdate(scopeForGraphUpdateRound(scope, round), strategy, graphStates)
		default:
			Panicf("invalid value for %q: valid values are \"tree\" or \"simultaneous\"", ParamGraphUpdateType)
		}
	}
	scopeReadout := scope.In("readout")
	visited := make(map[string]bool)
	for _, rule := range strategy.Seeds {
		seedState := graphStates[rule.Name]
		kernelScopeName := fmt.Sprintf("%s", rule.ConvKernelScopeName)
		var ruleScope *model.Scope
		if visited[kernelScopeName] {
			ruleScope = scopeReadout.Shared("%s", kernelScopeName)
		} else {
			visited[kernelScopeName] = true
			ruleScope = scopeReadout.In("%s", kernelScopeName)
		}
		seedState.Value = updateState(ruleScope, seedState.Value, seedState.Value, seedState.Mask)
	}
}

// scopeForGraphUpdateRound returns the scope with scope for the given round of graph update.
func scopeForGraphUpdateRound(scope *model.Scope, n int) *model.Scope {
	return scope.In("graph_update_%d", n)
}

// TreeGraphStateUpdate takes a `graphStates`, a map of name of node sets to their hidden states,
// and updates them by running graph convolutions on the reverse direction of the sampling
// rules in `strategy`, that is, from leaves back to the root of the trees -- trees rooted on the seed rules.
//
// All hyperparameters are read from the scope `scope`, and can be set to individual node sets,
// by setting them in the scope `"gnn:"+rule.Name`, where `rule.Name` is the name of the corresponding
// rule in the given `strategy` object.
//
// The object `strategy` defines the tree of node sets by its rules, and the convolutions
// run from the leaf nodes towards the root nodes (seeds).
//
// It updates all states in `graphStates` except the leaves. The masks are left unchanged.
func TreeGraphStateUpdate(scope *model.Scope, strategy *sampler.Strategy, graphStates map[string]*sampler.ValueMask[*Node]) {
	visited := make(map[string]bool)
	// Starting from the seed node sets, do updates recursively.
	for _, rule := range strategy.Seeds {
		recursivelyApplyGraphConvolution(scope, rule, nil, graphStates, true, visited)
	}
}

// SimultaneousGraphStateUpdate executes one step of state update on all node sets of the graph "simultaneously".
// If the graph has a tree like structure, one needs to call this function at least `N` times, where `N` is the depth
// of the tree, until the signal arrives from the leaf node to the root.
func SimultaneousGraphStateUpdate(scope *model.Scope, strategy *sampler.Strategy, graphStates map[string]*sampler.ValueMask[*Node]) {
	visited := make(map[string]bool)
	// Starting from the seed node sets, do updates recursively.
	for _, rule := range strategy.Seeds {
		recursivelyApplyGraphConvolution(scope, rule, nil, graphStates, false, visited)
	}
}

func recursivelyApplyGraphConvolution(scope *model.Scope, rule *sampler.Rule,
	pathToRootStates []*Node,
	graphStates map[string]*sampler.ValueMask[*Node], dependentsUpdateFirst bool, visited map[string]bool) {
	if rule.Name == "" || rule.ConvKernelScopeName == "" {
		Panicf("strategy's rule name=%q or kernel scope name=%q are empty, they both must be defined",
			rule.Name, rule.ConvKernelScopeName)
	}

	// Makes sure there is a state for the current dependent.
	state, found := graphStates[rule.Name]
	if !found {
		Panicf("state for sampling rule %q not given in `graphStates`, states given: %v", rule.Name, xslices.Keys(graphStates))
	}

	// Leaf nodes are not updated.
	if len(rule.Dependents) == 0 {
		return
	}

	var subPathToRootStates []*Node
	useRootAsScope := model.GetParamOr(scope, ParamUseRootAsScope, false)
	if model.GetParamOr(scope, ParamUsePathToRootStates, false) || useRootAsScope {
		// subPathToRootStates: passed to the children rules. They need to be expanded at each level to get the correct
		// broadcasting (the broadcasting to the right shapes will happen automatically).
		subPathToRootStates = make([]*Node, 0, len(pathToRootStates)+1)
		for _, scopeState := range pathToRootStates {
			// We need to expand, so it will be properly broadcast.
			// Noted the new axis is in between the "BatchSize" and following axes, and the last embedding
			// dimensions, which remains unchanged.
			subPathToRootStates = append(subPathToRootStates, InsertAxes(scopeState, -2))
		}
		if state.Value != nil {
			// If useRootAsScope, only takes the root state as model.
			if len(subPathToRootStates) == 0 || !useRootAsScope {
				// If the state of the current rule is not latent, include it as well.
				newScopeState := InsertAxes(state.Value, -2)
				subPathToRootStates = append(subPathToRootStates, newScopeState)
			}
		}
	}

	// Collections of inputs to be updated to update current hidden state.
	var hasNewUpdateInputs bool
	updateInputs := make([]*Node, 0, len(rule.Dependents)+1+len(pathToRootStates))
	if state.Value != nil { // state == nil for latent node types, at their initial state.
		updateInputs = append(updateInputs, state.Value)
	}
	for _, scopeState := range pathToRootStates {
		// Broadcast dimensions where needed.
		dims := make([]int, 0, rule.Shape.Rank()+1)
		dims = append(dims, rule.Shape.Dimensions...)
		dims = append(dims, scopeState.Shape().Dimensions[scopeState.Rank()-1])
		scopeState = BroadcastToDims(scopeState, dims...)
		updateInputs = append(updateInputs, scopeState)
		hasNewUpdateInputs = true
	}

	// Update dependents and calculate their convolved messages: it's a depth-first-search on dependents.
	for _, dependent := range rule.Dependents {
		if dependentsUpdateFirst {
			recursivelyApplyGraphConvolution(scope, dependent, subPathToRootStates, graphStates, dependentsUpdateFirst, visited)
		}
		dependentState, found := graphStates[dependent.Name]
		if !found {
			Panicf("state for sampling rule %q not given in `graphStates`, states given: %v", dependent.Name, xslices.Keys(graphStates))
		}
		dependentDegreePair := graphStates[sampler.NameForNodeDependentDegree(rule.Name, dependent.Name)]
		var dependentDegree *Node
		if dependentDegreePair != nil {
			dependentDegree = dependentDegreePair.Value
		}
		dependentConvKernelScopeName := fmt.Sprintf("%s", dependent.ConvKernelScopeName)
		var convolveScope *model.Scope
		if visited[dependentConvKernelScopeName] {
			convolveScope = scope.Shared("%s", dependentConvKernelScopeName).In("conv")
		} else {
			visited[dependentConvKernelScopeName] = true
			convolveScope = scope.In("%s", dependentConvKernelScopeName).In("conv")
		}
		if dependentState.Value != nil {
			updateInputs = append(updateInputs, sampledConvolveEdgeSet(convolveScope, dependentState.Value, dependentState.Mask, dependentDegree))
			hasNewUpdateInputs = true
		}
		if !dependentsUpdateFirst {
			recursivelyApplyGraphConvolution(scope, dependent, subPathToRootStates, graphStates, dependentsUpdateFirst, visited)
		}
	}

	// Update state of current rule: only update state if there was any new incoming input.
	if hasNewUpdateInputs {
		ruleUpdateKernelScopeName := fmt.Sprintf("%s", rule.UpdateKernelScopeName)
		var updateScope *model.Scope
		if visited[ruleUpdateKernelScopeName] {
			updateScope = scope.Shared("%s", ruleUpdateKernelScopeName).In("update")
		} else {
			visited[ruleUpdateKernelScopeName] = true
			updateScope = scope.In("%s", ruleUpdateKernelScopeName).In("update")
		}
		state.Value = updateState(updateScope, state.Value, Concatenate(updateInputs, -1), state.Mask)
	}
}

// sampledConvolveEdgeSet creates messages from a node set and aggregates them to the same prefix dimension of their target
// node sets. This runs a convolution over one of the edge sets that connects a Rule to its `SourceRule`.
//
// The scope `scope` must already have been properly scoped.
//
// This function should do the same as `layeredConvolveEdgeSet`, this function does it for sampled graphs.
// They must be aligned.
func sampledConvolveEdgeSet(scope *model.Scope, value, mask, degree *Node) *Node {
	messages, mask := edgeMessageGraph(scope.In("message"), value, mask)
	return poolMessagesWithFixedShape(scope, messages, mask, degree)
}

// edgeMessageGraph calculates the graph for messages being sent across edges.
// It takes as input the source node state already gathered for the edge: their shape should
// look like: `[batch_size, ..., num_edges, source_node_state_dim]`.
func edgeMessageGraph(scope *model.Scope, gatheredStates, gatheredMask *Node) (messages, mask *Node) {
	messageDim := model.GetParamOr(scope, ParamMessageDim, 128)

	useKan := model.GetParamOr(scope, "kan", false)
	if useKan {
		// KAN
		messages = kan.New(scope, gatheredStates, messageDim).NumHiddenLayers(0, messageDim).Done()

	} else {
		// Normal FNN
		messages = layers.DenseWithBias(scope, gatheredStates, messageDim)
		messages = activation.ApplyFromScope(scope, messages)
	}
	if NanLogger != nil {
		NanLogger.TraceFirstNaN(messages, fmt.Sprintf("(KAN)edgeMessageGraph(%s)", scope.Scope()))
	}

	mask = gatheredMask
	if mask != nil {
		edgeDropOutRate := model.GetParamOr(scope, ParamEdgeDropoutRate, 0.0)
		if edgeDropOutRate > 0 {
			// We apply edge dropout to the mask: values disabled here will mask the whole edge.
			mask = layers.DropoutStatic(scope, gatheredMask, edgeDropOutRate)
		}
	}
	return
}

// poolMessagesWithFixedShape will pool according to [ParamPoolingType].
//
// Let's say `value` is shaped `[d_0, d_1, ..., d_{n-1}, d_{n}, e]`: we assume `e` is the embedding dimension of the
// tensor, and we want to reduce the axis `n`.
// So the returned shape will be `[d_0, d_1, ..., d_{n-1}, k*e]`, where `k` is the number of pooling types configured.
// E.g.: If the pooling types (see [ParamPoolingType]) are configured to `mean|sum`, then `k=2`.
//
// The parameter `degree` is optional, but if given, the `sum` and `logsum` pooling will scale the sum to the given degree.
// It's expected to be of shape `[d_0, d_1, ..., d_{n-1}, 1]`.
//
// There are no training variables in this. The `scope` is only used for the hyperparameter configuration.
func poolMessagesWithFixedShape(scope *model.Scope, value, mask, degree *Node) *Node {
	poolTypes := model.GetParamOr(scope, ParamPoolingType, "mean|sum")
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
				pooled = Mul(pooled, ConvertDType(degree, pooled.DType()))
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
			Panicf("unknown graph convolution pooling type (%q) given in scope: value given %q (of %q) -- valid values are sum, mean and max, or a combination of them separated by '|'",
				ParamPoolingType, poolType, poolTypes)
		}
		parts = append(parts, pooled)
	}
	if len(parts) == 1 {
		return parts[0]
	}
	return Concatenate(parts, -1)
}

// poolMessagesWithAdjacency will pool according to [ParamPoolingType] using the adjacency information.
//
// Args:
//   - [scope] is the scope, used to read the [ParamPoolingType] hyperparameter, which defines the type(s) of
//     pooling to be used.
//   - [source] should be shaped `[num_source_nodes, embedding_size]` and have the `dtype` of the model.
//   - [edgesSource] and [edgesTarget] represent the adjacency: source and target indices of the connected nodes.
//     They must have the same shape: both shaped either `[num_edges]` or `[num_edges,1]` and should have some integer dtype.
//   - [targetDim] is the dimension of the first axis of the resulting target tensor.
//   - [degree] is optional. If given, the `sum` and `logsum` pooling will scale the sum to the given degree.
//     It's expected to be of shape `[targetDim, 1]`.
//
// It returns a tensor ([graph.Node]) shaped `[targetSize, pooled_embedded_size]`.
//
// There are no training variables in this. The `scope` is only used for the hyperparameter configuration.
func poolMessagesWithAdjacency(scope *model.Scope, source, edgesSource, edgesTarget *Node, targetSize int, degree *Node) *Node {
	poolTypes := model.GetParamOr(scope, ParamPoolingType, "mean|sum")
	if source.Rank() != 2 {
		Panicf("poolMessagesWithAdjacency(): source is expected to be shaped `[num_nodes, emb_size]`, instead got %s", source.Shape())
	}
	if (edgesSource.Rank() != 1 && edgesSource.Rank() != 2) || !edgesSource.Shape().Equal(edgesTarget.Shape()) ||
		(edgesSource.Rank() == 2 && edgesSource.Shape().Dimensions[1] != 1) {
		Panicf("poolMessagesWithAdjacency(): edgesSource and edgesTarget must have the same shape: either [num_edges] or [num_edges, 1] and be of "+
			"some integer dtype, instead got edgesSource.shape=%s edgesTarget.shape=%s", edgesSource.Shape(), edgesTarget.Shape())
	}
	if degree != nil && (degree.Rank() != 2 || degree.Shape().Dimensions[0] != targetSize || degree.Shape().Dimensions[1] != 1) {
		Panicf("poolMessagesWithAdjacency(): if degree is given (not nil) it is expected to be shaped `[targetSize=%d, 1]`, instead got %s", targetSize, degree.Shape())
	}
	g := source.Graph()
	dtype := source.DType()
	dtypePool := dtype
	if dtype.IsFloat16() {
		// Up-precision to 32 bits for pooling.
		dtypePool = dtypes.Float32
	}
	embSize := source.Shape().Dimensions[1]
	numEdges := edgesSource.Shape().Dimensions[0]
	if edgesSource.Rank() == 1 {
		edgesSource = InsertAxes(edgesSource, -1)
		edgesTarget = InsertAxes(edgesTarget, -1)
	}

	poolTypesList := strings.Split(poolTypes, "|")
	parts := make([]*Node, 0, len(poolTypesList))
	var pooled *Node
	for _, poolType := range poolTypesList {
		switch poolType {
		case "sum", "logsum", "mean":
			// Get values from the source to be pooled. Since a source may contribute to more than one target
			// node, a source value may appear more than once. Shaped `[num_edges, emb_size]`.
			values := Gather(source, edgesSource)
			if dtypePool != dtype {
				values = ConvertDType(values, dtypePool)
			}
			pooled = Scatter(edgesTarget, values, shapes.Make(dtypePool, targetSize, embSize), false, false)

			var pooledCount *Node
			if poolType == "mean" || degree != nil {
				// Get count of items pooled and take the mean.
				ones := Ones(g, shapes.Make(dtypePool, numEdges, 1))
				pooledCount = Scatter(edgesTarget, ones, shapes.Make(dtypePool, targetSize, 1), false, false)
				pooledCount = MaxScalar(pooledCount, 1) // To avoid division by 0.
				pooled = Div(pooled, pooledCount)
			}
			if poolType != "mean" && degree != nil {
				// Weight mean pooled value by `degree`.
				pooled = Mul(pooled, ConvertDType(degree, dtypePool))
			}
			if poolType == "logsum" {
				pooled = MirroredLog1p(pooled)
			}
		default:
			// Notice "max" is not implemented yet.
			Panicf("unknown graph convolution pooling type (%q) given in scope: value given %q (of %q) -- valid values are sum, mean and max, or a combination of them separated by '|'",
				ParamPoolingType, poolType, poolTypes)
		}
		parts = append(parts, pooled)
	}
	if len(parts) == 1 {
		return ConvertDType(parts[0], dtype)
	}
	all := Concatenate(parts, -1)
	if dtype != dtypePool {
		all = ConvertDType(all, dtype)
	}
	return all
}

var layersWithPaperEmbeddingsInput = []string{
	//"/model/readout/gnn:seeds",
	//"/model/graph_update_0/gnn:seedsAuthors/update",

	//"/model/graph_update_0/gnn:papersByAuthors/update",
	//"/model/graph_update_0/gnn:seedsBase/update",

	// These layers, if converted to KAN, incur in accuracy penalties.
	"/model/graph_update_0/gnn:seeds/update",
}

func hasPaperEmbeddingsInput(scope string) bool {
	return slices.Index(layersWithPaperEmbeddingsInput, scope) != -1
}

func noKANForScope(scope *model.Scope) bool {
	skip := model.GetParamOr(scope, ParamNoKanForLayers, "")
	if skip == "" {
		return false
	}
	noKANScopes := strings.Split(skip, ",")
	return slices.Index(noKANScopes, scope.Scope()) >= 0
}

// updateState of a node set, given the `input` (should be a concatenation of previous
// state and all pooled messages) and its `mask`.
func updateState(scope *model.Scope, prevState, input, mask *Node) *Node {
	useKAN := model.GetParamOr(scope, "kan", false)
	if useKAN && noKANForScope(scope) {
		useKAN = false
	}
	if useKAN {
		return kanUpdateState(scope, prevState, input, mask)
	}
	updateType := model.GetParamOr(scope, ParamUpdateStateType, "residual")
	if updateType != "residual" && updateType != "none" {
		Panicf("invalid GNN update type %q (given by parameter %q) -- valid values are 'residual' and 'none'",
			updateType, ParamUpdateStateType)
	}

	// Inputs: both previous state and pooled messages passes through a dropout first.
	input = layers.DropoutFromScope(scope, input)
	stateDim := model.GetParamOr(scope, ParamStateDim, 128)
	numHiddenLayers := model.GetParamOr(scope, ParamUpdateNumHiddenLayers, 0)
	state := input
	for ii := range numHiddenLayers {
		scopeHiddenLayer := scope.In("hidden_%d", ii)
		state = layers.DenseWithBias(scopeHiddenLayer, state, stateDim)
		state = activation.ApplyFromScope(scopeHiddenLayer, state)
	}
	state = layers.DenseWithBias(scope, state, stateDim)
	state = activation.ApplyFromScope(scope, state)
	state = layers.DropoutFromScope(scope, state)
	if updateType == "residual" && prevState.Shape().Equal(state.Shape()) {
		state = Add(state, prevState)
	}
	state = layers.MaskedNormalizeFromScope(scope.In("normalization"), state, mask)
	if NanLogger != nil {
		NanLogger.TraceFirstNaN(state, fmt.Sprintf("UpdateState(%s)", scope.Scope()))
	}
	return state
}

// kanUpdateState is a version of updateState using KAN networks.
func kanUpdateState(scope *model.Scope, prevState, input, mask *Node) *Node {
	stateDim := model.GetParamOr(scope, ParamStateDim, 128)
	numHiddenLayers := model.GetParamOr(scope, ParamUpdateNumHiddenLayers, 0)
	if false && hasPaperEmbeddingsInput(scope.Scope()) {
		fmt.Printf("\t> special KAN for %q\n", scope.Scope())
		inputDim := input.Shape().Dim(-1)
		return fnn.New(scope, input, inputDim).NumHiddenLayers(1, inputDim).Done()
		//g := input.Graph()
		//inputDim := input.Shape().Dim(-1)
		//a := scope.In("adjust").WithInitializer(initializer.One).
		//	VariableWithShape("a", shapes.Make(input.DType(), inputDim)).ValueGraph(g)
		//b := scope.In("adjust").WithInitializer(initializer.Zero).
		//	VariableWithShape("b", shapes.Make(input.DType(), inputDim)).ValueGraph(g)
		//a = ExpandLeftToRank(a, input.Rank())
		//b = ExpandLeftToRank(b, input.Rank())
		//input = Add(Mul(input, a), b)

		/*
			input = kan.New(scope.In("adjust"), input, input.Shape().Dim(-1)).
				//DiscreteInputRange(-1.6, 1.6).
				DiscreteInitialSplitPoints(tensors.FromValue([]float32{
					-1.439246,
					//-0.352688,
					-0.273454,
					//-0.217877,
					-0.172744,
					//-0.134441,
					-0.101113,
					//-0.071418,
					-0.044232,
					//-0.018693,
					0.005911,
					//0.03016,
					0.054653,
					//0.080034,
					0.107104,
					//0.137099,
					0.172041,
					//0.215929,
					0.278253,
					//-1.313336,
				})).Done()
		*/
	}
	kanLayer := kan.New(scope.In("kan_update_state"), input, stateDim).NumHiddenLayers(numHiddenLayers, stateDim)
	state := kanLayer.Done()
	state = layers.DropoutFromScope(scope, state)
	//state = layers.MaskedNormalizeFromScope(scope.In("normalization"), state, mask)

	updateType := model.GetParamOr(scope, ParamUpdateStateType, "residual")
	if updateType == "residual" && prevState.Shape().Equal(state.Shape()) {
		state = Add(state, prevState)
	}
	if NanLogger != nil {
		NanLogger.TraceFirstNaN(state, fmt.Sprintf("(KAN)UpdateState(%s)", scope.Scope()))
	}
	return state
}
