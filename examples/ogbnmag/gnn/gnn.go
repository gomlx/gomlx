// Package gnn implements a generic GNN modeling based on [TF-GNN MtAlbis].
//
// And it also includes and example for the OGBN-MAG dataset.
//
// [TF-GNN MtAlbis]: https://github.com/tensorflow/gnn/tree/main/tensorflow_gnn/models/mt_albis
package gnn

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"strings"
)

const (
	// ParamMessageDim context hyperparameter defines the dimension of the messages calculated per node.
	// The default is 128.
	ParamMessageDim = "message_dim"

	// ParamNodeStateDim context hyperparameter defines the dimension of the updated hidden states per node.
	// The default is 128.
	ParamNodeStateDim = "node_state_dim"

	// ParamActivation context hyperparameter defines the activation to use, see [layers.ParamActivation]
	ParamActivation = layers.ParamActivation

	// ParamDropoutRate is applied to FNN kernels, as usual, randomly turning off individual values.
	// Default is 0.0, meaning no dropout.
	ParamDropoutRate = layers.ParamDropoutRate

	// ParamEdgeDropoutRate is applied to full edges, disabling the whole edge.
	// Default is 0.0, meaning no edge dropout.
	ParamEdgeDropoutRate = "edge_dropout_rate"

	// ParamL2Regularization is an alias for layers.ParamL2Regularization, and adds regularizations to the NN kernels.
	// The default is `0.0`.
	ParamL2Regularization = layers.ParamL2Regularization

	// ParamPoolingType context hyperparameter defines how incoming messages in the graph convolutions are pooled (that is, reduced
	// or aggregated).
	// It can take values `mean`, `sum` or `max` or a combination of them separated by `|`.
	// The default is `mean|sum`.
	ParamPoolingType = "convolution_reduce_type"

	// ParamNormalizationType context hyperparameter can take values `none` ( or `""`), `batch` or `layer`.
	// The default is `layer`.
	ParamNormalizationType = "normalization_type"

	// ParamUpdateStateType context hyperparameter can take values `residual` or `none`.
	// The default is `residual`.
	ParamUpdateStateType = "update_state_type"
)

// GraphStateUpdate takes a `graphStates`, a map of name of node sets to their hidden states,
// and updates them by running graph convolutions on the reverse direction of the sampling
// rules in `strategy`, that is, from leaves back to the root of the tree.
//
// All hyperparameters are read from the context `ctx`, and can be set to individual node sets,
// by setting them in the scope `"gnn:"+rule.Name`, where `rule.Name` is the name of the corresponding
// rule in the given `strategy` object.
//
// The object `strategy` defines the tree of node sets by its rules, and the convolutions
// run from the leaf nodes towards the root nodes (seeds).
//
// It updates all states in `graphStates` except the leaves. The masks are left unchanged.
func GraphStateUpdate(ctx *context.Context, strategy *sampler.Strategy, graphStates map[string]*sampler.ValueMask[*Node]) {
	// Starting from the seed node sets, do updates recursively.
	for _, rule := range strategy.Seeds {
		recursivelyApplyGraphConvolution(ctx, strategy, rule, graphStates)
	}
}

func recursivelyApplyGraphConvolution(ctx *context.Context, strategy *sampler.Strategy, rule *sampler.Rule,
	graphStates map[string]*sampler.ValueMask[*Node]) {
	// Makes sure there is a state for the current dependent.
	state, found := graphStates[rule.Name]
	if !found {
		Panicf("state for node %q not given in `graphStates`, states given: %v", rule.Name, slices.Keys(graphStates))
	}

	// Leaf nodes are simply skip.
	if len(rule.Dependents) == 0 {
		return
	}

	// Update dependents and calculate their messages.
	updateInputs := make([]*Node, 0, len(rule.Dependents)+1)
	if state.Value != nil { // state == nil for latent node types, at their initial state.
		updateInputs = append(updateInputs, state.Value)
	}
	for _, dependent := range rule.Dependents {
		recursivelyApplyGraphConvolution(ctx, strategy, dependent, graphStates)
		dependentState := graphStates[dependent.Name]
		updateInputs = append(updateInputs,
			convolveNodeSet(ctx.In("gnn:"+dependent.Name).In("conv"), dependentState.Value, dependentState.Mask))
	}
	state.Value = updateState(
		ctx.In("gnn:"+rule.Name).In("update"),
		state.Value, Concatenate(updateInputs, -1), state.Mask)
}

// convolveNodeSet creates messages from a node set and aggregate them to the same prefix dimension of their source
// node sets. The context `ctx` must already have been properly scoped.
func convolveNodeSet(ctx *context.Context, value, mask *Node) *Node {
	messageDim := context.GetParamOr(ctx, ParamMessageDim, 128)
	messages := layers.DenseWithBias(ctx.In("message"), value, messageDim)
	messages = layers.ActivationFromContext(ctx, messages)
	messages = layers.DropoutFromContext(ctx, messages)
	edgeDropOutRate := context.GetParamOr(ctx, ParamEdgeDropoutRate, 0.0)
	if edgeDropOutRate > 0 {
		// We apply edge dropout to the mask.
		g := messages.Graph()
		mask = layers.DropoutNormalize(ctx, mask, Scalar(g, shapes.F32, edgeDropOutRate), false)
	}
	return poolMessages(ctx, messages, mask)
}

// poolMessages will pool according to [ParamPoolingType].
// Let's say `value` is shaped `[d_0, d_1, ..., d_{n-1}, d_{n}, e]`: we assume `e` is the embedding dimension of the
// tensor, and we want to reduce the axis `n`.
// So the returned shape will be `[d_0, d_1, ..., d_{n-1}, k*e]`, where `k` is the number of pooling types configured.
// E.g.: If the pooling types (see [ParamPoolingType]) are configured to `mean|sum`, then `k=2`.
func poolMessages(ctx *context.Context, value, mask *Node) *Node {
	poolTypes := context.GetParamOr(ctx, ParamPoolingType, "mean")
	poolTypesList := strings.Split(poolTypes, "|")
	parts := make([]*Node, 0, len(poolTypesList))
	var pooled *Node
	for _, poolType := range poolTypesList {
		reduceAxis := value.Rank() - 2
		switch poolType {
		case "sum":
			pooled = MaskedReduceSum(value, mask, reduceAxis)
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

	stateDim := context.GetParamOr(ctx, ParamNodeStateDim, 128)
	state := layers.DenseWithBias(ctx.In("message"), input, stateDim)
	state = layers.ActivationFromContext(ctx, state)
	state = layers.DropoutFromContext(ctx, state)
	if updateType == "residual" && prevState.Shape().Eq(state.Shape()) {
		state = Add(state, prevState)
	}
	state = layers.MaskedNormalizeFromContext(ctx, state, mask)
	return state
}
