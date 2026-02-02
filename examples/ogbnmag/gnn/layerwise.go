// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gnn

import (
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	. "github.com/gomlx/gomlx/pkg/support/exceptions"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
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
// It runs the layer-wise GNN once [NodePrediction] is called.
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

// NodePrediction computes the layer-wise GNN.
//
// `initialStates` should map node set names to its initial values, and be shaped `[num_nodes, initial_state_dim]`.
// Any preprocessing transformation (like normalizations) need to be calculated and materialized to tensors.
//
// It can be called more than once with different `initialStates`.
// Subsequent calls will by-step the JIT-compilation of the models.
func (lw *LayerWiseConfig) NodePrediction(ctx *context.Context, graphStates map[string]*Node, edges map[string]sampler.EdgePair[*Node]) {
	for round := range lw.numGraphUpdates {
		for _, rule := range lw.strategy.Seeds {
			lw.recursivelyApplyGraphConvolution(ctxForGraphUpdateRound(ctx, round), rule, graphStates, edges)
		}
	}
	ctxReadout := ctx.In("readout")
	for _, rule := range lw.strategy.Seeds {
		seedState := graphStates[rule.Name]
		seedState = updateState(ctxReadout.In(rule.ConvKernelScopeName), seedState, seedState, nil)
		graphStates[rule.Name] = seedState
	}
}

func (lw *LayerWiseConfig) recursivelyApplyGraphConvolution(
	ctx *context.Context, rule *sampler.Rule, graphStates map[string]*Node, edges map[string]sampler.EdgePair[*Node]) {
	if rule.Name == "" || rule.ConvKernelScopeName == "" {
		Panicf("strategy's rule name=%q or kernel scope name=%q are empty, they both must be defined",
			rule.Name, rule.ConvKernelScopeName)
	}

	// Leaf nodes are not updated.
	if len(rule.Dependents) == 0 {
		return
	}

	// Makes sure there is a state for the current dependent.
	state, found := graphStates[rule.Name]
	if !found {
		Panicf("state for rule %q not given in `graphStates`, states given for rules: %q", rule.Name, xslices.Keys(graphStates))
	}

	updateInputs := make([]*Node, 0, len(rule.Dependents)+1)
	if state != nil { // state size is 0 for latent node types, at their initial state.
		updateInputs = append(updateInputs, state)
	}

	// Update dependents and calculate their convolved messages: it's a depth-first-search on dependents.

	for _, dependent := range rule.Dependents {
		dependentEdges, found := edges[dependent.Name]
		if !found {
			Panicf("edges for rule %q not given in `edges`, edges given for rules: %q", dependent.Name, xslices.Keys(edges))
		}

		if lw.dependentsUpdateFirst {
			lw.recursivelyApplyGraphConvolution(ctx, dependent, graphStates, edges)
		}
		dependentState := graphStates[dependent.Name]
		convolveCtx := ctx.In(dependent.ConvKernelScopeName).In("conv")
		if dependentState != nil {
			// Notice that we are sending messages on the reverse order of the sampling.
			// E.g.: If paper->"HasTopic"->topic, the sampling direction is "paper is source, topic is target".
			// When evaluating the GNN we want the message to go from topic (source) to paper (target).
			update := lw.convolveEdgeSet(convolveCtx, dependent.Name, dependentState, dependentEdges.TargetIndices, dependentEdges.SourceIndices, int(rule.NumNodes))
			updateInputs = append(updateInputs, update)
		}
		if !lw.dependentsUpdateFirst {
			lw.recursivelyApplyGraphConvolution(ctx, dependent, graphStates, edges)
		}
	}

	// Update state of current rule: only update state if there was any new incoming input.
	updateCtx := ctx.In(rule.UpdateKernelScopeName).In("update")
	state = updateState(updateCtx, state, Concatenate(updateInputs, -1), nil)
	graphStates[rule.Name] = state
}

func (lw *LayerWiseConfig) convolveEdgeSet(ctx *context.Context, ruleName string, sourceState, edgesSource, edgesTarget *Node, numTargetNodes int) *Node {
	messages, _ := edgeMessageGraph(ctx.In("message"), sourceState, nil)
	pooled := poolMessagesWithAdjacency(ctx, messages, edgesSource, edgesTarget, numTargetNodes, nil)
	return pooled
}
