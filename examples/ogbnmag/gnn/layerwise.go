// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package gnn

import (
	"fmt"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	"github.com/gomlx/gomlx/ml/model"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
)

// LayerWiseConfig is a configuration object for [ComputeLayerWiseGNN].
// Once configured with its methods, call [LayerWiseConfig.Done] to actually run the layer-wise GNN.
type LayerWiseConfig struct {
	store                  *model.Store
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
func LayerWiseGNN(store *model.Store, strategy *sampler.Strategy) (*LayerWiseConfig, error) {
	lw := &LayerWiseConfig{
		store:                  store,
		strategy:               strategy,
		sampler:                strategy.Sampler,
		keepIntermediaryStates: true,
		freeAcceleratorMemory:  true,
		numGraphUpdates:        model.GetRootParamOr(store, ParamNumGraphUpdates, 2),
		graphUpdateType:        model.GetRootParamOr(store, ParamGraphUpdateType, "tree"),
	}
	lw.dependentsUpdateFirst = "tree" == lw.graphUpdateType
	if lw.graphUpdateType != "tree" && lw.graphUpdateType != "simultaneous" {
		return nil, errors.Errorf("unsupported graph update type: %s", lw.graphUpdateType)
	}
	if model.GetRootParamOr(store, ParamUsePathToRootStates, false) {
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
func (lw *LayerWiseConfig) NodePrediction(scope *model.Scope, graphStates map[string]*Node, edges map[string]sampler.EdgePair[*Node]) {
	for round := range lw.numGraphUpdates {
		visited := make(map[string]bool)
		for _, rule := range lw.strategy.Seeds {
			lw.recursivelyApplyGraphConvolution(scopeForGraphUpdateRound(scope, round), rule, graphStates, edges, visited)
		}
	}
	scopeReadout := scope.In("readout")
	visited := make(map[string]bool)
	for _, rule := range lw.strategy.Seeds {
		seedState := graphStates[rule.Name]
		kernelScopeName := fmt.Sprintf("%s", rule.ConvKernelScopeName)
		var ruleScope *model.Scope
		if visited[kernelScopeName] {
			ruleScope = scopeReadout.Shared("%s", kernelScopeName)
		} else {
			visited[kernelScopeName] = true
			ruleScope = scopeReadout.In("%s", kernelScopeName)
		}
		seedState = updateState(ruleScope, seedState, seedState, nil)
		graphStates[rule.Name] = seedState
	}
}

func (lw *LayerWiseConfig) recursivelyApplyGraphConvolution(
	scope *model.Scope, rule *sampler.Rule, graphStates map[string]*Node, edges map[string]sampler.EdgePair[*Node], visited map[string]bool) {
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
			lw.recursivelyApplyGraphConvolution(scope, dependent, graphStates, edges, visited)
		}
		dependentState := graphStates[dependent.Name]
		dependentConvKernelScopeName := fmt.Sprintf("%s", dependent.ConvKernelScopeName)
		var convolveScope *model.Scope
		if visited[dependentConvKernelScopeName] {
			convolveScope = scope.Shared("%s", dependentConvKernelScopeName).In("conv")
		} else {
			visited[dependentConvKernelScopeName] = true
			convolveScope = scope.In("%s", dependentConvKernelScopeName).In("conv")
		}
		if dependentState != nil {
			// Notice that we are sending messages on the reverse order of the sampling.
			// E.g.: If paper->"HasTopic"->topic, the sampling direction is "paper is source, topic is target".
			// When evaluating the GNN we want the message to go from topic (source) to paper (target).
			update := lw.convolveEdgeSet(convolveScope, dependent.Name, dependentState, dependentEdges.TargetIndices, dependentEdges.SourceIndices, int(rule.NumNodes))
			updateInputs = append(updateInputs, update)
		}
		if !lw.dependentsUpdateFirst {
			lw.recursivelyApplyGraphConvolution(scope, dependent, graphStates, edges, visited)
		}
	}

	// Update state of current rule: only update state if there was any new incoming input.
	ruleUpdateKernelScopeName := fmt.Sprintf("%s", rule.UpdateKernelScopeName)
	var updateScope *model.Scope
	if visited[ruleUpdateKernelScopeName] {
		updateScope = scope.Shared("%s", ruleUpdateKernelScopeName).In("update")
	} else {
		visited[ruleUpdateKernelScopeName] = true
		updateScope = scope.In("%s", ruleUpdateKernelScopeName).In("update")
	}
	state = updateState(updateScope, state, Concatenate(updateInputs, -1), nil)
	graphStates[rule.Name] = state
}

func (lw *LayerWiseConfig) convolveEdgeSet(scope *model.Scope, ruleName string, sourceState, edgesSource, edgesTarget *Node, numTargetNodes int) *Node {
	messages, _ := edgeMessageGraph(scope.In("message"), sourceState, nil)
	pooled := poolMessagesWithAdjacency(scope, messages, edgesSource, edgesTarget, numTargetNodes, nil)
	return pooled
}
