package main

// This file defines the BYOL-CNN (Bootstrap Your Own Latent) model, based on
// https://arxiv.org/abs/2006.07733

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/nanlogger"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/exceptions"
	"strings"
)

// This file defines the BYOL-CNN (Bootstrap Your Own Latent) model, based on
// https://arxiv.org/abs/2006.07733

var nanLogger *nanlogger.NanLogger

// ByolCnnModelGraph builds a BYOL-version of the CNN model of our demo.
//
// It returns the logit, not the predictions, which works with most losses.
// inputs: only one tensor, with shape `[batch_size, width, height, depth]`.
func ByolCnnModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	// Create two models: same structure, different initializations

	// "Online" model is the one that we'll take the predictions from.
	onlineCtx := ctx.In("online")
	onlineOutputs := CnnModelWithEmbedding(onlineCtx, spec, inputs)
	onlineLogit, onlineEmbedding := onlineOutputs[0], onlineOutputs[1]
	g := inputs[0].Graph() // Graph.
	if !ctx.IsTraining(g) {
		// For evaluation, we don't need the "target" model, and we can immediately return the "online" model prediction.
		return []*Node{onlineLogit} // Return only the logits.
	}
	onlineProjection := byolProjection(onlineCtx, onlineEmbedding)
	onlineTargetPrediction := layers.Dense(onlineCtx.In("target_prediction"), onlineProjection, true,
		context.GetParamOr(onlineCtx, "byol_num_nodes", 0))
	onlineTargetPrediction = L2NormalizeWithEpsilon(onlineTargetPrediction, 1e-12, -1)

	// "Target" model is the one used to regularize, and is updated by a moving
	// average towards the "Online" model.
	targetCtx := ctx.In("target")
	targetOutputs := CnnModelWithEmbedding(targetCtx, spec, inputs)
	targetEmbedding := targetOutputs[1]
	targetProjection := byolProjection(targetCtx, targetEmbedding)
	targetProjection = L2NormalizeWithEpsilon(targetProjection, 1e-12, -1)

	// Gradient descent does not update the "target" model, so we `StopGradient` and mark their
	// variables as not training.
	targetProjection = StopGradient(targetProjection)
	targetCtx.EnumerateVariablesInScope(func(v *context.Variable) {
		v.Trainable = false
	})

	// Add a loss term regularizing the "online" model projection towards the "target" one.
	targetRegularization := L2NormSquare(Sub(onlineTargetPrediction, targetProjection), -1)
	train.AddLoss(ctx, targetRegularization)
	//train.AddLoss(ctx, MulScalar(targetRegularization, 1.0))

	// Update "target" model with moving average to the "online" model.
	movingAverageRatio := context.GetParamOr(targetCtx, "byol_target_update_ratio", 0.999)
	onlineScope := onlineCtx.Scope()
	targetScope := targetCtx.Scope()
	targetCtx.EnumerateVariablesInScope(func(targetVar *context.Variable) {
		if !strings.HasPrefix(targetVar.Scope(), targetScope) {
			exceptions.Panicf("BYOL target model variable %q::%q has unexpected scope (not prefixed with %q)",
				targetVar.Scope(), targetVar.Name(), targetScope)
		}

		// Get corresponding variable in "online" model.
		onlineVarScope := onlineScope + targetVar.Scope()[len(targetScope):]
		onlineVar := ctx.InspectVariable(onlineVarScope, targetVar.Name())
		if onlineVar == nil {
			exceptions.Panicf("BYOL target model variable %q::%q has no corresponding variable %q::%q in online model",
				targetVar.Scope(), targetVar.Name(), onlineVarScope, targetVar.Name())
		}

		targetValue := targetVar.ValueGraph(g)
		onlineValue := onlineVar.ValueGraph(g)
		targetValue = Add(
			MulScalar(onlineValue, 1.0-movingAverageRatio),
			MulScalar(targetValue, movingAverageRatio))
		targetVar.SetValueGraph(targetValue)
	})

	return []*Node{onlineLogit} // Return only the logits.
}

func byolProjection(ctx *context.Context, embeddings *Node) *Node {
	// Re-use FnnOnTop: redefine its params based on BYOL ones, in the local scope.
	ctx = ctx.In("byol_projection")
	ctx.SetParam("hidden_layers", context.GetParamOr(ctx, "byol_hidden_layers", 2))
	ctx.SetParam("num_nodes", context.GetParamOr(ctx, "byol_num_nodes", 0))
	return FnnOnTop(ctx, embeddings)
}
