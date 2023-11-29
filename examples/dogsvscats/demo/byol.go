package main

// This file defines the BYOL-CNN (Bootstrap Your Own Latent) model, based on
// https://arxiv.org/abs/2006.07733

import (
	"flag"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/models/inceptionv3"
	"github.com/gomlx/gomlx/types/exceptions"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"strings"
)

// This file defines the BYOL-CNN (Bootstrap Your Own Latent) model, based on
// https://arxiv.org/abs/2006.07733

var (
	flagByolProjectionNumLayers = flag.Int("byol_hidden_layers", 2, "When using \"byol\" model, this is the number of layers in the projection to the target regularizing model.")
	flagByolProjectionNumNodes  = flag.Int("byol_num_nodes", 128, "When using \"byol\" model, this is the number of nodes (dimension) in the projection to the target regularizing model.")
	flagByolTargetUpdateRatio   = flag.Float64("byol_target_update_ratio", 0.9999, "Moving average update weight to the \"target\" sub-model for BYOL model.")
	flagByolRegularizationRate  = flag.Float64("byol_regularization_rate", 0.31, "BYOL regularization loss rate, a simple multiplier.")
	flagByolInception           = flag.Bool("byol_inception", false, "Instead of using a CNN model with BYOL, uses InceptionV3.")

	flagByolPretraining = flag.Bool("byol_pretrain", false, "Pre-train BYOL model, unsupervised.")
	flagByolFinetuning  = flag.Bool("byol_finetuning", false, "Finetune BYOL model. If set to false, only the linear model on top is trained.")
)

// byolModel is the core of the BYOL model.
// It's built twice, once for the "online" model once for the "target" model -- using contexts on different scopes.
//
// baseTrainable defines whether the base model should be trainable (set to false for the "target"
// model, or if fine-tuning is disabled)
func byolModel(ctx *context.Context, images *Node, baseTrainable bool) (logit, embeddings *Node) {
	isInceptionV3 := context.GetParamOr(ctx, "byol_inception", false)
	if isInceptionV3 {
		channelsConfig := timage.ChannelsLast
		images = inceptionv3.PreprocessImage(images, 1.0, channelsConfig) // Adjust image to format used by Inception.
		embeddings = inceptionv3.BuildGraph(ctx, images).
			SetPooling(inceptionv3.MaxPooling).
			Trainable(baseTrainable).Done()
	} else {
		// Simple CNN model -- we need an extra FNN on top, so we discard the original prediction.
		embeddings = CnnEmbeddings(ctx, images)
	}
	if !baseTrainable {
		embeddings = StopGradient(embeddings)
	}

	logit = layers.DenseWithBias(ctx.In("readout"), embeddings, 1)
	return
}

// ByolCnnModelGraph builds a BYOL-version of the CNN model of our demo.
//
// It returns the logit, not the predictions, which works with most losses.
// inputs: only one tensor, with shape `[batch_size, width, height, depth]`.
func ByolCnnModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec // Not used.

	// Create two models: same structure, different initializations, and if `--byol_use_pairs` is set,
	// different augmentations of the same image.
	onlineCtx := ctx.In("online")
	targetCtx := ctx.In("target")
	regularizationRate := context.GetParamOr(targetCtx, "byol_regularization_rate", 0.1)

	// Evaluation/Inference and if pre-training is over, we only use the "online" model, and return
	// its prediction.
	g := inputs[0].Graph() // Graph.
	if !ctx.IsTraining(g) || !*flagByolPretraining {
		baseTraining := ctx.IsTraining(g) && *flagByolFinetuning
		onlineLogit, _ := byolModel(onlineCtx, inputs[0], baseTraining)
		return []*Node{onlineLogit} // Return only the logits.
	}

	// No dropout on target model.
	targetCtx.SetParam("conv_dropout", 0.0) // Disable dropout on the target side.
	targetCtx.SetParam("dropout", 0.0)      // Disable dropout on the target side.

	// We use image pairs A, B -- the same image, but with different augmentations.
	// There are no predictions, we `trainer.AddLoss` our BYOL unsupervised loss.
	// But we flip the images twice: first Online(A) vs Target(B) then Online(B) vs Target(A).
	for flip := 0; flip < 2; flip++ {
		// "Online" model is the one that we'll take the predictions from.
		if flip > 0 {
			onlineCtx = onlineCtx.Reuse()
		}
		_, onlineEmbedding := byolModel(onlineCtx, inputs[0], true)

		onlineProjection := byolProjection(onlineCtx, onlineEmbedding)
		onlineTargetPrediction := layers.Dense(onlineCtx.In("online_target_prediction"), onlineProjection, true,
			context.GetParamOr(onlineCtx, "byol_num_nodes", 0))
		onlineTargetPrediction = L2NormalizeWithEpsilon(onlineTargetPrediction, 1e-12, -1)

		// "Target" model is the one used to regularize, and is updated by a moving
		// average towards the "Online" model.
		// Flip inputs for target:
		inputs[0], inputs[1] = inputs[1], inputs[0]
		if flip > 0 {
			targetCtx = targetCtx.Reuse()
		}
		_, targetEmbedding := byolModel(targetCtx, inputs[0], false)
		targetProjection := byolProjection(targetCtx, targetEmbedding)
		targetProjection = L2NormalizeWithEpsilon(targetProjection, 1e-12, -1)

		// Gradient descent does not update the "target" model, so we `StopGradient` and mark their
		// variables as not training.
		targetProjection = StopGradient(targetProjection)
		if flip == 0 {
			targetCtx.EnumerateVariablesInScope(func(v *context.Variable) {
				v.Trainable = false
			})
		}

		// Add a loss term regularizing the "online" model projection towards the "target" one.
		targetRegularization := L2NormSquare(Sub(onlineTargetPrediction, targetProjection), -1)
		targetRegularization = MulScalar(targetRegularization, 0.5) // 1/2 for each flip.

		//train.AddLoss(ctx, targetRegularization)
		train.AddLoss(ctx, MulScalar(targetRegularization, regularizationRate))
	}

	// Update "target" model with moving average to the "online" model.
	movingAverageRatio := context.GetParamOr(targetCtx, "byol_target_update_ratio", 0.999)
	if movingAverageRatio < 1.0 {
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
	}
	return []*Node{} // No prediction to return.
}

func byolProjection(ctx *context.Context, embeddings *Node) *Node {
	// Re-use FnnOnTop: redefine its params based on BYOL ones, in the local scope.
	ctx = ctx.In("byol_projection")
	numLayers := context.GetParamOr(ctx, "byol_hidden_layers", 2)
	if numLayers == 0 {
		return embeddings
	}
	ctx.SetParam("hidden_layers", numLayers)
	numNodes := context.GetParamOr(ctx, "byol_num_nodes", 0)
	ctx.SetParam("num_nodes", numNodes)
	return FnnOnTop(ctx, embeddings)
}
