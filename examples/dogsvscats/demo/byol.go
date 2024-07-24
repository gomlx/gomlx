package main

// This file defines the BYOL-CNN (Bootstrap Your Own Latent) model, based on
// https://arxiv.org/abs/2006.07733

import (
	"flag"
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/models/inceptionv3"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"strings"
)

// This file defines the BYOL-CNN (Bootstrap Your Own Latent) model, based on
// https://arxiv.org/abs/2006.07733

var (
	flagByolProjectionHiddenNodes = flag.Int("byol_hidden_nodes", 4096, "When using \"byol\" model, the number of nodes in the hidden layer.")
	flagByolProjectionNodes       = flag.Int("byol_projection_nodes", 256, "When using \"byol\" model, this is the number of nodes (dimension) in the projection to the target regularizing model.")
	flagByolTargetUpdateRatio     = flag.Float64("byol_target_update_ratio", 0.99, "Moving average update weight to the \"target\" sub-model for BYOL model.")
	flagByolRegularizationRate    = flag.Float64("byol_regularization_rate", 1.0, "BYOL regularization loss rate, a simple multiplier.")
	flagByolRegLenOne             = flag.Float64("byol_reg_len1", 0.01, "BYOL regularize projections to length 1.")
	flagByolInception             = flag.Bool("byol_inception", false, "Instead of using a CNN model with BYOL, uses InceptionV3.")

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
	targetCtx := ctx.In("target").WithInitializer(initializers.RandomNormalFn(0, 1.0))

	// No dropout for the "target" model, and a more random initialization.
	targetCtx.SetParam("conv_dropout", 0.0) // Disable dropout on the target side.
	targetCtx.SetParam("dropout", 0.0)      // Disable dropout on the target side.

	// Evaluation/Inference and if pre-training is over, we only use the "online" model, and return
	// its prediction.
	g := inputs[0].Graph() // Graph.
	if !ctx.IsTraining(g) || !*flagByolPretraining {
		baseTraining := ctx.IsTraining(g) && *flagByolFinetuning
		onlineLogit, _ := byolModel(onlineCtx, inputs[0], baseTraining)
		return []*Node{onlineLogit} // Return only the logits.
	}

	stackedImages12 := Concatenate([]*Node{inputs[0], inputs[1]}, 0) // For "online" model.
	stackedImages21 := Concatenate([]*Node{inputs[1], inputs[0]}, 0) // For "target" model.

	regularizationRate := context.GetParamOr(targetCtx, "byol_regularization_rate", 1.0)

	_, onlineEmbedding := byolModel(onlineCtx, stackedImages12, true)
	onlineProjection := byolProjection(onlineCtx, onlineEmbedding)
	onlinePrediction := byolOnlinePrediction(onlineCtx, onlineProjection)
	//byolRegularizeToLengthOne(onlineCtx, onlineTargetPrediction)

	_, targetEmbedding := byolModel(targetCtx, stackedImages21, false)
	targetProjection := byolProjection(targetCtx, targetEmbedding)
	targetCtx.EnumerateVariablesInScope(func(v *context.Variable) {
		v.Trainable = false
	})
	targetProjection = StopGradient(targetProjection)

	byolReg := byolLoss(onlinePrediction, targetProjection)
	ReduceAllMean(Sqrt(byolReg)).SetLogged("byolReg")
	train.AddLoss(ctx, MulScalar(byolReg, regularizationRate))

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
	projectionNodes := context.GetParamOr(ctx, "byol_projection_nodes", 256)
	projectionHiddenNodes := context.GetParamOr(ctx, "byol_hidden_nodes", 4096)

	// Re-use FnnOnTop: redefine its params based on BYOL ones, in the local scope.
	ctx = ctx.In("byol_projection")
	hiddenCtx := ctx.In("hidden")
	embeddings = layers.Dense(hiddenCtx, embeddings, true, projectionHiddenNodes)
	embeddings = normalizeFeatures(hiddenCtx, embeddings)
	embeddings = activations.Relu(embeddings)
	embeddings = layers.Dense(ctx.In("projection"), embeddings, true, projectionNodes)
	return embeddings
}

func byolOnlinePrediction(ctx *context.Context, projection *Node) *Node {
	projectionNodes := context.GetParamOr(ctx, "byol_projection_nodes", 256)
	projectionHiddenNodes := context.GetParamOr(ctx, "byol_hidden_nodes", 4096)

	ctx = ctx.In("byol_online_prediction")
	hiddenCtx := ctx.In("hidden")
	projection = layers.Dense(hiddenCtx, projection, true, projectionHiddenNodes)
	projection = normalizeFeatures(hiddenCtx, projection)
	projection = activations.Relu(projection)
	projection = layers.Dense(ctx.In("projection"), projection, true, projectionNodes)
	return projection
}

// byolLoss is based on the projections from the "online" model and "target" models -- the order
// doesn't matter.
func byolLoss(p0, p1 *Node) *Node {
	ReduceAllMean(L2Norm(p0, -1)).SetLogged("||online_prediction||_2=")
	ReduceAllMean(L2Norm(p1, -1)).SetLogged("||target_projection||_2=")

	p0 = L2NormalizeWithEpsilon(p0, 1e-12, -1)
	p1 = L2NormalizeWithEpsilon(p1, 1e-12, -1)
	return AddScalar(
		MulScalar(
			ReduceSum(Mul(p0, p1), -1),
			-2.0),
		2.0)
}

// Add a regularization term proportional to $(1 - L2(projection))^2$.
func byolRegularizeToLengthOne(ctx *context.Context, projection *Node) {
	regLenOne := context.GetParamOr(ctx, "byol_reg_len1", 0.0)
	if regLenOne <= 0.0 {
		return
	}

	g := projection.Graph()
	dtype := projection.DType()
	lengths := L2Norm(projection, -1)
	ReduceAllMean(lengths).SetLogged("MeanLengthProjection")
	regLength := Square(Sub(ScalarOne(g, dtype), lengths))
	regLength = MulScalar(regLength, regLenOne)
	train.AddLoss(ctx, regLength)
}
