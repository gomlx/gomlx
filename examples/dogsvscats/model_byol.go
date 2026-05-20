// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dogsvscats

// This file defines the BYOL-CNN (Bootstrap Your Own Latent) model, based on
// https://arxiv.org/abs/2006.07733

import (
	"path"

	. "github.com/gomlx/gomlx/core/graph"
	timage "github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/examples/inceptionv3"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/support/exceptions"
)

// byolModelEmbedding is the core of the BYOL ((Bootstrap Your Own Latent) model.
// It's built twice, once for the "online" model once for the "target" model -- using scopes on different scopes.
//
// baseTrainable defines whether the base model should be trainable (set to false for the "target"
// model, or if fine-tuning is disabled)
func byolModelEmbedding(scope *model.Scope, images *Node, baseTrainable bool) (embeddings *Node) {
	isInceptionV3 := model.GetParamOr(scope, "byol_inception", false)
	if isInceptionV3 {
		channelsConfig := timage.ChannelsLast
		images = inceptionv3.PreprocessImage(images, 1.0, channelsConfig) // Adjust image to format used by Inception.
		embeddings = inceptionv3.BuildGraph(scope, images).
			SetPooling(inceptionv3.MaxPooling).
			Trainable(baseTrainable).Done()
	} else {
		// Simple CNN model -- we need an extra FNN on top, so we discard the original prediction.
		embeddings = CnnEmbeddings(scope, images)
	}
	if !baseTrainable {
		embeddings = StopGradient(embeddings)
	}
	return
}

// ByolCnnModelGraph builds a BYOL-version of the CNN model of our demo.
//
// It returns the logit, not the predictions, which works with most losses.
// inputs: only one tensor, with shape `[batch_size, width, height, depth]`.
//
// Based on https://arxiv.org/abs/2006.07733
func ByolCnnModelGraph(scope *model.Scope, spec any, inputs []*Node) []*Node {
	_ = spec // Not used.

	// Create two models: same structure, different initializations, and if `--byol_use_pairs` is set,
	// different augmentations of the same image.
	onlineScope := scope.In("online")
	targetScope := scope.In("target").WithInitializer(initializer.RandomNormalFn(scope, 1.0))

	// No dropout for the "target" model.
	targetScope.SetParam("cnn_dropout_rate", 0.0)
	targetScope.SetParam(layers.ParamDropoutRate, 0.0)

	// Evaluation/Inference and if pre-training is over, we only use the "online" model, and return
	// its prediction.
	g := inputs[0].Graph()
	byolPretrain := model.GetParamOr(scope, "byol_pretrain", false)
	byolFinetune := model.GetParamOr(scope, "byol_finetune", false)
	if !scope.IsTraining(g) || !byolPretrain {
		// Normal model path, without byol regularization.
		baseTraining := scope.IsTraining(g) && byolFinetune
		embeddings := byolModelEmbedding(onlineScope, inputs[0], baseTraining)
		logit := fnn.New(scope.In("readout"), embeddings, 1).NumHiddenLayers(0, 0).Done()
		return []*Node{logit} // Return only the logits.
	}

	stackedImages12 := Concatenate([]*Node{inputs[0], inputs[1]}, 0) // For "online" model.
	stackedImages21 := Concatenate([]*Node{inputs[1], inputs[0]}, 0) // For "target" model.
	regularizationRate := model.GetParamOr(targetScope, "byol_regularization_rate", 1.0)

	onlineEmbedding := byolModelEmbedding(onlineScope, stackedImages12, true)
	onlineProjection := byolProjection(onlineScope, onlineEmbedding, 3)
	//byolRegularizeToLengthOne(onlineScope, onlineTargetPrediction)

	targetEmbedding := byolModelEmbedding(targetScope, stackedImages21, false)
	targetProjection := byolProjection(targetScope, targetEmbedding, 1)
	for v := range targetScope.IterVariables() {
		v.Trainable = false
	}
	targetProjection = StopGradient(targetProjection)

	byolReg := byolLoss(onlineProjection, targetProjection)
	ReduceAllMean(Sqrt(byolReg)).SetLogged("byolReg")
	train.AddLoss(MulScalar(byolReg, regularizationRate))

	// Update "target" model with moving average to the "online" model.
	movingAverageRatio := model.GetParamOr(targetScope, "byol_target_update_ratio", 0.999)
	moveTargetModel(onlineScope, targetScope, g, movingAverageRatio)
	return []*Node{} // No prediction to return.
}

func byolProjection(scope *model.Scope, embeddings *Node, numHiddenLayers int) *Node {
	projectionNodes := model.GetParamOr(scope, "byol_projection_nodes", 256)
	projectionHiddenNodes := model.GetParamOr(scope, "byol_hidden_nodes", 4096)
	return fnn.New(scope.In("byol_projection"), embeddings, projectionNodes).
		NumHiddenLayers(numHiddenLayers, projectionHiddenNodes).
		Done()
}

// moveTargetModel slowly move the targetModel towards the onlineModel.
// If movingAverageRatio > 1.0, it is a no-op.
//
// onlineScope and targetScope are the same scope, in different scopes.
func moveTargetModel(onlineScope, targetScope *model.Scope, g *Graph, movingAverageRatio float64) {
	if movingAverageRatio >= 1.0 {
		return
	}
	onlineScope := onlineScope.Scope()
	targetScope := targetScope.Scope()
	for targetVar := range targetScope.IterVariables() {
		// Get corresponding variable in "online" model.
		onlineVarScope := onlineScope + targetVar.Scope()[len(targetScope):]
		onlineVar := onlineScope.Store().GetVariable(path.Join(onlineVarScope, targetVar.Name()))
		if onlineVar == nil {
			exceptions.Panicf("BYOL target model variable %q::%q has no corresponding variable %q::%q in online model",
				targetVar.Scope(), targetVar.Name(), onlineVarScope, targetVar.Name())
		}
		targetValue := targetVar.NodeValue(g)
		onlineValue := onlineVar.NodeValue(g)
		targetValue = Add(
			MulScalar(onlineValue, 1.0-movingAverageRatio),
			MulScalar(targetValue, movingAverageRatio))
		targetVar.SetNodeValue(targetValue)
	}
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
func byolRegularizeToLengthOne(scope *model.Scope, projection *Node) {
	regLenOne := model.GetParamOr(scope, "byol_reg_len1", 0.0)
	if regLenOne <= 0.0 {
		return
	}

	g := projection.Graph()
	dtype := projection.DType()
	lengths := L2Norm(projection, -1)
	ReduceAllMean(lengths).SetLogged("MeanLengthProjection")
	regLength := Square(Sub(ScalarOne(g, dtype), lengths))
	regLength = MulScalar(regLength, regLenOne)
	train.AddLoss(regLength)
}
