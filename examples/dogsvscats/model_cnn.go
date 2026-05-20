// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dogsvscats

// This file implements the baseline CNN model, including the FNN layers on top.

import (
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/exceptions"
)

// CnnModelGraph builds the CNN model for our demo.
// It returns the logit, not the predictions, which works with most losses.
// inputs: only one tensor, with shape `[batch_size, width, height, depth]`.
func CnnModelGraph(scope *model.Scope, spec any, inputs []*Node) []*Node {
	scope = scope.In("model") // Create the model by default under the "/model" scope.
	embeddings := CnnEmbeddings(scope, inputs[0])
	logit := fnn.New(scope.In("readout"), embeddings, 1).NumHiddenLayers(0, 0).Done()
	return []*Node{logit}
}

func CnnEmbeddings(scope *model.Scope, images *Node) *Node {
	batchSize := images.Shape().Dimensions[0]
	numConvolutions := model.GetParamOr(scope, "cnn_num_layers", 5)

	// Dropout.
	dropoutRate := model.GetParamOr(scope, "cnn_dropout_rate", -1.0)
	if dropoutRate < 0 {
		dropoutRate = model.GetParamOr(scope, layers.ParamDropoutRate, 0.0)
	}
	var dropoutNode *Node
	if dropoutRate > 0.0 {
		dropoutNode = Scalar(images.Graph(), images.DType(), dropoutRate)
	}

	numChannels := 16
	logits := images
	imgSize := logits.Shape().Dimensions[1]
	for convIdx := range numConvolutions {
		scope := scope.In("%03d_conv", convIdx)
		if convIdx > 0 {
			logits = normalizeImage(scope, logits)
		}
		for repeat := range 2 {
			scope := scope.In("repeat_%02d", repeat)
			residual := logits
			logits = layers.Convolution(scope, logits).Channels(numChannels).KernelSize(3).PadSame().Done()
			logits = activation.ApplyFromScope(scope, logits)
			if dropoutNode != nil {
				logits = layers.Dropout(scope, logits, dropoutNode)
			}
			if residual.Shape().Equal(logits.Shape()) {
				logits = Add(logits, residual)
			}
		}
		if imgSize > 16 {
			// Reduce image size by 2 each time.
			logits = MaxPool(logits).Window(2).Done()
			imgSize = logits.Shape().Dimensions[1]
		}
		logits.AssertDims(batchSize, imgSize, imgSize, numChannels)
	}

	// Flatten the resulting image, and treat the convolved values as tabular.
	logits = Reshape(logits, batchSize, -1)
	return fnn.New(scope.In("%03d_fnn", numConvolutions), logits, model.GetParamOr(scope, "cnn_embeddings_size", 128)).Done()
}

func normalizeImage(scope *model.Scope, x *Node) *Node {
	x.AssertRank(4) // [batch_size, width, height, depth]
	norm := model.GetParamOr(scope, "cnn_normalization", "")
	if norm == "" {
		model.GetParamOr(scope, layers.ParamNormalization, "")
	}
	switch norm {
	case "layer":
		return norm.LayerNorm(scope, x, 1, 2).ScaleNormalization(false).Done()
	case "batch":
		return norm.BatchNorm(scope, x, -1).Done()
	case "none", "":
		return x
	}
	exceptions.Panicf("invalid normalization selected %q -- valid values are batch, layer, none", norm)
	return nil
}
