// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

/*
 *	Copyright 2025 Rener Castro
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package mnist

// This file implements the baseline CNN model, including the FNN layers on top.

import (
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
)

// LinearModelGraph builds a simple  model logistic model
// It returns the logit, not the predictions, which works with most losses with shape `[batch_size, NumClasses]`.
// inputs: only one tensor, with shape `[batch_size, width, height, depth]`.
func LinearModelGraph(scope *model.Scope, spec any, inputs []*Node) []*Node {
	scope = scope.In("model") // Create the model by default under the "/model" scope.
	batchSize := inputs[0].Shape().Dimensions[0]
	embeddings := Reshape(inputs[0], batchSize, -1)
	logits := layers.DenseWithBias(scope, embeddings, NumClasses)
	return []*Node{logits}
}

// CnnModelGraph builds the CNN model for our demo.
// It returns the logit, not the predictions, which works with most losses with shape `[batch_size, NumClasses]`.
// inputs: only one tensor, with shape `[batch_size, width, height, depth]`.
func CnnModelGraph(scope *model.Scope, spec any, inputs []*Node) []*Node {
	scope = scope.In("model") // Create the model by default under the "/model" scope.
	embeddings := CnnEmbeddings(scope, inputs[0])
	logits := layers.Dense(scope, embeddings, true, NumClasses)
	return []*Node{logits}
}

func CnnEmbeddings(scope *model.Scope, images *Node) *Node {
	batchSize := images.Shape().Dimensions[0]
	g := images.Graph()
	dtype := images.DType()

	layerIdx := 0
	nextScope := func(name string) *model.Scope {
		newScope := scope.In("%03d_%s", layerIdx, name)
		layerIdx++
		return newScope
	}
	// Dropout.
	dropoutRate := model.GetParamOr(scope, "cnn_dropout_rate", -1.0)
	if dropoutRate < 0 {
		dropoutRate = model.GetParamOr(scope, layers.ParamDropoutRate, 0.0)
	}
	var dropoutNode *Node
	if dropoutRate > 0.0 {
		dropoutNode = Scalar(g, dtype, dropoutRate)
	}

	images = layers.Convolution(nextScope("conv"), images).Channels(32).KernelSize(3).PadSame().Done()
	images.AssertDims(batchSize, 28, 28, 32)
	images = activation.Relu(images)
	images = normalizeCNN(nextScope("norm"), images)
	images = MaxPool(images).Window(2).Done()
	images.AssertDims(batchSize, 14, 14, 32)

	images = layers.Convolution(nextScope("conv"), images).Channels(64).KernelSize(3).PadSame().Done()
	images.AssertDims(batchSize, 14, 14, 64)
	images = activation.Relu(images)
	images = normalizeCNN(nextScope("norm"), images)
	images = MaxPool(images).Window(2).Done()
	images = layers.DropoutNormalize(nextScope("dropout"), images, dropoutNode, true)
	images.AssertDims(batchSize, 7, 7, 64)

	// Flatten images
	images = Reshape(images, batchSize, -1)
	return images
}

func normalizeCNN(scope *model.Scope, logits *Node) *Node {
	normalizationType := model.GetParamOr(scope, "cnn_normalization", "none")
	switch normalizationType {
	case "layer":
		if logits.Rank() == 2 {
			return norm.LayerNorm(scope, logits, -1).Done()
		} else if logits.Rank() == 4 {
			return norm.LayerNorm(scope, logits, 2, 3).Done()
		} else {
			return logits
		}
	case "batch":
		return norm.BatchNorm(scope, logits, -1).Done()
	case "none", "":
		return logits
	default:
		panic(errors.Errorf("invalid normalization type %q -- set it with parameter %q", normalizationType, "cnn_normalization"))
	}
}
