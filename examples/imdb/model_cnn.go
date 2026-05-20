// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package imdb

import (
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/exceptions"
)

// Conv1DModelGraph implements a convolution (1D) based model for the IMDB dataset.
func Conv1DModelGraph(scope *model.Scope, spec any, inputs []*Node) []*Node {
	_ = spec
	tokens := inputs[0]
	embed, _ := EmbedTokensGraph(scope, tokens)

	g := embed.Graph()
	dtype := embed.DType()
	embed.AssertRank(3)
	batchSize := embed.Shape().Dimensions[0]
	// contentLen := embed.Shape().Dimensions[1]
	embedSize := embed.Shape().Dimensions[2]

	// Dropout.
	dropoutRate := model.GetParamOr(scope, "cnn_dropout_rate", -1.0)
	if dropoutRate < 0 {
		dropoutRate = model.GetParamOr(scope, layers.ParamDropoutRate, 0.0)
	}
	var dropoutNode *Node
	if dropoutRate > 0.0 {
		dropoutNode = Scalar(g, dtype, dropoutRate)
	}

	// 1D Convolution: embed is [batch_size, content_len, embed_size].
	numConvolutions := model.GetParamOr(scope, "cnn_num_layers", 5)
	logits := embed
	for convIdx := range numConvolutions {
		scope := scope.In("%03d_conv", convIdx)
		residual := logits
		if convIdx > 0 {
			logits = NormalizeSequence(scope, logits)
		}
		logits = layers.Convolution(scope, embed).KernelSize(7).Channels(embedSize).Strides(1).Done()
		logits = activations.ApplyFromScope(scope, logits)
		if dropoutNode != nil {
			logits = layers.Dropout(scope, logits, dropoutNode)
		}
		if residual.Shape().Equal(logits.Shape()) {
			logits = Add(logits, residual)
		}
	}

	// Take the max over the content length, and put an FNN on top.
	// Shape transformation: [batch_size, content_len, embed_size] -> [batch_size, embed_size]
	logits = ReduceMax(logits, 1)
	logits = fnn.New(scope, logits, 1).Done()
	logits.AssertDims(batchSize, 1)
	return []*Node{logits}
}

// NormalizeSequence `x` according to "normalization" hyperparameter. Works for sequence nodes (rank-3).
func NormalizeSequence(scope *model.Scope, x *Node) *Node {
	x.AssertRank(3) // [batch_size, content_length, embed_size]
	norm := model.GetParamOr(scope, "cnn_normalization", "")
	if norm == "" {
		model.GetParamOr(scope, layers.ParamNormalization, "")
	}

	switch norm {
	case "layer":
		return layers.LayerNormalization(scope, x, -2).
			LearnedOffset(true).LearnedGain(true).ScaleNormalization(true).Done()
	case "batch":
		return batchnorm.New(scope, x, -1).Done()
	case "none", "":
		return x
	}
	exceptions.Panicf(`invalid normalization selected %q -- valid values are "batch", "layer", "none" or ""`, norm)
	return nil
}
