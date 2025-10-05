package imdb

import (
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	. "github.com/gomlx/gomlx/pkg/core/graph"
)

// Conv1DModelGraph implements a convolution (1D) based model for the IMDB dataset.
func Conv1DModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec
	tokens := inputs[0]
	embed, _ := EmbedTokensGraph(ctx, tokens)

	g := embed.Graph()
	dtype := embed.DType()
	embed.AssertRank(3)
	batchSize := embed.Shape().Dimensions[0]
	// contentLen := embed.Shape().Dimensions[1]
	embedSize := embed.Shape().Dimensions[2]

	// Dropout.
	dropoutRate := context.GetParamOr(ctx, "cnn_dropout_rate", -1.0)
	if dropoutRate < 0 {
		dropoutRate = context.GetParamOr(ctx, layers.ParamDropoutRate, 0.0)
	}
	var dropoutNode *Node
	if dropoutRate > 0.0 {
		dropoutNode = Scalar(g, dtype, dropoutRate)
	}

	// 1D Convolution: embed is [batch_size, content_len, embed_size].
	numConvolutions := context.GetParamOr(ctx, "cnn_num_layers", 5)
	logits := embed
	for convIdx := range numConvolutions {
		ctx := ctx.Inf("%03d_conv", convIdx)
		residual := logits
		if convIdx > 0 {
			logits = NormalizeSequence(ctx, logits)
		}
		logits = layers.Convolution(ctx, embed).KernelSize(7).Channels(embedSize).Strides(1).Done()
		logits = activations.ApplyFromContext(ctx, logits)
		if dropoutNode != nil {
			logits = layers.Dropout(ctx, logits, dropoutNode)
		}
		if residual.Shape().Equal(logits.Shape()) {
			logits = Add(logits, residual)
		}
	}

	// Take the max over the content length, and put an FNN on top.
	// Shape transformation: [batch_size, content_len, embed_size] -> [batch_size, embed_size]
	logits = ReduceMax(logits, 1)
	logits = fnn.New(ctx, logits, 1).Done()
	logits.AssertDims(batchSize, 1)
	return []*Node{logits}
}

// NormalizeSequence `x` according to "normalization" hyperparameter. Works for sequence nodes (rank-3).
func NormalizeSequence(ctx *context.Context, x *Node) *Node {
	x.AssertRank(3) // [batch_size, content_length, embed_size]
	norm := context.GetParamOr(ctx, "cnn_normalization", "")
	if norm == "" {
		context.GetParamOr(ctx, layers.ParamNormalization, "")
	}

	switch norm {
	case "layer":
		return layers.LayerNormalization(ctx, x, -2).
			LearnedOffset(true).LearnedGain(true).ScaleNormalization(true).Done()
	case "batch":
		return batchnorm.New(ctx, x, -1).Done()
	case "none", "":
		return x
	}
	exceptions.Panicf(`invalid normalization selected %q -- valid values are "batch", "layer", "none" or ""`, norm)
	return nil
}
