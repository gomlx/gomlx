package dogsvscats

// This file implements the baseline CNN model, including the FNN layers on top.

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/ml/layers/fnn"
)

// CnnModelGraph builds the CNN model for our demo.
// It returns the logit, not the predictions, which works with most losses.
// inputs: only one tensor, with shape `[batch_size, width, height, depth]`.
func CnnModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	ctx = ctx.In("model") // Create the model by default under the "/model" scope.
	embeddings := CnnEmbeddings(ctx, inputs[0])
	logit := fnn.New(ctx.In("readout"), embeddings, 1).NumHiddenLayers(0, 0).Done()
	return []*Node{logit}
}

func CnnEmbeddings(ctx *context.Context, images *Node) *Node {
	batchSize := images.Shape().Dimensions[0]
	numConvolutions := context.GetParamOr(ctx, "cnn_num_layers", 5)

	// Dropout.
	dropoutRate := context.GetParamOr(ctx, "cnn_dropout_rate", -1.0)
	if dropoutRate < 0 {
		dropoutRate = context.GetParamOr(ctx, layers.ParamDropoutRate, 0.0)
	}
	var dropoutNode *Node
	if dropoutRate > 0.0 {
		dropoutNode = Scalar(images.Graph(), images.DType(), dropoutRate)
	}

	numChannels := 16
	logits := images
	imgSize := logits.Shape().Dimensions[1]
	for convIdx := range numConvolutions {
		ctx := ctx.Inf("%03d_conv", convIdx)
		if convIdx > 0 {
			logits = normalizeImage(ctx, logits)
		}
		for repeat := range 2 {
			ctx := ctx.Inf("repeat_%02d", repeat)
			residual := logits
			logits = layers.Convolution(ctx, logits).Channels(numChannels).KernelSize(3).PadSame().Done()
			logits = activations.ApplyFromContext(ctx, logits)
			if dropoutNode != nil {
				logits = layers.Dropout(ctx, logits, dropoutNode)
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
	return fnn.New(ctx.Inf("%03d_fnn", numConvolutions), logits, context.GetParamOr(ctx, "cnn_embeddings_size", 128)).Done()
}

func normalizeImage(ctx *context.Context, x *Node) *Node {
	x.AssertRank(4) // [batch_size, width, height, depth]
	norm := context.GetParamOr(ctx, "cnn_normalization", "")
	if norm == "" {
		context.GetParamOr(ctx, layers.ParamNormalization, "")
	}
	switch norm {
	case "layer":
		return layers.LayerNormalization(ctx, x, 1, 2).ScaleNormalization(false).Done()
	case "batch":
		return batchnorm.New(ctx, x, -1).Done()
	case "none", "":
		return x
	}
	exceptions.Panicf("invalid normalization selected %q -- valid values are batch, layer, none", norm)
	return nil
}
