package main

// This file implements the baseline CNN model, including the FNN layers on top.

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
)

// CnnModelGraph builds the CNN model for our demo.
// It returns the logit, not the predictions, which works with most losses.
// inputs: only one tensor, with shape `[batch_size, width, height, depth]`.
func CnnModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	outputs := CnnModelWithEmbedding(ctx, spec, inputs)
	return outputs[:1] // Return only the logits.
}

// CnnModelWithEmbedding builds a CNN model and return the final logit of the binary classification and the last layer embeddings.
func CnnModelWithEmbedding(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec // Not needed.
	x := inputs[0]
	filterSize := 16
	batchSize := x.Shape().Dimensions[0]
	logits := x
	imgSize := x.Shape().Dimensions[1]
	numConvolutions := context.GetParamOr(ctx, "num_convolutions", 5)
	for convIdx := 0; convIdx < numConvolutions && imgSize > 16; convIdx++ {
		ctx := ctx.In(fmt.Sprintf("conv_%d", convIdx))
		residual := logits
		if convIdx > 0 {
			logits = layers.Relu(logits)
		}
		logits = layers.Convolution(ctx, logits).Filters(filterSize).KernelSize(3).PadSame().Done()
		logits = layers.Relu(logits)
		logits = normalizeImage(ctx, logits)
		if convIdx > 0 {
			logits = Add(logits, residual)
		}
		if imgSize > 16 {
			// Reduce image size by 2 each time.
			logits = MaxPool(logits).Window(2).Done()
			imgSize /= 2
		}
		logits.AssertDims(batchSize, imgSize, imgSize, filterSize)
	}

	// Flatten the resulting image, and treat the convolved values as tabular.
	logits = Reshape(logits, batchSize, -1)
	logits = FnnOnTop(ctx, logits)
	embedding := logits
	logits = layers.DenseWithBias(ctx.In("readout"), logits, 1)
	return []*Node{logits, embedding}
}

// FnnOnTop adds a feedforward neural network on top of the CNN layer and returns the "embedding" of the last layer.
func FnnOnTop(ctx *context.Context, logits *Node) *Node {
	numHiddenLayers := context.GetParamOr(ctx, "hidden_layers", 3)
	numNodes := context.GetParamOr(ctx, "num_nodes", 3)
	for ii := 0; ii < numHiddenLayers; ii++ {
		ctx := ctx.In(fmt.Sprintf("dense_%d", ii))
		residual := logits
		// Add layer with residual connection.
		logits = layers.Relu(logits)
		logits = layers.DenseWithBias(ctx, logits, numNodes)
		logits = normalizeFeatures(ctx, logits)
		if ii >= 1 {
			logits = Add(logits, residual)
		}
	}
	return logits
}
