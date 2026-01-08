// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package cifar

import (
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"
	"github.com/gomlx/gomlx/pkg/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/pkg/ml/layers/fnn"
	"github.com/gomlx/gomlx/pkg/ml/layers/kan"
)

// C10PlainModelGraph implements train.ModelFn, and returns the logit Node, given the input image.
// It's a basic FNN (Feedforward Neural Network), so no convolutions. It is meant only as an example.
func C10PlainModelGraph(ctx *context.Context, spec any, inputs []*graph.Node) []*graph.Node {
	batchedImages := inputs[0]
	batchSize := batchedImages.Shape().Dimensions[0]
	logits := graph.Reshape(batchedImages, batchSize, -1)
	numClasses := len(C10Labels)
	modelType := context.GetParamOr(ctx, "model", C10ValidModels[0])
	if modelType == "kan" {
		// Configuration of the KAN layer(s) use the context hyperparameters.
		// Re-scale logits to be from -1.0 to 1.0.
		logits = graph.AddScalar(graph.MulScalar(logits, 2), -1)
		//graph.ReduceMean(graph.ReduceVariance(logits, -1)).SetLogged("Mean input variance of the examples")
		logits = kan.New(ctx, logits, numClasses).Done()
	} else {
		// Configuration of the FNN layer(s) use the context hyperparameters.
		logits = fnn.New(ctx, logits, numClasses).Done()
	}
	logits.AssertDims(batchSize, numClasses)
	return []*graph.Node{logits}
}

const ParamCNNNormalization = "cnn_normalization"

func normalizeCNN(ctx *context.Context, logits *graph.Node) *graph.Node {
	normalizationType := context.GetParamOr(ctx, ParamCNNNormalization, "none")
	switch normalizationType {
	case "layer":
		if logits.Rank() == 2 {
			return layers.LayerNormalization(ctx, logits, -1).Done()
		} else if logits.Rank() == 4 {
			return layers.LayerNormalization(ctx, logits, 2, 3).Done()
		} else {
			return logits
		}
	case "batch":
		return batchnorm.New(ctx, logits, -1).Done()
	case "none", "":
		return logits
	default:
		exceptions.Panicf("invalid normalization type %q -- set it with parameter %q", normalizationType, ParamCNNNormalization)
		panic(nil)
	}
}

// C10ConvolutionModelGraph implements train.ModelFn and returns the logit Node, given the input image.
// It's a straight forward CNN (Convolution Neural Network) model.
//
// This is modeled after the Keras example in Kaggle:
// https://www.kaggle.com/code/ektasharma/simple-cifar10-cnn-keras-code-with-88-accuracy
// (Thanks @ektasharma)
func C10ConvolutionModelGraph(ctx *context.Context, spec any, inputs []*graph.Node) []*graph.Node {
	batchedImages := inputs[0]
	g := batchedImages.Graph()
	dtype := batchedImages.DType()
	batchSize := batchedImages.Shape().Dimensions[0]
	logits := batchedImages

	layerIdx := 0
	nextCtx := func(name string) *context.Context {
		newCtx := ctx.Inf("%03d_%s", layerIdx, name)
		layerIdx++
		return newCtx
	}

	logits = layers.Convolution(nextCtx("conv"), logits).Channels(32).KernelSize(3).PadSame().Done()
	logits.AssertDims(batchSize, 32, 32, 32)
	logits = activations.Relu(logits)
	logits = normalizeCNN(nextCtx("norm"), logits)
	logits = layers.Convolution(nextCtx("conv"), logits).Channels(32).KernelSize(3).PadSame().Done()
	logits = activations.Relu(logits)
	logits = normalizeCNN(nextCtx("norm"), logits)
	logits = graph.MaxPool(logits).Window(2).Done()
	logits = layers.DropoutNormalize(nextCtx("dropout"), logits, graph.Scalar(g, dtype, 0.3), true)
	logits.AssertDims(batchSize, 16, 16, 32)

	logits = layers.Convolution(nextCtx("conv"), logits).Channels(64).KernelSize(3).PadSame().Done()
	logits.AssertDims(batchSize, 16, 16, 64)
	logits = activations.Relu(logits)
	logits = normalizeCNN(nextCtx("norm"), logits)
	logits = layers.Convolution(nextCtx("conv"), logits).Channels(64).KernelSize(3).PadSame().Done()
	logits.AssertDims(batchSize, 16, 16, 64)
	logits = activations.Relu(logits)
	logits = normalizeCNN(nextCtx("norm"), logits)
	logits = graph.MaxPool(logits).Window(2).Done()
	logits = layers.DropoutNormalize(nextCtx("dropout"), logits, graph.Scalar(g, dtype, 0.5), true)
	logits.AssertDims(batchSize, 8, 8, 64)

	logits = layers.Convolution(nextCtx("conv"), logits).Channels(128).KernelSize(3).PadSame().Done()
	logits.AssertDims(batchSize, 8, 8, 128)
	logits = activations.Relu(logits)
	logits = normalizeCNN(nextCtx("norm"), logits)
	logits = layers.Convolution(nextCtx("conv"), logits).Channels(128).KernelSize(3).PadSame().Done()
	logits.AssertDims(batchSize, 8, 8, 128)
	logits = activations.Relu(logits)
	logits = normalizeCNN(nextCtx("norm"), logits)
	logits = graph.MaxPool(logits).Window(2).Done()
	logits = layers.DropoutNormalize(nextCtx("dropout"), logits, graph.Scalar(g, dtype, 0.5), true)
	logits.AssertDims(batchSize, 4, 4, 128)

	// Flatten logits, and we can use the usual FNN/KAN.
	logits = graph.Reshape(logits, batchSize, -1)
	logits = layers.Dense(nextCtx("dense"), logits, true, 128)
	logits = activations.Relu(logits)
	logits = normalizeCNN(nextCtx("norm"), logits)
	logits = layers.DropoutNormalize(nextCtx("dropout"), logits, graph.Scalar(g, dtype, 0.5), true)
	numClasses := len(C10Labels)
	logits = layers.Dense(nextCtx("dense"), logits, true, numClasses)
	return []*graph.Node{logits}
}
