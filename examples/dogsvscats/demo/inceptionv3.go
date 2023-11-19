package main

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/models/inceptionv3"
	timage "github.com/gomlx/gomlx/types/tensor/image"
)

// This file implements the baseline CNN model, including the FNN layers on top.

// InceptionV3ModelGraph uses an optionally pre-trained inception model.
//
// Results on validation after training 1000 steps, inception pre-trained weights used:
// - no scaling (from 0.0 to 1.0): 90.4% accuracy
// - with Keras scale (from -1.0 to 1.0): 90.6% accuracy
//
// Results if we don't use the pre-trained weights (it can probably get much better with more training):
// - no scaling (from 0.0 to 1.0): 62.5% accuracy
// - with Keras scale (from -1.0 to 1.0): 61.8% accuracy
func InceptionV3ModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec           // Not needed.
	image := inputs[0] // Images scaled from 0.0 to 1.0
	channelsConfig := timage.ChannelsLast
	image = inceptionv3.PreprocessImage(image, 1.0, channelsConfig) // Adjust image to format used by Inception.

	var preTrainedPath string
	if *flagInceptionPreTrained {
		// Use pre-trained
		preTrainedPath = *flagDataDir
		err := inceptionv3.DownloadAndUnpackWeights(*flagDataDir) // Only downloads/unpacks the first time.
		AssertNoError(err)
	}
	logits := inceptionv3.BuildGraph(ctx, image).
		PreTrained(preTrainedPath).
		SetPooling(inceptionv3.MaxPooling).
		Trainable(*flagInceptionFineTuning).Done()
	if !*flagInceptionFineTuning {
		logits = StopGradient(logits) // We don't want to train the inception model.
	}

	logits = FnnOnTop(ctx, logits)
	logits = layers.DenseWithBias(ctx.In("readout"), logits, 1)
	return []*Node{logits}
}
