// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dogsvscats

import (
	. "github.com/gomlx/gomlx/core/graph"
	timage "github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/examples/inceptionv3"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
)

// InceptionV3ModelPrep is executed before training: it downloads the inceptionv3 weights.
func InceptionV3ModelPrep(scope *model.Scope, dataDir string, checkpointHandler *checkpoint.Handler) error {
	scope.SetParam("data_dir", dataDir)
	if model.GetParamOr(scope, "inception_pretrained", true) {
		if err := inceptionv3.DownloadAndUnpackWeights(dataDir); err != nil {
			return err
		}
	}
	return nil
}

// InceptionV3ModelGraph uses an optionally pre-trained inception model.
//
// Results on validation after training 1000 steps, inception pre-trained weights used:
// - no scaling (from 0.0 to 1.0): 90.4% accuracy
// - with Keras scale (from -1.0 to 1.0): 90.6% accuracy
//
// Results if we don't use the pre-trained weights (it can probably get much better with more training):
// - no scaling (from 0.0 to 1.0): 62.5% accuracy
// - with Keras scale (from -1.0 to 1.0): 61.8% accuracy
func InceptionV3ModelGraph(scope *model.Scope, images *Node) *Node {
	scope = scope.In("model") // Create the model by default under the "/model" scope.
	channelsConfig := timage.ChannelsLast
	images = inceptionv3.PreprocessImage(images, 1.0, channelsConfig) // Adjust image to format used by Inception.
	dataDir := model.GetParamOr(scope, "data_dir", ".")
	var preTrainedPath string
	if model.GetParamOr(scope, "inception_pretrained", true) {
		// Use pre-trained
		preTrainedPath = dataDir
	}
	logits := inceptionv3.BuildGraph(scope, images).
		PreTrained(preTrainedPath).
		SetPooling(inceptionv3.MaxPooling).
		Trainable(model.GetParamOr(scope, "inception_finetuning", false)).
		Done()
	logits = fnn.New(scope.In("fnn"), logits, 1).Done()
	return logits
}
