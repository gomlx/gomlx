package dogsvscats

import (
	"github.com/gomlx/gomlx/examples/inceptionv3"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	timage "github.com/gomlx/gomlx/types/tensors/images"
)

// InceptionV3ModelPrep is executed before training: it downloads the inceptionv3 weights.
func InceptionV3ModelPrep(ctx *context.Context, dataDir string, checkpoint *checkpoints.Handler) {
	ctx.SetParam("data_dir", dataDir)
	if context.GetParamOr(ctx, "inception_pretrained", true) {
		must.M(inceptionv3.DownloadAndUnpackWeights(dataDir))
	}
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
func InceptionV3ModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	ctx = ctx.In("model") // Create the model by default under the "/model" scope.
	_ = spec              // Not needed.
	images := inputs[0]   // Images scaled from 0.0 to 1.0
	channelsConfig := timage.ChannelsLast
	images = inceptionv3.PreprocessImage(images, 1.0, channelsConfig) // Adjust image to format used by Inception.
	dataDir := context.GetParamOr(ctx, "data_dir", ".")
	var preTrainedPath string
	if context.GetParamOr(ctx, "inception_pretrained", true) {
		// Use pre-trained
		preTrainedPath = dataDir
	}
	logits := inceptionv3.BuildGraph(ctx, images).
		PreTrained(preTrainedPath).
		SetPooling(inceptionv3.MaxPooling).
		Trainable(context.GetParamOr(ctx, "inception_finetuning", false)).
		Done()
	logits = fnn.New(ctx.In("fnn"), logits, 1).Done()
	return []*Node{logits}
}
