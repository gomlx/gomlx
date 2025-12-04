package main

import (
	"fmt"

	"github.com/gomlx/gomlx/examples/adult"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers"
	"github.com/gomlx/gomlx/pkg/ml/layers/fnn"
	"github.com/gomlx/gomlx/pkg/ml/layers/kan"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule"
)

// Model outputs the logits (not the probabilities). The parameter inputs should contain 3 tensors:
//
// - categorical inputs, shaped  `(int64)[batch_size, len(VocabulariesFeatures)]`
// - continuous inputs, shaped `(float32)[batch_size, len(Quantiles)]`
// - weights: not currently used, but shaped `(float32)[batch_size, 1]`.
func Model(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec // Not used, since the dataset is always the same.
	g := inputs[0].Graph()
	dtype := inputs[1].DType() // From continuous features.
	ctx = ctx.In("model")

	// Use Cosine schedule of the learning rate, if hyperparameter is set to a value > 0.
	cosineschedule.New(ctx, g, dtype).FromContext().Done()

	categorical, continuous := inputs[0], inputs[1]
	batchSize := categorical.Shape().Dimensions[0]

	// Feature preprocessing:
	var allEmbeddings []*Node
	if *flagUseCategorical {
		// Embedding of categorical values, each with its own vocabulary.
		numCategorical := categorical.Shape().Dimensions[1]
		for catIdx := range numCategorical {
			// Take one column at a time of the categorical values.
			split := Slice(categorical, AxisRange(), AxisRange(catIdx, catIdx+1))
			// Embed it accordingly.
			embedCtx := ctx.In(fmt.Sprintf("categorical_%d_%s", catIdx, adult.Data.VocabulariesFeatures[catIdx]))
			vocab := adult.Data.Vocabularies[catIdx]
			vocabSize := len(vocab)
			embedding := layers.Embedding(embedCtx, split, ModelDType, vocabSize, *flagEmbeddingDim, false)
			embedding.AssertDims(batchSize, *flagEmbeddingDim)
			allEmbeddings = append(allEmbeddings, embedding)
		}
	}

	if *flagUseContinuous {
		// Piecewise-linear calibration of the continuous values. Each feature has its own number of quantiles.
		numContinuous := continuous.Shape().Dimensions[1]
		for contIdx := range numContinuous {
			// Take one column at a time of the continuous values.
			split := Slice(continuous, AxisRange(), AxisRange(contIdx, contIdx+1))
			featureName := adult.Data.QuantilesFeatures[contIdx]
			calibrationCtx := ctx.In(fmt.Sprintf("continuous_%d_%s", contIdx, featureName))
			quantiles := adult.Data.Quantiles[contIdx]
			layers.AssertQuantilesForPWLCalibrationValid(quantiles)
			calibrated := layers.PieceWiseLinearCalibration(calibrationCtx, split, Const(g, quantiles),
				*flagTrainableCalibration)
			calibrated.AssertDims(batchSize, 1) // 2-dim tensor, with batch size as the leading dimension.
			allEmbeddings = append(allEmbeddings, calibrated)
		}
	}
	logits := Concatenate(allEmbeddings, -1)
	logits.AssertDims(batchSize, -1)

	// Model itself is an FNN or a KAN.
	if context.GetParamOr(ctx, "kan", false) {
		// Use KAN, all configured by context hyperparameters. See createDefaultContext for defaults.
		logits = kan.New(ctx.In("kan"), logits, 1).Done()
	} else {
		// Normal FNN, all configured by context hyperparameters. See createDefaultContext for defaults.
		logits = fnn.New(ctx.In("fnn"), logits, 1).Done()
	}
	logits.AssertDims(batchSize, 1) // 2-dim tensor, with batch size as the leading dimension.
	return []*Node{logits}
}
