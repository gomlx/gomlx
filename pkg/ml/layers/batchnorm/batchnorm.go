/*
 *	Copyright 2023 Jan Pfeifer
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

// Package batchnorm implements a batch normalization layer, and associated tools.
// It's a very common normalization technique that greatly facilitates training of deeper models.
//
// See details and examples in New.
//
// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
package batchnorm

import (
	"strings"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/initializers"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

// Config for a batch normalization layer.
// Create it with New, set the desired parameters, and when all is set, call Done.
type Config struct {
	ctx                       *context.Context
	x                         *Node
	featureAxis               int
	momentum, epsilon         float64
	center, scale             bool
	newScope                  bool
	trainable, frozenAverages bool
	useBackendInference       bool
}

const (
	// BatchNormalizationScopeName is used as sub-scope for all batch normalization variables.
	BatchNormalizationScopeName = "batch_normalization"
)

// New creates builder performs a batch normalization layer on the input. It includes a scaling and offset factor,
// and normalization over the batch entries.
// It maintains a moving average mean and variance of the inputs which is later used during inference.
//
// featureAxis is the axis over which **not to normalize**: this will normalize over the other dimensions,
// calculating the mean and variance by reducing all other dimensions.
// E.g: if your input is `[batch_size, features]` you should use featureAxis=1 (same as -1) to normalize over
// the batch; if your input is an image of shape `[batch_size, height, width, channels]` you should use
// featureAxis=3 (same as -1) to normalize over the batch and all the pixels, so each channel is
// normalized differently, but normalization happens over all the pixes of the whole batch.
//
// Notice the difference between LayerNormalization, that normalizes over the feature dimensions, as opposed
// to the batch dimension.
//
// To ease setting its parameters, it returns a Config object for configuration. Once it is
// set up call `Config.Done` and it will return the normalized x. Browse through Config to see the
// capabilities and the defaults.
//
// Batch normalization behaves differently during training and inference: during training it normalizes over
// the batch (so it likely won't work well for very small batch sizes), and in inference, it normalizes
// using the collected moving average of the mean and variance.
//
// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
//
// See also UpdateAverages to update the running averages after the training of the model -- or just
// before evaluations. Because during training the target averages of the mean and variances are moving (as the
// model changes), they are often biases and suboptimal, UpdateAverages fixes that and often provides some
// significant gains.
//
// FutureWork:
// 1. Support padding by not normalizing parts that weren't touched.
// 2. Support selection of multiple features axes.
func New(ctx *context.Context, x *Node, featureAxis int) *Config {
	return &Config{
		ctx:                 ctx,
		x:                   x,
		featureAxis:         featureAxis,
		momentum:            0.99,
		epsilon:             1e-03,
		center:              true,
		scale:               true,
		newScope:            true,
		trainable:           true,
		useBackendInference: true,
	}
}

// Momentum sets the moment of the moving averages collected for the mean and variance of the values.
// New maintains moving averages for the mean and variance during training.
// This averaged mean and variance is used during inference for normalization.
// The default is 0.99.
//
// Notice Keras default is 0.99 (the one we use), but PyTorch's default of 0.9.
//
// This has no effect if one sets `Trainable(false)`.
func (builder *Config) Momentum(value float64) *Config {
	builder.momentum = value
	return builder
}

// Epsilon is a small float added to variance to avoid dividing by zero.
// It defaults to 1e-3.
//
// Notice Keras default is 1e-3 (the one we use), but PyTorch's default of 1e-05.
func (builder *Config) Epsilon(value float64) *Config {
	builder.epsilon = value
	return builder
}

// Center defines whether the batch normalization tries to center the input by adding a learned offset.
// Default to true.
//
// This is also called the β (beta) parameter, and referred to as a "learnable offset".
func (builder *Config) Center(value bool) *Config {
	builder.center = value
	return builder
}

// Scale defines whether the batch normalization tries to scale the input by adding a learned scale. Default to true.
//
// This is also called the	γ (gamma) parameter.
func (builder *Config) Scale(value bool) *Config {
	builder.scale = value
	return builder
}

// CurrentScope configures New not to create a new sub-scope named BatchNormalizationScopeName for its variables.
// This allows more control on scope names, but it breaks things that rely on batch normalization variables to be under
// BatchNormalizationScopeName (e.g.: ResetWeights).
func (builder *Config) CurrentScope() *Config {
	builder.newScope = false
	return builder
}

// Trainable defines whether the batch normalization is trainable.
// If set to `false` it is frozen, and none of its parameters are changeable.
// The default is `true`.
//
// Independent of the value set here, if the context is not set for training (
// see `context.Context.IsTraining()`) like during evaluation and inference,
// the Config will generate code for inference only.
func (builder *Config) Trainable(trainable bool) *Config {
	builder.trainable = trainable
	return builder
}

// FrozenAverages defines whether the moving averages for mean and variance should be kept frozen
// for this layer.
//
// This is useful in transfer learning when the sub-model being incorporated was trained on a different
// distribution, and we don't want to impact that.
func (builder *Config) FrozenAverages(frozen bool) *Config {
	builder.frozenAverages = frozen
	return builder
}

// UseBackendInference uses a backend version of batch normalization inference.
// The alternative is a manually defined batch normalization inference, which is differentiable.
// Only used if training is false.
//
// The default is true.
func (builder *Config) UseBackendInference(value bool) *Config {
	builder.useBackendInference = value
	return builder
}

// Done finishes configuring the New and generates the graph computation to normalize the input.
func (builder *Config) Done() *Node {
	x := builder.x
	g := x.Graph()
	dtype := x.DType()

	// Set about batch normalization usage.
	if builder.trainable && !builder.frozenAverages {
		builder.ctx.InAbsPath(context.RootScope).SetParam(AveragesUpdatesTriggerParam, true)
	}

	// Creates new scope for variables.
	if builder.newScope {
		builder.ctx = builder.ctx.In("batch_normalization")
	}
	ctx := builder.ctx

	featureAxis := AdjustAxisToOperandRank(x, builder.featureAxis)
	featureDim := x.Shape().Dimensions[featureAxis]

	// Scale and offset applied to the normalized value.
	var scale, offset *Node
	varShape := shapes.Make(dtype, featureDim)
	var scaleVar *context.Variable
	if builder.scale {
		scaleVar = ctx.WithInitializer(initializers.One).VariableWithShape("scale", varShape).SetTrainable(true)
		scale = scaleVar.ValueGraph(g)
	} else {
		scale = Ones(g, varShape)
	}

	if builder.center {
		offsetVar := ctx.WithInitializer(initializers.Zero).VariableWithShape("offset", varShape).SetTrainable(true)
		offset = offsetVar.ValueGraph(g)
	} else {
		offset = Zeros(g, varShape)
	}

	// Normalization moving average of the mean and variance.
	meanAverageVar := ctx.WithInitializer(initializers.Zero).VariableWithShape("mean", varShape).SetTrainable(false)
	varianceAverageVar := ctx.WithInitializer(initializers.One).
		VariableWithShape("variance", varShape).
		SetTrainable(false)
	weightVar := ctx.WithInitializer(initializers.Zero).VariableWithShape("avg_weight", varShape).SetTrainable(false)

	var normalized *Node
	mean, variance := meanAverageVar.ValueGraph(g), varianceAverageVar.ValueGraph(g)
	averagesUpdatePhase := context.GetGraphParamOr(ctx, g, train.BatchNormalizationUpdatePhase, -1)
	if averagesUpdatePhase >= 0 {
		if builder.frozenAverages {
			normalized = builder.directBatchNormGraph(x, scale, offset, mean, variance)
		} else {
			var batchMean, batchVariance *Node
			batchMean, batchVariance = builder.batchMeanAndVariance(x)
			builder.updateMeanAndVariance(ctx, g, batchMean, batchVariance, meanAverageVar, varianceAverageVar, weightVar)
			if averagesUpdatePhase > 0 {
				// Use updated mean, variance to normalize.
				mean, variance = meanAverageVar.ValueGraph(g), varianceAverageVar.ValueGraph(g)
			} else {
				mean, variance = batchMean, batchVariance
			}
			if builder.useBackendInference {
				normalized = InternalBatchNormForInference(x, scale, offset, mean, variance, float32(builder.epsilon), featureAxis)
			} else {
				normalized = builder.directBatchNormGraph(x, scale, offset, mean, variance)
			}
		}

	} else if builder.trainable && ctx.IsTraining(g) {
		if builder.frozenAverages {
			// Here we want to use the frozen mean and variance, and not the batch one, since these are different
			// quantities and in inference we will use the frozen ones.
			normalized = builder.directBatchNormGraph(x, scale, offset, mean, variance)

		} else if builder.useBackendInference {
			// Training: take batch's mean and variance and use it to update averages.
			var batchMean, batchVariance *Node
			normalized, batchMean, batchVariance = InternalBatchNormForTraining(x, scale, offset, float32(builder.epsilon), featureAxis)
			builder.updateMeanAndVariance(ctx, g, batchMean, batchVariance, meanAverageVar, varianceAverageVar, weightVar)
		} else {
			batchMean, batchVariance := builder.batchMeanAndVariance(x)
			builder.updateMeanAndVariance(ctx, g, batchMean, batchVariance, meanAverageVar, varianceAverageVar, weightVar)
			normalized = builder.directBatchNormGraph(x, scale, offset, batchMean, batchVariance)
		}

	} else {
		// Inference: use stored mean/variance.
		if builder.useBackendInference {
			// Uses dedicated op.
			normalized = InternalBatchNormForInference(x, scale, offset, mean, variance, float32(builder.epsilon), featureAxis)

		} else {
			// Direct batch normalization: it is differentiable.
			normalized = builder.directBatchNormGraph(x, scale, offset, mean, variance)
		}
	}

	// Add regularization to scale.
	if scaleVar != nil {
		if l2 := context.GetParamOr(ctx, regularizers.ParamL2, 0.0); l2 > 0 {
			reg := regularizers.L2(l2)
			reg(ctx, g, scaleVar)
		}
	}
	return normalized
}

func (builder *Config) batchMeanAndVariance(x *Node) (batchMean, batchVariance *Node) {
	featureAxis := AdjustAxisToOperandRank(x, builder.featureAxis)
	nonFeatureAxes := make([]int, 0, x.Rank()-1)
	for ii := range x.Rank() {
		if ii != featureAxis {
			nonFeatureAxes = append(nonFeatureAxes, ii)
		}
	}
	batchMean = ReduceAndKeep(x, ReduceMean, nonFeatureAxes...)
	batchVariance = ReduceMean(Square(Sub(x, batchMean)), nonFeatureAxes...)
	batchMean = Reshape(batchMean, batchVariance.Shape().Dimensions...)
	batchMean = StopGradient(batchMean)
	batchVariance = StopGradient(batchVariance)
	return
}

// directBatchNormGraph calculates the batch normalized x without using XLA -- so it's differentiable.
func (builder *Config) directBatchNormGraph(x, scale, offset, mean, variance *Node) *Node {
	featureAxis := AdjustAxisToOperandRank(x, builder.featureAxis)
	featureDim := x.Shape().Dimensions[featureAxis]

	dims := xslices.SliceWithValue(x.Rank(), 1)
	dims[featureAxis] = featureDim
	expandedMean := Reshape(mean, dims...)
	expandedVariance := Reshape(variance, dims...)
	normalized := Div(
		Sub(x, expandedMean),
		Sqrt(AddScalar(expandedVariance, builder.epsilon)))
	if builder.scale {
		expandedScale := Reshape(scale, dims...)
		normalized = Mul(normalized, expandedScale)
	}
	if builder.center {
		expandedOffset := Reshape(offset, dims...)
		normalized = Add(normalized, expandedOffset)
	}
	return normalized
}

// batchNormUpdater implements context.PerStepUpdateHandler, which is called by optimizers at every step
type batchNormUpdater struct {
	builder                            *Config
	meanAverageVar, varianceAverageVar *context.Variable
	weightVar                          *context.Variable
	mean, variance                     *Node
}

// updateMeanAndVariance values that will be used in inference later. It's a moving average, where weight is how many
// examples have been seen so far -- it's incremented at every step.
func (builder *Config) updateMeanAndVariance(
	ctx *context.Context,
	graph *Graph,
	batchMean, batchVariance *Node,
	meanAverageVar, varianceAverageVar, weightVar *context.Variable,
) {
	_ = ctx
	if builder.frozenAverages {
		// We are not changing the averages.
		return
	}
	momentum := Scalar(batchMean.Graph(), batchMean.DType(), builder.momentum)

	weight := OnePlus(weightVar.ValueGraph(graph))
	weightVar.SetValueGraph(weight)
	debiasedMomentum := Min(momentum, OneMinus(Reciprocal(weight)))

	meanAverage := meanAverageVar.ValueGraph(graph)
	meanAverage = Add(
		Mul(debiasedMomentum, meanAverage),         // Current mean.
		Mul(OneMinus(debiasedMomentum), batchMean)) // New batch mean.
	meanAverageVar.SetValueGraph(meanAverage)

	varianceAverage := varianceAverageVar.ValueGraph(graph)
	varianceAverage = Add(
		Mul(debiasedMomentum, varianceAverage),         // Current variance.
		Mul(OneMinus(debiasedMomentum), batchVariance)) // New variance.
	varianceAverageVar.SetValueGraph(varianceAverage)
}

// ResetWeights reset the weights of the moving averages, forcing them to be reinitialized to 0.
// It searches for all variables under scope named "batch_normalization"
//
// It is a no-op if no batch-normalization was used.
//
// Usually this method is not used directly, instead use UpdateAverages.
func ResetWeights(ctx *context.Context) error {
	suffix := "/" + BatchNormalizationScopeName
	for v := range ctx.IterVariablesInScope() {
		if strings.HasSuffix(v.Scope(), suffix) && v.Name() == "avg_weight" {
			zeros := tensors.FromShape(v.Shape())
			err := v.SetValue(zeros)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

const (
	// AveragesUpdatesTriggerParam is a boolean parameter set in case batch normalization was used.
	// See UpdateAverages.
	AveragesUpdatesTriggerParam = "batch_normalization_averages_updates_trigger"
)

// UpdateAverages resets the weights of the moving averages and recalculate them over the
// given oneEpochDS dataset and the trainer.
// It uses the context assigned to the trainer.
//
// It is a no-op if no batch-normalization was used.
//
// The oneEpochDS dataset (typically, the same as a training data evaluation dataset) should be a 1-epoch training
// data dataset, and it can use evaluation batch sizes.
// If oneEpochDS is nil, it disabled the updating of the averages.
//
// It returns whether batch normalization was used and averages were updated.
//
// An error is only returned if it attempts to update the averages.
//
// See discussions:
// - https://www.mindee.com/blog/batch-normalization
// - https://discuss.pytorch.org/t/batch-norm-instability/32159/14
func UpdateAverages(trainer *train.Trainer, oneEpochDS train.Dataset) (bool, error) {
	ctx := trainer.Context()
	if !context.GetParamOr(ctx, AveragesUpdatesTriggerParam, false) {
		// No-op.
		return false, nil
	}

	err := ResetWeights(ctx)
	if err != nil {
		return true, err
	}
	return true, trainer.BatchNormalizationAveragesUpdate(oneEpochDS)
}
