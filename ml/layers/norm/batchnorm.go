// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package norm implements normalization layers for GoMLX, including Batch, Layer, and RMS normalization.
package norm

import (
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/regularizer"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
	"github.com/gomlx/gomlx/ml/train"
)

// BatchNormBuilder is a builder for configuring a batch normalization layer.
// Create it with BatchNorm, set the desired parameters, and when all is set, call Done.
type BatchNormBuilder struct {
	scope                     *model.Scope
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

// BatchNorm creates a builder for a batch normalization layer on the input. It includes a scaling and offset factor,
// and normalization over the batch entries.
// It maintains a moving average mean and variance of the inputs which is later used during inference.
//
// featureAxis is the axis over which **not to normalize**: this will normalize over the other dimensions,
// calculating the mean and variance by reducing all other dimensions.
// E.g: if your input is `[batch_size, features]` you should use featureAxis=1 (same as -1) to normalize over
// the batch; if your input is an image of shape `[batch_size, height, width, channels]` you should use
// featureAxis=3 (same as -1) to normalize over the batch and all the pixels, so each channel is
// normalized differently, but normalization happens over all the pixels of the whole batch.
//
// Notice the difference between LayerNorm, that normalizes over the feature dimensions, as opposed
// to the batch dimension.
//
// To ease setting its parameters, it returns a BatchNormBuilder object for configuration. Once it is
// set up call `BatchNormBuilder.Done` and it will return the normalized x. Browse through BatchNormBuilder to see the
// capabilities and the defaults.
//
// Batch normalization behaves differently during training and inference: during training it normalizes over
// the batch (so it likely won't work well for very small batch sizes), and in inference, it normalizes
// using the collected moving average of the mean and variance.
//
// Based on paper "Batch Normalization: Accelerating Deep Network Training by Reducing
// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
//
// See also UpdateBatchNormAverages to update the running averages after the training of the model -- or just
// before evaluations. Because during training the target averages of the mean and variances are moving (as the
// model changes), they are often biased and suboptimal, UpdateBatchNormAverages fixes that and often provides some
// significant gains.
//
// FutureWork:
// 1. Support padding by not normalizing parts that weren't touched.
// 2. Support selection of multiple features axes.
func BatchNorm(scope *model.Scope, x *Node, featureAxis int) *BatchNormBuilder {
	return &BatchNormBuilder{
		scope:               scope,
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

// Momentum sets the momentum of the moving averages collected for the mean and variance of the values.
// BatchNorm maintains moving averages for the mean and variance during training.
// This averaged mean and variance is used during inference for normalization.
// The default is 0.99.
//
// Notice Keras default is 0.99 (the one we use), but PyTorch's default of 0.9.
//
// This has no effect if one sets `Trainable(false)`.
func (builder *BatchNormBuilder) Momentum(value float64) *BatchNormBuilder {
	builder.momentum = value
	return builder
}

// Epsilon is a small float added to variance to avoid dividing by zero.
// It defaults to 1e-3.
//
// Notice Keras default is 1e-3 (the one we use), but PyTorch's default of 1e-05.
func (builder *BatchNormBuilder) Epsilon(value float64) *BatchNormBuilder {
	builder.epsilon = value
	return builder
}

// Center defines whether the batch normalization tries to center the input by adding a learned offset.
// Default to true.
//
// This is also called the β (beta) parameter, and referred to as a "learnable offset".
func (builder *BatchNormBuilder) Center(value bool) *BatchNormBuilder {
	builder.center = value
	return builder
}

// Scale defines whether the batch normalization tries to scale the input by adding a learned scale. Default to true.
//
// This is also called the γ (gamma) parameter.
func (builder *BatchNormBuilder) Scale(value bool) *BatchNormBuilder {
	builder.scale = value
	return builder
}

// CurrentScope configures BatchNorm not to create a new sub-scope named BatchNormalizationScopeName for its variables.
// This allows more control on scope names, but it breaks things that rely on batch normalization variables to be under
// BatchNormalizationScopeName (e.g.: ResetBatchNormWeights).
func (builder *BatchNormBuilder) CurrentScope() *BatchNormBuilder {
	builder.newScope = false
	return builder
}

// Trainable defines whether the batch normalization is trainable.
// If set to `false` it is frozen, and none of its parameters are changeable.
// The default is `true`.
//
// Independent of the value set here, if the scope is not set for training (
// see `model.Store.IsTraining()`) like during evaluation and inference,
// the BatchNormBuilder will generate code for inference only.
func (builder *BatchNormBuilder) Trainable(trainable bool) *BatchNormBuilder {
	builder.trainable = trainable
	return builder
}

// FrozenAverages defines whether the moving averages for mean and variance should be kept frozen
// for this layer.
//
// This is useful in transfer learning when the sub-model being incorporated was trained on a different
// distribution, and we don't want to impact that.
func (builder *BatchNormBuilder) FrozenAverages(frozen bool) *BatchNormBuilder {
	builder.frozenAverages = frozen
	return builder
}

// UseBackendInference uses a backend version of batch normalization inference.
// The alternative is a manually defined batch normalization inference, which is differentiable.
// Only used if training is false.
//
// The default is true.
func (builder *BatchNormBuilder) UseBackendInference(value bool) *BatchNormBuilder {
	builder.useBackendInference = value
	return builder
}

// Done finishes configuring the BatchNorm and generates the graph computation to normalize the input.
func (builder *BatchNormBuilder) Done() *Node {
	x := builder.x
	g := x.Graph()
	dtype := x.DType()

	// Set about batch normalization usage.
	if builder.trainable && !builder.frozenAverages {
		builder.scope.Store().Scope(model.RootScopePath).SetParam(AveragesUpdatesTriggerParam, true)
	}

	// Creates new scope for variables.
	if builder.newScope {
		builder.scope = builder.scope.In("batch_normalization")
	}
	scope := builder.scope

	featureAxis := MustAdjustAxis(builder.featureAxis, x)
	featureDim := x.Shape().Dimensions[featureAxis]

	// Scale and offset applied to the normalized value.
	var scale, offset *Node
	varShape := shapes.Make(dtype, featureDim)
	var scaleVar *model.Variable
	if builder.scale {
		scaleVar = scope.WithInitializer(initializer.One).VariableWithShape("scale", varShape).SetTrainable(true)
		scale = scaleVar.NodeValue(g)
	} else {
		scale = Ones(g, varShape)
	}

	if builder.center {
		offsetVar := scope.WithInitializer(initializer.Zero).VariableWithShape("offset", varShape).SetTrainable(true)
		offset = offsetVar.NodeValue(g)
	} else {
		offset = Zeros(g, varShape)
	}

	// Normalization moving average of the mean and variance.
	meanAverageVar := scope.WithInitializer(initializer.Zero).VariableWithShape("mean", varShape).SetTrainable(false)
	varianceAverageVar := scope.WithInitializer(initializer.One).
		VariableWithShape("variance", varShape).
		SetTrainable(false)
	weightVar := scope.WithInitializer(initializer.Zero).VariableWithShape("avg_weight", varShape).SetTrainable(false)

	var normalized *Node
	mean, variance := meanAverageVar.NodeValue(g), varianceAverageVar.NodeValue(g)
	averagesUpdatePhase := model.GetGraphParamOr(scope, g, train.BatchNormalizationUpdatePhase, -1)
	if averagesUpdatePhase >= 0 {
		if builder.frozenAverages {
			normalized = builder.directBatchNormGraph(x, scale, offset, mean, variance)
		} else {
			var batchMean, batchVariance *Node
			batchMean, batchVariance = builder.batchMeanAndVariance(x)
			builder.updateMeanAndVariance(scope, g, batchMean, batchVariance, meanAverageVar, varianceAverageVar, weightVar)
			if averagesUpdatePhase > 0 {
				// Use updated mean, variance to normalize.
				mean, variance = meanAverageVar.NodeValue(g), varianceAverageVar.NodeValue(g)
			} else {
				mean, variance = batchMean, batchVariance
			}
			if builder.useBackendInference {
				normalized = InternalBatchNormForInference(x, scale, offset, mean, variance, float32(builder.epsilon), featureAxis)
			} else {
				normalized = builder.directBatchNormGraph(x, scale, offset, mean, variance)
			}
		}

	} else if builder.trainable && scope.IsTraining(g) {
		if builder.frozenAverages {
			// Here we want to use the frozen mean and variance, and not the batch one, since these are different
			// quantities and in inference we will use the frozen ones.
			normalized = builder.directBatchNormGraph(x, scale, offset, mean, variance)

		} else if builder.useBackendInference {
			// Training: take batch's mean and variance and use it to update averages.
			var batchMean, batchVariance *Node
			normalized, batchMean, batchVariance = InternalBatchNormForTraining(x, scale, offset, float32(builder.epsilon), featureAxis)
			builder.updateMeanAndVariance(scope, g, batchMean, batchVariance, meanAverageVar, varianceAverageVar, weightVar)
		} else {
			batchMean, batchVariance := builder.batchMeanAndVariance(x)
			builder.updateMeanAndVariance(scope, g, batchMean, batchVariance, meanAverageVar, varianceAverageVar, weightVar)
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
		if l2 := model.GetParamOr(scope, regularizer.ParamL2, 0.0); l2 > 0 {
			reg := regularizer.L2(l2)
			reg(g, scaleVar)
		}
	}
	return normalized
}

func (builder *BatchNormBuilder) batchMeanAndVariance(x *Node) (batchMean, batchVariance *Node) {
	featureAxis := MustAdjustAxis(builder.featureAxis, x)
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
func (builder *BatchNormBuilder) directBatchNormGraph(x, scale, offset, mean, variance *Node) *Node {
	featureAxis := MustAdjustAxis(builder.featureAxis, x)
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

// updateMeanAndVariance values that will be used in inference later. It's a moving average, where weight is how many
// examples have been seen so far -- it's incremented at every step.
func (builder *BatchNormBuilder) updateMeanAndVariance(
	scope *model.Scope,
	graph *Graph,
	batchMean, batchVariance *Node,
	meanAverageVar, varianceAverageVar, weightVar *model.Variable,
) {
	_ = scope
	if builder.frozenAverages {
		// We are not changing the averages.
		return
	}
	momentum := Scalar(batchMean.Graph(), batchMean.DType(), builder.momentum)

	weight := OnePlus(weightVar.NodeValue(graph))
	weightVar.SetNodeValue(weight)
	debiasedMomentum := Min(momentum, OneMinus(Reciprocal(weight)))

	meanAverage := meanAverageVar.NodeValue(graph)
	meanAverage = Add(
		Mul(debiasedMomentum, meanAverage),         // Current mean.
		Mul(OneMinus(debiasedMomentum), batchMean)) // New batch mean.
	meanAverageVar.SetNodeValue(meanAverage)

	varianceAverage := varianceAverageVar.NodeValue(graph)
	varianceAverage = Add(
		Mul(debiasedMomentum, varianceAverage),         // Current variance.
		Mul(OneMinus(debiasedMomentum), batchVariance)) // New variance.
	varianceAverageVar.SetNodeValue(varianceAverage)
}

// ResetBatchNormWeights resets the weights of the moving averages, forcing them to be reinitialized to 0.
// It searches for all variables under scope named "batch_normalization"
//
// This is not a graph building function, it sets the materialized values to zero.
//
// It is a no-op if no batch-normalization was used.
//
// Usually this method is not used directly, instead use UpdateBatchNormAverages.
func ResetBatchNormWeights(backend compute.Backend, store *model.Store) error {
	suffix := "/" + BatchNormalizationScopeName
	for v := range store.IterVariables() {
		scope, name := model.SplitPath(v.Path())
		if strings.HasSuffix(scope, suffix) && name == "avg_weight" {
			zeros, err := tensors.FromShapeForBackend(backend, 0, v.Shape())
			if err != nil {
				return err
			}
			err = v.SetValue(zeros)
			if err != nil {
				return err
			}
			err = v.SetValue(zeros)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

const (
	// AveragesUpdatesTriggerParam is a boolean parameter set in case batch normalization was used.
	// See UpdateBatchNormAverages.
	AveragesUpdatesTriggerParam = "batch_normalization_averages_updates_trigger"
)

// UpdateBatchNormAverages resets the weights of the moving averages and recalculates them over the
// given oneEpochDS dataset and the trainer.
// It uses the scope assigned to the trainer.
//
// It is a no-op if no batch-normalization was used.
//
// The oneEpochDS dataset (typically, the same as a training data evaluation dataset) should be a 1-epoch training
// data dataset, and it can use evaluation batch sizes.
// If oneEpochDS is nil, it disables the updating of the averages.
//
// It returns whether batch normalization was used and averages were updated.
//
// An error is only returned if it attempts to update the averages.
//
// See discussions:
// - https://www.mindee.com/blog/batch-normalization
// - https://discuss.pytorch.org/t/batch-norm-instability/32159/14
func UpdateBatchNormAverages(trainer *train.Trainer, oneEpochDS train.Dataset) (bool, error) {
	rootScope := trainer.Store().RootScope()
	if !model.GetParamOr(rootScope, AveragesUpdatesTriggerParam, false) {
		// No-op.
		return false, nil
	}

	err := ResetBatchNormWeights(trainer.Backend(), trainer.Store())
	if err != nil {
		return true, err
	}
	return true, trainer.BatchNormalizationAveragesUpdate(oneEpochDS)
}
