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

package layers

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/shapes"
)

// BatchNormBuilder is a helper to build a batch normalization computation. Create it with BatchNormalization, set the
// desired parameters, and when all is set, call Done.
type BatchNormBuilder struct {
	ctx                        *context.Context
	x                          *Node
	featureAxis                int
	momentum, epsilon          float64
	center, scale              bool
	newScope                   bool
	trainable, useXlaInference bool
}

// BatchNormalization performs a batch normalization layer on the input. It includes a scaling and offset factor,
// and normalization over the batch entries. It maintains a moving average mean and variance of the inputs
// which is later used during inference.
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
// FutureWork:
// 1. Support padding by not normalizing parts that weren't touched.
// 2. Support selection of multiple features axes.
func BatchNormalization(ctx *context.Context, x *Node, featureAxis int) *BatchNormBuilder {
	return &BatchNormBuilder{
		ctx:         ctx,
		x:           x,
		featureAxis: featureAxis,
		momentum:    0.99,
		epsilon:     1e-03,
		center:      true,
		scale:       true,
		newScope:    true,
		trainable:   true,
	}
}

// Momentum sets the moment of the moving averages collected for the mean and variance of the values.
// BatchNormalization maintains moving averages for the mean and variance during training.
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
// This is also called the	γ (gamma) parameter.
func (builder *BatchNormBuilder) Scale(value bool) *BatchNormBuilder {
	builder.scale = value
	return builder
}

// CurrentScope configures the convolution not to create a sub-scope for the kernel weights it needs,
// and instead use the current one provided in Convolution.
//
// By default, Convolution will create a sub-scope named "conv".
func (builder *BatchNormBuilder) CurrentScope() *BatchNormBuilder {
	builder.newScope = false
	return builder
}

// Trainable defines whether the batch normalization is trainable.
// If set to `false` it is frozen, and none of its parameters are changeable.
// The default is `true`.
//
// Independent of the value set here, if the context is not set for training (
// see `context.Context.IsTraining()`) like during evaluation and inference,
// the BatchNormBuilder will generate code for inference only.
func (builder *BatchNormBuilder) Trainable(trainable bool) *BatchNormBuilder {
	builder.trainable = trainable
	return builder
}

// UseXLAInference uses a dedicated XLA op for batch normalization inference.
// XLA offers it, but it is not necessarily faster, and it is **not differentiable**.
// The default is false.
func (builder *BatchNormBuilder) UseXLAInference(value bool) *BatchNormBuilder {
	builder.useXlaInference = value
	return builder
}

// Done finishes configuring the BatchNormalization and generates the graph computation to normalize the input.
func (builder *BatchNormBuilder) Done() *Node {
	x := builder.x
	g := x.Graph()
	dtype := x.DType()

	// Creates new scope for variables.
	if builder.newScope {
		builder.ctx = builder.ctx.In("batch_normalization")
	}
	ctx := builder.ctx

	featureAxis := AdjustAxis(x, builder.featureAxis)
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
	varianceAverageVar := ctx.WithInitializer(initializers.One).VariableWithShape("variance", varShape).SetTrainable(false)
	weightVar := ctx.WithInitializer(initializers.Zero).VariableWithShape("avg_weight", varShape).SetTrainable(false)

	var normalized *Node
	if builder.trainable && ctx.IsTraining(g) {
		// Training: take batch's mean and variance and use it to update averages.
		var batchMean, batchVariance *Node
		normalized, batchMean, batchVariance = BatchNormTrainingXLA(x, scale, offset, float32(builder.epsilon), featureAxis)
		builder.updateMeanAndVariance(ctx, g, batchMean, batchVariance, meanAverageVar, varianceAverageVar, weightVar)

	} else {
		// Inference: use stored mean/variance.
		mean, variance := meanAverageVar.ValueGraph(g), varianceAverageVar.ValueGraph(g)
		if builder.useXlaInference {
			// Uses dedicated op.
			normalized = BatchNormInferenceXLA(x, scale, offset, mean, variance, float32(builder.epsilon), featureAxis)

		} else {
			// Direct batch normalization: it is differentiable.
			normalized = builder.directBatchNormGraph(x, scale, offset, mean, variance)
		}
	}

	// Add regularization to scale.
	if scaleVar != nil {
		if l2any, found := ctx.GetParam(ParamL2Regularization); found {
			l2 := l2any.(float64)
			if l2 > 0 {
				l2Node := ConstAs(x, l2)
				AddL2Regularization(ctx, l2Node, scaleVar.ValueGraph(g))
			}
		}
	}

	return normalized
}

// directBatchNormGraph calculates the batch normalized x without using XLA -- so it's differentiable.
func (builder *BatchNormBuilder) directBatchNormGraph(x, scale, offset, mean, variance *Node) *Node {
	featureAxis := AdjustAxis(x, builder.featureAxis)
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
	builder                            *BatchNormBuilder
	meanAverageVar, varianceAverageVar *context.Variable
	weightVar                          *context.Variable
	mean, variance                     *Node
}

// updateMeanAndVariance values that will be used in inference later. It's a moving average, where weight is how many
// examples have been seen so far -- it's incremented at every step.
func (builder *BatchNormBuilder) updateMeanAndVariance(ctx *context.Context, graph *Graph, batchMean, batchVariance *Node, meanAverageVar, varianceAverageVar, weightVar *context.Variable) {
	_ = ctx
	momentum := ConstAs(batchMean, builder.momentum)

	weight := weightVar.ValueGraph(graph)
	weight = Add(weight, OnesLike(weight))
	weightVar.SetValueGraph(weight)
	debiasedMomentum := Min(momentum, OneMinus(Inverse(weight)))

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
