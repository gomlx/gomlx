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

package optimizers

import (
	"fmt"
	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

const (
	// AdamDefaultLearningRate is used by Adam if no learning rate is set.
	AdamDefaultLearningRate = 0.001

	// AdamDefaultScope is the default scope name for moments and step used by Adam.
	AdamDefaultScope = "AdamOptimizer"

	// ParamAdamEpsilon can be used to configure the default value of epsilon. It must be a float64.
	ParamAdamEpsilon = "adam_epsilon"

	// ParamAdamDType can be used to specify the dtype to be used by Adam's temporary variables and computations.
	// The default or if set to empty is to use the same dtype as the value of the loss provided.
	// This was created for the case of training with `float16` or `bfloat16`, which is not enough resolution
	// for Adam calculations.
	// Valid values: "" (empty), "float32", "float64".
	ParamAdamDType = "adam_dtype"

	// ParamAdamWeightDecay defaults to 0.0. See AdamConfig.WeightDecay.
	ParamAdamWeightDecay = "adam_weight_decay"
)

// Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and
// second-order moments. According to [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
// the method is "*computationally efficient, has little memory requirement, invariant to diagonal rescaling of
// gradients, and is well suited for problems that are large in terms of data/parameters*".
//
// It returns a configuration object that can be used to set its parameters. Once configured call IsNil, and it
// will return an optimizer.Interface.
//
// See [AdamConfig.FromContext] to configure it from the context hyperparameters.
//
// Clipping of the gradient updates available by setting the context hyperparameter [ParamClipStepByValue]("clip_step_by_value").
func Adam() *AdamConfig {
	return &AdamConfig{
		scopeName:    AdamDefaultScope,
		learningRate: -1, // < 0 means use the default.
		beta1:        0.9,
		beta2:        0.999,
		epsilon:      1e-7,
		amsGrad:      false,
		dtype:        dtypes.InvalidDType,
	}
}

// AdamConfig holds the configuration for an Adam configuration, create using Adam(), and once configured
// call Done to create an Adam based optimizer.Interface.
type AdamConfig struct {
	scopeName    string
	dtype        dtypes.DType // If invalid, use the loss type instead.
	learningRate float64
	beta1, beta2 float64
	epsilon      float64
	amsGrad      bool
	adamax       bool    // Works as Adamax.
	weightDecay  float64 // Works as AdamW.
}

// FromContext will configure Adam with hyperparameters set in the given context.
// E.g.: "adam_epsilon" (see [ParamAdamEpsilon]) is used to set [AdamConfig.Epsilon].
func (c *AdamConfig) FromContext(ctx *context.Context) *AdamConfig {
	c.Epsilon(context.GetParamOr(ctx, ParamAdamEpsilon, c.epsilon))
	dtypeStr := context.GetParamOr(ctx, ParamAdamDType, "")
	if dtypeStr != "" {
		dtype, err := dtypes.DTypeString(dtypeStr)
		if err != nil || !dtype.IsFloat() {
			Panicf("Invalid hyperparameter value %s=%q", ParamAdamDType, dtypeStr)
		}
		c.DType(dtype)
	}
	c.WeightDecay(context.GetParamOr(ctx, ParamAdamWeightDecay, 0.0))
	return c
}

// Scope defines the top-level scope to use to store the 1st and 2nd order moments of the gradients and the step number
// used by Adam optimizer. Generally this doesn't need to be changed, but if one is using multiple schedules,
// potentially with different loss functions (so the moments should be different), one can change.
//
// It defaults to AdamDefaultScope.
func (c *AdamConfig) Scope(name string) *AdamConfig {
	c.scopeName = name
	return c
}

// DType sets the dtype to use for Adam calculation and temporary variables.
// This can be useful if training using `float16`, which is not enough resolution for Adam calculations in some cases.
//
// If set to `shapes.InvalidDType` it will use the dtype of the `loss` used to optimize.
//
// This can also be set from context using [ParamAdamDType]("adam_dtype") hyperparameter.
func (c *AdamConfig) DType(dtype dtypes.DType) *AdamConfig {
	c.dtype = dtype
	return c
}

// LearningRate sets the base learning rate as a floating point value -- eventually converted to the same dtype as the loss.
//
// Default is either the value of ParamLearningRate ("learning_rate") global parameter in Context if defined, or 0.001 if not.
func (c *AdamConfig) LearningRate(value float64) *AdamConfig {
	c.learningRate = value
	return c
}

// Betas sets the two moving averages constants (exponential decays). They default to 0.9 and 0.999.
func (c *AdamConfig) Betas(beta1, beta2 float64) *AdamConfig {
	c.beta1, c.beta2 = beta1, beta2
	return c
}

// Epsilon used on the denominator as a small constant for stability.
// For low precision numbers like float16, try a larger value here, like 1e-3.
func (c *AdamConfig) Epsilon(epsilon float64) *AdamConfig {
	c.epsilon = epsilon
	return c
}

// Adamax configure Adam to use a L-infinity (== max, which gives the name) for
// the second moment, instead of L2, as described in the same Adam paper.
func (c *AdamConfig) Adamax() *AdamConfig {
	c.adamax = true
	return c
}

// WeightDecay configure optimizer to work as AdamW, with the given static weight decay.
// This is because L2 regularization doesn't work well with Adam.
//
// Defaults to the value given in the AdamWeightDecay hyperparameter.
//
// TODO: (1) Allow certain variables to be excluded from weight decay (e.g: biases); (2) Allow dynamically calculated weight decay.
func (c *AdamConfig) WeightDecay(weightDecay float64) *AdamConfig {
	c.weightDecay = weightDecay
	return c
}

/* TODO: implement AMSGrad.
// AMSGrad defines whether to use the AMSGrad variant described in the paper "On the Convergence of Adam and Beyond".
// Defaults to false.
// Although the description in https://towardsdatascience.com/adam-latest-trends-in-deep-learning-optimization-6be9a291375c
// suggests while it is correct, it has little impact in practice.
func (c *AdamConfig) AMSGrad(amsGrad bool) *AdamConfig {
	c.amsGrad = amsGrad
	return c
}
*/

// Done will finish the configuration and construct an optimizer.Interface that implements Adam to specification.
func (c *AdamConfig) Done() Interface {
	return &adam{config: c}
}

// adam implements the Adam algorithm as an optimizer.Interface.
type adam struct {
	config *AdamConfig
}

// UpdateGraph builds the graph to update the weights for one training step.
// It implements optimizers.Interface.
func (o *adam) UpdateGraph(ctx *context.Context, g *Graph, loss *Node) {
	if !loss.Shape().IsScalar() {
		Panicf("optimizer requires a scalar loss to optimize, got loss.shape=%s instead", loss.Shape())
		return
	}
	dtype := o.config.dtype
	if dtype == dtypes.InvalidDType {
		dtype = loss.DType()
	}

	// Set up learning-rate.
	lrValue := o.config.learningRate
	if lrValue < 0 {
		lrValue = context.GetParamOr(ctx, ParamLearningRate, AdamDefaultLearningRate)
	}
	lrVar := LearningRateVar(ctx, dtype, lrValue)
	learningRate := lrVar.ValueGraph(g)

	_ = IncrementGlobalStepGraph(ctx, g, dtype) // LoopStep, not used by this optimizer, but updated.
	adamStep := IncrementGlobalStepGraph(ctx.In(o.config.scopeName), g, dtype)
	beta1 := Const(g, shapes.CastAsDType(o.config.beta1, dtype))
	debiasTermBeta1 := Inverse(OneMinus(Pow(beta1, adamStep)))
	beta2 := Const(g, shapes.CastAsDType(o.config.beta2, dtype))
	debiasTermBeta2 := Inverse(OneMinus(Pow(beta2, adamStep)))
	epsilon := Const(g, shapes.CastAsDType(o.config.epsilon, dtype))

	grads := ctx.BuildTrainableVariablesGradientsGraph(loss)
	if len(grads) == 0 {
		Panicf("Context.BuildTrainableVariablesGradientsGraph returned 0 gradients, are there any trainable variables ?")
	}

	// Apply gradient one variable at a time.
	numTrainable := len(grads)
	varIdx := 0
	ctx.EnumerateVariables(func(v *context.Variable) {
		if v.Trainable && v.InUseByGraph(g) {
			if varIdx < numTrainable {
				o.applyAdamGraph(ctx, g, v, dtype, grads[varIdx], learningRate, beta1, debiasTermBeta1, beta2, debiasTermBeta2, epsilon)
			}
			varIdx++
		}
	})
	if varIdx != numTrainable {
		Panicf("Context.BuildTrainableVariablesGradientsGraph returned gradients for %d variables, but "+
			"Adam only sees %d variables -- were new variables created in between ?",
			numTrainable, varIdx)
	}

	return // Errors reported in Context or Graph.
}

// applyAdamGraph calculates variable and its 1st and 2nd order moments updates.
// If `Adamax` is set, we use instead moment2 to store the L-infinity (the max) of the gradient.
func (o *adam) applyAdamGraph(ctx *context.Context, g *Graph, v *context.Variable, dtype dtypes.DType, grad *Node,
	learningRate, beta1, debiasTermBeta1, beta2, debiasTermBeta2, epsilon *Node) {
	m1Var, m2Var := o.getMomentVariables(ctx, v, dtype)
	moment1, moment2 := m1Var.ValueGraph(g), m2Var.ValueGraph(g)

	// Adam runs on a fixed dtype -- defaults to the dtype of the loss, but it can be configured.
	// We convert the grad to the dtype used by Adam for its computation.
	if grad.DType() != dtype {
		grad = ConvertDType(grad, dtype)
	}
	grad = ClipNaNsInGradients(ctx, grad)

	// Do gradient step with momentum.
	moment1 = Add(
		Mul(beta1, moment1),
		Mul(OneMinus(beta1), grad))
	m1Var.SetValueGraph(moment1)
	debiasedMoment1 := Mul(moment1, debiasTermBeta1)

	var denominator *Node
	if o.config.adamax {
		// Adamax
		moment2 = Max(
			Mul(beta2, moment2),
			Abs(grad)) // L-infinity norm. Notice Abs() can change dtypes for complex numbers.
		m2Var.SetValueGraph(moment2)
		denominator = Add(moment2, epsilon)

	} else {
		// Normal Adam.
		moment2 = Add(
			Mul(beta2, moment2),
			Mul(OneMinus(beta2), Square(grad)))
		m2Var.SetValueGraph(moment2)
		debiasedMoment2 := Mul(moment2, debiasTermBeta2)
		denominator = Add(Sqrt(debiasedMoment2), epsilon)
	}

	value := v.ValueGraph(g)
	if value.DType() != dtype {
		value = ConvertDType(value, dtype)
	}
	stepDirection := Mul(learningRate, debiasedMoment1)
	stepDirection = Div(stepDirection, denominator)

	// Weight decay: also scaled by the learning rate.
	if o.config.weightDecay > 0 {
		stepDirection = Add(stepDirection, Mul(learningRate, MulScalar(value, o.config.weightDecay)))
	}

	// Clip step value, if requested.
	clipByValue := context.GetParamOr(ctx, ParamClipStepByValue, 0.0)
	if clipByValue > 0 {
		stepDirection = ClipScalar(stepDirection, -clipByValue, clipByValue)
	}

	// Update variable.
	updated := Sub(value, stepDirection)
	updated = ClipNaNsInUpdates(ctx, value, updated) // If selected, clip NaN updates.
	if v.Shape().DType != dtype {
		// Convert back to the variable type.
		updated = ConvertDType(updated, v.Shape().DType)
	}
	v.SetValueGraph(updated)
	return
}

// getMomentVariables returns the moment variables corresponding to the trainable variable give.
//
// If g is not nil, it creates the moments variables if they don't exist. Otherwise, it just tries to
// fetch the presumably existing variables.
func (o *adam) getMomentVariables(ctx *context.Context, trainable *context.Variable, dtype dtypes.DType) (m1, m2 *context.Variable) {
	originalScope := trainable.Scope()
	originalName := trainable.Name()
	scopePath := fmt.Sprintf("%s%s%s", context.ScopeSeparator, o.config.scopeName, originalScope)
	m1Name := fmt.Sprintf("%s_1st_moment", originalName)
	m2Name := fmt.Sprintf("%s_2nd_moment", originalName)
	shape := trainable.Shape().Clone()
	shape.DType = dtype
	ctx = ctx.Checked(false) // It shouldn't matter if it's the first time or not creating the variable.
	m1 = ctx.InAbsPath(scopePath).WithInitializer(initializers.Zero).VariableWithShape(m1Name, shape).SetTrainable(false)
	m2 = ctx.InAbsPath(scopePath).WithInitializer(initializers.Zero).VariableWithShape(m2Name, shape).SetTrainable(false)
	return
}

// Clear all optimizer variables.
// It implements optimizers.Interface.
func (o *adam) Clear(ctx *context.Context) {
	ctxAdam := ctx.In(o.config.scopeName)
	ctxAdam.DeleteVariablesInScope()
}
