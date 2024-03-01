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
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
)

const (
	// AdamDefaultLearningRate is used by Adam if no learning rate is set.
	AdamDefaultLearningRate = 0.001

	// AdamDefaultScope is the default scope name for moments and step used by Adam.
	AdamDefaultScope = "AdamOptimizer"
)

// Adam optimization is a stochastic gradient descent method that is based on adaptive estimation of first-order and
// second-order moments. According to [Kingma et al., 2014](http://arxiv.org/abs/1412.6980),
// the method is "*computationally efficient, has little memory requirement, invariant to diagonal rescaling of
// gradients, and is well suited for problems that are large in terms of data/parameters*".
//
// It returns a configuration object that can be used to set its parameters. Once configured call IsNil, and it
// will return an optimizer.Interface.
func Adam() *AdamConfig {
	return &AdamConfig{
		scopeName:    AdamDefaultScope,
		learningRate: -1, // < 0 means use the default.
		beta1:        0.9,
		beta2:        0.999,
		epsilon:      1e-7,
		amsGrad:      false,
	}
}

// AdamConfig holds the configuration for an Adam configuration, create using Adam(), and once configured
// call Done to create an Adam based optimizer.Interface.
type AdamConfig struct {
	scopeName    string
	learningRate float64
	beta1, beta2 float64
	epsilon      float64
	amsGrad      bool
	adamax       bool    // Works as Adamax.
	weightDecay  float64 // Works as AdamW.
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
	dtype := loss.DType()

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
				o.applyAdamGraph(ctx, g, v, grads[varIdx], learningRate, beta1, debiasTermBeta1, beta2, debiasTermBeta2, epsilon)
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
func (o *adam) applyAdamGraph(ctx *context.Context, g *Graph, v *context.Variable, grad *Node,
	learningRate, beta1, debiasTermBeta1, beta2, debiasTermBeta2, epsilon *Node) {
	m1Var, m2Var := o.getMomentVariables(ctx, v)
	moment1, moment2 := m1Var.ValueGraph(g), m2Var.ValueGraph(g)

	// Notice beta1, beta2 and debias terms are of the dtype of the loss. Since a model can have operations
	// with different dtypes, we need to convert it to the variable's dtype (same as the moment). We create
	// this closure to perform this.
	varDType := moment1.DType()
	castToVar := func(n *Node) *Node {
		if n.DType() == varDType {
			// No-op.
			return n
		}
		return ConvertType(n, varDType)
	}

	// Do gradient step with momentum.
	moment1 = Add(
		Mul(castToVar(beta1), moment1),
		Mul(OneMinus(castToVar(beta1)), grad))
	m1Var.SetValueGraph(moment1)
	debiasedMoment1 := Mul(moment1, castToVar(debiasTermBeta1))

	var denominator *Node
	if o.config.adamax {
		// Adamax
		moment2 = Max(
			Mul(castToVar(beta2), moment2),
			castToVar(Abs(grad))) // L-infinity norm. Notice Abs() can change dtypes for complex numbers.
		m2Var.SetValueGraph(moment2)
		denominator = Add(moment2, castToVar(epsilon))
	} else {
		// Normal Adam.
		moment2 = Add(
			Mul(castToVar(beta2), moment2),
			Mul(OneMinus(castToVar(beta2)), Square(grad)))
		m2Var.SetValueGraph(moment2)
		debiasedMoment2 := Mul(moment2, castToVar(debiasTermBeta2))
		denominator = Add(Sqrt(debiasedMoment2), castToVar(epsilon))
	}

	value := v.ValueGraph(g)
	stepDirection := Div(debiasedMoment1, denominator)
	if o.config.weightDecay > 0 {
		stepDirection = Add(stepDirection, MulScalar(value, o.config.weightDecay))
	}
	updated := Sub(value, Mul(castToVar(learningRate), stepDirection))

	// Update variables.
	v.SetValueGraph(updated)
	return
}

// getMomentVariables returns the moment variables corresponding to the trainable variable give.
//
// If g is not nil, it creates the moments variables if they don't exist. Otherwise, it just tries to
// fetch the presumably existing variables.
func (o *adam) getMomentVariables(ctx *context.Context, trainable *context.Variable) (m1, m2 *context.Variable) {
	originalScope := trainable.Scope()
	originalName := trainable.Name()
	scopePath := fmt.Sprintf("%s%s%s", context.ScopeSeparator, o.config.scopeName, originalScope)
	m1Name := fmt.Sprintf("%s_1st_moment", originalName)
	m2Name := fmt.Sprintf("%s_2nd_moment", originalName)
	shape := trainable.Shape()
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
