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

package metrics

import (
	"fmt"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

// takeFirstFn wraps the given `metricFn` with a function that takes a single node for labels and predictions as
// opposed to slices of nodes.
func takeFirstFn(
	metricFn func(ctx *context.Context, labels, predictions []*Node) *Node,
) func(*context.Context, *Node, *Node) *Node {
	return func(ctx *context.Context, labels, predictions *Node) *Node {
		return metricFn(ctx, []*Node{labels}, []*Node{predictions})
	}
}

// takeLabelsMaskWeightPredictionsFn wraps the given `metricFn` with a function that takes labels, mask, weights and predictions.
func takeLabelsMaskWeightPredictionsFn(
	metricFn func(ctx *context.Context, labels, predictions []*Node) *Node,
) func(ctx *context.Context, labels, mask, weights, predictions *Node) *Node {
	return func(ctx *context.Context, labels, mask, weights, predictions *Node) *Node {
		return metricFn(ctx, []*Node{labels, mask, weights}, []*Node{predictions})
	}
}

func TestBinaryAccuracyGraph(t *testing.T) {
	manager := graphtest.BuildTestBackend()
	ctx := context.New()
	accuracyExec := context.MustNewExec(manager, ctx, takeFirstFn(BinaryAccuracyGraph))
	labels, probs := []float32{0, 1, 0, 1, 0, 1}, []float32{0.1, 0.1, 0.5, 0.5, 0.8, 0.8}
	results := accuracyExec.MustExec(labels, probs)
	got := results[0].Value().(float32)
	if got != float32(2.0/6.0) {
		t.Errorf("TestBinaryAccuracyGraph: wanted 2/6=0.333..., got %.4f", got)
	}
}

func TestNewMeanBinaryAccuracy(t *testing.T) {
	manager := graphtest.BuildTestBackend()
	ctx := context.New().Checked(false)
	accMetric := NewMeanBinaryAccuracy("accuracy", "acc")
	accExec := context.MustNewExec(manager, ctx, func(ctx *context.Context, labels, predictions *Node) *Node {
		return accMetric.UpdateGraph(ctx, []*Node{labels}, []*Node{predictions})
	})

	// First batch:
	labels, probs := []float32{0, 1, 0, 1, 0, 1}, []float32{0.1, 0.1, 0.5, 0.5, 0.8, 0.8}
	results := accExec.MustExec(labels, probs)
	meanAccT := results[0]
	meanAcc := meanAccT.Value().(float32)
	assert.Equal(t, float32(2.0/6.0), meanAcc, "MeanBinaryAccuracy")

	// List and check variables.
	fmt.Println("Variables:")
	ctx.EnumerateVariables(func(v *context.Variable) {
		fmt.Printf("\t%s / %s=%s\n", v.Scope(), v.Name(), v.MustValue())
	})

	metricScope := ctx.In(Scope).In(accMetric.ScopeName()).Scope()
	totalVar := ctx.GetVariableByScopeAndName(metricScope, "total")
	require.NotNilf(t, totalVar, "Variable \"total\" was not created in %s / total", metricScope)
	total := totalVar.MustValue().Value().(float32)
	assert.Equal(t, float32(2), total, "MeanBinaryAccuracy total value")

	weightVar := ctx.GetVariableByScopeAndName(metricScope, "weight")
	require.NotNilf(t, weightVar, "Variable \"weight\" was not created in %s / total", metricScope)
	weight := weightVar.MustValue().Value().(float32)
	assert.Equal(t, float32(6), weight, "MeanBinaryAccuracy weight value")

	// Second batch:
	labels, probs = []float32{0, 0, 0, 1, 1, 1}, []float32{0.1, 0.4, 0.7, 0.8, 0.9, 0.09}
	results = accExec.MustExec(labels, probs)
	meanAccT = results[0]
	meanAcc = meanAccT.Value().(float32)
	assert.Equal(t, float32(6.0/12.0), meanAcc, "MeanBinaryAccuracy after batch #2")

	// Zeros the state.
	accMetric.Reset(ctx)
	total = totalVar.MustValue().Value().(float32)
	weight = weightVar.MustValue().Value().(float32)
	assert.Zero(t, total, "Expected total variable to be 0 after Reset()")
	assert.Zero(t, weight, "Expected weight variable to be 0 after Reset()")
}

func TestBinaryLogitsAccuracyGraph(t *testing.T) {
	manager := graphtest.BuildTestBackend()
	ctx := context.New()
	accuracyExec := context.MustNewExec(manager, ctx, takeFirstFn(BinaryLogitsAccuracyGraph))
	labels, logits := []float32{0, 1, 0, 1, 0, 1}, []float32{-0.1, -0.1, 0, 0, 0.2, 10.0}
	results := accuracyExec.MustExec(labels, logits)
	got, _ := results[0].Value().(float32)
	assert.Equal(t, float32(2.0/6.0), got, "TestBinaryAccuracyGraph")
}

func TestSparseCategoricalAccuracyGraph(t *testing.T) {
	manager := graphtest.BuildTestBackend()
	ctx := context.New()
	{
		accuracyExec := context.MustNewExec(manager, ctx, takeFirstFn(SparseCategoricalAccuracyGraph))
		labels, logits := [][]int{{0}, {1}, {2}}, [][]float32{
			{0, 0, 1},     // Tie, should be a miss.
			{-2, -1, -3},  // Correct, even if negative.
			{100, 90, 80}, // Wrong even if positive.
		}
		results := accuracyExec.MustExec(labels, logits)
		got, _ := results[0].Value().(float32)
		assert.Equal(t, float32(1.0/3.0), got, "TestSparseCategoricalAccuracyGraph")
	}
	{
		accuracyExec := context.MustNewExec(
			manager,
			ctx,
			takeLabelsMaskWeightPredictionsFn(SparseCategoricalAccuracyGraph),
		)
		labels := [][]int{{0}, {1}, {0}, {2}}
		mask := []bool{true, true, false, true}
		weights := []float32{1.0, 2.0, 100.0, 0.5}
		logits := [][]float32{
			{0, 0, 1},      // Tie, should be a miss.
			{-2, -1, -3},   // Correct, even if negative.
			{-100, 20, 80}, // Disabled by mask.
			{100, 90, 80},  // Wrong even if positive.
		}
		results := accuracyExec.MustExec(labels, mask, weights, logits)
		got, _ := results[0].Value().(float32)
		assert.Equal(t, float32((2.0*1.0)/(1.0+2.0+0.5)), got, "TestSparseCategoricalAccuracyGraph[with mask/weights]")
	}
}
