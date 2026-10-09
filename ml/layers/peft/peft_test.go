// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package peft

import (
	"math"
	"testing"

	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestLoRAUsesScopeParametersAndFreezesBase(t *testing.T) {
	store := model.NewStore()
	store.RootScope().SetParams(map[string]any{ParamRank: 1, ParamAlpha: float32(2)})
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		scope = scope.In("projection")
		weight := scope.VariableWithValue("weight", [][]float32{{1, 2}, {3, 4}})
		return New(scope, input, weight, nil).Done()
	})
	if err != nil {
		t.Fatal(err)
	}
	base := call(t, exec)
	if !closeEnough(base, []float32{7, 10}) {
		t.Fatalf("base forward = %v", base)
	}
	weight := store.GetVariable("/projection/weight")
	a, b := store.GetVariable("/projection/lora/A"), store.GetVariable("/projection/lora/B")
	if weight.Trainable || a == nil || b == nil || !a.Trainable || !b.Trainable {
		t.Fatal("LoRA must freeze the base and create only trainable A/B variables")
	}
	if err := a.SetValue(tensors.FromFlatDataAndDimensions([]float32{1, 1}, 2, 1)); err != nil {
		t.Fatal(err)
	}
	if err := b.SetValue(tensors.FromFlatDataAndDimensions([]float32{1, 1}, 1, 2)); err != nil {
		t.Fatal(err)
	}
	if updated := call(t, exec); closeEnough(updated, base) {
		t.Fatal("trained LoRA update did not affect the forward pass")
	}
}

func TestRankZeroIsTheOriginalProjection(t *testing.T) {
	store := model.NewStore()
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		scope = scope.In("projection")
		weight := scope.VariableWithValue("weight", [][]float32{{1, 2}, {3, 4}})
		return Apply(scope, input, weight, nil)
	})
	if err != nil {
		t.Fatal(err)
	}
	if got := call(t, exec); !closeEnough(got, []float32{7, 10}) {
		t.Fatalf("rank-zero forward = %v", got)
	}
	if !store.GetVariable("/projection/weight").Trainable || store.GetVariable("/projection/lora/A") != nil {
		t.Fatal("rank zero must not freeze the base or create adapter variables")
	}
}

func TestLoRAGradientsExcludeFrozenBase(t *testing.T) {
	store := model.NewStore()
	store.RootScope().SetParam(ParamRank, 1)
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) []*graph.Node {
		scope = scope.In("projection")
		weight := scope.VariableWithValue("weight", [][]float32{{1, 2}, {3, 4}})
		output := New(scope, input, weight, nil).Done()
		return scope.BuildTrainableVariablesGradientsGraph(graph.ReduceAllSum(output))
	})
	if err != nil {
		t.Fatal(err)
	}
	gradients, err := exec.Call([][]float32{{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(gradients) != 2 {
		t.Fatalf("gradient count = %d, want A/B only", len(gradients))
	}
}

func TestNF4QLoRAUsesScopeParameters(t *testing.T) {
	weight, err := QuantizeNF4Double(2, 4, 2, 2, []float32{1, -1, 0.5, -0.5, -1, 1, -0.5, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	store := model.NewStore()
	store.RootScope().SetParam(ParamRank, 1)
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		return NewNF4(scope.In("projection"), input, weight, nil).Done()
	})
	if err != nil {
		t.Fatal(err)
	}
	output := call(t, exec)
	for index, want := range []float32{-1, 1, -0.5, 0.5} {
		if math.Abs(float64(output[index]-want)) > 0.02 {
			t.Fatalf("output[%d] = %v, want %v", index, output[index], want)
		}
	}
	if store.GetVariable("/projection/qlora/lora/A") == nil || store.GetVariable("/projection/qlora/lora/B") == nil {
		t.Fatal("QLoRA must create adapter variables in the model store")
	}
}

func TestQuantizeNF4RejectsOddOutput(t *testing.T) {
	if _, err := QuantizeNF4(2, 3, 2, make([]float32, 6)); err == nil {
		t.Fatal("odd NF4 output must be rejected")
	}
}

func TestInvalidScopeConfigurationFailsDuringGraphBuilding(t *testing.T) {
	store := model.NewStore()
	store.RootScope().SetParam(ParamRank, -1)
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		scope = scope.In("projection")
		weight := scope.VariableWithValue("weight", [][]float32{{1, 2}, {3, 4}})
		return New(scope, input, weight, nil).Done()
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := exec.Call([][]float32{{1, 2}}); err == nil {
		t.Fatal("invalid graph configuration must raise an exception")
	}
}

func call(t *testing.T, exec *model.Exec) []float32 {
	t.Helper()
	outputs, err := exec.Call([][]float32{{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	values, err := tensors.CopyFlatData[float32](outputs[0])
	if err != nil {
		t.Fatal(err)
	}
	return values
}

func closeEnough(left, right []float32) bool {
	if len(left) != len(right) {
		return false
	}
	for index := range left {
		if math.Abs(float64(left[index]-right[index])) > 1e-5 {
			return false
		}
	}
	return true
}

func TestFNNWithAutomaticPEFT(t *testing.T) {
	store := model.NewStore()
	store.RootScope().SetParams(map[string]any{
		ParamAdapter: "test_lora",
		ParamRank:    2,
		ParamAlpha:   float32(4),
	})
	backend := gobackend.GetBackend()
	exec, err := model.NewExec(backend, store, func(scope *model.Scope, input *graph.Node) []*graph.Node {
		output := fnn.New(scope.In("fnn"), input, 2).
			NumHiddenLayers(1, 4).
			Done()
		loss := graph.ReduceAllSum(output)
		return scope.BuildTrainableVariablesGradientsGraph(loss)
	})
	require.NoError(t, err)

	input := [][]float32{{1, 2, 3}}
	grads, err := exec.Call(input)
	require.NoError(t, err)

	// Base weights must be frozen:
	wHidden := store.GetVariable("/fnn/fnn_hidden_layer_0/weights")
	wOutput := store.GetVariable("/fnn/fnn_output_layer/weights")
	require.NotNil(t, wHidden)
	require.NotNil(t, wOutput)
	assert.False(t, wHidden.Trainable)
	assert.False(t, wOutput.Trainable)

	// LoRA adapters must exist and be trainable:
	aHidden := store.GetVariable("/fnn/fnn_hidden_layer_0/test_lora/A")
	bHidden := store.GetVariable("/fnn/fnn_hidden_layer_0/test_lora/B")
	aOutput := store.GetVariable("/fnn/fnn_output_layer/test_lora/A")
	bOutput := store.GetVariable("/fnn/fnn_output_layer/test_lora/B")
	require.NotNil(t, aHidden)
	require.NotNil(t, bHidden)
	require.NotNil(t, aOutput)
	require.NotNil(t, bOutput)
	assert.True(t, aHidden.Trainable)
	assert.True(t, bHidden.Trainable)
	assert.True(t, aOutput.Trainable)
	assert.True(t, bOutput.Trainable)

	// Gradients must ONLY be for the 4 LoRA adapter variables (2 layers x A/B):
	assert.Len(t, grads, 4)
}

func TestFNNTargetModules(t *testing.T) {
	store := model.NewStore()
	store.RootScope().SetParams(map[string]any{
		ParamAdapter:       "task_lora",
		ParamRank:          2,
		ParamTargetModules: "fnn_hidden_layer_0",
	})
	backend := gobackend.GetBackend()
	exec, err := model.NewExec(backend, store, func(scope *model.Scope, input *graph.Node) []*graph.Node {
		output := fnn.New(scope.In("fnn"), input, 2).
			NumHiddenLayers(1, 4).
			Done()
		loss := graph.ReduceAllSum(output)
		return scope.BuildTrainableVariablesGradientsGraph(loss)
	})
	require.NoError(t, err)

	input := [][]float32{{1, 2, 3}}
	grads, err := exec.Call(input)
	require.NoError(t, err)

	// Both layers should have base weights frozen:
	assert.False(t, store.GetVariable("/fnn/fnn_hidden_layer_0/weights").Trainable)
	assert.False(t, store.GetVariable("/fnn/fnn_output_layer/weights").Trainable)

	// Target module (hidden layer 0) has LoRA:
	assert.NotNil(t, store.GetVariable("/fnn/fnn_hidden_layer_0/task_lora/A"))
	assert.NotNil(t, store.GetVariable("/fnn/fnn_hidden_layer_0/task_lora/B"))

	// Non-target module (output layer) does NOT have LoRA:
	assert.Nil(t, store.GetVariable("/fnn/fnn_output_layer/task_lora/A"))
	assert.Nil(t, store.GetVariable("/fnn/fnn_output_layer/task_lora/B"))

	// Gradients should only be for hidden layer 0's A and B:
	assert.Len(t, grads, 2)
}

func TestFNNAdapterDisabled(t *testing.T) {
	store := model.NewStore()
	store.RootScope().SetParams(map[string]any{
		ParamAdapter: "off",
		ParamRank:    2,
	})
	backend := gobackend.GetBackend()
	exec, err := model.NewExec(backend, store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		return fnn.New(scope.In("fnn"), input, 2).
			NumHiddenLayers(1, 4).
			Done()
	})
	require.NoError(t, err)

	_, err = exec.Call([][]float32{{1, 2, 3}})
	require.NoError(t, err)

	// Base weights must be frozen:
	assert.False(t, store.GetVariable("/fnn/fnn_hidden_layer_0/weights").Trainable)
	assert.False(t, store.GetVariable("/fnn/fnn_output_layer/weights").Trainable)

	// No adapter variables should exist:
	assert.Nil(t, store.GetVariable("/fnn/fnn_hidden_layer_0/off/A"))
	assert.Nil(t, store.GetVariable("/fnn/fnn_hidden_layer_0/lora/A"))
}

func TestFNNMultiAdapter(t *testing.T) {
	store := model.NewStore()
	store.RootScope().SetParam(ParamRank, 2)
	backend := gobackend.GetBackend()

	// Task A
	store.RootScope().SetParam(ParamAdapter, "task_a")
	execA, err := model.NewExec(backend, store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		return fnn.New(scope.In("fnn"), input, 2).NumHiddenLayers(1, 4).Done()
	})
	require.NoError(t, err)
	_, err = execA.Call([][]float32{{1, 2, 3}})
	require.NoError(t, err)

	// Task B
	store.RootScope().SetParam(ParamAdapter, "task_b")
	execB, err := model.NewExec(backend, store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		return fnn.New(scope.In("fnn"), input, 2).NumHiddenLayers(1, 4).Done()
	})
	require.NoError(t, err)
	_, err = execB.Call([][]float32{{1, 2, 3}})
	require.NoError(t, err)

	// Both adapters should exist side-by-side in store:
	assert.NotNil(t, store.GetVariable("/fnn/fnn_hidden_layer_0/task_a/A"))
	assert.NotNil(t, store.GetVariable("/fnn/fnn_hidden_layer_0/task_a/B"))
	assert.NotNil(t, store.GetVariable("/fnn/fnn_hidden_layer_0/task_b/A"))
	assert.NotNil(t, store.GetVariable("/fnn/fnn_hidden_layer_0/task_b/B"))
}

func TestDenseWithAutomaticPEFT(t *testing.T) {
	store := model.NewStore()
	store.RootScope().SetParams(map[string]any{
		ParamAdapter: "dense_lora",
		ParamRank:    2,
	})
	backend := gobackend.GetBackend()
	exec, err := model.NewExec(backend, store, func(scope *model.Scope, input *graph.Node) []*graph.Node {
		output := layers.Dense(scope, input, true, 4)
		return scope.BuildTrainableVariablesGradientsGraph(graph.ReduceAllSum(output))
	})
	require.NoError(t, err)

	grads, err := exec.Call([][]float32{{1, 2}})
	require.NoError(t, err)

	assert.False(t, store.GetVariable("/dense/weights").Trainable)
	assert.NotNil(t, store.GetVariable("/dense/dense_lora/A"))
	assert.NotNil(t, store.GetVariable("/dense/dense_lora/B"))
	assert.Len(t, grads, 2) // A and B only
}

func TestAttentionWithAutomaticPEFT(t *testing.T) {
	store := model.NewStore()
	store.RootScope().SetParams(map[string]any{
		ParamAdapter:       "attn_lora",
		ParamRank:          2,
		ParamTargetModules: "query,value",
	})
	backend := gobackend.GetBackend()
	exec, err := model.NewExec(backend, store, func(scope *model.Scope, input *graph.Node) []*graph.Node {
		output := attention.MultiHeadAttention(scope, input, input, input, 2, 4).Done()
		return scope.BuildTrainableVariablesGradientsGraph(graph.ReduceAllSum(output))
	})
	require.NoError(t, err)

	// input: [batch=1, seq=3, dim=8]
	input := [][][]float32{{{1, 2, 3, 4, 5, 6, 7, 8}, {2, 3, 4, 5, 6, 7, 8, 9}, {3, 4, 5, 6, 7, 8, 9, 10}}}
	grads, err := exec.Call(input)
	require.NoError(t, err)

	// Query and Value should have LoRA adapters:
	assert.NotNil(t, store.GetVariable("/MultiHeadAttention/query/dense/attn_lora/A"))
	assert.NotNil(t, store.GetVariable("/MultiHeadAttention/query/dense/attn_lora/B"))
	assert.NotNil(t, store.GetVariable("/MultiHeadAttention/value/dense/attn_lora/A"))
	assert.NotNil(t, store.GetVariable("/MultiHeadAttention/value/dense/attn_lora/B"))

	// Key and Output base weights should be frozen without LoRA adapters:
	assert.False(t, store.GetVariable("/MultiHeadAttention/key/dense/weights").Trainable)
	assert.Nil(t, store.GetVariable("/MultiHeadAttention/key/dense/attn_lora/A"))
	assert.False(t, store.GetVariable("/MultiHeadAttention/output/dense/weights").Trainable)
	assert.Nil(t, store.GetVariable("/MultiHeadAttention/output/dense/attn_lora/A"))

	// Gradients should only be for Query (A, B) and Value (A, B) -> 4 variables:
	assert.Len(t, grads, 4)
}

