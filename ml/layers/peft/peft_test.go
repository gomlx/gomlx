// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package peft

import (
	"math"
	"testing"

	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
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
