// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package peft

import (
	"errors"
	"math"
	"math/rand"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
)

type testModel struct {
	modules      []Module
	replacements []Replacement
	replaceErr   error
}

func (m *testModel) LinearModules() ([]Module, error) { return m.modules, nil }
func (m *testModel) ReplaceLoRALinearModules(replacements []Replacement) error {
	if m.replaceErr != nil {
		return m.replaceErr
	}
	m.replacements = append([]Replacement(nil), replacements...)
	return nil
}

func TestDefaultLayoutSupportsSequenceInputs(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("projection")
	weight := scope.VariableWithValue("weight", [][]float32{{1, 0, 1}, {0, 1, 1}}) // [input, output]
	layer, err := NewLinear("projection", scope, weight, nil, Config{Rank: 1, Alpha: 1, TargetModules: []string{"projection"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node { return layer.Apply(scope, input) })
	if err != nil {
		t.Fatal(err)
	}
	outputs, err := exec.Call([][][]float32{{{1, 2}, {3, 4}}})
	if err != nil {
		t.Fatal(err)
	}
	if got := outputs[0].Shape().Dimensions; len(got) != 3 || got[0] != 1 || got[1] != 2 || got[2] != 3 {
		t.Fatalf("sequence output shape = %v", got)
	}
}

func TestLoRARetainsFloat16AndAppliesDropoutOnlyDuringTraining(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("projection")
	weight := scope.VariableWithValue("weight", tensors.FromFlatDataAndDimensions([]float16.Float16{float16.FromFloat32(1), float16.FromFloat32(0), float16.FromFloat32(0), float16.FromFloat32(1)}, 2, 2))
	layer, err := NewLinear("projection", scope, weight, nil, Config{Rank: 1, Alpha: 1, Dropout: 0.5, TargetModules: []string{"projection"}}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if layer.A().DType() != dtypes.Float16 || layer.B().DType() != dtypes.Float16 {
		t.Fatal("LoRA must retain the base floating-point dtype")
	}
	if err := layer.A().SetValue(tensors.FromFlatDataAndDimensions([]float16.Float16{float16.FromFloat32(1), float16.FromFloat32(1)}, 2, 1)); err != nil {
		t.Fatal(err)
	}
	if err := layer.B().SetValue(tensors.FromFlatDataAndDimensions([]float16.Float16{float16.FromFloat32(1), float16.FromFloat32(1)}, 1, 2)); err != nil {
		t.Fatal(err)
	}
	inputs := make([][]float16.Float16, 64)
	for index := range inputs {
		inputs[index] = []float16.Float16{float16.FromFloat32(1), float16.FromFloat32(2)}
	}
	evalExec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		scope.SetTraining(input.Graph(), false)
		return layer.Apply(scope, input)
	})
	if err != nil {
		t.Fatal(err)
	}
	trainExec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node {
		scope.SetTraining(input.Graph(), true)
		return layer.Apply(scope, input)
	})
	if err != nil {
		t.Fatal(err)
	}
	eval, err := evalExec.Call(inputs)
	if err != nil {
		t.Fatal(err)
	}
	train, err := trainExec.Call(inputs)
	if err != nil {
		t.Fatal(err)
	}
	evalValues, err := tensors.CopyFlatData[float16.Float16](eval[0])
	if err != nil {
		t.Fatal(err)
	}
	trainValues, err := tensors.CopyFlatData[float16.Float16](train[0])
	if err != nil {
		t.Fatal(err)
	}
	same := true
	for index := range evalValues {
		same = same && evalValues[index] == trainValues[index]
	}
	if same {
		t.Fatal("LoRA dropout must affect the training graph but not inference")
	}
}

func TestNamedAdaptersShareFrozenBaseAndCanBeSelected(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	weight := scope.VariableWithValue("weight", [][]float32{{1, 2}, {3, 4}})
	host := &testModel{modules: []Module{{Name: "q_proj", Scope: scope, Weight: weight}}}
	config := Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}, WeightLayout: compute.DenseLayoutOutputsInput}
	first, err := Inject("support", host, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	layer := host.replacements[0].Layer
	host.modules = []Module{{Name: "q_proj", Scope: scope, Layer: layer}}
	second, err := Inject("billing", host, config, rand.New(rand.NewSource(2)))
	if err != nil {
		t.Fatal(err)
	}
	if len(first.TrainableVariables()) != 2 || len(second.TrainableVariables()) != 2 || weight.Trainable {
		t.Fatal("named adapters must expose independent A/B while retaining one frozen base")
	}
	if _, ok := layer.Adapter("support"); !ok {
		t.Fatal("support adapter missing")
	}
	if _, ok := layer.Adapter("billing"); !ok {
		t.Fatal("billing adapter missing")
	}
	if err := layer.SetActiveAdapters("billing"); err != nil {
		t.Fatal(err)
	}
}

func TestInjectRollsBackWhenHostReplacementFails(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	weight := scope.VariableWithValue("weight", [][]float32{{1, 2}, {3, 4}})
	host := &testModel{modules: []Module{{Name: "q_proj", Scope: scope, Weight: weight}}, replaceErr: errors.New("replace failed")}
	_, err := Inject("support", host, Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}, WeightLayout: compute.DenseLayoutOutputsInput}, rand.New(rand.NewSource(1)))
	if err == nil {
		t.Fatal("expected replacement failure")
	}
	if !weight.Trainable {
		t.Fatal("failed injection must restore base trainability")
	}
}

type testNF4Model struct {
	modules      []NF4Module
	replacements []NF4Replacement
	replaceErr   error
}

func (m *testNF4Model) NF4LinearModules() ([]NF4Module, error) { return m.modules, nil }
func (m *testNF4Model) ReplaceNF4LinearModules(replacements []NF4Replacement) error {
	if m.replaceErr != nil {
		return m.replaceErr
	}
	m.replacements = append([]NF4Replacement(nil), replacements...)
	return nil
}

func TestInjectFreezesBaseAndChangesForward(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("layers").In("0").In("q_proj")
	weight := scope.VariableWithValue("weight", [][]float32{{1, 2}, {3, 4}})
	host := &testModel{modules: []Module{{Name: "layers.0.q_proj", Scope: scope, Weight: weight}}}
	adapter, err := Inject("support", host, Config{Rank: 1, Alpha: 2, TargetModules: []string{"q_proj"}, WeightLayout: compute.DenseLayoutOutputsInput}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if weight.Trainable || len(host.replacements) != 1 || len(adapter.TrainableVariables()) != 2 {
		t.Fatal("injection must freeze the base and select only A/B")
	}
	layer := host.replacements[0].Layer
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node { return layer.Apply(scope, input) })
	if err != nil {
		t.Fatal(err)
	}
	base := call(t, exec)
	if !closeEnough(base, []float32{5, 11}) {
		t.Fatalf("base forward = %v", base)
	}
	if err := layer.A().SetValue(tensors.FromFlatDataAndDimensions([]float32{1, 1}, 2, 1)); err != nil {
		t.Fatal(err)
	}
	if err := layer.B().SetValue(tensors.FromFlatDataAndDimensions([]float32{1, 1}, 1, 2)); err != nil {
		t.Fatal(err)
	}
	updatedExec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node { return layer.Apply(scope, input) })
	if err != nil {
		t.Fatal(err)
	}
	if trained := call(t, updatedExec); closeEnough(trained, base) {
		t.Fatalf("adapter update did not affect forward: %v", trained)
	}
}

func TestAdapterGradientsExcludeFrozenBase(t *testing.T) {
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	weight := scope.VariableWithValue("weight", [][]float32{{1, 2}, {3, 4}})
	host := &testModel{modules: []Module{{Name: "q_proj", Scope: scope, Weight: weight}}}
	adapter, err := Inject("support", host, Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}, WeightLayout: compute.DenseLayoutOutputsInput}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if len(adapter.TrainableVariables()) != 2 {
		t.Fatal("adapter must expose exactly A/B to an optimizer")
	}
	layer := host.replacements[0].Layer
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) []*graph.Node {
		return scope.BuildTrainableVariablesGradientsGraph(graph.ReduceAllSum(layer.Apply(scope, input)))
	})
	if err != nil {
		t.Fatal(err)
	}
	gradients, err := exec.Call([][]float32{{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(gradients) != 2 {
		t.Fatalf("gradient count = %d, want only A/B", len(gradients))
	}
}

func TestNF4QLoRAForwardAndInjection(t *testing.T) {
	weight, err := QuantizeNF4Double(2, 4, 2, 2, []float32{1, -1, 0.5, -0.5, -1, 1, -0.5, 0.5})
	if err != nil {
		t.Fatal(err)
	}
	if len(weight.Scales) != 0 || len(weight.ScaleCodes) == 0 {
		t.Fatal("expected double-quantized primary scales")
	}
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	host := &testNF4Model{modules: []NF4Module{{Name: "q_proj", Scope: scope, Weight: weight}}}
	adapter, err := InjectNF4("support", host, QLoRAConfig{LoRA: Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, BlockSize: 2, DoubleQuant: true, ScaleBlockSize: 2}, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	if len(host.replacements) != 1 || len(adapter.TrainableVariables()) != 2 {
		t.Fatal("QLoRA must select only A/B")
	}
	layer := host.replacements[0].Layer
	exec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) *graph.Node { return layer.Apply(scope, input) })
	if err != nil {
		t.Fatal(err)
	}
	output := call(t, exec)
	for index, want := range []float32{-1, 1, -0.5, 0.5} {
		if math.Abs(float64(output[index]-want)) > 0.02 {
			t.Fatalf("output[%d] = %v, want %v", index, output[index], want)
		}
	}
	gradientExec, err := model.NewExec(gobackend.GetBackend(), store, func(scope *model.Scope, input *graph.Node) []*graph.Node {
		return scope.BuildTrainableVariablesGradientsGraph(graph.ReduceAllSum(layer.Apply(scope, input)))
	})
	if err != nil {
		t.Fatal(err)
	}
	gradients, err := gradientExec.Call([][]float32{{1, 2}})
	if err != nil {
		t.Fatal(err)
	}
	if len(gradients) != 2 {
		t.Fatalf("QLoRA gradient count = %d, want A/B only", len(gradients))
	}
}

func TestNF4NamedAdaptersShareOneFrozenBase(t *testing.T) {
	weight, err := QuantizeNF4(2, 2, 2, []float32{1, -1, -1, 1})
	if err != nil {
		t.Fatal(err)
	}
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	host := &testNF4Model{modules: []NF4Module{{Name: "q_proj", Scope: scope, Weight: weight}}}
	config := QLoRAConfig{LoRA: Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}}, BlockSize: 2}
	first, err := InjectNF4("support", host, config, rand.New(rand.NewSource(1)))
	if err != nil {
		t.Fatal(err)
	}
	layer := host.replacements[0].Layer
	host.modules = []NF4Module{{Name: "q_proj", Scope: scope, Layer: layer}}
	second, err := InjectNF4("billing", host, config, rand.New(rand.NewSource(2)))
	if err != nil {
		t.Fatal(err)
	}
	if len(first.TrainableVariables()) != 2 || len(second.TrainableVariables()) != 2 {
		t.Fatal("each QLoRA adapter must expose only its own A/B variables")
	}
	if _, ok := layer.Adapter("support"); !ok {
		t.Fatal("support QLoRA adapter missing")
	}
	if _, ok := layer.Adapter("billing"); !ok {
		t.Fatal("billing QLoRA adapter missing")
	}
	if err := layer.SetActiveAdapters("billing"); err != nil {
		t.Fatal(err)
	}
	if err := layer.SetActiveAdapters(); err != nil || layer.A() != nil || layer.B() != nil {
		t.Fatal("QLoRA must allow disabling all named adapters")
	}
}

func TestNF4InjectionRollsBackWhenHostReplacementFails(t *testing.T) {
	weight, err := QuantizeNF4(2, 2, 2, []float32{1, -1, -1, 1})
	if err != nil {
		t.Fatal(err)
	}
	store := model.NewStore()
	scope := store.RootScope().In("q_proj")
	bias := scope.VariableWithValue("bias", []float32{0, 0})
	host := &testNF4Model{
		modules:    []NF4Module{{Name: "q_proj", Scope: scope, Weight: weight, Bias: bias}},
		replaceErr: errors.New("replace failed"),
	}
	_, err = InjectNF4("support", host, QLoRAConfig{LoRA: Config{Rank: 1, Alpha: 1, Bias: BiasAll, TargetModules: []string{"q_proj"}}, BlockSize: 2}, rand.New(rand.NewSource(1)))
	if err == nil {
		t.Fatal("expected replacement failure")
	}
	if !bias.Trainable {
		t.Fatal("failed QLoRA injection must restore bias trainability")
	}
}

func TestRejectsOddNF4Output(t *testing.T) {
	if _, err := QuantizeNF4(2, 3, 2, make([]float32, 6)); err == nil {
		t.Fatal("odd NF4 output must be rejected")
	}
}

func TestRejectsQLoRAWeightConfigurationMismatch(t *testing.T) {
	weight, err := QuantizeNF4(2, 2, 2, []float32{1, -1, -1, 1})
	if err != nil {
		t.Fatal(err)
	}
	store := model.NewStore()
	_, err = NewNF4Linear("q_proj", store.RootScope().In("q_proj"), weight, nil, QLoRAConfig{
		LoRA:      Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}},
		BlockSize: 64,
	}, rand.New(rand.NewSource(1)))
	if !errors.Is(err, ErrInvalidQLoRAConfig) {
		t.Fatalf("mismatched block size error = %v", err)
	}
}

func TestRejectsOutputInputLayoutForNF4QLoRA(t *testing.T) {
	config := QLoRAConfig{
		LoRA:      Config{Rank: 1, Alpha: 1, TargetModules: []string{"q_proj"}, WeightLayout: compute.DenseLayoutOutputsInput},
		BlockSize: 2,
	}
	if !errors.Is(config.Validate(), ErrInvalidQLoRAConfig) {
		t.Fatalf("QLoRA output-input layout error = %v", config.Validate())
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
