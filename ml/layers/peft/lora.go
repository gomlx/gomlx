// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package peft

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/pkg/errors"
)

// Linear is a frozen dense projection plus a trainable LoRA update. It honors
// Config.WeightLayout; A and B always use GoMLX's [input, output] layout.
type Linear struct {
	name             string
	scope            *model.Scope
	weight, bias     *model.Variable
	scaling, dropout float32
	config           Config
	input, output    int
	adapters         map[string]*AdapterWeights
	active           []string
}

// AdapterWeights is one named LoRA update attached to a Linear projection.
type AdapterWeights struct {
	name    string
	config  Config
	a, b    *model.Variable
	scaling float32
}

// NewLinear creates a native LoRA layer under scope/lora. The caller retains
// ownership of the model registry and uses Apply wherever its base projection
// would have been used.
func NewLinear(name string, scope *model.Scope, weight, bias *model.Variable, config Config, rng *rand.Rand) (*Linear, error) {
	return NewLinearWithAdapter("default", name, scope, weight, bias, config, rng)
}

// NewLinearWithAdapter creates a layer and its first named adapter. Use
// AddAdapter to attach further named adapters to the same frozen base.
func NewLinearWithAdapter(adapterName, name string, scope *model.Scope, weight, bias *model.Variable, config Config, rng *rand.Rand) (*Linear, error) {
	if scope == nil || weight == nil || rng == nil {
		return nil, errors.New("peft: scope, weight, and random source are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	shape := weight.Shape()
	if shape.Rank() != 2 || !supportedDType(shape.DType) || shape.Dimensions[0] <= 0 || shape.Dimensions[1] <= 0 {
		return nil, errors.New("peft: base weight must be a non-empty F32/F16/BF16 matrix")
	}
	in, out := shape.Dimensions[0], shape.Dimensions[1]
	if config.WeightLayout == compute.DenseLayoutOutputsInput {
		out, in = shape.Dimensions[0], shape.Dimensions[1]
	}
	if bias != nil && (bias.Shape().Rank() != 1 || bias.Shape().Dimensions[0] != out) {
		return nil, errors.New("peft: bias shape must be [output]")
	}

	layer := &Linear{name: name, scope: scope, weight: weight, bias: bias, config: cloneConfig(config), input: in, output: out, adapters: make(map[string]*AdapterWeights)}
	weight.SetTrainable(false)
	if _, err := layer.AddAdapter(adapterName, config, rng); err != nil {
		weight.SetTrainable(true)
		return nil, err
	}
	return layer, nil
}

// AddAdapter adds a named LoRA update without duplicating the frozen base
// projection. Adapter names are unique per Linear layer.
func (l *Linear) AddAdapter(name string, config Config, rng *rand.Rand) (*AdapterWeights, error) {
	if l == nil || name == "" || rng == nil {
		return nil, errors.New("peft: layer, adapter name, and random source are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if config.WeightLayout != l.config.WeightLayout {
		return nil, ErrInvalidWeightLayout
	}
	if _, exists := l.adapters[name]; exists {
		return nil, fmt.Errorf("peft: adapter %q already exists", name)
	}
	aValues := make([]float32, config.Rank*l.input)
	bound := float32(1 / math.Sqrt(float64(l.input)))
	for index := range aValues {
		aValues[index] = (2*rng.Float32() - 1) * bound
	}
	adapterScope := l.scope.At("peft").In("%s", name)
	aValue, err := matrixValue(l.weight.DType(), l.input, config.Rank, aValues)
	if err != nil {
		return nil, err
	}
	bValue, err := matrixValue(l.weight.DType(), config.Rank, l.output, make([]float32, config.Rank*l.output))
	if err != nil {
		return nil, err
	}
	adapter := &AdapterWeights{name: name, config: cloneConfig(config), a: adapterScope.VariableWithValue("A", aValue).SetTrainable(true), b: adapterScope.VariableWithValue("B", bValue).SetTrainable(true), scaling: config.Alpha / float32(config.Rank)}
	l.adapters[name] = adapter
	l.active = append(l.active, name)
	l.refreshBiasTrainability()
	return adapter, nil
}

// SetActiveAdapters chooses which attached updates participate in Apply.
func (l *Linear) SetActiveAdapters(names ...string) error {
	seen := make(map[string]struct{}, len(names))
	for _, name := range names {
		if _, ok := l.adapters[name]; !ok {
			return fmt.Errorf("peft: unknown adapter %q", name)
		}
		if _, duplicate := seen[name]; duplicate {
			return fmt.Errorf("peft: duplicate active adapter %q", name)
		}
		seen[name] = struct{}{}
	}
	l.active = append([]string(nil), names...)
	return nil
}

// Adapter returns one named update and its trainable variables.
func (l *Linear) Adapter(name string) (*AdapterWeights, bool) {
	adapter, ok := l.adapters[name]
	return adapter, ok
}
func (a *AdapterWeights) Name() string                          { return a.name }
func (a *AdapterWeights) A() *model.Variable                    { return a.a }
func (a *AdapterWeights) B() *model.Variable                    { return a.b }
func (a *AdapterWeights) TrainableVariables() []*model.Variable { return []*model.Variable{a.a, a.b} }

func (l *Linear) refreshBiasTrainability() {
	if l.bias == nil {
		return
	}
	trainable := false
	for _, adapter := range l.adapters {
		trainable = trainable || adapter.config.Bias != BiasNone
	}
	l.bias.SetTrainable(trainable)
}

func (l *Linear) removeAdapter(name string) {
	if _, ok := l.adapters[name]; !ok {
		return
	}
	delete(l.adapters, name)
	active := l.active[:0]
	for _, current := range l.active {
		if current != name {
			active = append(active, current)
		}
	}
	l.active = active
	_ = l.scope.At("peft").At("%s", name).DeleteVariablesInScope()
	l.refreshBiasTrainability()
}

// Apply builds base(input) + scale * B(A(dropout(input))). It supports batch
// dimensions such as [batch, sequence, hidden] through nn.Dense.
func (l *Linear) Apply(scope *model.Scope, input *Node) *Node {
	base := nn.Dense(input, l.weight.NodeValue(input), nodeValue(l.bias, input), l.config.WeightLayout)
	for _, name := range l.active {
		adapter := l.adapters[name]
		update := nn.Dense(input, adapter.a.NodeValue(input), nil, compute.DenseLayoutInputOutputs)
		update = layers.DropoutStatic(scope, update, float64(adapter.config.Dropout))
		update = nn.Dense(update, adapter.b.NodeValue(input), nil, compute.DenseLayoutInputOutputs)
		base = Add(base, MulScalar(update, adapter.scaling))
	}
	return base
}

func (l *Linear) Name() string            { return l.name }
func (l *Linear) Weight() *model.Variable { return l.weight }
func (l *Linear) Bias() *model.Variable   { return l.bias }
func (l *Linear) A() *model.Variable {
	if len(l.active) == 0 {
		return nil
	}
	return l.adapters[l.active[0]].a
}
func (l *Linear) B() *model.Variable {
	if len(l.active) == 0 {
		return nil
	}
	return l.adapters[l.active[0]].b
}

// TrainableVariables returns exactly the variables an optimizer should update.
func (l *Linear) TrainableVariables() []*model.Variable {
	variables := make([]*model.Variable, 0, len(l.adapters)*2+1)
	for _, adapter := range l.adapters {
		variables = append(variables, adapter.a, adapter.b)
	}
	if l.bias != nil && l.bias.Trainable {
		variables = append(variables, l.bias)
	}
	return variables
}

// Module describes a host-model projection available for adapter injection.
type Module struct {
	Name         string
	Scope        *model.Scope
	Weight, Bias *model.Variable
	// Layer is set by hosts that already replaced this module with a PEFT layer.
	// It enables attaching another named adapter without replacing the base again.
	Layer *Linear
}

// Replacement associates a host module with its LoRA replacement.
type Replacement struct {
	Name  string
	Layer *Linear
}

// Model lets an architecture expose and replace its named linear projections.
type Model interface {
	LinearModules() ([]Module, error)
	ReplaceLoRALinearModules([]Replacement) error
}

// Adapter owns injected layers and gives the host an exact optimizer variable set.
type Adapter struct {
	name   string
	config Config
	layers []*Linear
	index  map[string]*Linear
}

// Inject creates all matching layers before asking the host to replace them.
func Inject(name string, host Model, config Config, rng *rand.Rand) (*Adapter, error) {
	if name == "" || host == nil || rng == nil {
		return nil, errors.New("peft: adapter name, model, and random source are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	modules, err := host.LinearModules()
	if err != nil {
		return nil, errors.Wrap(err, "peft: discover linear modules")
	}
	replacements := make([]Replacement, 0, len(modules))
	index := make(map[string]*Linear, len(modules))
	layers := make([]*Linear, 0, len(modules))
	type mutation struct {
		layer                          *Linear
		newLayer                       bool
		weight, bias                   *model.Variable
		weightTrainable, biasTrainable bool
	}
	mutations := make([]mutation, 0, len(modules))
	committed := false
	defer func() {
		if committed {
			return
		}
		for index := len(mutations) - 1; index >= 0; index-- {
			mutation := mutations[index]
			mutation.layer.removeAdapter(name)
			if mutation.newLayer {
				mutation.weight.SetTrainable(mutation.weightTrainable)
				if mutation.bias != nil {
					mutation.bias.SetTrainable(mutation.biasTrainable)
				}
			}
		}
	}()
	for _, module := range modules {
		if !Matches(module.Name, config.TargetModules) {
			continue
		}
		if module.Name == "" || module.Scope == nil || (module.Weight == nil && module.Layer == nil) {
			return nil, fmt.Errorf("peft: invalid module %q", module.Name)
		}
		if _, exists := index[module.Name]; exists {
			return nil, fmt.Errorf("peft: duplicate module %q", module.Name)
		}
		layer := module.Layer
		if layer != nil {
			if _, err := layer.AddAdapter(name, config, rng); err != nil {
				return nil, fmt.Errorf("peft: attach %q: %w", module.Name, err)
			}
			mutations = append(mutations, mutation{layer: layer})
		} else {
			var err error
			weightTrainable := module.Weight.Trainable
			biasTrainable := false
			if module.Bias != nil {
				biasTrainable = module.Bias.Trainable
			}
			layer, err = NewLinearWithAdapter(name, module.Name, module.Scope, module.Weight, module.Bias, config, rng)
			if err != nil {
				return nil, fmt.Errorf("peft: inject %q: %w", module.Name, err)
			}
			mutations = append(mutations, mutation{layer: layer, newLayer: true, weight: module.Weight, bias: module.Bias, weightTrainable: weightTrainable, biasTrainable: biasTrainable})
			replacements = append(replacements, Replacement{Name: module.Name, Layer: layer})
		}
		index[module.Name] = layer
		layers = append(layers, layer)
	}
	if len(layers) == 0 {
		return nil, ErrNoTargetModules
	}
	if len(replacements) > 0 {
		if err := host.ReplaceLoRALinearModules(replacements); err != nil {
			return nil, errors.Wrap(err, "peft: replace linear modules")
		}
	}
	committed = true
	return &Adapter{name: name, config: cloneConfig(config), layers: layers, index: index}, nil
}

func (a *Adapter) Name() string                      { return a.name }
func (a *Adapter) Config() Config                    { return cloneConfig(a.config) }
func (a *Adapter) Layers() []*Linear                 { return append([]*Linear(nil), a.layers...) }
func (a *Adapter) Layer(name string) (*Linear, bool) { layer, ok := a.index[name]; return layer, ok }

// TrainableVariables returns adapter A/B variables and configured selected biases.
func (a *Adapter) TrainableVariables() []*model.Variable {
	variables := make([]*model.Variable, 0, len(a.layers)*2)
	for _, layer := range a.layers {
		adapter, ok := layer.Adapter(a.name)
		if ok {
			variables = append(variables, adapter.TrainableVariables()...)
		}
		if layer.bias != nil && layer.bias.Trainable {
			variables = append(variables, layer.bias)
		}
	}
	return variables
}

func supportedDType(dtype dtypes.DType) bool {
	return dtype == dtypes.Float32 || dtype == dtypes.Float16 || dtype == dtypes.BFloat16
}

func matrixValue(dtype dtypes.DType, rows, cols int, values []float32) (any, error) {
	switch dtype {
	case dtypes.Float32:
		return tensors.FromFlatDataAndDimensions(values, rows, cols), nil
	case dtypes.Float16:
		converted := make([]float16.Float16, len(values))
		for index := range values {
			converted[index] = float16.FromFloat32(values[index])
		}
		return tensors.FromFlatDataAndDimensions(converted, rows, cols), nil
	case dtypes.BFloat16:
		converted := make([]bfloat16.BFloat16, len(values))
		for index := range values {
			converted[index] = bfloat16.FromFloat32(values[index])
		}
		return tensors.FromFlatDataAndDimensions(converted, rows, cols), nil
	default:
		return nil, errors.New("peft: unsupported adapter dtype")
	}
}
