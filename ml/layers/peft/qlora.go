// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package peft

import (
	"fmt"
	"math"
	"math/rand"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/nn"
	"github.com/pkg/errors"
)

// NF4Weight stores a frozen [input, output] NormalFloat4 matrix. Packed holds
// two NF4 codes per byte. Scales are per-input, per-output-block absolute-max
// scales; ScaleCodes/ScaleScales replace Scales when double quantization is on.
type NF4Weight struct {
	InputFeatures, OutputFeatures int
	BlockSize                     int
	Packed                        []byte
	Scales                        []float32
	ScaleCodes                    []byte
	ScaleScales                   []float32
	ScaleBlockSize                int
}

// QuantizeNF4 packs a [input, output] F32 matrix. output must be even because
// GoMLX's packed Uint4 graph representation cannot safely drop one nibble.
func QuantizeNF4(inputFeatures, outputFeatures, blockSize int, values []float32) (*NF4Weight, error) {
	return quantizeNF4(inputFeatures, outputFeatures, blockSize, 0, values)
}

// QuantizeNF4Double additionally stores primary scales as uint8 values grouped
// by an F32 second-level scale.
func QuantizeNF4Double(inputFeatures, outputFeatures, blockSize, scaleBlockSize int, values []float32) (*NF4Weight, error) {
	return quantizeNF4(inputFeatures, outputFeatures, blockSize, scaleBlockSize, values)
}

func quantizeNF4(inputFeatures, outputFeatures, blockSize, scaleBlockSize int, values []float32) (*NF4Weight, error) {
	if inputFeatures <= 0 || outputFeatures <= 0 || outputFeatures%2 != 0 || blockSize <= 0 || scaleBlockSize < 0 || inputFeatures > math.MaxInt/outputFeatures || len(values) != inputFeatures*outputFeatures {
		return nil, ErrInvalidNF4Weight
	}
	blocks := (outputFeatures + blockSize - 1) / blockSize
	weight := &NF4Weight{
		InputFeatures: inputFeatures, OutputFeatures: outputFeatures, BlockSize: blockSize,
		Packed: make([]byte, inputFeatures*(outputFeatures/2)), Scales: make([]float32, inputFeatures*blocks),
	}
	for input := range inputFeatures {
		row := values[input*outputFeatures : (input+1)*outputFeatures]
		for block := range blocks {
			start, end := block*blockSize, min((block+1)*blockSize, outputFeatures)
			var scale float32
			for _, value := range row[start:end] {
				if absolute := float32(math.Abs(float64(value))); absolute > scale {
					scale = absolute
				}
			}
			weight.Scales[input*blocks+block] = scale
			if scale == 0 {
				continue
			}
			for output := start; output < end; output++ {
				weight.setCode(input, output, nf4NearestCode(row[output]/scale))
			}
		}
	}
	if scaleBlockSize > 0 {
		weight.doubleQuantizeScales(scaleBlockSize)
	}
	return weight, nil
}

func (w *NF4Weight) validate() error {
	if w == nil || w.InputFeatures <= 0 || w.OutputFeatures <= 0 || w.OutputFeatures%2 != 0 || w.BlockSize <= 0 {
		return ErrInvalidNF4Weight
	}
	blocks := (w.OutputFeatures + w.BlockSize - 1) / w.BlockSize
	if len(w.Packed) != w.InputFeatures*(w.OutputFeatures/2) || (len(w.Scales) != w.InputFeatures*blocks && len(w.ScaleCodes) != w.InputFeatures*blocks) {
		return ErrInvalidNF4Weight
	}
	if len(w.ScaleCodes) > 0 && (w.ScaleBlockSize <= 0 || len(w.ScaleScales) != w.InputFeatures*((blocks+w.ScaleBlockSize-1)/w.ScaleBlockSize)) {
		return ErrInvalidNF4Weight
	}
	return nil
}

func (w *NF4Weight) setCode(input, output int, code byte) {
	index := input*(w.OutputFeatures/2) + output/2
	if output%2 == 0 {
		w.Packed[index] = w.Packed[index]&0xf0 | code
		return
	}
	w.Packed[index] = w.Packed[index]&0x0f | code<<4
}

func (w *NF4Weight) doubleQuantizeScales(scaleBlockSize int) {
	blocks := (w.OutputFeatures + w.BlockSize - 1) / w.BlockSize
	w.ScaleBlockSize = scaleBlockSize
	w.ScaleCodes = make([]byte, len(w.Scales))
	w.ScaleScales = make([]float32, w.InputFeatures*((blocks+scaleBlockSize-1)/scaleBlockSize))
	for input := range w.InputFeatures {
		for group := 0; group*scaleBlockSize < blocks; group++ {
			start, end := group*scaleBlockSize, min((group+1)*scaleBlockSize, blocks)
			var maximum float32
			for _, scale := range w.Scales[input*blocks+start : input*blocks+end] {
				if scale > maximum {
					maximum = scale
				}
			}
			groupScale := maximum / 255
			w.ScaleScales[input*((blocks+scaleBlockSize-1)/scaleBlockSize)+group] = groupScale
			if groupScale == 0 {
				continue
			}
			for block := start; block < end; block++ {
				code := min(int(math.Round(float64(w.Scales[input*blocks+block]/groupScale))), 255)
				w.ScaleCodes[input*blocks+block] = byte(code)
			}
		}
	}
	w.Scales = nil
}

// NF4Linear combines a frozen NF4 projection and one or more trainable F32
// LoRA adapters.
type NF4Linear struct {
	name                                          string
	scope                                         *model.Scope
	packed, scales, scaleCodes, scaleScales, bias *model.Variable
	in, out, block                                int
	config                                        QLoRAConfig
	adapters                                      map[string]*NF4AdapterWeights
	active                                        []string
}

// NF4AdapterWeights is one named F32 LoRA update attached to an NF4 base.
type NF4AdapterWeights struct {
	name    string
	config  QLoRAConfig
	a, b    *model.Variable
	scaling float32
}

// NewNF4Linear creates a QLoRA layer with its default named adapter. Its
// adapter matrices stay F32 even when the base projection is packed NF4.
func NewNF4Linear(name string, scope *model.Scope, weight *NF4Weight, bias *model.Variable, config QLoRAConfig, rng *rand.Rand) (*NF4Linear, error) {
	return NewNF4LinearWithAdapter("default", name, scope, weight, bias, config, rng)
}

// NewNF4LinearWithAdapter creates an NF4 projection and its first named
// adapter. Use AddAdapter to attach additional adapters to the same base.
func NewNF4LinearWithAdapter(adapterName, name string, scope *model.Scope, weight *NF4Weight, bias *model.Variable, config QLoRAConfig, rng *rand.Rand) (*NF4Linear, error) {
	if scope == nil || rng == nil {
		return nil, errors.New("peft: scope and random source are required")
	}
	if err := weight.validate(); err != nil {
		return nil, err
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if config.BlockSize != weight.BlockSize || config.DoubleQuant != (len(weight.ScaleCodes) > 0) || (config.DoubleQuant && config.ScaleBlockSize != weight.ScaleBlockSize) {
		return nil, ErrInvalidQLoRAConfig
	}
	if bias != nil && (bias.Shape().Rank() != 1 || bias.Shape().Dimensions[0] != weight.OutputFeatures) {
		return nil, errors.New("peft: bias shape must be [output]")
	}
	qScope := scope.In("qlora")
	packed := qScope.VariableWithValue("nf4_packed", tensors.FromFlatDataAndDimensions(weight.Packed, weight.InputFeatures, weight.OutputFeatures/2)).SetTrainable(false)
	blocks := (weight.OutputFeatures + weight.BlockSize - 1) / weight.BlockSize
	var scales, scaleCodes, scaleScales *model.Variable
	if len(weight.ScaleCodes) == 0 {
		scales = qScope.VariableWithValue("nf4_scales", tensors.FromFlatDataAndDimensions(weight.Scales, weight.InputFeatures, blocks)).SetTrainable(false)
	} else {
		scaleCodes = qScope.VariableWithValue("nf4_scale_codes", tensors.FromFlatDataAndDimensions(weight.ScaleCodes, weight.InputFeatures, blocks)).SetTrainable(false)
		scaleScales = qScope.VariableWithValue("nf4_scale_scales", tensors.FromFlatDataAndDimensions(weight.ScaleScales, weight.InputFeatures, (blocks+weight.ScaleBlockSize-1)/weight.ScaleBlockSize)).SetTrainable(false)
	}
	layer := &NF4Linear{
		name: name, scope: scope, packed: packed, scales: scales, scaleCodes: scaleCodes,
		scaleScales: scaleScales, bias: bias, in: weight.InputFeatures, out: weight.OutputFeatures,
		block: weight.BlockSize, config: cloneQLoRAConfig(config), adapters: make(map[string]*NF4AdapterWeights),
	}
	if _, err := layer.AddAdapter(adapterName, config, rng); err != nil {
		_ = qScope.DeleteVariablesInScope()
		return nil, err
	}
	return layer, nil
}

// AddAdapter adds a named QLoRA update without duplicating the frozen NF4
// projection. The quantization settings must match the projection.
func (l *NF4Linear) AddAdapter(name string, config QLoRAConfig, rng *rand.Rand) (*NF4AdapterWeights, error) {
	if l == nil || name == "" || rng == nil {
		return nil, errors.New("peft: layer, adapter name, and random source are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	if config.BlockSize != l.config.BlockSize || config.DoubleQuant != l.config.DoubleQuant || config.ScaleBlockSize != l.config.ScaleBlockSize {
		return nil, ErrInvalidQLoRAConfig
	}
	if _, exists := l.adapters[name]; exists {
		return nil, fmt.Errorf("peft: adapter %q already exists", name)
	}
	aValues := make([]float32, config.LoRA.Rank*l.in)
	bound := float32(1 / math.Sqrt(float64(l.in)))
	for index := range aValues {
		aValues[index] = (2*rng.Float32() - 1) * bound
	}
	adapterScope := l.scope.At("peft").In("%s", name)
	adapter := &NF4AdapterWeights{
		name: name, config: cloneQLoRAConfig(config),
		a:       adapterScope.VariableWithValue("A", tensors.FromFlatDataAndDimensions(aValues, l.in, config.LoRA.Rank)).SetTrainable(true),
		b:       adapterScope.VariableWithValue("B", tensors.FromFlatDataAndDimensions(make([]float32, config.LoRA.Rank*l.out), config.LoRA.Rank, l.out)).SetTrainable(true),
		scaling: config.LoRA.Alpha / float32(config.LoRA.Rank),
	}
	l.adapters[name] = adapter
	l.active = append(l.active, name)
	l.refreshBiasTrainability()
	return adapter, nil
}

// SetActiveAdapters chooses which attached updates participate in Apply.
func (l *NF4Linear) SetActiveAdapters(names ...string) error {
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

// Adapter returns one named QLoRA update and its trainable variables.
func (l *NF4Linear) Adapter(name string) (*NF4AdapterWeights, bool) {
	adapter, ok := l.adapters[name]
	return adapter, ok
}

func (a *NF4AdapterWeights) Name() string       { return a.name }
func (a *NF4AdapterWeights) A() *model.Variable { return a.a }
func (a *NF4AdapterWeights) B() *model.Variable { return a.b }
func (a *NF4AdapterWeights) TrainableVariables() []*model.Variable {
	return []*model.Variable{a.a, a.b}
}

func (l *NF4Linear) refreshBiasTrainability() {
	if l.bias == nil {
		return
	}
	trainable := false
	for _, adapter := range l.adapters {
		trainable = trainable || adapter.config.LoRA.Bias != BiasNone
	}
	l.bias.SetTrainable(trainable)
}

func (l *NF4Linear) removeAdapter(name string) {
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

// Apply builds a quantized base projection plus all selected QLoRA updates.
func (l *NF4Linear) Apply(scope *model.Scope, input *Node) *Node {
	packed := Bitcast(l.packed.NodeValue(input), dtypes.Uint4)
	weights := Reshape(packed, l.in, l.out)
	base := nn.QuantizedDense(input, weights, &Quantization{Scheme: compute.QuantNF4, Scale: l.quantizationScales(input), BlockAxis: 1, BlockSize: l.block}, nodeValue(l.bias, input))
	for _, name := range l.active {
		adapter := l.adapters[name]
		update := MatMul(input, adapter.a.NodeValue(input))
		update = layers.DropoutStatic(scope, update, float64(adapter.config.LoRA.Dropout))
		update = MatMul(update, adapter.b.NodeValue(input))
		base = Add(base, MulScalar(update, adapter.scaling))
	}
	return base
}

func (l *NF4Linear) quantizationScales(input *Node) *Node {
	if l.scales != nil {
		return l.scales.NodeValue(input)
	}
	blocks := (l.out + l.block - 1) / l.block
	groups := l.scaleScales.Shape().Dimensions[1]
	groupSize := (blocks + groups - 1) / groups
	indices := make([]int32, blocks)
	for index := range indices {
		indices[index] = int32(index / groupSize)
	}
	group := Reshape(Const(input.Graph(), indices), blocks, 1)
	expanded := Gather(Transpose(l.scaleScales.NodeValue(input), 0, 1), group)
	expanded = Transpose(expanded, 0, 1)
	return Mul(ConvertDType(l.scaleCodes.NodeValue(input), dtypes.Float32), expanded)
}

func (l *NF4Linear) Name() string          { return l.name }
func (l *NF4Linear) Bias() *model.Variable { return l.bias }
func (l *NF4Linear) A() *model.Variable {
	if len(l.active) == 0 {
		return nil
	}
	return l.adapters[l.active[0]].a
}
func (l *NF4Linear) B() *model.Variable {
	if len(l.active) == 0 {
		return nil
	}
	return l.adapters[l.active[0]].b
}
func (l *NF4Linear) TrainableVariables() []*model.Variable {
	variables := make([]*model.Variable, 0, len(l.adapters)*2+1)
	for _, adapter := range l.adapters {
		variables = append(variables, adapter.a, adapter.b)
	}
	if l.bias != nil && l.bias.Trainable {
		variables = append(variables, l.bias)
	}
	return variables
}

// NF4Module is a host-discovered base projection for QLoRA injection.
type NF4Module struct {
	Name   string
	Scope  *model.Scope
	Weight *NF4Weight
	Bias   *model.Variable
	// Layer is set by hosts that already replaced this module with QLoRA.
	// It enables attaching another named adapter without duplicating NF4 data.
	Layer *NF4Linear
}

// NF4Replacement associates a model module with its QLoRA replacement.
type NF4Replacement struct {
	Name  string
	Layer *NF4Linear
}

// NF4Model lets an architecture expose quantized projections for replacement.
type NF4Model interface {
	NF4LinearModules() ([]NF4Module, error)
	ReplaceNF4LinearModules([]NF4Replacement) error
}

// NF4Adapter owns injected QLoRA layers.
type NF4Adapter struct {
	name   string
	config QLoRAConfig
	layers []*NF4Linear
	index  map[string]*NF4Linear
}

// InjectNF4 creates layers for matching modules then replaces them atomically.
func InjectNF4(name string, host NF4Model, config QLoRAConfig, rng *rand.Rand) (*NF4Adapter, error) {
	if name == "" || host == nil || rng == nil {
		return nil, errors.New("peft: adapter name, model, and random source are required")
	}
	if err := config.Validate(); err != nil {
		return nil, err
	}
	modules, err := host.NF4LinearModules()
	if err != nil {
		return nil, errors.Wrap(err, "peft: discover NF4 linear modules")
	}
	replacements := make([]NF4Replacement, 0, len(modules))
	index := make(map[string]*NF4Linear, len(modules))
	layers := make([]*NF4Linear, 0, len(modules))
	type mutation struct {
		layer         *NF4Linear
		newLayer      bool
		bias          *model.Variable
		biasTrainable bool
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
			if mutation.newLayer && mutation.bias != nil {
				mutation.bias.SetTrainable(mutation.biasTrainable)
			}
		}
	}()
	for _, module := range modules {
		if !Matches(module.Name, config.LoRA.TargetModules) {
			continue
		}
		if module.Name == "" || module.Scope == nil || (module.Weight == nil && module.Layer == nil) {
			return nil, fmt.Errorf("peft: invalid NF4 module %q", module.Name)
		}
		if _, exists := index[module.Name]; exists {
			return nil, fmt.Errorf("peft: duplicate NF4 module %q", module.Name)
		}
		layer := module.Layer
		if layer != nil {
			if _, err := layer.AddAdapter(name, config, rng); err != nil {
				return nil, fmt.Errorf("peft: attach NF4 %q: %w", module.Name, err)
			}
			mutations = append(mutations, mutation{layer: layer})
		} else {
			biasTrainable := module.Bias != nil && module.Bias.Trainable
			var err error
			layer, err = NewNF4LinearWithAdapter(name, module.Name, module.Scope, module.Weight, module.Bias, config, rng)
			if err != nil {
				return nil, fmt.Errorf("peft: inject NF4 %q: %w", module.Name, err)
			}
			mutations = append(mutations, mutation{layer: layer, newLayer: true, bias: module.Bias, biasTrainable: biasTrainable})
			replacements = append(replacements, NF4Replacement{Name: module.Name, Layer: layer})
		}
		index[module.Name] = layer
		layers = append(layers, layer)
	}
	if len(layers) == 0 {
		return nil, ErrNoTargetModules
	}
	if len(replacements) > 0 {
		if err := host.ReplaceNF4LinearModules(replacements); err != nil {
			return nil, errors.Wrap(err, "peft: replace NF4 linear modules")
		}
	}
	committed = true
	return &NF4Adapter{name: name, config: cloneQLoRAConfig(config), layers: layers, index: index}, nil
}

func (a *NF4Adapter) Name() string         { return a.name }
func (a *NF4Adapter) Config() QLoRAConfig  { return cloneQLoRAConfig(a.config) }
func (a *NF4Adapter) Layers() []*NF4Linear { return append([]*NF4Linear(nil), a.layers...) }
func (a *NF4Adapter) Layer(name string) (*NF4Linear, bool) {
	layer, ok := a.index[name]
	return layer, ok
}
func (a *NF4Adapter) TrainableVariables() []*model.Variable {
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

func nodeValue(variable *model.Variable, input *Node) *Node {
	if variable == nil {
		return nil
	}
	return variable.NodeValue(input)
}

var nf4Codebook = [...]float32{-1, -0.6961928, -0.52507305, -0.3949175, -0.28444138, -0.18477343, -0.09105004, 0, 0.0795803, 0.1609302, 0.2461123, 0.33791524, 0.44070983, 0.562617, 0.72295684, 1}

func nf4NearestCode(value float32) byte {
	best, distance := 0, float32(math.MaxFloat32)
	for index, code := range nf4Codebook {
		if candidate := float32(math.Abs(float64(value - code))); candidate < distance {
			best, distance = index, candidate
		}
	}
	return byte(best)
}

func min(left, right int) int {
	if left < right {
		return left
	}
	return right
}
