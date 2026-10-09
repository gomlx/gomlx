// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package peft

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/initializer"
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
		return nil, errors.New("peft: invalid NF4 weight")
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
		return errors.New("peft: invalid NF4 weight")
	}
	blocks := (w.OutputFeatures + w.BlockSize - 1) / w.BlockSize
	if len(w.Packed) != w.InputFeatures*(w.OutputFeatures/2) || (len(w.Scales) != w.InputFeatures*blocks && len(w.ScaleCodes) != w.InputFeatures*blocks) {
		return errors.New("peft: invalid NF4 weight")
	}
	if len(w.ScaleCodes) > 0 && (w.ScaleBlockSize <= 0 || len(w.ScaleScales) != w.InputFeatures*((blocks+w.ScaleBlockSize-1)/w.ScaleBlockSize)) {
		return errors.New("peft: invalid NF4 weight")
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
				maximum = max(maximum, scale)
			}
			groupScale := maximum / 255
			w.ScaleScales[input*((blocks+scaleBlockSize-1)/scaleBlockSize)+group] = groupScale
			if groupScale == 0 {
				continue
			}
			for block := start; block < end; block++ {
				w.ScaleCodes[input*blocks+block] = byte(min(int(math.Round(float64(w.Scales[input*blocks+block]/groupScale))), 255))
			}
		}
	}
	w.Scales = nil
}

// NF4 is a graph-building QLoRA projection. Configure its LoRA update with
// the same scope parameters as New, then call Done.
type NF4 struct {
	scope  *model.Scope
	input  *Node
	weight *NF4Weight
	bias   *model.Variable
	config config
}

// NewNF4 configures a QLoRA projection around a packed NF4 base weight.
func NewNF4(scope *model.Scope, input *Node, weight *NF4Weight, bias *model.Variable) *NF4 {
	if scope == nil || input == nil {
		panic(errors.New("peft.NewNF4 requires non-nil scope and input"))
	}
	if err := weight.validate(); err != nil {
		panic(err)
	}
	config := configFromScope(scope)
	config.validate()
	if config.layout != compute.DenseLayoutInputOutputs {
		panic(errors.Errorf("peft.NewNF4 only supports DenseLayoutInputOutputs"))
	}
	return &NF4{scope: scope, input: input, weight: weight, bias: bias, config: config}
}

// Done builds the frozen NF4 base projection plus the optional LoRA update.
func (l *NF4) Done() *Node {
	if l.bias != nil && (l.bias.Shape().Rank() != 1 || l.bias.Shape().Dimensions[0] != l.weight.OutputFeatures) {
		panic(errors.Errorf("peft: bias shape must be [%d], got %s", l.weight.OutputFeatures, l.bias.Shape()))
	}
	qScope := l.scope.In("qlora")
	packed := qScope.VariableWithValue("nf4_packed", tensors.FromFlatDataAndDimensions(l.weight.Packed, l.weight.InputFeatures, l.weight.OutputFeatures/2)).SetTrainable(false)
	blocks := (l.weight.OutputFeatures + l.weight.BlockSize - 1) / l.weight.BlockSize
	var scales, scaleCodes, scaleScales *model.Variable
	if len(l.weight.ScaleCodes) == 0 {
		scales = qScope.VariableWithValue("nf4_scales", tensors.FromFlatDataAndDimensions(l.weight.Scales, l.weight.InputFeatures, blocks)).SetTrainable(false)
	} else {
		scaleCodes = qScope.VariableWithValue("nf4_scale_codes", tensors.FromFlatDataAndDimensions(l.weight.ScaleCodes, l.weight.InputFeatures, blocks)).SetTrainable(false)
		scaleScales = qScope.VariableWithValue("nf4_scale_scales", tensors.FromFlatDataAndDimensions(l.weight.ScaleScales, l.weight.InputFeatures, (blocks+l.weight.ScaleBlockSize-1)/l.weight.ScaleBlockSize)).SetTrainable(false)
	}
	base := nn.QuantizedDense(l.input, Reshape(Bitcast(packed.NodeValue(l.input), dtypes.Uint4), l.weight.InputFeatures, l.weight.OutputFeatures), &Quantization{Scheme: compute.QuantNF4, Scale: nf4Scales(l.input, scales, scaleCodes, scaleScales, l.weight), BlockAxis: 1, BlockSize: l.weight.BlockSize}, nodeValue(l.bias, l.input))
	if l.config.rank == 0 {
		return base
	}
	if l.bias != nil {
		l.bias.SetTrainable(l.config.bias != BiasNone)
	}
	adapterScope := qScope.In("lora")
	bound := 1 / math.Sqrt(float64(l.weight.InputFeatures))
	a := adapterScope.WithInitializer(initializer.RandomUniformFn(adapterScope, -bound, bound)).VariableWithShape("A", shapes.Make(dtypes.Float32, l.weight.InputFeatures, l.config.rank))
	b := adapterScope.WithInitializer(initializer.Zero).VariableWithShape("B", shapes.Make(dtypes.Float32, l.config.rank, l.weight.OutputFeatures))
	update := MatMul(l.input, a.NodeValue(l.input))
	update = layers.DropoutStatic(l.scope, update, float64(l.config.dropout))
	update = MatMul(update, b.NodeValue(l.input))
	return Add(base, MulScalar(update, l.config.alpha/float32(l.config.rank)))
}

func nf4Scales(input *Node, scales, scaleCodes, scaleScales *model.Variable, weight *NF4Weight) *Node {
	if scales != nil {
		return scales.NodeValue(input)
	}
	blocks := (weight.OutputFeatures + weight.BlockSize - 1) / weight.BlockSize
	groups := scaleScales.Shape().Dimensions[1]
	groupSize := (blocks + groups - 1) / groups
	indices := make([]int32, blocks)
	for index := range indices {
		indices[index] = int32(index / groupSize)
	}
	group := Reshape(Const(input.Graph(), indices), blocks, 1)
	expanded := Gather(Transpose(scaleScales.NodeValue(input), 0, 1), group)
	expanded = Transpose(expanded, 0, 1)
	return Mul(ConvertDType(scaleCodes.NodeValue(input), dtypes.Float32), expanded)
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
