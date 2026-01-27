// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package highway

import (
	"unsafe"

	"github.com/ajroetker/go-highway/hwy"
	"github.com/ajroetker/go-highway/hwy/contrib/algo"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/x448/float16"
)

func init() {
	// Register SIMD-accelerated unary operations with architecture priority.
	simplego.SetNodeExecutor(backends.OpTypeExp, simplego.RegisterPriorityArch, execExpHighway)
	simplego.SetNodeExecutor(backends.OpTypeLog, simplego.RegisterPriorityArch, execLogHighway)
	simplego.SetNodeExecutor(backends.OpTypeSin, simplego.RegisterPriorityArch, execSinHighway)
	simplego.SetNodeExecutor(backends.OpTypeCos, simplego.RegisterPriorityArch, execCosHighway)
	simplego.SetNodeExecutor(backends.OpTypeTanh, simplego.RegisterPriorityArch, execTanhHighway)
	simplego.SetNodeExecutor(backends.OpTypeLogistic, simplego.RegisterPriorityArch, execSigmoidHighway)
	simplego.SetNodeExecutor(backends.OpTypeErf, simplego.RegisterPriorityArch, execErfHighway)
}

// execExpHighway executes the Exp operation using SIMD.
func execExpHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	input, output := simplego.UnaryOperandAndOutput(backend, inputs, inputsOwned)
	applyTransform(input, output, algo.ExpTransformFloat32, algo.ExpTransformFloat64,
		algo.ExpTransformFloat16, algo.ExpTransformBFloat16)
	return output, nil
}

// execLogHighway executes the Log operation using SIMD.
func execLogHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	input, output := simplego.UnaryOperandAndOutput(backend, inputs, inputsOwned)
	applyTransform(input, output, algo.LogTransformFloat32, algo.LogTransformFloat64,
		algo.LogTransformFloat16, algo.LogTransformBFloat16)
	return output, nil
}

// execSinHighway executes the Sin operation using SIMD.
func execSinHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	input, output := simplego.UnaryOperandAndOutput(backend, inputs, inputsOwned)
	applyTransform(input, output, algo.SinTransformFloat32, algo.SinTransformFloat64,
		algo.SinTransformFloat16, algo.SinTransformBFloat16)
	return output, nil
}

// execCosHighway executes the Cos operation using SIMD.
func execCosHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	input, output := simplego.UnaryOperandAndOutput(backend, inputs, inputsOwned)
	applyTransform(input, output, algo.CosTransformFloat32, algo.CosTransformFloat64,
		algo.CosTransformFloat16, algo.CosTransformBFloat16)
	return output, nil
}

// execTanhHighway executes the Tanh operation using SIMD.
func execTanhHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	input, output := simplego.UnaryOperandAndOutput(backend, inputs, inputsOwned)
	applyTransform(input, output, algo.TanhTransformFloat32, algo.TanhTransformFloat64,
		algo.TanhTransformFloat16, algo.TanhTransformBFloat16)
	return output, nil
}

// execSigmoidHighway executes the Logistic (sigmoid) operation using SIMD.
func execSigmoidHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	input, output := simplego.UnaryOperandAndOutput(backend, inputs, inputsOwned)
	applyTransform(input, output, algo.SigmoidTransformFloat32, algo.SigmoidTransformFloat64,
		algo.SigmoidTransformFloat16, algo.SigmoidTransformBFloat16)
	return output, nil
}

// execErfHighway executes the Erf operation using SIMD.
func execErfHighway(backend *simplego.Backend, node *simplego.Node, inputs []*simplego.Buffer, inputsOwned []bool) (*simplego.Buffer, error) {
	input, output := simplego.UnaryOperandAndOutput(backend, inputs, inputsOwned)
	applyTransform(input, output, algo.ErfTransformFloat32, algo.ErfTransformFloat64,
		algo.ErfTransformFloat16, algo.ErfTransformBFloat16)
	return output, nil
}

// applyTransform applies the appropriate SIMD transform based on the input dtype.
func applyTransform(input, output *simplego.Buffer,
	f32Fn func([]float32, []float32),
	f64Fn func([]float64, []float64),
	f16Fn func([]hwy.Float16, []hwy.Float16),
	bf16Fn func([]hwy.BFloat16, []hwy.BFloat16)) {

	switch input.DType() {
	case dtypes.Float32:
		f32Fn(input.Flat().([]float32), output.Flat().([]float32))
	case dtypes.Float64:
		f64Fn(input.Flat().([]float64), output.Flat().([]float64))
	case dtypes.Float16:
		// Convert between x448/float16.Float16 and hwy.Float16 using unsafe
		inSlice := input.Flat().([]float16.Float16)
		outSlice := output.Flat().([]float16.Float16)
		inHwy := unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(inSlice))), len(inSlice))
		outHwy := unsafe.Slice((*hwy.Float16)(unsafe.Pointer(unsafe.SliceData(outSlice))), len(outSlice))
		f16Fn(inHwy, outHwy)
	case dtypes.BFloat16:
		// Convert between bfloat16.BFloat16 and hwy.BFloat16 using unsafe
		inSlice := input.Flat().([]bfloat16.BFloat16)
		outSlice := output.Flat().([]bfloat16.BFloat16)
		inHwy := unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(inSlice))), len(inSlice))
		outHwy := unsafe.Slice((*hwy.BFloat16)(unsafe.Pointer(unsafe.SliceData(outSlice))), len(outSlice))
		bf16Fn(inHwy, outHwy)
	}
}
