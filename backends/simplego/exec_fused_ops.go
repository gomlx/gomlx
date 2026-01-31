// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"math"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego/packgemm"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

func init() {
	setNodeExecutor(backends.OpTypeFusedSoftmax, priorityTyped, execFusedSoftmax)
	setNodeExecutor(backends.OpTypeFusedGelu, priorityTyped, execFusedGelu)
	setNodeExecutor(backends.OpTypeFusedLayerNorm, priorityTyped, execFusedLayerNorm)
	setNodeExecutor(backends.OpTypeFusedDense, priorityTyped, execFusedDense)
}

// computeAxisStrides returns the outer size, axis size, and inner size for iterating
// over an axis of the given shape. This decomposition allows softmax (and similar
// axis-based ops) to operate on any axis.
func computeAxisStrides(shape shapes.Shape, axis int) (outerSize, axisSize, innerSize int) {
	dims := shape.Dimensions
	outerSize = 1
	for i := 0; i < axis; i++ {
		outerSize *= dims[i]
	}
	axisSize = dims[axis]
	innerSize = 1
	for i := axis + 1; i < len(dims); i++ {
		innerSize *= dims[i]
	}
	return
}

// execFusedSoftmax implements optimized softmax with better cache locality.
// Three passes over the axis: find max, compute exp(x-max) and sum, then normalize.
func execFusedSoftmax(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	data := node.data.(*nodeFusedSoftmax)
	axis := data.axis
	input := inputs[0]
	output := backend.getBufferForShape(node.shape)

	switch input.shape.DType {
	case dtypes.Float32:
		softmax(input.flat.([]float32), output.flat.([]float32), axis, node.shape)
	case dtypes.Float64:
		softmax(input.flat.([]float64), output.flat.([]float64), axis, node.shape)
	default:
		return nil, errors.Wrapf(backends.ErrUnsupportedDType, "FusedSoftmax: dtype %s", input.shape.DType)
	}
	return output, nil
}

func softmax[T float32 | float64](input, output []T, axis int, shape shapes.Shape) {
	outerSize, axisSize, innerSize := computeAxisStrides(shape, axis)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			baseIdx := outer*axisSize*innerSize + inner

			// Pass 1: Find max.
			maxVal := T(math.Inf(-1))
			for i := 0; i < axisSize; i++ {
				idx := baseIdx + i*innerSize
				if input[idx] > maxVal {
					maxVal = input[idx]
				}
			}

			// Pass 2: Exp and sum.
			var sum T
			for i := 0; i < axisSize; i++ {
				idx := baseIdx + i*innerSize
				output[idx] = T(math.Exp(float64(input[idx] - maxVal)))
				sum += output[idx]
			}

			// Pass 3: Normalize.
			invSum := 1.0 / sum
			for i := 0; i < axisSize; i++ {
				idx := baseIdx + i*innerSize
				output[idx] *= invSum
			}
		}
	}
}

// execFusedGelu implements GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
func execFusedGelu(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	input := inputs[0]
	output := backend.getBufferForShape(node.shape)

	switch input.shape.DType {
	case dtypes.Float32:
		gelu(input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		gelu(input.flat.([]float64), output.flat.([]float64))
	default:
		return nil, errors.Wrapf(backends.ErrUnsupportedDType, "FusedGelu: dtype %s", input.shape.DType)
	}
	return output, nil
}

func gelu[T float32 | float64](input, output []T) {
	sqrt2Inv := T(1.0 / math.Sqrt(2.0))
	for i, x := range input {
		output[i] = x * 0.5 * (1.0 + T(math.Erf(float64(x*sqrt2Inv))))
	}
}

// execFusedLayerNorm implements layer normalization.
// For each sample: y = (x - mean) / sqrt(var + epsilon) * gamma + beta
func execFusedLayerNorm(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	data := node.data.(*nodeFusedLayerNorm)
	input := inputs[0]
	output := backend.getBufferForShape(node.shape)

	// Determine gamma/beta. inputs[0]=x, inputs[1]=gamma (optional), inputs[2]=beta (optional).
	var gamma, beta *Buffer
	if len(inputs) > 1 {
		gamma = inputs[1]
	}
	if len(inputs) > 2 {
		beta = inputs[2]
	}

	switch input.shape.DType {
	case dtypes.Float32:
		layerNorm[float32](input, output, gamma, beta, data.axes, data.epsilon)
	case dtypes.Float64:
		layerNorm[float64](input, output, gamma, beta, data.axes, data.epsilon)
	default:
		return nil, errors.Wrapf(backends.ErrUnsupportedDType, "FusedLayerNorm: dtype %s", input.shape.DType)
	}
	return output, nil
}

// layerNorm dispatches to the trailing-axes fast path or the general case.
func layerNorm[T float32 | float64](input, output, gamma, beta *Buffer, axes []int, epsilon float64) {
	inData := input.flat.([]T)
	outData := output.flat.([]T)
	dims := input.shape.Dimensions
	rank := len(dims)

	normSize := 1
	for _, a := range axes {
		normSize *= dims[a]
	}

	// Check for trailing axes fast path.
	isTrailingAxes := true
	for i, a := range axes {
		if a != rank-len(axes)+i {
			isTrailingAxes = false
			break
		}
	}

	var gammaData, betaData []T
	if gamma != nil {
		gammaData = gamma.flat.([]T)
	}
	if beta != nil {
		betaData = beta.flat.([]T)
	}

	if isTrailingAxes {
		trailingAxesLayerNorm(inData, outData, gammaData, betaData, normSize, epsilon)
	} else {
		arbitraryAxesLayerNorm(inData, outData, gammaData, betaData, dims, axes, normSize, epsilon)
	}
}

// trailingAxesLayerNorm handles the common case where normalization axes are the last N axes.
// Each contiguous block of normSize elements is one normalization group.
func trailingAxesLayerNorm[T float32 | float64](inData, outData, gammaData, betaData []T, normSize int, epsilon float64) {
	normSizeF := T(normSize)
	outerSize := len(inData) / normSize

	for outer := 0; outer < outerSize; outer++ {
		base := outer * normSize

		// Compute mean.
		var sum T
		for i := 0; i < normSize; i++ {
			sum += inData[base+i]
		}
		mean := sum / normSizeF

		// Compute variance.
		var varSum T
		for i := 0; i < normSize; i++ {
			diff := inData[base+i] - mean
			varSum += diff * diff
		}
		variance := varSum / normSizeF
		invStd := T(1.0 / math.Sqrt(float64(variance)+epsilon))

		// Normalize and apply scale/offset.
		for i := 0; i < normSize; i++ {
			normalized := (inData[base+i] - mean) * invStd
			if gammaData != nil {
				normalized *= gammaData[i]
			}
			if betaData != nil {
				normalized += betaData[i]
			}
			outData[base+i] = normalized
		}
	}
}

// arbitraryAxesLayerNorm handles normalization over arbitrary (non-trailing) axes
// using Shape.IterOnAxes for index iteration.
func arbitraryAxesLayerNorm[T float32 | float64](inData, outData, gammaData, betaData []T, dims, axes []int, normSize int, epsilon float64) {
	normSizeF := T(normSize)
	rank := len(dims)

	// Build set of norm axes for fast lookup.
	isNormAxis := make([]bool, rank)
	for _, a := range axes {
		isNormAxis[a] = true
	}

	// Build outer axes (those not in normalization set).
	outerAxes := make([]int, 0, rank-len(axes))
	for i := 0; i < rank; i++ {
		if !isNormAxis[i] {
			outerAxes = append(outerAxes, i)
		}
	}

	// Create shape for iteration. DType is irrelevant for IterOnAxes.
	shape := shapes.Make(dtypes.Float32, dims...)
	strides := shape.Strides()
	indices := make([]int, rank)

	for outerFlatIdx := range shape.IterOnAxes(outerAxes, strides, indices) {
		// Compute mean over norm axes.
		var sum T
		for flatIdx := range shape.IterOnAxes(axes, strides, indices) {
			sum += inData[flatIdx]
		}
		mean := sum / normSizeF

		// Compute variance.
		var varSum T
		for flatIdx := range shape.IterOnAxes(axes, strides, indices) {
			diff := inData[flatIdx] - mean
			varSum += diff * diff
		}
		variance := varSum / normSizeF
		invStd := T(1.0 / math.Sqrt(float64(variance)+epsilon))

		// Normalize and apply scale/offset.
		normFlatIdx := 0
		for flatIdx := range shape.IterOnAxes(axes, strides, indices) {
			normalized := (inData[flatIdx] - mean) * invStd
			if gammaData != nil {
				normalized *= gammaData[normFlatIdx]
			}
			if betaData != nil {
				normalized += betaData[normFlatIdx]
			}
			outData[flatIdx] = normalized
			normFlatIdx++
		}
		_ = outerFlatIdx
	}
}

// execFusedDense implements y = activation(x @ W + b).
// x: [..., in_features], weight: [in_features, out_features], bias: [out_features] (optional)
// Uses packGemm for the matmul. When bias is present, the output is pre-filled
// with broadcast bias and GEMM is called with beta=1 so that C = 1*A*B + 1*C.
func execFusedDense(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	x := inputs[0]
	weight := inputs[1]
	var bias *Buffer
	if len(inputs) > 2 {
		bias = inputs[2]
	}

	output := backend.getBufferForShape(node.shape)
	data := node.data.(*nodeFusedDense)

	switch x.shape.DType {
	case dtypes.Float32:
		dense(backend, x, weight, bias, output)
		applyActivation(output.flat.([]float32), data.activation)
	case dtypes.Float64:
		dense(backend, x, weight, bias, output)
		applyActivation(output.flat.([]float64), data.activation)
	default:
		return nil, errors.Wrapf(backends.ErrUnsupportedDType, "FusedDense: dtype %s", x.shape.DType)
	}
	return output, nil
}

// dense computes output = x @ weight + bias using packGemm.
// When bias is present, output is pre-filled with broadcast bias and GEMM uses beta=1.
func dense(backend *Backend, x, weight, bias, output *Buffer) {
	xShape := x.shape
	wShape := weight.shape
	inFeatures := xShape.Dimensions[xShape.Rank()-1]
	outFeatures := wShape.Dimensions[1]
	batchSize := xShape.Size() / inFeatures

	// Pre-fill output with bias so GEMM accumulates: C = 1*A*B + beta*C.
	var beta float64
	if bias != nil {
		broadcastBias(bias, output, batchSize, outFeatures)
		beta = 1
	}

	inputDType := x.shape.DType
	outputDType := output.shape.DType
	packgemm.GEMMDynamic(inputDType, outputDType, 1, beta,
		x.flat, weight.flat,
		1, batchSize, outFeatures, inFeatures,
		output.flat,
		getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.workers)
}

// broadcastBias fills output with bias repeated across the batch dimension.
func broadcastBias(bias, output *Buffer, batchSize, outFeatures int) {
	switch biasData := bias.flat.(type) {
	case []float32:
		outData := output.flat.([]float32)
		for b := 0; b < batchSize; b++ {
			copy(outData[b*outFeatures:(b+1)*outFeatures], biasData)
		}
	case []float64:
		outData := output.flat.([]float64)
		for b := 0; b < batchSize; b++ {
			copy(outData[b*outFeatures:(b+1)*outFeatures], biasData)
		}
	}
}

func applyActivation[T float32 | float64](data []T, activation backends.ActivationType) {
	switch activation {
	case backends.ActivationNone:
		// No-op.
	case backends.ActivationGelu:
		gelu(data, data) // in-place
	case backends.ActivationRelu:
		for i, x := range data {
			if x < 0 {
				data[i] = 0
			}
		}
	case backends.ActivationSilu:
		for i, x := range data {
			data[i] = x / (1.0 + T(math.Exp(float64(-x))))
		}
	case backends.ActivationTanh:
		for i, x := range data {
			data[i] = T(math.Tanh(float64(x)))
		}
	}
}
