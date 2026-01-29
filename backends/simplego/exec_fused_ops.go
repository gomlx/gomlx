// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"math"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

func init() {
	setNodeExecutor(backends.OpTypeSoftmax, priorityTyped, execFusedSoftmax)
	setNodeExecutor(backends.OpTypeGelu, priorityTyped, execFusedGelu)
	setNodeExecutor(backends.OpTypeLayerNorm, priorityTyped, execFusedLayerNorm)
	setNodeExecutor(backends.OpTypeLinear, priorityTyped, execFusedLinear)
	setNodeExecutor(backends.OpTypeLinearActivation, priorityTyped, execFusedLinearActivation)
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
		softmaxFloat32(input.flat.([]float32), output.flat.([]float32), axis, node.shape)
	case dtypes.Float64:
		softmaxFloat64(input.flat.([]float64), output.flat.([]float64), axis, node.shape)
	default:
		return nil, errors.Errorf("FusedSoftmax: unsupported dtype %s", input.shape.DType)
	}
	return output, nil
}

func softmaxFloat32(input, output []float32, axis int, shape shapes.Shape) {
	outerSize, axisSize, innerSize := computeAxisStrides(shape, axis)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			baseIdx := outer*axisSize*innerSize + inner

			// Pass 1: Find max.
			maxVal := float32(math.Inf(-1))
			for i := 0; i < axisSize; i++ {
				idx := baseIdx + i*innerSize
				if input[idx] > maxVal {
					maxVal = input[idx]
				}
			}

			// Pass 2: Exp and sum.
			var sum float32
			for i := 0; i < axisSize; i++ {
				idx := baseIdx + i*innerSize
				output[idx] = float32(math.Exp(float64(input[idx] - maxVal)))
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

func softmaxFloat64(input, output []float64, axis int, shape shapes.Shape) {
	outerSize, axisSize, innerSize := computeAxisStrides(shape, axis)
	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			baseIdx := outer*axisSize*innerSize + inner

			maxVal := math.Inf(-1)
			for i := 0; i < axisSize; i++ {
				idx := baseIdx + i*innerSize
				if input[idx] > maxVal {
					maxVal = input[idx]
				}
			}

			var sum float64
			for i := 0; i < axisSize; i++ {
				idx := baseIdx + i*innerSize
				output[idx] = math.Exp(input[idx] - maxVal)
				sum += output[idx]
			}

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
		geluFloat32(input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		geluFloat64(input.flat.([]float64), output.flat.([]float64))
	default:
		return nil, errors.Errorf("FusedGelu: unsupported dtype %s", input.shape.DType)
	}
	return output, nil
}

func geluFloat32(input, output []float32) {
	sqrt2Inv := float32(1.0 / math.Sqrt(2.0))
	for i, x := range input {
		output[i] = x * 0.5 * (1.0 + float32(math.Erf(float64(x*sqrt2Inv))))
	}
}

func geluFloat64(input, output []float64) {
	sqrt2Inv := 1.0 / math.Sqrt(2.0)
	for i, x := range input {
		output[i] = x * 0.5 * (1.0 + math.Erf(x*sqrt2Inv))
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
		layerNormFloat32(input, output, gamma, beta, data.axes, data.epsilon)
	case dtypes.Float64:
		layerNormFloat64(input, output, gamma, beta, data.axes, data.epsilon)
	default:
		return nil, errors.Errorf("FusedLayerNorm: unsupported dtype %s", input.shape.DType)
	}
	return output, nil
}

func layerNormFloat32(input, output, gamma, beta *Buffer, axes []int, epsilon float64) {
	inData := input.flat.([]float32)
	outData := output.flat.([]float32)
	dims := input.shape.Dimensions

	// Compute the size of the normalization region (product of axes dimensions).
	normSize := 1
	for _, a := range axes {
		normSize *= dims[a]
	}
	normSizeF := float32(normSize)
	totalSize := input.shape.Size()

	// Compute the outer size (number of independent normalizations).
	outerSize := totalSize / normSize

	// Compute stride pattern: for each outer index, which elements belong together.
	// For the common case of normalizing over the last N axes, we can use a fast path.
	isTrailingAxes := true
	rank := len(dims)
	for i, a := range axes {
		if a != rank-len(axes)+i {
			isTrailingAxes = false
			break
		}
	}

	var gammaData, betaData []float32
	if gamma != nil {
		gammaData = gamma.flat.([]float32)
	}
	if beta != nil {
		betaData = beta.flat.([]float32)
	}

	if isTrailingAxes {
		// Fast path: normalizing over trailing axes.
		// Each contiguous block of normSize elements is one normalization group.
		for outer := 0; outer < outerSize; outer++ {
			base := outer * normSize

			// Compute mean.
			var sum float32
			for i := 0; i < normSize; i++ {
				sum += inData[base+i]
			}
			mean := sum / normSizeF

			// Compute variance.
			var varSum float32
			for i := 0; i < normSize; i++ {
				diff := inData[base+i] - mean
				varSum += diff * diff
			}
			variance := varSum / normSizeF
			invStd := float32(1.0 / math.Sqrt(float64(variance)+epsilon))

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
	} else {
		// General case: copy input, normalize element-wise using index decomposition.
		// This is slower but handles arbitrary axis combinations.
		copy(outData, inData)

		// For the general case, we need to compute mean/variance over the specified axes.
		// Use the same approach as the fast path but with strided access.
		// For simplicity, fall back to computing per-element with index decomposition.
		// This is a reference implementation; the fast path above handles the common case.
		eps32 := float32(epsilon)

		// Compute strides.
		strides := make([]int, rank)
		strides[rank-1] = 1
		for i := rank - 2; i >= 0; i-- {
			strides[i] = strides[i+1] * dims[i+1]
		}

		// Build set of norm axes for fast lookup.
		isNormAxis := make([]bool, rank)
		for _, a := range axes {
			isNormAxis[a] = true
		}

		// For each outer group, compute mean and variance across the norm axes.
		// An "outer group" is identified by fixing the non-norm axes.
		outerDims := make([]int, 0, rank-len(axes))
		outerStrides := make([]int, 0, rank-len(axes))
		normDims := make([]int, 0, len(axes))
		normStrides := make([]int, 0, len(axes))
		for i := 0; i < rank; i++ {
			if isNormAxis[i] {
				normDims = append(normDims, dims[i])
				normStrides = append(normStrides, strides[i])
			} else {
				outerDims = append(outerDims, dims[i])
				outerStrides = append(outerStrides, strides[i])
			}
		}

		// Iterate over outer indices.
		outerIdx := make([]int, len(outerDims))
		for {
			// Compute base index for this outer group.
			outerBase := 0
			for i, idx := range outerIdx {
				outerBase += idx * outerStrides[i]
			}

			// Compute mean over norm axes.
			var sum float32
			normIdx := make([]int, len(normDims))
			for {
				offset := 0
				for i, idx := range normIdx {
					offset += idx * normStrides[i]
				}
				sum += inData[outerBase+offset]

				// Increment normIdx.
				carry := true
				for i := len(normIdx) - 1; i >= 0 && carry; i-- {
					normIdx[i]++
					if normIdx[i] < normDims[i] {
						carry = false
					} else {
						normIdx[i] = 0
					}
				}
				if carry {
					break
				}
			}
			mean := sum / normSizeF

			// Compute variance.
			var varSum float32
			normIdx = make([]int, len(normDims))
			for {
				offset := 0
				for i, idx := range normIdx {
					offset += idx * normStrides[i]
				}
				diff := inData[outerBase+offset] - mean
				varSum += diff * diff

				carry := true
				for i := len(normIdx) - 1; i >= 0 && carry; i-- {
					normIdx[i]++
					if normIdx[i] < normDims[i] {
						carry = false
					} else {
						normIdx[i] = 0
					}
				}
				if carry {
					break
				}
			}
			variance := varSum / normSizeF
			invStd := float32(1.0 / math.Sqrt(float64(variance+eps32)))

			// Normalize.
			normIdx = make([]int, len(normDims))
			normFlatIdx := 0
			for {
				offset := 0
				for i, idx := range normIdx {
					offset += idx * normStrides[i]
				}
				normalized := (inData[outerBase+offset] - mean) * invStd
				if gammaData != nil {
					normalized *= gammaData[normFlatIdx]
				}
				if betaData != nil {
					normalized += betaData[normFlatIdx]
				}
				outData[outerBase+offset] = normalized

				normFlatIdx++
				carry := true
				for i := len(normIdx) - 1; i >= 0 && carry; i-- {
					normIdx[i]++
					if normIdx[i] < normDims[i] {
						carry = false
					} else {
						normIdx[i] = 0
					}
				}
				if carry {
					break
				}
			}

			// Increment outerIdx.
			carry := true
			for i := len(outerIdx) - 1; i >= 0 && carry; i-- {
				outerIdx[i]++
				if outerIdx[i] < outerDims[i] {
					carry = false
				} else {
					outerIdx[i] = 0
				}
			}
			if carry {
				break
			}
		}
	}
}

func layerNormFloat64(input, output, gamma, beta *Buffer, axes []int, epsilon float64) {
	inData := input.flat.([]float64)
	outData := output.flat.([]float64)
	dims := input.shape.Dimensions

	normSize := 1
	for _, a := range axes {
		normSize *= dims[a]
	}
	normSizeF := float64(normSize)

	// Check for trailing axes fast path.
	isTrailingAxes := true
	rank := len(dims)
	for i, a := range axes {
		if a != rank-len(axes)+i {
			isTrailingAxes = false
			break
		}
	}

	var gammaData, betaData []float64
	if gamma != nil {
		gammaData = gamma.flat.([]float64)
	}
	if beta != nil {
		betaData = beta.flat.([]float64)
	}

	if isTrailingAxes {
		totalSize := input.shape.Size()
		outerSize := totalSize / normSize
		for outer := 0; outer < outerSize; outer++ {
			base := outer * normSize

			var sum float64
			for i := 0; i < normSize; i++ {
				sum += inData[base+i]
			}
			mean := sum / normSizeF

			var varSum float64
			for i := 0; i < normSize; i++ {
				diff := inData[base+i] - mean
				varSum += diff * diff
			}
			variance := varSum / normSizeF
			invStd := 1.0 / math.Sqrt(variance+epsilon)

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
	} else {
		// General case: similar to float32 version above.
		copy(outData, inData)

		strides := make([]int, rank)
		strides[rank-1] = 1
		for i := rank - 2; i >= 0; i-- {
			strides[i] = strides[i+1] * dims[i+1]
		}

		isNormAxis := make([]bool, rank)
		for _, a := range axes {
			isNormAxis[a] = true
		}

		outerDims := make([]int, 0, rank-len(axes))
		outerStrides := make([]int, 0, rank-len(axes))
		normDims := make([]int, 0, len(axes))
		normStrides := make([]int, 0, len(axes))
		for i := 0; i < rank; i++ {
			if isNormAxis[i] {
				normDims = append(normDims, dims[i])
				normStrides = append(normStrides, strides[i])
			} else {
				outerDims = append(outerDims, dims[i])
				outerStrides = append(outerStrides, strides[i])
			}
		}

		outerIdx := make([]int, len(outerDims))
		for {
			outerBase := 0
			for i, idx := range outerIdx {
				outerBase += idx * outerStrides[i]
			}

			var sum float64
			normIdx := make([]int, len(normDims))
			for {
				offset := 0
				for i, idx := range normIdx {
					offset += idx * normStrides[i]
				}
				sum += inData[outerBase+offset]
				carry := true
				for i := len(normIdx) - 1; i >= 0 && carry; i-- {
					normIdx[i]++
					if normIdx[i] < normDims[i] {
						carry = false
					} else {
						normIdx[i] = 0
					}
				}
				if carry {
					break
				}
			}
			mean := sum / normSizeF

			var varSum float64
			normIdx = make([]int, len(normDims))
			for {
				offset := 0
				for i, idx := range normIdx {
					offset += idx * normStrides[i]
				}
				diff := inData[outerBase+offset] - mean
				varSum += diff * diff
				carry := true
				for i := len(normIdx) - 1; i >= 0 && carry; i-- {
					normIdx[i]++
					if normIdx[i] < normDims[i] {
						carry = false
					} else {
						normIdx[i] = 0
					}
				}
				if carry {
					break
				}
			}
			variance := varSum / normSizeF
			invStd := 1.0 / math.Sqrt(variance+epsilon)

			normIdx = make([]int, len(normDims))
			normFlatIdx := 0
			for {
				offset := 0
				for i, idx := range normIdx {
					offset += idx * normStrides[i]
				}
				normalized := (inData[outerBase+offset] - mean) * invStd
				if gammaData != nil {
					normalized *= gammaData[normFlatIdx]
				}
				if betaData != nil {
					normalized += betaData[normFlatIdx]
				}
				outData[outerBase+offset] = normalized
				normFlatIdx++
				carry := true
				for i := len(normIdx) - 1; i >= 0 && carry; i-- {
					normIdx[i]++
					if normIdx[i] < normDims[i] {
						carry = false
					} else {
						normIdx[i] = 0
					}
				}
				if carry {
					break
				}
			}

			carry := true
			for i := len(outerIdx) - 1; i >= 0 && carry; i-- {
				outerIdx[i]++
				if outerIdx[i] < outerDims[i] {
					carry = false
				} else {
					outerIdx[i] = 0
				}
			}
			if carry {
				break
			}
		}
	}
}

// execFusedLinear implements y = x @ W^T + b.
// x: [..., in_features], weight: [out_features, in_features], bias: [out_features] (optional)
func execFusedLinear(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	x := inputs[0]
	weight := inputs[1]
	var bias *Buffer
	if len(inputs) > 2 {
		bias = inputs[2]
	}

	output := backend.getBufferForShape(node.shape)

	switch x.shape.DType {
	case dtypes.Float32:
		linearFloat32(x, weight, bias, output)
	case dtypes.Float64:
		linearFloat64(x, weight, bias, output)
	default:
		return nil, errors.Errorf("FusedLinear: unsupported dtype %s", x.shape.DType)
	}
	return output, nil
}

func linearFloat32(x, weight, bias, output *Buffer) {
	xData := x.flat.([]float32)
	wData := weight.flat.([]float32)
	outData := output.flat.([]float32)

	inFeatures := x.shape.Dimensions[x.shape.Rank()-1]
	outFeatures := weight.shape.Dimensions[0]
	batchSize := x.shape.Size() / inFeatures

	var biasData []float32
	if bias != nil {
		biasData = bias.flat.([]float32)
	}

	// y = x @ W^T + b
	// For each batch element, compute matmul with weight transposed.
	for b := 0; b < batchSize; b++ {
		xBase := b * inFeatures
		oBase := b * outFeatures
		for o := 0; o < outFeatures; o++ {
			var sum float32
			wBase := o * inFeatures
			for i := 0; i < inFeatures; i++ {
				sum += xData[xBase+i] * wData[wBase+i]
			}
			if biasData != nil {
				sum += biasData[o]
			}
			outData[oBase+o] = sum
		}
	}
}

func linearFloat64(x, weight, bias, output *Buffer) {
	xData := x.flat.([]float64)
	wData := weight.flat.([]float64)
	outData := output.flat.([]float64)

	inFeatures := x.shape.Dimensions[x.shape.Rank()-1]
	outFeatures := weight.shape.Dimensions[0]
	batchSize := x.shape.Size() / inFeatures

	var biasData []float64
	if bias != nil {
		biasData = bias.flat.([]float64)
	}

	for b := 0; b < batchSize; b++ {
		xBase := b * inFeatures
		oBase := b * outFeatures
		for o := 0; o < outFeatures; o++ {
			var sum float64
			wBase := o * inFeatures
			for i := 0; i < inFeatures; i++ {
				sum += xData[xBase+i] * wData[wBase+i]
			}
			if biasData != nil {
				sum += biasData[o]
			}
			outData[oBase+o] = sum
		}
	}
}

// execFusedLinearActivation implements y = activation(x @ W^T + b).
func execFusedLinearActivation(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	x := inputs[0]
	weight := inputs[1]
	var bias *Buffer
	if len(inputs) > 2 {
		bias = inputs[2]
	}

	output := backend.getBufferForShape(node.shape)
	data := node.data.(*nodeFusedLinearActivation)

	switch x.shape.DType {
	case dtypes.Float32:
		linearFloat32(x, weight, bias, output)
		applyActivationFloat32(output.flat.([]float32), data.activation)
	case dtypes.Float64:
		linearFloat64(x, weight, bias, output)
		applyActivationFloat64(output.flat.([]float64), data.activation)
	default:
		return nil, errors.Errorf("FusedLinearActivation: unsupported dtype %s", x.shape.DType)
	}
	return output, nil
}

func applyActivationFloat32(data []float32, activation backends.ActivationType) {
	sqrt2Inv := float32(1.0 / math.Sqrt(2.0))
	switch activation {
	case backends.ActivationNone:
		// No-op.
	case backends.ActivationGelu:
		for i, x := range data {
			data[i] = x * 0.5 * (1.0 + float32(math.Erf(float64(x*sqrt2Inv))))
		}
	case backends.ActivationRelu:
		for i, x := range data {
			if x < 0 {
				data[i] = 0
			}
		}
	case backends.ActivationSilu:
		for i, x := range data {
			data[i] = x / (1.0 + float32(math.Exp(float64(-x))))
		}
	case backends.ActivationTanh:
		for i, x := range data {
			data[i] = float32(math.Tanh(float64(x)))
		}
	}
}

func applyActivationFloat64(data []float64, activation backends.ActivationType) {
	sqrt2Inv := 1.0 / math.Sqrt(2.0)
	switch activation {
	case backends.ActivationNone:
		// No-op.
	case backends.ActivationGelu:
		for i, x := range data {
			data[i] = x * 0.5 * (1.0 + math.Erf(x*sqrt2Inv))
		}
	case backends.ActivationRelu:
		for i, x := range data {
			if x < 0 {
				data[i] = 0
			}
		}
	case backends.ActivationSilu:
		for i, x := range data {
			data[i] = x / (1.0 + math.Exp(-x))
		}
	case backends.ActivationTanh:
		for i, x := range data {
			data[i] = math.Tanh(x)
		}
	}
}
