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
		return nil, errors.Errorf("FusedSoftmax: unsupported dtype %s", input.shape.DType)
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
		return nil, errors.Errorf("FusedGelu: unsupported dtype %s", input.shape.DType)
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
		return nil, errors.Errorf("FusedDense: unsupported dtype %s", x.shape.DType)
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
	sqrt2Inv := T(1.0 / math.Sqrt(2.0))
	switch activation {
	case backends.ActivationNone:
		// No-op.
	case backends.ActivationGelu:
		for i, x := range data {
			data[i] = x * 0.5 * (1.0 + T(math.Erf(float64(x*sqrt2Inv))))
		}
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
