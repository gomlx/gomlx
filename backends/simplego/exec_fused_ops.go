// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"math"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

func init() {
	setNodeExecutor(backends.OpTypeFusedSoftmax, priorityTyped, execFusedSoftmax)
	setNodeExecutor(backends.OpTypeFusedGelu, priorityTyped, execFusedGelu)
	setNodeExecutor(backends.OpTypeFusedLayerNorm, priorityTyped, execFusedLayerNorm)
	setNodeExecutor(backends.OpTypeFusedDense, priorityTyped, execFusedDense)
	setNodeExecutor(backends.OpTypeFusedMultiHeadSDPA, priorityTyped, execFusedMultiHeadSDPA)
	multiOutputsNodeExecutors[backends.OpTypeFusedQKVDense] = execFusedQKVDense
}

// computeAxisStrides returns the outer size, axis size, and inner size for iterating
// over an axis of the given shape. This decomposition allows softmax (and similar
// axis-based ops) to operate on any axis.
func computeAxisStrides(shape shapes.Shape, axis int) (outerSize, axisSize, innerSize int) {
	dims := shape.Dimensions
	outerSize = 1
	for i := range axis {
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
		return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedSoftmax: dtype %s", input.shape.DType)
	}
	return output, nil
}

func softmax[T float32 | float64](input, output []T, axis int, shape shapes.Shape) {
	outerSize, axisSize, innerSize := computeAxisStrides(shape, axis)
	for outer := range outerSize {
		for inner := range innerSize {
			baseIdx := outer*axisSize*innerSize + inner

			// Pass 1: Find max.
			maxVal := T(math.Inf(-1))
			for i := range axisSize {
				idx := baseIdx + i*innerSize
				if input[idx] > maxVal {
					maxVal = input[idx]
				}
			}

			// Pass 2: Exp and sum.
			var sum T
			for i := range axisSize {
				idx := baseIdx + i*innerSize
				output[idx] = T(math.Exp(float64(input[idx] - maxVal)))
				sum += output[idx]
			}

			// Pass 3: Normalize.
			invSum := 1.0 / sum
			for i := range axisSize {
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
		gelu(backend, input.flat.([]float32), output.flat.([]float32))
	case dtypes.Float64:
		gelu(backend, input.flat.([]float64), output.flat.([]float64))
	default:
		return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedGelu: dtype %s", input.shape.DType)
	}
	return output, nil
}

// minParallelizeChunk is the minimum number of elements to parallelize over.
const minParallelizeChunk = 4096

func gelu[T float32 | float64](backend *Backend, input, output []T) {
	n := len(input)
	if backend != nil && backend.workers.IsEnabled() && n > minParallelizeChunk {
		var wg sync.WaitGroup
		for ii := 0; ii < n; ii += minParallelizeChunk {
			iiEnd := min(ii+minParallelizeChunk, n)
			wg.Add(1)
			backend.workers.WaitToStart(func() {
				geluChunk(input[ii:iiEnd], output[ii:iiEnd])
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		geluChunk(input, output)
	}
}

func geluChunk[T float32 | float64](input, output []T) {
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
		return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedLayerNorm: dtype %s", input.shape.DType)
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

	for outer := range outerSize {
		base := outer * normSize

		// Compute mean.
		var sum T
		for i := range normSize {
			sum += inData[base+i]
		}
		mean := sum / normSizeF

		// Compute variance.
		var varSum T
		for i := range normSize {
			diff := inData[base+i] - mean
			varSum += diff * diff
		}
		variance := varSum / normSizeF
		invStd := T(1.0 / math.Sqrt(float64(variance)+epsilon))

		// Normalize and apply scale/offset.
		for i := range normSize {
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
	for i := range rank {
		if !isNormAxis[i] {
			outerAxes = append(outerAxes, i)
		}
	}

	// Create shape for iteration. DType is irrelevant for IterOnAxes.
	shape := shapes.Make(dtypes.Float32, dims...)
	strides := shape.Strides()
	outerIndices := make([]int, rank)
	normIndices := make([]int, rank)

	for outerFlatIdx := range shape.IterOnAxes(outerAxes, strides, outerIndices) {
		// Compute mean over norm axes.
		var sum T
		copy(normIndices, outerIndices)
		for flatIdx := range shape.IterOnAxes(axes, strides, normIndices) {
			sum += inData[flatIdx]
		}
		mean := sum / normSizeF

		// Compute variance.
		var varSum T
		copy(normIndices, outerIndices)
		for flatIdx := range shape.IterOnAxes(axes, strides, normIndices) {
			diff := inData[flatIdx] - mean
			varSum += diff * diff
		}
		variance := varSum / normSizeF
		invStd := T(1.0 / math.Sqrt(float64(variance)+epsilon))

		// Normalize and apply scale/offset.
		normFlatIdx := 0
		copy(normIndices, outerIndices)
		for flatIdx := range shape.IterOnAxes(axes, strides, normIndices) {
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

// execFusedDense implements y = activation(matmul + bias).
// inputs layout: [dotResult, x, weight, bias?]
// inputs[0] is the DotGeneral result (matmul already computed by the backend).
// inputs[1] is x, inputs[2] is weight (unused by this executor).
// inputs[3] is the optional bias.
func execFusedDense(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	matmul := inputs[0]
	// inputs layout: [dotResult, x, weight, bias?]
	var bias *Buffer
	if len(inputs) > 3 {
		bias = inputs[3]
	}

	data := node.data.(*nodeFusedDense)

	// If no bias and no activation, just return the matmul result directly.
	if bias == nil && data.activation == backends.ActivationNone {
		if inputsOwned[0] {
			inputs[0] = nil // Signal to executor that we reused the input.
			return matmul, nil
		}
		output := backend.getBufferForShape(node.shape)
		copyFlat(output.flat, matmul.flat)
		return output, nil
	}

	// Try to reuse the matmul buffer if owned; otherwise allocate.
	var output *Buffer
	if inputsOwned[0] {
		output = matmul
		inputs[0] = nil // Signal to executor that we reused the input.
	} else {
		output = backend.getBufferForShape(node.shape)
		copyFlat(output.flat, matmul.flat)
	}

	switch output.shape.DType {
	case dtypes.Float32:
		outData := output.flat.([]float32)
		if bias != nil {
			addBias(outData, bias.flat.([]float32))
		}
		applyActivation(backend, outData, data.activation)
	case dtypes.Float64:
		outData := output.flat.([]float64)
		if bias != nil {
			addBias(outData, bias.flat.([]float64))
		}
		applyActivation(backend, outData, data.activation)
	default:
		return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedDense: dtype %s", output.shape.DType)
	}
	return output, nil
}

// addBias adds bias to each row of the output in-place.
// output shape is [..., outFeatures], bias shape is [outFeatures].
func addBias[T float32 | float64](output, bias []T) {
	outFeatures := len(bias)
	for i, v := range output {
		output[i] = v + bias[i%outFeatures]
	}
}

func applyActivation[T float32 | float64](backend *Backend, data []T, activation backends.ActivationType) {
	switch activation {
	case backends.ActivationNone:
		// No-op.
	case backends.ActivationGelu:
		gelu(backend, data, data) // in-place
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

// computeMaskStrides returns (batchStride, headStride) for indexing into a mask
// tensor based on its rank. Dimensions of size 1 are broadcast (stride 0).
//
//	rank 2: [seqLen, kvLen]                     → (0, 0)
//	rank 3: [batch, seqLen, kvLen]              → (seqLen*kvLen, 0) or (0, 0) if dim[0]==1
//	rank 4: [batch, heads, seqLen, kvLen]       → strides computed per dim
func computeMaskStrides(dims []int) (batchStride, headStride int) {
	switch len(dims) {
	case 2:
		return 0, 0
	case 3:
		if dims[0] <= 1 {
			return 0, 0
		}
		return dims[1] * dims[2], 0
	case 4:
		if dims[0] > 1 {
			batchStride = dims[1] * dims[2] * dims[3]
		}
		if dims[1] > 1 {
			headStride = dims[2] * dims[3]
		}
		return batchStride, headStride
	default:
		return 0, 0
	}
}

// execFusedMultiHeadSDPA implements multi-head scaled dot-product attention.
// q: [batch, numHeads, seqLen, headDim], k/v: [batch, numKVHeads, kvLen, headDim]
// mask: optional additive mask of rank 2–4 (broadcasting via strides)
// output: [batch, numHeads, seqLen, headDim]
func execFusedMultiHeadSDPA(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	data := node.data.(*nodeFusedMultiHeadSDPA)
	q := inputs[0]
	k := inputs[1]
	v := inputs[2]
	var mask *Buffer
	if len(inputs) > 3 {
		mask = inputs[3]
	}
	output := backend.getBufferForShape(node.shape)

	// Compute mask strides for broadcasting.
	var maskBatchStride, maskHeadStride int
	if mask != nil {
		maskBatchStride, maskHeadStride = computeMaskStrides(mask.shape.Dimensions)
	}

	switch q.shape.DType {
	case dtypes.Float32:
		var maskData []float32
		if mask != nil {
			maskData = mask.flat.([]float32)
		}
		multiHeadSDPA(
			q.flat.([]float32), k.flat.([]float32), v.flat.([]float32), maskData, output.flat.([]float32),
			q.shape.Dimensions[0], data.numHeads, data.numKVHeads,
			q.shape.Dimensions[2], k.shape.Dimensions[2], q.shape.Dimensions[3],
			maskBatchStride, maskHeadStride,
			float32(data.scale), data.causal,
		)
	case dtypes.Float64:
		var maskData []float64
		if mask != nil {
			maskData = mask.flat.([]float64)
		}
		multiHeadSDPA(
			q.flat.([]float64), k.flat.([]float64), v.flat.([]float64), maskData, output.flat.([]float64),
			q.shape.Dimensions[0], data.numHeads, data.numKVHeads,
			q.shape.Dimensions[2], k.shape.Dimensions[2], q.shape.Dimensions[3],
			maskBatchStride, maskHeadStride,
			data.scale, data.causal,
		)
	default:
		return nil, errors.Errorf("FusedMultiHeadSDPA: unsupported dtype %s", q.shape.DType)
	}
	return output, nil
}

func sdpa[T float32 | float64](q, k, v, mask, scores, output []T, seqLen, kvLen, headDim int, scale T, causal bool) {
	// scores[i][j] = sum_d(q[i][d] * k[j][d]) * scale + mask[i][j]
	for i := range seqLen {
		rowMax := T(math.Inf(-1))
		for j := range kvLen {
			if causal && j > i {
				scores[i*kvLen+j] = T(math.Inf(-1))
				continue
			}
			var dot T
			for d := range headDim {
				dot += q[i*headDim+d] * k[j*headDim+d]
			}
			s := dot * scale
			if mask != nil {
				s += mask[i*kvLen+j]
			}
			scores[i*kvLen+j] = s
			if s > rowMax {
				rowMax = s
			}
		}

		// Softmax: exp(scores - max) and sum.
		var sum T
		for j := range kvLen {
			scores[i*kvLen+j] = T(math.Exp(float64(scores[i*kvLen+j] - rowMax)))
			sum += scores[i*kvLen+j]
		}
		invSum := 1.0 / sum
		for j := range kvLen {
			scores[i*kvLen+j] *= invSum
		}

		// output[i][d] = sum_j(scores[i][j] * v[j][d])
		for d := range headDim {
			var acc T
			for j := range kvLen {
				acc += scores[i*kvLen+j] * v[j*headDim+d]
			}
			output[i*headDim+d] = acc
		}
	}
}

func multiHeadSDPA[T float32 | float64](q, k, v, mask, output []T,
	batchSize, numHeads, numKVHeads, seqLen, kvLen, headDim int,
	maskBatchStride, maskHeadStride int,
	scale T, causal bool,
) {
	headsPerKV := numHeads / numKVHeads
	scores := make([]T, seqLen*kvLen)
	headSize := seqLen * headDim
	kvHeadSize := kvLen * headDim
	maskSliceLen := seqLen * kvLen
	for b := range batchSize {
		for h := range numHeads {
			kvH := h / headsPerKV
			qOff := (b*numHeads + h) * headSize
			kOff := (b*numKVHeads + kvH) * kvHeadSize
			vOff := kOff
			oOff := qOff
			var maskSlice []T
			if mask != nil {
				maskOff := b*maskBatchStride + h*maskHeadStride
				maskSlice = mask[maskOff : maskOff+maskSliceLen]
			}
			sdpa(
				q[qOff:qOff+headSize], k[kOff:kOff+kvHeadSize], v[vOff:vOff+kvHeadSize],
				maskSlice, scores, output[oOff:oOff+headSize],
				seqLen, kvLen, headDim, scale, causal,
			)
		}
	}
}

// execFusedQKVDense implements fused QKV projection.
// x: [batch, inFeatures], wQKV: [inFeatures, qDim+2*kvDim] (Q/K/V weights concatenated along last axis)
// biasQ: [qDim] (opt), biasK: [kvDim] (opt), biasV: [kvDim] (opt)
// outputs: q [batch, qDim], k [batch, kvDim], v [batch, kvDim]
func execFusedQKVDense(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	data := node.data.(*nodeFusedQKVDense)
	x := inputs[0]
	wQKV := inputs[1]

	// Determine bias buffers by position.
	var biasQ, biasK, biasV *Buffer
	biasIdx := 2
	if biasIdx < len(inputs) {
		biasQ = inputs[biasIdx]
		biasIdx++
	}
	if biasIdx < len(inputs) {
		biasK = inputs[biasIdx]
		biasIdx++
	}
	if biasIdx < len(inputs) {
		biasV = inputs[biasIdx]
	}

	qShape := node.multiOutputsShapes[0]
	kShape := node.multiOutputsShapes[1]
	vShape := node.multiOutputsShapes[2]
	qBuf := backend.getBufferForShape(qShape)
	kBuf := backend.getBufferForShape(kShape)
	vBuf := backend.getBufferForShape(vShape)

	inFeatures := x.shape.Dimensions[x.shape.Rank()-1]
	batchSize := x.shape.Size() / inFeatures

	switch x.shape.DType {
	case dtypes.Float32:
		var bqData, bkData, bvData []float32
		if biasQ != nil {
			bqData = biasQ.flat.([]float32)
		}
		if biasK != nil {
			bkData = biasK.flat.([]float32)
		}
		if biasV != nil {
			bvData = biasV.flat.([]float32)
		}
		qkvDense(
			x.flat.([]float32), wQKV.flat.([]float32),
			bqData, bkData, bvData,
			qBuf.flat.([]float32), kBuf.flat.([]float32), vBuf.flat.([]float32),
			batchSize, inFeatures, data.qDim, data.kvDim,
		)
	case dtypes.Float64:
		var bqData, bkData, bvData []float64
		if biasQ != nil {
			bqData = biasQ.flat.([]float64)
		}
		if biasK != nil {
			bkData = biasK.flat.([]float64)
		}
		if biasV != nil {
			bvData = biasV.flat.([]float64)
		}
		qkvDense(
			x.flat.([]float64), wQKV.flat.([]float64),
			bqData, bkData, bvData,
			qBuf.flat.([]float64), kBuf.flat.([]float64), vBuf.flat.([]float64),
			batchSize, inFeatures, data.qDim, data.kvDim,
		)
	default:
		return nil, errors.Errorf("FusedQKVDense: unsupported dtype %s", x.shape.DType)
	}

	return []*Buffer{qBuf, kBuf, vBuf}, nil
}

func qkvDense[T float32 | float64](x, wQKV, biasQ, biasK, biasV, q, k, v []T,
	batchSize, inFeatures, qDim, kvDim int,
) {
	totalOut := qDim + 2*kvDim
	// wQKV is [inFeatures, totalOut] row-major.
	// Column layout: [0..qDim) = Q, [qDim..qDim+kvDim) = K, [qDim+kvDim..totalOut) = V.
	for b := range batchSize {
		xBase := b * inFeatures
		qBase := b * qDim
		kBase := b * kvDim
		vBase := b * kvDim

		// Q = x @ wQ + biasQ, where wQ = wQKV[:, 0:qDim]
		for o := range qDim {
			var sum T
			for i := range inFeatures {
				sum += x[xBase+i] * wQKV[i*totalOut+o]
			}
			if biasQ != nil {
				sum += biasQ[o]
			}
			q[qBase+o] = sum
		}
		// K = x @ wK + biasK, where wK = wQKV[:, qDim:qDim+kvDim]
		for o := range kvDim {
			var sum T
			for i := range inFeatures {
				sum += x[xBase+i] * wQKV[i*totalOut+qDim+o]
			}
			if biasK != nil {
				sum += biasK[o]
			}
			k[kBase+o] = sum
		}
		// V = x @ wV + biasV, where wV = wQKV[:, qDim+kvDim:]
		for o := range kvDim {
			var sum T
			for i := range inFeatures {
				sum += x[xBase+i] * wQKV[i*totalOut+qDim+kvDim+o]
			}
			if biasV != nil {
				sum += biasV[o]
			}
			v[vBase+o] = sum
		}
	}
}
