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
	setNodeExecutor(backends.OpTypeFusedScaledDotProductAttention, priorityTyped, execFusedScaledDotProductAttention)
	multiOutputsNodeExecutors[backends.OpTypeFusedAttentionQKVProjection] = execFusedAttentionQKVProjection
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

// sdpaComputeMaskStrides returns (batchStride, headStride) for indexing into a mask
// tensor based on its rank. Dimensions of size 1 are broadcast (stride 0).
//
//	rank 2: [seqLen, kvLen]                     → (0, 0)
//	rank 3: [batch, seqLen, kvLen]              → (seqLen*kvLen, 0) or (0, 0) if dim[0]==1
//	rank 4: [batch, heads, seqLen, kvLen]       → strides computed per dim
func sdpaComputeMaskStrides(dims []int) (batchStride, headStride int) {
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
		panic(errors.Errorf("sdpaComputeMaskStrides: unsupported mask rank %d (dims=%v), expected rank 2, 3, or 4", len(dims), dims))
	}
}

// transposeBuffer transposes a buffer according to the given axis permutation,
// reusing the existing transposeIterator and transposeDTypeMap infrastructure.
func transposeBuffer(backend *Backend, buf *Buffer, permutations []int) *Buffer {
	output := backend.getBuffer(buf.shape.DType, buf.shape.Size())
	// Compute the output shape by permuting dimensions.
	dims := buf.shape.Dimensions
	outDims := make([]int, len(dims))
	for i, p := range permutations {
		outDims[i] = dims[p]
	}
	output.shape = shapes.Make(buf.shape.DType, outDims...)
	it := newTransposeIterator(buf.shape, permutations)
	transposeFn := transposeDTypeMap.Get(buf.shape.DType).(func(operand, output *Buffer, it *transposeIterator))
	transposeFn(buf, output, it)
	return output
}

// execFusedScaledDotProductAttention implements multi-head scaled dot-product attention.
// The internal computation always uses BHSD [batch, heads, seq, dim] layout.
// For BSHD inputs, the executor transposes to BHSD, runs SDPA, and transposes back.
// mask: optional additive mask of rank 2–4 (broadcasting via strides). Boolean masks are not
// supported; the graph-level caller must convert them to additive form before reaching here.
func execFusedScaledDotProductAttention(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	data := node.data.(*nodeFusedScaledDotProductAttention)
	query := inputs[0]
	key := inputs[1]
	value := inputs[2]
	var mask *Buffer
	if len(inputs) > 3 {
		mask = inputs[3]
	}

	// Boolean masks are not supported at the fused-op level; the caller must convert to additive.
	if mask != nil && mask.shape.DType == dtypes.Bool {
		return nil, errors.Errorf("FusedScaledDotProductAttention: boolean masks are not supported; convert to additive mask before calling")
	}

	// Transpose BSHD → BHSD so the inner loop always operates on BHSD layout.
	isBSHD := data.axesLayout == backends.AxesLayoutBSHD
	if isBSHD {
		perm := []int{0, 2, 1, 3}
		query = transposeBuffer(backend, query, perm)
		key = transposeBuffer(backend, key, perm)
		value = transposeBuffer(backend, value, perm)
		if mask != nil && mask.shape.Rank() == 4 {
			mask = transposeBuffer(backend, mask, perm)
		}
	}

	output := backend.getBufferForShape(query.shape.Clone())

	// Compute mask strides for broadcasting (always BHSD convention now).
	var maskBatchStride, maskHeadStride int
	if mask != nil {
		maskBatchStride, maskHeadStride = sdpaComputeMaskStrides(mask.shape.Dimensions)
	}

	switch query.shape.DType {
	case dtypes.Float32:
		sdpaMultiHeadGeneric[float32](query, key, value, mask, output, data, maskBatchStride, maskHeadStride)
	case dtypes.Float64:
		sdpaMultiHeadGeneric[float64](query, key, value, mask, output, data, maskBatchStride, maskHeadStride)
	default:
		return nil, errors.Errorf("FusedScaledDotProductAttention: unsupported dtype %s", query.shape.DType)
	}

	// Transpose output back to BSHD if needed.
	if isBSHD {
		output = transposeBuffer(backend, output, []int{0, 2, 1, 3})
	}
	return output, nil
}

func sdpaGeneric[T float32 | float64](q, k, v, mask, scores, output []T, seqLen, kvLen, headDim int, scale T, causal bool) {
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

func sdpaMultiHeadGeneric[T float32 | float64](query, key, value, mask, output *Buffer, data *nodeFusedScaledDotProductAttention, maskBatchStride, maskHeadStride int) {
	q := query.flat.([]T)
	k := key.flat.([]T)
	v := value.flat.([]T)
	out := output.flat.([]T)
	var maskData []T
	if mask != nil {
		maskData = mask.flat.([]T)
	}

	batchSize := query.shape.Dimensions[0]
	numHeads := data.numHeads
	numKVHeads := data.numKVHeads
	seqLen := query.shape.Dimensions[2]
	kvLen := key.shape.Dimensions[2]
	headDim := query.shape.Dimensions[3]
	scale := T(data.scale)
	causal := data.causal

	headsPerKV := numHeads / numKVHeads
	scores := make([]T, seqLen*kvLen)
	headSize := seqLen * headDim
	kvHeadSize := kvLen * headDim
	maskSliceLen := seqLen * kvLen
	for batchIdx := range batchSize {
		for headIdx := range numHeads {
			kvHeadIdx := headIdx / headsPerKV
			qOffset := (batchIdx*numHeads + headIdx) * headSize
			kOffset := (batchIdx*numKVHeads + kvHeadIdx) * kvHeadSize
			vOffset := kOffset
			outOffset := qOffset
			var maskSlice []T
			if maskData != nil {
				maskOffset := batchIdx*maskBatchStride + headIdx*maskHeadStride
				maskSlice = maskData[maskOffset : maskOffset+maskSliceLen]
			}
			sdpaGeneric(
				q[qOffset:qOffset+headSize], k[kOffset:kOffset+kvHeadSize], v[vOffset:vOffset+kvHeadSize],
				maskSlice, scores, out[outOffset:outOffset+headSize],
				seqLen, kvLen, headDim, scale, causal,
			)
		}
	}
}

// execFusedAttentionQKVProjection implements fused QKV projection.
// inputs[0]: pre-computed DotGeneral result [batch, qDim+2*kvDim]
// inputs[1..]: biasQ, biasK, biasV (optional, determined by node data flags)
// outputs: q [batch, qDim], k [batch, kvDim], v [batch, kvDim]
//
// The matmul (x @ wQKV) is already computed by the DotGeneral sub-node.
// This executor just splits the combined result into Q/K/V and adds biases.
func execFusedAttentionQKVProjection(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) ([]*Buffer, error) {
	data := node.data.(*nodeFusedAttentionQKVProjection)
	combined := inputs[0] // DotGeneral result: [batch, qDim+2*kvDim]

	// Determine bias buffers using flags from node data, not positional indexing.
	var biasQ, biasK, biasV *Buffer
	biasIdx := 1
	if data.hasBiasQ {
		biasQ = inputs[biasIdx]
		biasIdx++
	}
	if data.hasBiasK {
		biasK = inputs[biasIdx]
		biasIdx++
	}
	if data.hasBiasV {
		biasV = inputs[biasIdx]
	}

	qShape := node.multiOutputsShapes[0]
	kShape := node.multiOutputsShapes[1]
	vShape := node.multiOutputsShapes[2]
	qBuf := backend.getBufferForShape(qShape)
	kBuf := backend.getBufferForShape(kShape)
	vBuf := backend.getBufferForShape(vShape)

	qDim := data.qDim
	kvDim := data.kvDim

	switch combined.shape.DType {
	case dtypes.Float32:
		qkvSplitBiasGeneric[float32](combined, biasQ, biasK, biasV, qBuf, kBuf, vBuf, qDim, kvDim)
	case dtypes.Float64:
		qkvSplitBiasGeneric[float64](combined, biasQ, biasK, biasV, qBuf, kBuf, vBuf, qDim, kvDim)
	default:
		return nil, errors.Errorf("FusedAttentionQKVProjection: unsupported dtype %s", combined.shape.DType)
	}

	return []*Buffer{qBuf, kBuf, vBuf}, nil
}

// qkvSplitBiasGeneric splits the pre-computed matmul result [batch, totalOut] into
// Q [batch, qDim], K [batch, kvDim], V [batch, kvDim] and adds optional biases.
func qkvSplitBiasGeneric[T float32 | float64](combined, biasQBuf, biasKBuf, biasVBuf, qBuf, kBuf, vBuf *Buffer, qDim, kvDim int) {
	src := combined.flat.([]T)
	q := qBuf.flat.([]T)
	k := kBuf.flat.([]T)
	v := vBuf.flat.([]T)
	var biasQ, biasK, biasV []T
	if biasQBuf != nil {
		biasQ = biasQBuf.flat.([]T)
	}
	if biasKBuf != nil {
		biasK = biasKBuf.flat.([]T)
	}
	if biasVBuf != nil {
		biasV = biasVBuf.flat.([]T)
	}

	totalOut := qDim + 2*kvDim
	batchSize := len(src) / totalOut
	for batchIdx := range batchSize {
		srcBase := batchIdx * totalOut
		qBase := batchIdx * qDim
		kBase := batchIdx * kvDim
		vBase := batchIdx * kvDim

		// Copy Q columns and add bias.
		copy(q[qBase:qBase+qDim], src[srcBase:srcBase+qDim])
		if biasQ != nil {
			for o := range qDim {
				q[qBase+o] += biasQ[o]
			}
		}

		// Copy K columns and add bias.
		copy(k[kBase:kBase+kvDim], src[srcBase+qDim:srcBase+qDim+kvDim])
		if biasK != nil {
			for o := range kvDim {
				k[kBase+o] += biasK[o]
			}
		}

		// Copy V columns and add bias.
		copy(v[vBase:vBase+kvDim], src[srcBase+qDim+kvDim:srcBase+totalOut])
		if biasV != nil {
			for o := range kvDim {
				v[vBase+o] += biasV[o]
			}
		}
	}
}
