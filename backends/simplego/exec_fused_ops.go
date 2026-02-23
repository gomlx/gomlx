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

// FusedSoftmax =====================================================================================================

// execFusedSoftmax implements optimized softmax with better cache locality.
// Three passes over the axis: find max, compute exp(x-max) and sum, then normalize.
func execFusedSoftmax(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	data := node.data.(*nodeFusedSoftmax)
	axis := data.axis
	input := inputs[0]
	output, err := backend.getBufferForShape(node.shape)
	if err != nil {
		return nil, err
	}

	switch input.shape.DType {
	case dtypes.Float32:
		fusedSoftmax(input.flat.([]float32), output.flat.([]float32), axis, node.shape)
	case dtypes.Float64:
		fusedSoftmax(input.flat.([]float64), output.flat.([]float64), axis, node.shape)
	default:
		return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedSoftmax: dtype %s", input.shape.DType)
	}
	return output, nil
}

// fusedSoftmaxComputeAxisStrides returns the outer size, axis size, and inner size for iterating
// over an axis of the given shape. This decomposition allows softmax (and similar
// axis-based ops) to operate on any axis.
func fusedSoftmaxComputeAxisStrides(shape shapes.Shape, axis int) (outerSize, axisSize, innerSize int) {
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

func fusedSoftmax[T float32 | float64](input, output []T, axis int, shape shapes.Shape) {
	outerSize, axisSize, innerSize := fusedSoftmaxComputeAxisStrides(shape, axis)
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

// FusedGelu =======================================================================================================

// execFusedGelu implements GELU: x * 0.5 * (1 + erf(x / sqrt(2)))
func execFusedGelu(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	input := inputs[0]
	output, err := backend.getBufferForShape(node.shape)
	if err != nil {
		return nil, err
	}

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

// FusedLayerNorm ===================================================================================================

// execFusedLayerNorm implements layer normalization.
// For each sample: y = (x - mean) / sqrt(var + epsilon) * gamma + beta
func execFusedLayerNorm(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	data := node.data.(*nodeFusedLayerNorm)
	input := inputs[0]
	output, err := backend.getBufferForShape(node.shape)
	if err != nil {
		return nil, err
	}

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
		layerNormTrailingAxes(inData, outData, gammaData, betaData, normSize, epsilon)
	} else {
		layerNormArbitraryAxes(inData, outData, gammaData, betaData, dims, axes, normSize, epsilon)
	}
}

// layerNormTrailingAxes handles the common case where normalization axes are the last N axes.
// Each contiguous block of normSize elements is one normalization group.
func layerNormTrailingAxes[T float32 | float64](
	inData, outData, gammaData, betaData []T,
	normSize int,
	epsilon float64,
) {
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

// layerNormArbitraryAxes handles normalization over arbitrary (non-trailing) axes
// using Shape.IterOnAxes for index iteration.
func layerNormArbitraryAxes[T float32 | float64](
	inData, outData, gammaData, betaData []T,
	dims, axes []int,
	normSize int,
	epsilon float64,
) {
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

// FusedDense =======================================================================================================

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
		output, err := backend.getBufferForShape(node.shape)
		if err != nil {
			return nil, err
		}
		copyFlat(output.flat, matmul.flat)
		return output, nil
	}

	// Try to reuse the matmul buffer if owned; otherwise allocate.
	var output *Buffer
	if inputsOwned[0] {
		output = matmul
		inputs[0] = nil // Signal to the executor that we reused the input.
	} else {
		var err error
		output, err = backend.getBufferForShape(node.shape)
		if err != nil {
			return nil, err
		}
		copyFlat(output.flat, matmul.flat)
	}

	switch output.shape.DType {
	case dtypes.Float32:
		outData := output.flat.([]float32)
		if bias != nil {
			fusedDenseAddBias(outData, bias.flat.([]float32))
		}
		fusedDenseApplyActivation(backend, outData, data.activation)
	case dtypes.Float64:
		outData := output.flat.([]float64)
		if bias != nil {
			fusedDenseAddBias(outData, bias.flat.([]float64))
		}
		fusedDenseApplyActivation(backend, outData, data.activation)
	default:
		return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedDense: dtype %s", output.shape.DType)
	}
	return output, nil
}

// fusedDenseAddBias adds bias to each row of the output in-place.
// output shape is [..., outFeatures], bias shape is [outFeatures].
func fusedDenseAddBias[T float32 | float64](output, bias []T) {
	outFeatures := len(bias)
	for i, v := range output {
		output[i] = v + bias[i%outFeatures]
	}
}

func fusedDenseApplyActivation[T float32 | float64](backend *Backend, data []T, activation backends.ActivationType) {
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
	case backends.ActivationHardSwish:
		const scale = 1.0 / 6.0
		const bias = 0.5
		for i, x := range data {
			shapeX := min(max(x*scale+bias, 0), 1)
			data[i] = x * shapeX
		}
	case backends.ActivationTanh:
		for i, x := range data {
			data[i] = T(math.Tanh(float64(x)))
		}
	}
}

// FusedScaledDotProductAttention ===================================================================================

// execFusedScaledDotProductAttention implements multi-head scaled dot-product attention.
// Both BHSD and BSHD layouts are handled directly via stride-based indexing in
// sdpaGeneric/sdpaMultiHeadGeneric, avoiding expensive transpose operations.
// mask: optional additive mask of rank 2–4 (broadcasting via strides). Boolean masks are not
// supported; the graph-level caller must convert them to additive form before reaching here.
func execFusedScaledDotProductAttention(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (
	*Buffer, error) {
	data := node.data.(*nodeFusedScaledDotProductAttention)
	query := inputs[0]
	key := inputs[1]
	value := inputs[2]
	var mask *Buffer
	if len(inputs) > 3 {
		mask = inputs[3]
	}

	// For rank-4 BSHD masks [batch, seq, heads, kvLen], transpose to BHSD so that
	// per-head mask data is contiguous [seqLen, kvLen]. The mask is small (no headDim
	// axis), so this is cheap. Rank ≤ 3 masks have no head dimension and work as-is.
	if data.axesLayout == backends.AxesLayoutBSHD && mask != nil && mask.shape.Rank() == 4 {
		var err error
		mask, err = transposeBuffer(backend, mask, []int{0, 2, 1, 3})
		if err != nil {
			return nil, err
		}
	}

	output, err := backend.getBufferForShape(query.shape.Clone())
	if err != nil {
		return nil, err
	}

	// Compute mask strides for broadcasting (BHSD convention for the mask).
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

	return output, nil
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
func transposeBuffer(backend *Backend, buf *Buffer, permutations []int) (*Buffer, error) {
	output, err := backend.getBuffer(buf.shape.DType, buf.shape.Size())
	if err != nil {
		return nil, err
	}
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
	return output, nil
}

// sdpaGeneric computes scaled dot-product attention for a group of query heads
// that share the same key/value head (Grouped Query Attention / GQA).
// For standard multi-head attention, groupSize is 1.
//
// The q/k/v/output slices are the full flat arrays for the tensor; qOff and kvOff
// give the element-offset to the first element of this group at seq=0 (for Q, this
// is the first query head in the group). qSeqStride and kvSeqStride are the element
// stride between consecutive sequence positions for a single head (headDim for BHSD
// contiguous layout, numHeads*headDim for BSHD interleaved layout). qGroupStride is
// the element stride between consecutive query heads within the group.
// The output uses qOff/qSeqStride/qGroupStride (same layout as query).
//
// scores is a dense [groupSize, seqLen, kvLen] scratch buffer.
// Masks are dense per-head [seqLen, kvLen] buffers, shared across the group when
// maskGroupStride is 0, or offset by maskGroupStride per group member for per-head masks.
func sdpaGeneric[T float32 | float64](
	q, k, v []T, qOff, kvOff, qSeqStride, kvSeqStride, qGroupStride int,
	additiveMask []T,
	booleanMask []bool,
	maskGroupStride int,
	scores []T,
	output []T,
	groupSize, seqLen, kvLen, headDim int, scale T, causal bool,
) {
	for gIdx := range groupSize {
		gQOff := qOff + gIdx*qGroupStride
		gMaskOff := gIdx * maskGroupStride
		for qIdx := range seqLen {
			rowMax := T(math.Inf(-1))
			qBase := gQOff + qIdx*qSeqStride
			scoreIdxBase := (gIdx*seqLen + qIdx) * kvLen
			maskIdxBase := gMaskOff + qIdx*kvLen

			kvLenUnmasked := kvLen
			if causal {
				kvLenUnmasked = min(kvLen, qIdx+1)
			}
			for kvIdx := range kvLenUnmasked {
				scoreIdx := scoreIdxBase + kvIdx
				maskIdx := maskIdxBase + kvIdx
				if len(booleanMask) > 0 {
					if !booleanMask[maskIdx] {
						continue
					}
				}
				var dot T
				kBase := kvOff + kvIdx*kvSeqStride
				for d := range headDim {
					dot += q[qBase+d] * k[kBase+d]
				}
				s := dot * scale
				if len(additiveMask) > 0 {
					s += additiveMask[maskIdx]
				}
				scores[scoreIdx] = s
				if s > rowMax {
					rowMax = s
				}
			}

			// Softmax: exp(scores - max) and sum.
			var sum T
			scoreIdx := scoreIdxBase
			maskIdx := maskIdxBase
			if len(booleanMask) > 0 {
				for range kvLenUnmasked {
					if booleanMask[maskIdx] {
						scores[scoreIdx] = T(math.Exp(float64(scores[scoreIdx] - rowMax)))
						sum += scores[scoreIdx]
					}
					scoreIdx++
					maskIdx++
				}
			} else {
				// No boolean mask, so we can use the fast path.
				for range kvLenUnmasked {
					scores[scoreIdx] = T(math.Exp(float64(scores[scoreIdx] - rowMax)))
					sum += scores[scoreIdx]
					scoreIdx++
				}
			}
			invSum := 1.0 / sum
			scoreIdx = scoreIdxBase
			for range kvLenUnmasked {
				scores[scoreIdx] *= invSum
				scoreIdx++
			}

			// output[qIdx][d] = sum_kvIdx(scores[qIdx][kvIdx] * v[kvIdx][d])
			outBase := gQOff + qIdx*qSeqStride
			for d := range headDim {
				scoreIdx := scoreIdxBase
				maskIdx := maskIdxBase
				var acc T
				if len(booleanMask) > 0 {
					for kvIdx := range kvLenUnmasked {
						if booleanMask[maskIdx] {
							acc += scores[scoreIdx] * v[kvOff+kvIdx*kvSeqStride+d]
						}
						scoreIdx++
						maskIdx++
					}
				} else {
					for kvIdx := range kvLenUnmasked {
						acc += scores[scoreIdx] * v[kvOff+kvIdx*kvSeqStride+d]
						scoreIdx++
					}
				}
				output[outBase+d] = acc
			}
		}
	}
}

func sdpaMultiHeadGeneric[T float32 | float64](query, key, value, mask, output *Buffer, data *nodeFusedScaledDotProductAttention, maskBatchStride, maskHeadStride int) {
	q := query.flat.([]T)
	k := key.flat.([]T)
	v := value.flat.([]T)
	out := output.flat.([]T)
	var additiveMask []T
	var booleanMask []bool
	if mask != nil {
		if mask.shape.DType == dtypes.Bool {
			booleanMask = mask.flat.([]bool)
		} else {
			additiveMask = mask.flat.([]T)
		}
	}

	dims := query.shape.Dimensions
	batchSize := dims[0]
	numHeads := data.numHeads
	numKVHeads := data.numKVHeads
	scale := T(data.scale)
	causal := data.causal
	groupSize := numHeads / numKVHeads

	// Layout-dependent axis indices and strides.
	var seqLen, kvLen, headDim int
	var qSeqStride, kvSeqStride int     // element stride between consecutive seq positions for one head
	var qBatchStride, kvBatchStride int // element stride between consecutive batches
	var qHeadStride, kvHeadStride int   // element stride between consecutive heads at seq=0

	if data.axesLayout == backends.AxesLayoutBSHD {
		// [batch, seq, heads, dim]
		seqLen = dims[1]
		headDim = dims[3]
		kvDims := key.shape.Dimensions
		kvLen = kvDims[1]
		qSeqStride = numHeads * headDim
		kvSeqStride = numKVHeads * headDim
		qHeadStride = headDim
		kvHeadStride = headDim
		qBatchStride = seqLen * numHeads * headDim
		kvBatchStride = kvLen * numKVHeads * headDim
	} else {
		// BHSD: [batch, heads, seq, dim]
		seqLen = dims[2]
		headDim = dims[3]
		kvDims := key.shape.Dimensions
		kvLen = kvDims[2]
		qSeqStride = headDim
		kvSeqStride = headDim
		qHeadStride = seqLen * headDim
		kvHeadStride = kvLen * headDim
		qBatchStride = numHeads * seqLen * headDim
		kvBatchStride = numKVHeads * kvLen * headDim
	}

	scores := make([]T, groupSize*seqLen*kvLen)
	maskSliceLen := seqLen * kvLen
	for batchIdx := range batchSize {
		for kvHeadIdx := range numKVHeads {
			qOff := batchIdx*qBatchStride + kvHeadIdx*groupSize*qHeadStride
			kvOff := batchIdx*kvBatchStride + kvHeadIdx*kvHeadStride

			// Compute mask slice and group stride for this KV head group.
			var additiveMaskSlice []T
			var booleanMaskSlice []bool
			maskGroupStride := 0
			if len(additiveMask) > 0 {
				maskOffset := batchIdx*maskBatchStride + kvHeadIdx*groupSize*maskHeadStride
				maskEnd := maskOffset + maskSliceLen
				if maskHeadStride > 0 && groupSize > 1 {
					maskEnd = maskOffset + (groupSize-1)*maskHeadStride + maskSliceLen
					maskGroupStride = maskHeadStride
				}
				additiveMaskSlice = additiveMask[maskOffset:maskEnd]
			}
			if len(booleanMask) > 0 {
				maskOffset := batchIdx*maskBatchStride + kvHeadIdx*groupSize*maskHeadStride
				maskEnd := maskOffset + maskSliceLen
				if maskHeadStride > 0 && groupSize > 1 {
					maskEnd = maskOffset + (groupSize-1)*maskHeadStride + maskSliceLen
					maskGroupStride = maskHeadStride
				}
				booleanMaskSlice = booleanMask[maskOffset:maskEnd]
			}
			sdpaGeneric(
				q, k, v, qOff, kvOff, qSeqStride, kvSeqStride, qHeadStride,
				additiveMaskSlice, booleanMaskSlice, maskGroupStride,
				scores,
				out,
				groupSize, seqLen, kvLen, headDim, scale, causal,
			)
		}
	}
}

// FusedAttentionQKVProjection ===================================================================================

// execFusedAttentionQKVProjection implements fused QKV projection.
// inputs[0]: pre-computed DotGeneral result [batch, qDim+2*kvDim]
// inputs[1..]: biasQ, biasK, biasV (optional, determined by node data flags)
// outputs: q [batch, qDim], k [batch, kvDim], v [batch, kvDim]
//
// The matmul (x @ wQKV) is already computed by the DotGeneral sub-node.
// This executor just splits the combined result into Q/K/V and adds biases.
func execFusedAttentionQKVProjection(backend *Backend, node *Node, inputs []*Buffer, _ []bool) ([]*Buffer, error) {
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
	qBuf, err := backend.getBufferForShape(qShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", qShape)
	}
	kBuf, err := backend.getBufferForShape(kShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", kShape)
	}
	vBuf, err := backend.getBufferForShape(vShape)
	if err != nil {
		return nil, errors.Wrapf(err, "fail to get buffer for shape %s", vShape)
	}
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
