// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

func init() {
	setNodeExecutor(backends.OpTypeFusedQuantizedDense, priorityTyped, execFusedQuantizedDense)
	setNodeExecutor(backends.OpTypeQuantizedEmbeddingLookup, priorityTyped, execQuantizedEmbeddingLookup)
}

// execFusedQuantizedDense implements scalar dequant + matmul + bias + activation.
// inputs layout: [x, weights, scales, zeroPoints?, bias?]
//
// Weights have their dtype set to reflect the storage type (Int4, Int8, etc.).
// Int4/Uint4 weights may be in packed form ([]byte, 2 nibbles per byte) when
// produced by Bitcast, or unpacked ([]int8/[]uint8, one value per element) when
// produced by ConvertDType. Both forms are supported.
func execFusedQuantizedDense(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	data := node.data.(*nodeFusedQuantizedDense)

	// GGML has a different input layout: [x, weights, bias?] (no scales/zeroPoints).
	if data.scheme == backends.QuantGGML {
		return execFusedQuantizedDenseGGML(backend, node, inputs, data)
	}

	xBuf := inputs[0]
	wBuf := inputs[1]
	sBuf := inputs[2]

	// Determine zeroPoints and bias from remaining inputs using explicit flags.
	// Inputs: [x, weights, scales, zeroPoints?, bias?]
	var zeroPointsBuf, biasBuf *Buffer
	nextIdx := 3
	if data.hasZeroPoint {
		zeroPointsBuf = inputs[nextIdx]
		nextIdx++
	}
	if data.hasBias {
		biasBuf = inputs[nextIdx]
	}

	if xBuf.shape.DType != dtypes.Float32 {
		return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedQuantizedDense: only float32 input supported, got %s", xBuf.shape.DType)
	}

	output, err := backend.getBufferForShape(node.shape)
	if err != nil {
		return nil, err
	}
	x := xBuf.flat.([]float32)
	scales := sBuf.flat.([]float32)
	out := output.flat.([]float32)

	K := xBuf.shape.Dimensions[xBuf.shape.Rank()-1]
	N := wBuf.shape.Dimensions[1]
	M := xBuf.shape.Size() / K
	blockSize := data.blockSize
	numBlocks := (N + blockSize - 1) / blockSize

	var bias []float32
	if biasBuf != nil {
		bias = biasBuf.flat.([]float32)
	}
	var zeroPoints []float32
	if zeroPointsBuf != nil {
		zeroPoints = zeroPointsBuf.flat.([]float32)
	}

	// For packed sub-byte weights (from Bitcast), unpack nibbles via the buffer pool
	// and ConvertDType infrastructure. Non-sub-byte types pass through unchanged.
	unpackedBuf, isUnpackedOwned, err := unpackWeightsToBuffer(backend, wBuf)
	if err != nil {
		return nil, err
	}
	if isUnpackedOwned {
		defer backend.putBuffer(unpackedBuf)
	}
	wFlat := unpackedBuf.flat

	switch data.scheme {
	case backends.QuantNF4:
		// NF4 weights are nibble indices [0..15]. Supports Int4/Int8/Uint4/Uint8.
		switch wFlat := wFlat.(type) {
		case []uint8:
			quantizedDenseNF4(backend, x, wFlat, scales, bias, out, M, K, N, blockSize, numBlocks)
		case []int8:
			quantizedDenseNF4(backend, x, wFlat, scales, bias, out, M, K, N, blockSize, numBlocks)
		default:
			return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedQuantizedDense: NF4 unsupported weight type %T", wFlat)
		}
	case backends.QuantLinear:
		switch wFlat := wFlat.(type) {
		case []int8:
			quantizedDenseLinearInt(backend, x, wFlat, scales, zeroPoints, bias, out, M, K, N, blockSize, numBlocks)
		case []uint8:
			quantizedDenseLinearInt(backend, x, wFlat, scales, zeroPoints, bias, out, M, K, N, blockSize, numBlocks)
		default:
			return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedQuantizedDense: Linear unsupported weight type %T", wFlat)
		}
	default:
		return nil, errors.Wrapf(backends.ErrNotImplemented, "FusedQuantizedDense: unknown quantization scheme %d", data.scheme)
	}

	fusedDenseApplyActivation(backend, out, data.activation)
	return output, nil
}

// unpackWeightsToBuffer unpacks sub-byte weight data (Int4, Uint4) into a pooled
// buffer using the ConvertDType infrastructure. For non-sub-byte types, returns the
// original buffer unchanged.
//
// Returns the (possibly new) buffer, whether it was allocated from the pool
// (caller must putBuffer), and any error.
func unpackWeightsToBuffer(backend *Backend, wBuf *Buffer) (*Buffer, bool, error) {
	var targetDType dtypes.DType
	switch wBuf.shape.DType {
	case dtypes.Int4, dtypes.Int2:
		targetDType = dtypes.Int8
	case dtypes.Uint4, dtypes.Uint2:
		targetDType = dtypes.Uint8
	default:
		return wBuf, false, nil
	}

	outBuf, err := backend.getBuffer(targetDType, wBuf.shape.Size())
	if err != nil {
		return nil, false, err
	}
	outBuf.shape = shapes.Make(targetDType, wBuf.shape.Dimensions...)

	convertFnAny, err := convertDTypePairMap.Get(wBuf.shape.DType, targetDType)
	if err != nil {
		backend.putBuffer(outBuf)
		return nil, false, err
	}
	convertFn := convertFnAny.(convertFnType)
	convertFn(wBuf, outBuf)
	return outBuf, true, nil
}

// execQuantizedEmbeddingLookup performs quantized embedding lookup.
// Inputs: [data, indices]. Data is [vocabSize, bytesPerRow] Uint8.
// Indices are integer with last dim = 1. Output is [batch..., K] Float32.
func execQuantizedEmbeddingLookup(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	data := node.data.(*nodeQuantizedEmbeddingLookup)
	dataBuf := inputs[0]
	indicesBuf := inputs[1]

	output, err := backend.getBufferForShape(node.shape)
	if err != nil {
		return nil, err
	}

	dataBytes := dataBuf.flat.([]uint8)
	out := output.flat.([]float32)
	K := data.ggmlK
	bytesPerRow := dataBuf.shape.Dimensions[1]

	dequantFn, err := ggmlDequantFunc(data.ggmlType)
	if err != nil {
		return nil, err
	}

	// Last dim is pre-validated to be 1, so total elements == number of indices.
	numIndices := indicesBuf.shape.Size()

	// Convert indices to int64 via the buffer pool and ConvertDType infrastructure.
	idxBuf, isIdxOwned, err := convertIndicesToInt64(backend, indicesBuf)
	if err != nil {
		return nil, errors.Wrapf(err, "QuantizedEmbeddingLookup")
	}
	if isIdxOwned {
		defer backend.putBuffer(idxBuf)
	}
	indices := idxBuf.flat.([]int64)

	vocabSize := int64(dataBuf.shape.Dimensions[0])
	for i, rowIdx := range indices[:numIndices] {
		if rowIdx < 0 || rowIdx >= vocabSize {
			return nil, errors.Errorf("QuantizedEmbeddingLookup: index %d out of range [0, %d)", rowIdx, vocabSize)
		}
		rowStart := rowIdx * int64(bytesPerRow)
		rowData := dataBytes[rowStart : rowStart+int64(bytesPerRow)]
		dequantFn(rowData, out[i*K:(i+1)*K])
	}

	return output, nil
}

// convertIndicesToInt64 converts an integer index buffer to int64 via the buffer
// pool and ConvertDType infrastructure. If the buffer is already int64, it is
// returned as-is.
//
// Returns the (possibly new) buffer, whether it was allocated from the pool
// (caller must putBuffer), and any error.
func convertIndicesToInt64(backend *Backend, indicesBuf *Buffer) (*Buffer, bool, error) {
	if indicesBuf.shape.DType == dtypes.Int64 {
		return indicesBuf, false, nil
	}
	outBuf, err := backend.getBuffer(dtypes.Int64, indicesBuf.shape.Size())
	if err != nil {
		return nil, false, err
	}
	outBuf.shape = shapes.Make(dtypes.Int64, indicesBuf.shape.Dimensions...)

	convertFnAny, err := convertDTypePairMap.Get(indicesBuf.shape.DType, dtypes.Int64)
	if err != nil {
		backend.putBuffer(outBuf)
		return nil, false, err
	}
	convertFn := convertFnAny.(convertFnType)
	convertFn(indicesBuf, outBuf)
	return outBuf, true, nil
}

// quantizedDenseParallelTileCount returns the number of parallel work units that
// quantizedDenseParallel will dispatch for the given dimensions.
func quantizedDenseParallelTileCount(backend *Backend, M, K, N int) int {
	totalWork := M * K * N
	if backend == nil || !backend.workers.IsEnabled() || totalWork <= minParallelizeChunk {
		return M
	}
	if M > 1 {
		return M
	}
	tileSize := max(minParallelizeChunk/K, 1)
	return (N + tileSize - 1) / tileSize
}

// quantizedDenseParallel parallelizes over M rows, or tiles over N columns when M=1.
// workerIdx is a dense index in [0, quantizedDenseParallelTileCount) identifying the work unit.
func quantizedDenseParallel(backend *Backend, M, K, N int, rowFn func(workerIdx, m, nStart, nEnd int)) {
	totalWork := M * K * N
	if backend == nil || !backend.workers.IsEnabled() || totalWork <= minParallelizeChunk {
		for m := range M {
			rowFn(m, m, 0, N)
		}
		return
	}

	if M > 1 {
		// Parallelize over M rows.
		var wg sync.WaitGroup
		for m := range M {
			wg.Add(1)
			backend.workers.WaitToStart(func() {
				rowFn(m, m, 0, N)
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		// M=1: tile over N columns for single-token inference.
		tileSize := max(minParallelizeChunk/K, 1)
		var wg sync.WaitGroup
		workerIdx := 0
		for nStart := 0; nStart < N; nStart += tileSize {
			nEnd := min(nStart+tileSize, N)
			idx := workerIdx
			wg.Add(1)
			backend.workers.WaitToStart(func() {
				rowFn(idx, 0, nStart, nEnd)
				wg.Done()
			})
			workerIdx++
		}
		wg.Wait()
	}
}

// quantizedDenseNF4 performs NF4 dequant + matmul + bias for Int4 (int8) or Uint4 (uint8) weights.
// NF4 does not support zeroPoints (validated by the builder).
//
// Uses cache-friendly (m, k, n) loop order so both weights[k*N+n] and out[m*N+n] are
// accessed with stride-1 in the innermost loop.
func quantizedDenseNF4[T int8 | uint8](backend *Backend, x []float32, weights []T, scales, bias, out []float32, M, K, N, blockSize, numBlocks int) {
	quantizedDenseParallel(backend, M, K, N, func(_, m, nStart, nEnd int) {
		outSlice := out[m*N+nStart : m*N+nEnd]
		if bias != nil {
			copy(outSlice, bias[nStart:nEnd])
		} else {
			clear(outSlice)
		}
		for k := range K {
			xVal := x[m*K+k]
			wRow := weights[k*N:]
			sRow := scales[k*numBlocks:]
			blockIdx := nStart / blockSize
			nextBlock := (blockIdx + 1) * blockSize
			for n := nStart; n < nEnd; n++ {
				if n >= nextBlock {
					blockIdx++
					nextBlock += blockSize
				}
				outSlice[n-nStart] += xVal * backends.NF4LookupTable[uint8(wRow[n])&0x0F] * sRow[blockIdx]
			}
		}
	})
}

// quantizedDenseLinearInt performs linear dequant + matmul + bias for integer weights.
// float_value = int_value * scale + zeroPoint
//
// Uses cache-friendly (m, k, n) loop order so both weights[k*N+n] and out[m*N+n] are
// accessed with stride-1 in the innermost loop.
func quantizedDenseLinearInt[T int8 | uint8](backend *Backend, x []float32, weights []T, scales, zeroPoints, bias, out []float32, M, K, N, blockSize, numBlocks int) {
	quantizedDenseParallel(backend, M, K, N, func(_, m, nStart, nEnd int) {
		outSlice := out[m*N+nStart : m*N+nEnd]
		if bias != nil {
			copy(outSlice, bias[nStart:nEnd])
		} else {
			clear(outSlice)
		}
		for k := range K {
			xVal := x[m*K+k]
			wRow := weights[k*N:]
			sRow := scales[k*numBlocks:]
			blockIdx := nStart / blockSize
			nextBlock := (blockIdx + 1) * blockSize
			if zeroPoints != nil {
				zpRow := zeroPoints[k*numBlocks:]
				for n := nStart; n < nEnd; n++ {
					if n >= nextBlock {
						blockIdx++
						nextBlock += blockSize
					}
					outSlice[n-nStart] += xVal * (float32(wRow[n])*sRow[blockIdx] + zpRow[blockIdx])
				}
			} else {
				for n := nStart; n < nEnd; n++ {
					if n >= nextBlock {
						blockIdx++
						nextBlock += blockSize
					}
					outSlice[n-nStart] += xVal * float32(wRow[n]) * sRow[blockIdx]
				}
			}
		}
	})
}
