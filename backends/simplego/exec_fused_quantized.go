// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/pkg/errors"
)

func init() {
	setNodeExecutor(backends.OpTypeFusedQuantizedDense, priorityTyped, execFusedQuantizedDense)
	setNodeExecutor(backends.OpTypeFusedQuantizedGather, priorityTyped, execFusedQuantizedGather)
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

	// For packed sub-byte weights (from Bitcast), unpack nibbles before processing.
	// Packed buffers have len(flat) < shape.Size() (2 nibbles per byte).
	wFlat := unpackWeightsToInt8(wBuf)

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

// unpackWeightsToInt8 unpacks sub-byte weight data (Int4, Uint4) from packed
// []byte storage into []int8 (one value per element) for the matmul kernel.
// For non-sub-byte types, returns the flat data as-is.
func unpackWeightsToInt8(wBuf *Buffer) any {
	var unpackFn unpackNibblesFn
	switch wBuf.shape.DType {
	case dtypes.Uint4:
		unpackFn = unpackUint4Nibbles
	case dtypes.Int4:
		unpackFn = unpackInt4Nibbles
	default:
		return wBuf.flat
	}
	unpacked := make([]int8, wBuf.shape.Size())
	unpackFn(wBuf.flat.([]byte), unpacked)
	return unpacked
}

// execFusedQuantizedGather performs quantized embedding lookup.
// Inputs: [data, indices]. Data is [vocabSize, bytesPerRow] Uint8.
// Indices are integer with last dim = 1. Output is [batch..., K] Float32.
func execFusedQuantizedGather(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	data := node.data.(*nodeFusedQuantizedGather)
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

	numIndices := indicesBuf.shape.Size() / indicesBuf.shape.Dimensions[indicesBuf.shape.Rank()-1]
	dequantRow := make([]float32, K)

	indices, err := flatToIntSlice(indicesBuf.flat, numIndices)
	if err != nil {
		return nil, errors.Wrapf(err, "FusedQuantizedGather")
	}
	vocabSize := dataBuf.shape.Dimensions[0]
	for i, rowIdx := range indices {
		if rowIdx < 0 || rowIdx >= vocabSize {
			return nil, errors.Errorf("FusedQuantizedGather: index %d out of range [0, %d)", rowIdx, vocabSize)
		}
		rowData := dataBytes[rowIdx*bytesPerRow : (rowIdx+1)*bytesPerRow]
		dequantFn(rowData, dequantRow)
		copy(out[i*K:(i+1)*K], dequantRow)
	}

	return output, nil
}

// flatToIntSlice converts a flat index slice ([]int32, []int64, or []int) to []int.
func flatToIntSlice(flat any, n int) ([]int, error) {
	switch s := flat.(type) {
	case []int32:
		out := make([]int, n)
		for i := range n {
			out[i] = int(s[i])
		}
		return out, nil
	case []int64:
		out := make([]int, n)
		for i := range n {
			out[i] = int(s[i])
		}
		return out, nil
	case []int:
		return s[:n], nil
	default:
		return nil, errors.Errorf("unsupported indices type %T", flat)
	}
}

// quantizedDenseParallel runs rowFn(m) for each row m in [0, M), parallelizing over M rows
// when workers are available. For M=1 with large workloads, it tiles over N columns instead.
func quantizedDenseParallel(backend *Backend, M, K, N int, rowFn func(m, nStart, nEnd int)) {
	totalWork := M * K * N
	if backend == nil || !backend.workers.IsEnabled() || totalWork <= minParallelizeChunk {
		for m := range M {
			rowFn(m, 0, N)
		}
		return
	}

	if M > 1 {
		// Parallelize over M rows.
		var wg sync.WaitGroup
		for m := range M {
			wg.Add(1)
			backend.workers.WaitToStart(func() {
				rowFn(m, 0, N)
				wg.Done()
			})
		}
		wg.Wait()
	} else {
		// M=1: tile over N columns for single-token inference.
		tileSize := max(minParallelizeChunk/K, 1)
		var wg sync.WaitGroup
		for nStart := 0; nStart < N; nStart += tileSize {
			nEnd := min(nStart+tileSize, N)
			wg.Add(1)
			backend.workers.WaitToStart(func() {
				rowFn(0, nStart, nEnd)
				wg.Done()
			})
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
	quantizedDenseParallel(backend, M, K, N, func(m, nStart, nEnd int) {
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
	quantizedDenseParallel(backend, M, K, N, func(m, nStart, nEnd int) {
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
