// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/pkg/errors"
)

func init() {
	setNodeExecutor(backends.OpTypeBitcast, priorityGeneric, execBitcast)
}

func execBitcast(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	src := inputs[0]
	targetDType := node.shape.DType
	srcDType := src.shape.DType
	srcBits := srcDType.Bits()
	dstBits := targetDType.Bits()

	output := backend.getBufferForShape(node.shape)

	switch {
	case srcBits == dstBits:
		return execBitcastSameSize(src, output)
	case srcBits > dstBits:
		return execBitcastUnpack(src, output, srcDType, targetDType)
	default:
		return execBitcastPack(src, output, srcDType, targetDType)
	}
}

// execBitcastSameSize handles bitcast between types with the same bit-width (e.g., uint8 ↔ int8).
func execBitcastSameSize(src, output *Buffer) (*Buffer, error) {
	srcBytes := src.mutableBytes()
	dstBytes := output.mutableBytes()
	copy(dstBytes, srcBytes)
	return output, nil
}

// execBitcastUnpack handles bitcast from a larger element type to a smaller one
// (e.g., uint8 → int4: each byte becomes 2 nibble values).
func execBitcastUnpack(src, output *Buffer, srcDType, dstDType dtypes.DType) (*Buffer, error) {
	if srcDType == dtypes.Uint8 && (dstDType == dtypes.Int4 || dstDType == dtypes.Uint4) {
		return execBitcastUint8ToNibbles(src, output, dstDType)
	}
	return nil, errors.Errorf("Bitcast: unsupported unpack conversion %s → %s", srcDType, dstDType)
}

// execBitcastPack handles bitcast from a smaller element type to a larger one
// (e.g., int4 → uint8: each pair of nibble values becomes 1 byte).
func execBitcastPack(src, output *Buffer, srcDType, dstDType dtypes.DType) (*Buffer, error) {
	if (srcDType == dtypes.Int4 || srcDType == dtypes.Uint4) && dstDType == dtypes.Uint8 {
		return execBitcastNibblesToUint8(src, output, srcDType)
	}
	return nil, errors.Errorf("Bitcast: unsupported pack conversion %s → %s", srcDType, dstDType)
}

// execBitcastUint8ToNibbles unpacks uint8 bytes into individual nibble values.
// Low nibble (bits 0-3) comes first, high nibble (bits 4-7) second.
func execBitcastUint8ToNibbles(src, output *Buffer, dstDType dtypes.DType) (*Buffer, error) {
	srcData := src.flat.([]uint8)
	switch dstDType {
	case dtypes.Int4:
		dstData := output.flat.([]int8)
		for i, b := range srcData {
			lo := int8(b & 0x0F)
			hi := int8(b >> 4)
			// Sign-extend 4-bit values: values 8-15 map to -8 to -1.
			if lo >= 8 {
				lo -= 16
			}
			if hi >= 8 {
				hi -= 16
			}
			dstData[2*i] = lo
			dstData[2*i+1] = hi
		}
	case dtypes.Uint4:
		dstData := output.flat.([]uint8)
		for i, b := range srcData {
			dstData[2*i] = b & 0x0F
			dstData[2*i+1] = b >> 4
		}
	}
	return output, nil
}

// execBitcastNibblesToUint8 packs individual nibble values into uint8 bytes.
// Low nibble first, high nibble second.
func execBitcastNibblesToUint8(src, output *Buffer, srcDType dtypes.DType) (*Buffer, error) {
	dstData := output.flat.([]uint8)
	switch srcDType {
	case dtypes.Int4:
		srcData := src.flat.([]int8)
		for i := range dstData {
			lo := uint8(srcData[2*i]) & 0x0F
			hi := uint8(srcData[2*i+1]) & 0x0F
			dstData[i] = lo | (hi << 4)
		}
	case dtypes.Uint4:
		srcData := src.flat.([]uint8)
		for i := range dstData {
			lo := srcData[2*i] & 0x0F
			hi := srcData[2*i+1] & 0x0F
			dstData[i] = lo | (hi << 4)
		}
	}
	return output, nil
}
