// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

func init() {
	setNodeExecutor(backends.OpTypeBitcast, priorityGeneric, execBitcast)
}

// execBitcast implements Bitcast as a pure bit reinterpretation: the raw bytes are
// unchanged, only the shape and dtype are updated.
//
// For same-bit-width types (e.g. int8 ↔ uint8), the buffer is reused if owned
// and the Go storage type matches, otherwise a byte-level copy is performed.
//
// For different-bit-width types (e.g. uint8 → Int4), the raw bytes stay identical.
// A uint8[N] buffer bitcast to Int4[2*N] keeps its []uint8 flat data — the 2*N
// nibbles are stored packed (2 per byte). To unpack into one value per element,
// use ConvertDType (e.g. ConvertDType(Int4 → Int8)).
func execBitcast(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	src := inputs[0]
	targetDType := node.shape.DType
	srcDType := src.shape.DType
	sameBitWidth := srcDType.Bits() == targetDType.Bits()

	// If owned, reuse the buffer directly. For same-bit-width types we also
	// require matching Go storage types (e.g. int8 vs uint8) since downstream
	// code type-asserts the flat slice. For different-bit-width types the raw
	// bytes stay as-is and downstream code (ConvertDType, FusedQuantizedDense)
	// handles both slice types via type-switch.
	// Reuse owned buffer only when the flat slice's Go element type is compatible
	// with downstream operations on the target dtype. For same-bit-width types
	// with matching Go types this is trivially true. For different-bit-width types
	// involving sub-byte types (e.g. Uint8 ↔ Int4), both use []uint8 storage so
	// reuse is safe. But Uint8 → Float16 must allocate because Float16 operations
	// expect []float16.Float16, not []uint8.
	canReuse := false
	if inputsOwned[0] {
		if sameBitWidth && srcDType.GoType() == targetDType.GoType() {
			canReuse = true
		} else if !sameBitWidth {
			// For different-bit-width bitcasts, reuse only when both source and
			// target use the same underlying Go storage type. Sub-byte types
			// (Int2, Uint2, Int4, Uint4) and Uint8 all store as []uint8.
			_, srcIsUint8 := src.flat.([]uint8)
			dstIsUint8Storage := targetDType == dtypes.Uint8 || targetDType.Bits() < 8
			canReuse = srcIsUint8 && dstIsUint8Storage
		}
	}
	if canReuse {
		src.shape = node.shape
		inputs[0] = nil // signal to executor that input buffer was reused
		return src, nil
	}

	// Not owned or Go type mismatch: allocate via the standard pool and copy
	// raw bytes. For sub-byte types, getBufferForShape allocates packed storage
	// (e.g. Int4[2N] gets []uint8 of length N), matching the source byte count.
	output, err := backend.getBufferForShape(node.shape)
	if err != nil {
		return nil, err
	}
	copy(output.mutableBytes(), src.mutableBytes())
	return output, nil
}
