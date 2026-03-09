// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
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
	if inputsOwned[0] && (!sameBitWidth || srcDType.GoType() == targetDType.GoType()) {
		src.shape = node.shape
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
