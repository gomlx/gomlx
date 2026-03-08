// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
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
	srcBits := srcDType.Bits()
	dstBits := targetDType.Bits()

	switch {
	case srcBits == dstBits:
		// Same bit-width: reuse the input buffer if owned and the Go storage
		// type matches (e.g. Int4↔Int8 both use []int8, Uint4↔Uint8 both use []uint8).
		if inputsOwned[0] && srcDType.GoType() == targetDType.GoType() {
			src.shape = node.shape
			return src, nil
		}
		output, err := backend.getBufferForShape(node.shape)
		if err != nil {
			return nil, err
		}
		return execBitcastSameSize(src, output)
	default:
		// Different bit-widths: pure bit reinterpretation.
		// The raw bytes stay the same, only shape/dtype changes.
		// E.g., uint8[N] → Int4[2*N]: the N bytes now represent 2*N packed nibbles.
		if inputsOwned[0] {
			src.shape = node.shape
			return src, nil
		}
		// Not owned: copy raw bytes into a new buffer with the same flat type.
		srcBytes := src.mutableBytes()
		if len(srcBytes) == 0 {
			return &Buffer{shape: node.shape, flat: make([]uint8, 0), inUse: true}, nil
		}
		newFlat := make([]uint8, len(srcBytes))
		copy(newFlat, srcBytes)
		return &Buffer{shape: node.shape, flat: newFlat, inUse: true}, nil
	}
}

// execBitcastSameSize handles bitcast between types with the same bit-width (e.g., uint8 ↔ int8).
func execBitcastSameSize(src, output *Buffer) (*Buffer, error) {
	srcBytes := src.mutableBytes()
	dstBytes := output.mutableBytes()
	if len(srcBytes) != len(dstBytes) {
		return nil, errors.Errorf("Bitcast: source (%d bytes) and destination (%d bytes) size mismatch", len(srcBytes), len(dstBytes))
	}
	copy(dstBytes, srcBytes)
	return output, nil
}
