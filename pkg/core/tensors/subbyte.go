package tensors

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

// UnpackSubBytes unpacks packed sub-byte data (Uint4, Int4, Uint2 and Int2, packed into []uint8)
// into a one-value-per-element []int8 slice -- enough bits to represent the unsigned types as well.
//
// It unpacks only the first numValues. If numValues is <= 0, it will unpack all the values in packed.
//
// It can panic if a non-packed dtype is given.
func UnpackSubBytes(packed []uint8, dtype dtypes.DType, numValues int) []int8 {
	bitsPerValue := dtype.Bits()
	valuesPerUnit := dtype.ValuesPerStorageUnit()
	numPackedValues := len(packed) * valuesPerUnit
	if numValues <= 0 {
		numValues = numPackedValues
	} else {
		numValues = min(numValues, numPackedValues)
	}
	mask := uint8((1 << bitsPerValue) - 1) // 0x0F for 4-bit, 0x03 for 2-bit
	signBit := uint8(1 << (bitsPerValue - 1))
	signExtend := ^mask // 0xF0 for 4-bit, 0xFC for 2-bit

	// Extract the raw unsigned value at logical index i.
	extract := func(i int) uint8 {
		b := packed[i/valuesPerUnit]
		shift := uint(i%valuesPerUnit) * uint(bitsPerValue)
		return (b >> shift) & mask
	}

	switch dtype {
	case dtypes.Uint4, dtypes.Uint2:
		out := make([]int8, numValues)
		for i := range numValues {
			out[i] = int8(extract(i))
		}
		return out
	case dtypes.Int4, dtypes.Int2:
		out := make([]int8, numValues)
		for i := range numValues {
			val := extract(i)
			if val&signBit != 0 {
				val |= signExtend
			}
			out[i] = int8(val)
		}
		return out
	default:
		exceptions.Panicf("UnpackSubBytes: unsupported packed dtype %s", dtype)
		panic(nil) // needed for lint.
	}
}

// UnpackSubByteAt returns the packed value at the given upacked position -- only one value is unpacked.
// The packed slice represents sub-byte data (Uint4, Int4, Uint2 and Int2, packed into []uint8).
// It returns the value as an int8, enough bits for both unsigned and signed values.
//
// It can panic if non-packed dtype is given, or an out-of-bound unpackedIdx.
func UnpackSubByteAt(packed []uint8, dtype dtypes.DType, unpackedIdx int) int8 {
	bitsPerValue := dtype.Bits()
	valuesPerUnit := dtype.ValuesPerStorageUnit()
	numPackedValues := len(packed) * valuesPerUnit
	if unpackedIdx < 0 || unpackedIdx >= numPackedValues {
		exceptions.Panicf("UnpackSubByteAt: invalid index %d -- valid range is 0 < unpackedIdx < %d",
			unpackedIdx, numPackedValues)
	}
	mask := uint8((1 << bitsPerValue) - 1) // 0x0F for 4-bit, 0x03 for 2-bit
	b := packed[unpackedIdx/valuesPerUnit]
	shift := uint(unpackedIdx%valuesPerUnit) * uint(bitsPerValue)
	rawValue := (b >> shift) & mask
	switch dtype {
	case dtypes.Uint4, dtypes.Uint2:
		return int8(rawValue)
	case dtypes.Int4, dtypes.Int2:
		signBit := uint8(1 << (bitsPerValue - 1))
		signExtend := ^mask // 0xF0 for 4-bit, 0xFC for 2-bit
		if rawValue&signBit != 0 {
			rawValue |= signExtend
		}
		return int8(rawValue)
	default:
		exceptions.Panicf("UnpackSubByteAt: unsupported packed dtype %s", dtype)
		panic(nil) // needed for lint.
	}
}
