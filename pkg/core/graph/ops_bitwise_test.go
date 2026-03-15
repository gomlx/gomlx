// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"fmt"
	"slices"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
)

func TestBitwiseShifts(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Shifts", func(g *Graph) (inputs, outputs []*Node) {
		operand := Const(g, []int8{-3, -2, -1, 1, 2, 4})
		inputs = []*Node{operand}
		outputs = []*Node{
			BitwiseShiftLeftScalar(operand, 1),
			BitwiseShiftRightArithmeticScalar(operand, 1),
			BitwiseShiftRightLogicalScalar(operand, 1),
		}
		return
	}, []any{
		[]int8{-6, -4, -2, 2, 4, 8},
		[]int8{-2, -1, -1, 0, 1, 2},
		[]int8{126, 127, 127, 0, 1, 2},
	}, -1)
}

func TestBitcastSubByteDTypes(t *testing.T) {
	// Int2: each uint8 unpacks to 4 Int2 values in order [bits 0-1, bits 2-3, bits 4-5, bits 6-7]
	// 0x00 = 0b00000000 -> [0, 0, 0, 0] (all bits are 00)
	// 0x03 = 0b00000011 -> [-1, 0, 0, 0] (bits 0-1 = 11 = -1)
	// 0x0C = 0b00001100 -> [0, -1, 0, 0] (bits 2-3 = 11 = -1)
	// 0x30 = 0b00110000 -> [0, 0, -1, 0] (bits 4-5 = 11 = -1)
	// 0xC0 = 0b11000000 -> [0, 0, 0, -1] (bits 6-7 = 11 = -1)
	// 0xFF = 0b11111111 -> [-1, -1, -1, -1] (all bits are 11 = -1)
	// 0x12 = 0b00010010 -> [-2, 0, 1, 0] (bits 0-1=10(-2), bits 2-3=00(0), bits 4-5=01(1), bits 6-7=00(0))
	int2Value := []uint8{0x00, 0x03, 0x0C, 0x30, 0xC0, 0xFF, 0x12}
	int2ToInt8Converted := []int8{
		0, 0, 0, 0, // 0x00: all bits are 00
		-1, 0, 0, 0, // 0x03: bits 0-1=11(-1)
		0, -1, 0, 0, // 0x0C: bits 2-3=11(-1)
		0, 0, -1, 0, // 0x30: bits 4-5=11(-1)
		0, 0, 0, -1, // 0xC0: bits 6-7=11(-1)
		-1, -1, -1, -1, // 0xFF: all bits are 11(-1)
		-2, 0, 1, 0, // 0x12: bits 0-1=10(-2), bits 2-3=00(0), bits 4-5=01(1), bits 6-7=00(0)
	}

	int4Value := []uint8{0x00, 0x0F, 0xF0, 0xFF, 0x12, 0x87, 0x78}
	int4ToInt8Converted := []int8{
		0, 0, // 0x00: all nibbles are 0000
		-1, 0, // 0x0F: bits 0-3=1111(-1)
		0, -1, // 0xF0: bits 4-7=1111(-1)
		-1, -1, // 0xFF: all nibbles are 1111(-1)
		2, 1, // 0x12: bits 0-3=0010(2), bits 4-7=0001(1)
		7, -8, // 0x87: bits 0-3=0111(7), bits 4-7=1000(-8)
		-8, 7, // 0x78: bits 0-3=1000(-8), bits 4-7=0111(7)
	}

	uint2Value := []uint8{0x00, 0x03, 0x0C, 0x30, 0xC0, 0xFF, 0x12}
	uint2ToUint8Converted := []uint8{
		0, 0, 0, 0, // 0x00: all bits are 00
		3, 0, 0, 0, // 0x03: bits 0-1=11(3)
		0, 3, 0, 0, // 0x0C: bits 2-3=11(3)
		0, 0, 3, 0, // 0x30: bits 4-5=11(3)
		0, 0, 0, 3, // 0xC0: bits 6-7=11(3)
		3, 3, 3, 3, // 0xFF: all bits are 11(3)
		2, 0, 1, 0, // 0x12: bits 0-1=10(2), bits 2-3=00(0), bits 4-5=01(1), bits 6-7=00(0)
	}

	uint4Value := []uint8{0x00, 0x0F, 0xF0, 0xFF, 0x12, 0x87, 0x78}
	uint4ToUint8Converted := []uint8{
		0, 0, // 0x00: all nibbles are 0000
		15, 0, // 0x0F: bits 0-3=1111(15)
		0, 15, // 0xF0: bits 4-7=1111(15)
		15, 15, // 0xFF: all nibbles are 1111(15)
		2, 1, // 0x12: bits 0-3=0010(2), bits 4-7=0001(1)
		7, 8, // 0x87: bits 0-3=0111(7), bits 4-7=1000(8)
		8, 7, // 0x78: bits 0-3=1000(8), bits 4-7=0111(7)
	}

	genericValue := []uint8{0x12}
	genericInt2Converted := []int8{-2, 0, 1, 0}
	genericInt4Converted := []int8{2, 1}
	genericUint2Converted := []uint8{2, 0, 1, 0}
	genericUint4Converted := []uint8{2, 1}

	graphtest.TestOfficialBackends(t, func(t *testing.T, backend backends.Backend) {
		t.Run("Int2-AsInputParam", func(t *testing.T) {
			output := MustExecOnce(backend, func(packed *Node) *Node {
				return ConvertDType(Bitcast(packed, dtypes.Int2), dtypes.Int8)
			}, int2Value)
			if output.DType() != dtypes.Int8 {
				exceptions.Panicf("expected dtype Int8, got %s", output.DType())
			}
			output.Shape().CheckDims(7, 4)
			fmt.Printf("Int2 value %X -> %s\n", int2Value, output)
			flatData, err := tensors.CopyFlatData[int8](output)
			if err != nil {
				t.Fatalf("Failed to copy result: %+v", err)
			}
			if !slices.Equal(flatData, int2ToInt8Converted) {
				t.Errorf("expected %v, got %v", int2ToInt8Converted, flatData)
			}
		})

		t.Run("Int2-AsConst", func(t *testing.T) {
			if true {
				t.Skip("Skipping broken test: see https://github.com/openxla/xla/issues/38964")
			}
			output := MustExecOnce(backend, func(g *Graph) *Node {
				packed := Const(g, int2Value)
				packed.Shape().Check(dtypes.Uint8, 7)
				return ConvertDType(Bitcast(packed, dtypes.Int2), dtypes.Int8)
			})
			if output.DType() != dtypes.Int8 {
				exceptions.Panicf("expected dtype Int8, got %s", output.DType())
			}
			output.Shape().CheckDims(7, 4)
			fmt.Printf("Int2 value %X -> %s\n", int2Value, output)
			flatData, err := tensors.CopyFlatData[int8](output)
			if err != nil {
				t.Fatalf("Failed to copy result: %+v", err)
			}
			if !slices.Equal(flatData, int2ToInt8Converted) {
				t.Errorf("expected %v, got %v", int2ToInt8Converted, flatData)
			}
		})

		t.Run("Int4-AsInputParam", func(t *testing.T) {
			output := MustExecOnce(backend, func(packed *Node) *Node {
				return ConvertDType(Bitcast(packed, dtypes.Int4), dtypes.Int8)
			}, int4Value)
			if output.DType() != dtypes.Int8 {
				exceptions.Panicf("expected dtype Int8, got %s", output.DType())
			}
			output.Shape().CheckDims(7, 2)
			fmt.Printf("Int4 value %X -> %s\n", int4Value, output)
			flatData, err := tensors.CopyFlatData[int8](output)
			if err != nil {
				t.Fatalf("Failed to copy result: %+v", err)
			}
			if !slices.Equal(flatData, int4ToInt8Converted) {
				t.Errorf("expected %v, got %v", int4ToInt8Converted, flatData)
			}
		})

		t.Run("Int4-AsConst", func(t *testing.T) {
			if true {
				t.Skip("Skipping broken test: see https://github.com/openxla/xla/issues/38964")
			}
			output := MustExecOnce(backend, func(g *Graph) *Node {
				packed := Const(g, int4Value)
				packed.Shape().Check(dtypes.Uint8, 7)
				return ConvertDType(Bitcast(packed, dtypes.Int4), dtypes.Int8)
			})
			if output.DType() != dtypes.Int8 {
				exceptions.Panicf("expected dtype Int8, got %s", output.DType())
			}
			output.Shape().CheckDims(7, 2)
			fmt.Printf("Int4 value %X -> %s\n", int4Value, output)
			flatData, err := tensors.CopyFlatData[int8](output)
			if err != nil {
				t.Fatalf("Failed to copy result: %+v", err)
			}
			if !slices.Equal(flatData, int4ToInt8Converted) {
				t.Errorf("expected %v, got %v", int4ToInt8Converted, flatData)
			}
		})

		t.Run("Uint2-AsInputParam", func(t *testing.T) {
			output := MustExecOnce(backend, func(packed *Node) *Node {
				return ConvertDType(Bitcast(packed, dtypes.Uint2), dtypes.Uint8)
			}, uint2Value)
			if output.DType() != dtypes.Uint8 {
				exceptions.Panicf("expected dtype Uint8, got %s", output.DType())
			}
			output.Shape().CheckDims(7, 4)
			fmt.Printf("Uint2 value %X -> %s\n", uint2Value, output)
			flatData, err := tensors.CopyFlatData[uint8](output)
			if err != nil {
				t.Fatalf("Failed to copy result: %+v", err)
			}
			if !slices.Equal(flatData, uint2ToUint8Converted) {
				t.Errorf("expected %v, got %v", uint2ToUint8Converted, flatData)
			}
		})

		t.Run("Uint2-AsConst", func(t *testing.T) {
			if true {
				t.Skip("Skipping broken test: see https://github.com/openxla/xla/issues/38964")
			}
			output := MustExecOnce(backend, func(g *Graph) *Node {
				packed := Const(g, uint2Value)
				packed.Shape().Check(dtypes.Uint8, 7)
				return ConvertDType(Bitcast(packed, dtypes.Uint2), dtypes.Uint8)
			})
			if output.DType() != dtypes.Uint8 {
				exceptions.Panicf("expected dtype Uint8, got %s", output.DType())
			}
			output.Shape().CheckDims(7, 4)
			fmt.Printf("Uint2 value %X -> %s\n", uint2Value, output)
			flatData, err := tensors.CopyFlatData[uint8](output)
			if err != nil {
				t.Fatalf("Failed to copy result: %+v", err)
			}
			if !slices.Equal(flatData, uint2ToUint8Converted) {
				t.Errorf("expected %v, got %v", uint2ToUint8Converted, flatData)
			}
		})

		t.Run("Uint4-AsInputParam", func(t *testing.T) {
			output := MustExecOnce(backend, func(packed *Node) *Node {
				return ConvertDType(Bitcast(packed, dtypes.Uint4), dtypes.Uint8)
			}, uint4Value)
			if output.DType() != dtypes.Uint8 {
				exceptions.Panicf("expected dtype Uint8, got %s", output.DType())
			}
			output.Shape().CheckDims(7, 2)
			fmt.Printf("Uint4 value %X -> %s\n", uint4Value, output)
			flatData, err := tensors.CopyFlatData[uint8](output)
			if err != nil {
				t.Fatalf("Failed to copy result: %+v", err)
			}
			if !slices.Equal(flatData, uint4ToUint8Converted) {
				t.Errorf("expected %v, got %v", uint4ToUint8Converted, flatData)
			}
		})

		t.Run("Uint4-AsConst", func(t *testing.T) {
			if true {
				t.Skip("Skipping broken test: see https://github.com/openxla/xla/issues/38964")
			}
			output := MustExecOnce(backend, func(g *Graph) *Node {
				packed := Const(g, uint4Value)
				packed.Shape().Check(dtypes.Uint8, 7)
				return ConvertDType(Bitcast(packed, dtypes.Uint4), dtypes.Uint8)
			})
			if output.DType() != dtypes.Uint8 {
				exceptions.Panicf("expected dtype Uint8, got %s", output.DType())
			}
			output.Shape().CheckDims(7, 2)
			fmt.Printf("Uint4 value %X -> %s\n", uint4Value, output)
			flatData, err := tensors.CopyFlatData[uint8](output)
			if err != nil {
				t.Fatalf("Failed to copy result: %+v", err)
			}
			if !slices.Equal(flatData, uint4ToUint8Converted) {
				t.Errorf("expected %v, got %v", uint4ToUint8Converted, flatData)
			}
		})

		t.Run("Generic-AsInputParam", func(t *testing.T) {
			outputs := MustExecOnceN(backend, func(packed *Node) []*Node {
				return []*Node{
					ConvertDType(Bitcast(packed, dtypes.Int2), dtypes.Int8),
					ConvertDType(Bitcast(packed, dtypes.Int4), dtypes.Int8),
					ConvertDType(Bitcast(packed, dtypes.Uint2), dtypes.Uint8),
					ConvertDType(Bitcast(packed, dtypes.Uint4), dtypes.Uint8),
				}
			}, genericValue)

			flatData1, _ := tensors.CopyFlatData[int8](outputs[0])
			if !slices.Equal(flatData1, genericInt2Converted) {
				t.Errorf("Int2 expected %v, got %v", genericInt2Converted, flatData1)
			}

			flatData2, _ := tensors.CopyFlatData[int8](outputs[1])
			if !slices.Equal(flatData2, genericInt4Converted) {
				t.Errorf("Int4 expected %v, got %v", genericInt4Converted, flatData2)
			}

			flatData3, _ := tensors.CopyFlatData[uint8](outputs[2])
			if !slices.Equal(flatData3, genericUint2Converted) {
				t.Errorf("Uint2 expected %v, got %v", genericUint2Converted, flatData3)
			}

			flatData4, _ := tensors.CopyFlatData[uint8](outputs[3])
			if !slices.Equal(flatData4, genericUint4Converted) {
				t.Errorf("Uint4 expected %v, got %v", genericUint4Converted, flatData4)
			}
		})

		t.Run("Generic-AsConst", func(t *testing.T) {
			if true {
				t.Skip("Skipping broken test: see https://github.com/openxla/xla/issues/38964")
			}
			outputs := MustExecOnceN(backend, func(g *Graph) []*Node {
				packed := Const(g, genericValue)
				return []*Node{
					ConvertDType(Bitcast(packed, dtypes.Int2), dtypes.Int8),
					ConvertDType(Bitcast(packed, dtypes.Int4), dtypes.Int8),
					ConvertDType(Bitcast(packed, dtypes.Uint2), dtypes.Uint8),
					ConvertDType(Bitcast(packed, dtypes.Uint4), dtypes.Uint8),
				}
			})

			flatData1, _ := tensors.CopyFlatData[int8](outputs[0])
			if !slices.Equal(flatData1, genericInt2Converted) {
				t.Errorf("Int2 expected %v, got %v", genericInt2Converted, flatData1)
			}

			flatData2, _ := tensors.CopyFlatData[int8](outputs[1])
			if !slices.Equal(flatData2, genericInt4Converted) {
				t.Errorf("Int4 expected %v, got %v", genericInt4Converted, flatData2)
			}

			flatData3, _ := tensors.CopyFlatData[uint8](outputs[2])
			if !slices.Equal(flatData3, genericUint2Converted) {
				t.Errorf("Uint2 expected %v, got %v", genericUint2Converted, flatData3)
			}

			flatData4, _ := tensors.CopyFlatData[uint8](outputs[3])
			if !slices.Equal(flatData4, genericUint4Converted) {
				t.Errorf("Uint4 expected %v, got %v", genericUint4Converted, flatData4)
			}
		})

	}, /* Exclude backends: */ "go")
}
