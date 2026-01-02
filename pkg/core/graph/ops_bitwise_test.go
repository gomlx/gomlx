package graph_test

import (
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
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

func TestUnpack(t *testing.T) {
	t.Run("Int2", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "UnpackInt2", func(g *Graph) (inputs, outputs []*Node) {
			// Test cases: each uint8 unpacks to 4 Int2 values in order [bits 0-1, bits 2-3, bits 4-5, bits 6-7]
			// 0x00 = 0b00000000 -> [0, 0, 0, 0] (all bits are 00)
			// 0x03 = 0b00000011 -> [-1, 0, 0, 0] (bits 0-1 = 11 = -1)
			// 0x0C = 0b00001100 -> [0, -1, 0, 0] (bits 2-3 = 11 = -1)
			// 0x30 = 0b00110000 -> [0, 0, -1, 0] (bits 4-5 = 11 = -1)
			// 0xC0 = 0b11000000 -> [0, 0, 0, -1] (bits 6-7 = 11 = -1)
			// 0xFF = 0b11111111 -> [-1, -1, -1, -1] (all bits are 11 = -1)
			// 0x12 = 0b00010010 -> [-2, 0, 1, 0] (bits 0-1=10(-2), bits 2-3=00(0), bits 4-5=01(1), bits 6-7=00(0))
			operand := Const(g, []uint8{0x00, 0x03, 0x0C, 0x30, 0xC0, 0xFF, 0x12})
			inputs = []*Node{operand}
			outputs = []*Node{UnpackInt2(operand)}
			return
		}, []any{
			// Expected output shape: [7, 4] with dtype Int2
			// Each uint8 unpacks to 4 Int2 values in order: [bits 0-1, bits 2-3, bits 4-5, bits 6-7]
			[][]int8{
				{0, 0, 0, 0},     // 0x00: all bits are 00
				{-1, 0, 0, 0},    // 0x03: bits 0-1=11(-1)
				{0, -1, 0, 0},    // 0x0C: bits 2-3=11(-1)
				{0, 0, -1, 0},    // 0x30: bits 4-5=11(-1)
				{0, 0, 0, -1},    // 0xC0: bits 6-7=11(-1)
				{-1, -1, -1, -1}, // 0xFF: all bits are 11(-1)
				{-2, 0, 1, 0},    // 0x12: bits 0-1=10(-2), bits 2-3=00(0), bits 4-5=01(1), bits 6-7=00(0)
			},
		}, -1)
	})
}
