package graph_test

import (
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
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
