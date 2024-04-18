package gnn

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"testing"
)

func TestPoolMessages(t *testing.T) {
	manager := graphtest.BuildTestManager()
	ctx := context.NewContext(manager)
	ctx.SetParam(ParamPoolingType, "sum|max")
	graphtest.RunTestGraphFn(
		t, "poolMessages()",
		func(g *Graph) (inputs, outputs []*Node) {
			mask := Const(g, [][]bool{
				{true, false, true},
				{false, false, false}})
			x := IotaFull(g, shapes.Make(shapes.F32, append(mask.Shape().Dimensions, 5)...))
			degree := Const(g, [][]float32{{10}, {7}})
			outputNoDegree := poolMessages(ctx, x, mask, nil)
			outputWithDegree := poolMessages(ctx, x, mask, degree)
			inputs = []*Node{x, mask}
			outputs = []*Node{outputNoDegree, outputWithDegree}
			return
		}, []any{
			[][]float32{
				{ /* sum */ 10, 12, 14, 16, 18 /* max */, 10, 11, 12, 13, 14},
				{ /* sum */ 0, 0, 0, 0, 0 /* max */, 0, 0, 0, 0, 0},
			},
			[][]float32{
				{ /* sum */ 50, 60, 70, 80, 90 /* max */, 10, 11, 12, 13, 14},
				{ /* sum */ 0, 0, 0, 0, 0 /* max */, 0, 0, 0, 0, 0},
			},
		}, slices.Epsilon)
}
