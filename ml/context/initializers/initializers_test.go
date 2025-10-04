package initializers

import (
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestBroadcastTensorToShape(t *testing.T) {
	graphtest.RunTestGraphFn(t, "BroadcastTensorToShape",
		func(g *Graph) (inputs, outputs []*Node) {
			t0 := tensors.FromValue([]float64{7.0, 11.0, 17.0})
			t1 := tensors.FromScalar(3.0)
			inputs = []*Node{ConstCachedTensor(g, t0), ConstCachedTensor(g, t1)}
			initFn0 := BroadcastTensorToShape(t0)
			initFn1 := BroadcastTensorToShape(t1)
			outputs = []*Node{
				initFn0(g, shapes.Make(dtypes.Float32, 4, 3)),
				initFn0(g, shapes.Make(dtypes.Int32, 3)),
				initFn1(g, shapes.Make(dtypes.Int8, 1, 2, 3)),
			}
			return
		}, []any{
			[][]float32{
				{7.0, 11.0, 17.0},
				{7.0, 11.0, 17.0},
				{7.0, 11.0, 17.0},
				{7.0, 11.0, 17.0},
			},
			[]int32{7, 11, 17},
			[][][]int8{{{3, 3, 3}, {3, 3, 3}}},
		}, 0)
}
