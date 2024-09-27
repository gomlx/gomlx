package initializers

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

func TestBroadcastTensorToShape(t *testing.T) {
	graphtest.RunTestGraphFn(t, "BroadcastTensorToShape",
		func(g *Graph) (inputs, outputs []*Node) {
			t := tensors.FromValue([]float64{7.0, 11.0, 17.0})
			inputs = []*Node{ConstCachedTensor(g, t)}
			initFn := BroadcastTensorToShape(t)
			outputs = []*Node{
				initFn(g, shapes.Make(dtypes.Float32, 4, 3)),
				initFn(g, shapes.Make(dtypes.Int32, 3)),
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
		}, 0)
}
