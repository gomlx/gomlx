package rational

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/ctxtest"
	"github.com/gomlx/gomlx/ml/layers/activations"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

func TestConfig_Approximate(t *testing.T) {
	ctxtest.RunTestGraphFn(t, "Swish-Approximate",
		func(ctx *context.Context, g *Graph) (inputs, outputs []*Node) {
			batchSize := 1_000
			minX, maxX := -10.0, 10.0
			xs := Iota(g, shapes.Make(dtypes.Float64, batchSize), 0)
			xs = MulScalar(xs, (maxX-minX)/float64(batchSize-1))
			xs = AddScalar(xs, minX)
			y := New(ctx, xs).
				Approximate("swish").
				Version("B").
				WithInputGroups(1).
				WithDegrees(6, 5).
				Done()
			want := activations.Swish(xs)
			//want := activations.Relu(xs)
			rmse := Sqrt(ReduceAllMean(Square(Sub(want, y))))
			inputs = []*Node{}
			outputs = []*Node{rmse}
			return
		}, []any{
			0.016326732768871793,
		}, 0.01)

	// Add a multiplier that tries to preserve the variance.
	ctxtest.RunTestGraphFn(t, "Relu-Variance",
		func(ctx *context.Context, g *Graph) (inputs, outputs []*Node) {
			batchSize := 1_000
			numOutputs := 1000
			xs := ctx.RandomNormal(g, shapes.Make(dtypes.Float64, batchSize, 1))
			y := New(ctx, xs).
				Approximate("swish").
				Version("B").
				WithMultipleOutputs(numOutputs).
				WithDegrees(5, 4).
				WithMultiplier(true).
				Done()
			y = ReduceSum(y, -2)
			yMean := ReduceAllMean(y)
			variance := DivScalar(ReduceAllSum(Square(Sub(y, yMean))), float64(batchSize*numOutputs-1))
			inputs = []*Node{}
			outputs = []*Node{yMean, variance}
			return
		}, []any{
			0.0,
			1.0,
		}, 1.0)
}
