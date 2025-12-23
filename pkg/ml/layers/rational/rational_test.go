package rational

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/ctxtest"
	"github.com/gomlx/gomlx/pkg/ml/layers/activations"

	_ "github.com/gomlx/gomlx/backends/default"
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
			y = ReduceSum(y, -1)
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

func TestMultipleOutputs(t *testing.T) {
	ctxtest.RunTestGraphFn(t, "Identity: multiple outputs, numInputGroups=2",
		func(ctx *context.Context, g *Graph) (inputs, outputs []*Node) {
			batchSize := 3
			x := IotaFull(g, shapes.Make(dtypes.Float32, batchSize, 6))
			y := New(ctx, x).
				Approximate("identity").
				Version("B").
				WithInputGroups(2).
				WithMultipleOutputs(2).
				Done()
			inputs = []*Node{x}
			outputs = []*Node{y}
			return
		}, []any{
			// Output: shape [batchSize, outputDim=2, inputDim=6]
			[][][]float32{
				{{0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 5}},
				{{6, 7, 8, 9, 10, 11}, {6, 7, 8, 9, 10, 11}},
				{{12, 13, 14, 15, 16, 17}, {12, 13, 14, 15, 16, 17}},
			},
		}, 0.01)
}
