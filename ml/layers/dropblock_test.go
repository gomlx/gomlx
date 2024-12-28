package layers

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
	"math"
	"testing"
)

func TestDropBlock(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.RngStateFromSeed(42) // Always the same result.
	batchSize := 10
	width, height := 100, 100
	numChannels := 3
	shape := shapes.Make(dtypes.Float32, batchSize, width, height, numChannels)
	for _, dropRate := range []float64{0.1, 0.2} {
		for _, blockSize := range []int{1, 3, 4} {
			gotT := context.ExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
				ctx.SetTraining(g, true)
				batch := ctx.RandomUniform(g, shape)
				batch = DropBlock(ctx, batch).
					ChannelsAxis(images.ChannelsLast).
					WithDropoutProbability(dropRate).
					WithBlockSize(blockSize).
					Done()
				// ReduceSum values of channels: it only counts as zero if all channel values are zero.
				countZeros := ReduceSum(batch, -1)
				countZeros = Equal(countZeros, ScalarZero(g, shape.DType))
				countZeros = ConvertDType(countZeros, shape.DType) // Every pixel will be 1 or 0.
				return ReduceAllMean(countZeros)                   // Return mean of zeros
			})
			got := gotT.Value().(float32)
			name := fmt.Sprintf("DropBlock(blockSize=%d, dropRate=%.3f)", blockSize, dropRate)
			want := 1.0 - math.Pow(1.0-dropRate/float64(blockSize*blockSize), float64(blockSize*blockSize))
			fmt.Printf("%s dropped %.1f%%, wanted %.1f%%\n", name, 100.0*got, 100.0*want)
			require.InDelta(t, want, float64(got), 0.01)
		}
	}
}
