// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package layers

import (
	"fmt"
	"math"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

func TestDropBlock(t *testing.T) {
	backend := testutil.BuildTestBackend()
	scope := model.NewStore()
	scope.SetRNGStateFromSeed(42) // Always the same result.
	batchSize := 10
	width, height := 100, 100
	numChannels := 3
	shape := shapes.Make(dtypes.Float32, batchSize, width, height, numChannels)
	for _, dropRate := range []float64{0.1, 0.2} {
		for _, blockSize := range []int{1, 3, 4} {
			gotT := model.MustCallOnce(backend, scope, func(scope *model.Scope, g *Graph) *Node {
				scope.Store().SetTraining(g, true)
				batch := scope.RandomUniform(g, shape)
				batch = DropBlock(scope, batch).
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

func TestDropPath(t *testing.T) {
	backend := testutil.BuildTestBackend()
	scope := model.NewStore()
	scope.SetRNGStateFromSeed(42) // Always the same result.
	gotT := model.MustCallOnce(backend, scope, func(scope *model.Scope, g *Graph) *Node {
		scope.Store().SetTraining(g, true)
		ones := Ones(g, shapes.Make(dtypes.Float32, 10_000, 10, 10))
		masked := DropPath(scope, ones, Const(g, 0.07))
		require.NoError(t, masked.Shape().CheckDims(10_000, 10, 10))

		// Makes sure masks happens on all axes but the batch axis.
		reduced := ReduceSum(masked, 1, 2)
		maskedExamples := ConvertDType(GreaterThan(reduced, ScalarZero(g, reduced.DType())), dtypes.Float32)
		return ReduceAllMean(maskedExamples) // Ratio of examples that were not masked (not zero).
	})
	got := gotT.Value().(float32)
	fmt.Printf("DropPath with drop probability of 7%%: %.2f%% examples survived\n", 100.0*got)
	require.InDelta(t, float32(0.93), gotT.Value(), 0.01)
}
