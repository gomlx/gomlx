/*
 *	Copyright 2025 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package graph_test

import (
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func TestPadToBucketSize(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("Pow2Bucketing", func(t *testing.T) {
		exec := MustNewExec(backend, func(x *Node) *Node {
			return PadToPow2(x, Const(x.Graph(), float32(0)), 0)
		})
		defer exec.Finalize()

		// Input [7, 512] -> Output [8, 512]
		input := tensors.FromValue(make([][]float32, 7))
		for i := range input.Value().([][]float32) {
			input.Value().([][]float32)[i] = make([]float32, 512)
		}
		output := exec.MustExec(input)[0]
		require.Equal(t, []int{8, 512}, output.Shape().Dimensions)

		// Verify the output contains the input data plus padding
		require.Equal(t, dtypes.Float32, output.DType())
	})

	t.Run("LinearBucketing", func(t *testing.T) {
		exec := MustNewExec(backend, func(x *Node) *Node {
			return PadToMultiple(x, Const(x.Graph(), float32(0)), 8, 0)
		})
		defer exec.Finalize()

		// Input [7, 512] -> Output [8, 512]
		input := tensors.FromValue(make([][]float32, 7))
		for i := range input.Value().([][]float32) {
			input.Value().([][]float32)[i] = make([]float32, 512)
		}
		output := exec.MustExec(input)[0]
		require.Equal(t, []int{8, 512}, output.Shape().Dimensions)

		// Input [9, 512] -> Output [16, 512]
		input2 := tensors.FromValue(make([][]float32, 9))
		for i := range input2.Value().([][]float32) {
			input2.Value().([][]float32)[i] = make([]float32, 512)
		}
		output2 := exec.MustExec(input2)[0]
		require.Equal(t, []int{16, 512}, output2.Shape().Dimensions)
	})

	t.Run("MultipleAxes", func(t *testing.T) {
		exec := MustNewExec(backend, func(x *Node) *Node {
			return PadToPow2(x, Const(x.Graph(), float32(0)), 0, 1)
		})
		defer exec.Finalize()

		// Input [7, 100] -> Output [8, 128]
		input := tensors.FromValue(make([][]float32, 7))
		for i := range input.Value().([][]float32) {
			input.Value().([][]float32)[i] = make([]float32, 100)
		}
		output := exec.MustExec(input)[0]
		require.Equal(t, []int{8, 128}, output.Shape().Dimensions)
	})

	t.Run("NoPaddingNeeded", func(t *testing.T) {
		exec := MustNewExec(backend, func(x *Node) *Node {
			return PadToPow2(x, Const(x.Graph(), float32(0)), 0)
		})
		defer exec.Finalize()

		// Input [8, 512] -> Output [8, 512] (already power of 2)
		input := tensors.FromValue(make([][]float32, 8))
		for i := range input.Value().([][]float32) {
			input.Value().([][]float32)[i] = make([]float32, 512)
		}
		output := exec.MustExec(input)[0]
		require.Equal(t, []int{8, 512}, output.Shape().Dimensions)
	})

	t.Run("NegativeAxisIndex", func(t *testing.T) {
		exec := MustNewExec(backend, func(x *Node) *Node {
			// -1 means last axis
			return PadToPow2(x, Const(x.Graph(), float32(0)), -1)
		})
		defer exec.Finalize()

		// Input [7, 100] -> Output [7, 128] (padding last axis only)
		input := tensors.FromValue(make([][]float32, 7))
		for i := range input.Value().([][]float32) {
			input.Value().([][]float32)[i] = make([]float32, 100)
		}
		output := exec.MustExec(input)[0]
		require.Equal(t, []int{7, 128}, output.Shape().Dimensions)
	})

	t.Run("AllAxesByDefault", func(t *testing.T) {
		exec := MustNewExec(backend, func(x *Node) *Node {
			// No axes specified = pad all axes
			return PadToPow2(x, Const(x.Graph(), float32(0)))
		})
		defer exec.Finalize()

		// Input [7, 100] -> Output [8, 128] (padding both axes)
		input := tensors.FromValue(make([][]float32, 7))
		for i := range input.Value().([][]float32) {
			input.Value().([][]float32)[i] = make([]float32, 100)
		}
		output := exec.MustExec(input)[0]
		require.Equal(t, []int{8, 128}, output.Shape().Dimensions)
	})

	t.Run("WithDifferentDTypes", func(t *testing.T) {
		exec := MustNewExec(backend, func(x *Node) *Node {
			return PadToPow2(x, Const(x.Graph(), int32(0)), 0)
		})
		defer exec.Finalize()

		// Input [7, 512] -> Output [8, 512]
		input := tensors.FromValue(make([][]int32, 7))
		for i := range input.Value().([][]int32) {
			input.Value().([][]int32)[i] = make([]int32, 512)
		}
		output := exec.MustExec(input)[0]
		require.Equal(t, []int{8, 512}, output.Shape().Dimensions)
		require.Equal(t, dtypes.Int32, output.DType())
	})
}

func TestPadToBucketSize_Values(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("VerifyPaddingValues", func(t *testing.T) {
		exec := MustNewExec(backend, func(x *Node) *Node {
			return PadToPow2(x, Const(x.Graph(), float32(-1.0)), 0)
		})
		defer exec.Finalize()

		// Create input [3, 2] with known values
		input := tensors.FromValue([][]float32{
			{1.0, 2.0},
			{3.0, 4.0},
			{5.0, 6.0},
		})

		// Expected output [4, 2] with -1.0 padding
		output := exec.MustExec(input)[0]
		require.Equal(t, []int{4, 2}, output.Shape().Dimensions)

		// Check that original values are preserved
		outputData := output.Value().([][]float32)
		require.Equal(t, float32(1.0), outputData[0][0])
		require.Equal(t, float32(2.0), outputData[0][1])
		require.Equal(t, float32(3.0), outputData[1][0])
		require.Equal(t, float32(4.0), outputData[1][1])
		require.Equal(t, float32(5.0), outputData[2][0])
		require.Equal(t, float32(6.0), outputData[2][1])

		// Check that padding is -1.0
		require.Equal(t, float32(-1.0), outputData[3][0])
		require.Equal(t, float32(-1.0), outputData[3][1])
	})
}
