// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDynamicShapes(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		if !backend.Capabilities().DynamicAxes {
			t.Skipf("Backend %q does not support DynamicAxes", backend.Name())
		}

		t.Run("BasicExec", func(t *testing.T) {
			// x is [batch, 3], returns x + x
			exec, err := NewExec(backend, func(x *Node) *Node {
				return Add(x, x)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			// Run with batch=2
			input2 := [][]float32{{1, 2, 3}, {4, 5, 6}}
			out2, err := exec.Call(input2)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{2, 4, 6}, {8, 10, 12}}, out2[0].Value())

			// Run with batch=3
			input3 := [][]float32{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}
			out3, err := exec.Call(input3)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{2, 2, 2}, {4, 4, 4}, {6, 6, 6}}, out3[0].Value())

			// Verify that they used the same cached graph!
			exec.cacheMu.Lock()
			cacheLen := len(exec.cache)
			exec.cacheMu.Unlock()
			assert.Equal(t, 1, cacheLen, "Graph should only be compiled once for dynamic batch dimension")
		})

		t.Run("Broadcast", func(t *testing.T) {
			// x has shape [batch, 1], broadcast to target shape [batch, 3]
			exec, err := NewExec(backend, func(x *Node) *Node {
				// Target shape has dynamic dimension "batch"
				targetShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
				return BroadcastToShape(x, targetShape)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			input := [][]float32{{1}, {2}}
			out, err := exec.Call(input)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{1, 1, 1}, {2, 2, 2}}, out[0].Value())
		})

		t.Run("Reshape", func(t *testing.T) {
			// x has shape [batch, 4], reshaped to [batch, 2, 2]
			exec, err := NewExec(backend, func(x *Node) *Node {
				targetShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 2, 2}, []string{"batch", "", ""})
				return ReshapeWithShape(x, targetShape)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			input := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
			out, err := exec.Call(input)
			require.NoError(t, err)
			assert.Equal(t, [][][]float32{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, out[0].Value())
		})

		t.Run("DynamicDimensionSizeAndShape", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) []*Node {
				dimSize := DynamicDimensionSize(x, 0)
				shapeNode := DynamicShape(x)
				return []*Node{dimSize, shapeNode}
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			// Run with batch=2
			input2 := [][]float32{{1, 2, 3}, {4, 5, 6}}
			out2, err := exec.Call(input2)
			require.NoError(t, err)
			assert.Equal(t, int32(2), out2[0].Value())
			assert.Equal(t, []int32{2, 3}, out2[1].Value())
		})

		t.Run("RankMismatchPanic", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) *Node {
				return Add(x, x)
			})
			require.NoError(t, err)

			// x is 2D but dynamicAxes expects 3D
			exec.WithDynamicAxes([]string{"batch", "time", ""})

			input := [][]float32{{1, 2}, {3, 4}}
			_, err = exec.Call(input)
			assert.Error(t, err)
			assert.Contains(t, err.Error(), "rank mismatch")
		})

		t.Run("DynamicReshape", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) *Node {
				// x has shape [batch, 4].
				// We reshape it to [batch, 2, 2].
				// Get the dynamic batch size.
				batchSize := DynamicDimensionSize(x, 0)
				batchSpec := NamedDynamicDim("batch", batchSize)
				return DynamicReshape(x, batchSpec, StaticDim(2), NamedInferredDim("inferred_dim"))
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			// Run with batch = 2
			input := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
			out, err := exec.Call(input)
			require.NoError(t, err)
			// Target shape should be [2, 2, 2]
			assert.Equal(t, [][][]float32{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, out[0].Value())
		})

		t.Run("DynamicReshapeLike", func(t *testing.T) {
			exec, err := NewExec(backend, func(x, ref *Node) *Node {
				// x has flat shape [batch*4], ref has dynamic shape [batch, 2, 2]
				return DynamicReshapeLike(x, ref)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes(
				[]string{"batch", ""},
				[]string{"batch", "", ""},
			)

			xInput := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
			refInput := [][][]float32{{{0, 0}, {0, 0}}, {{0, 0}, {0, 0}}}
			out, err := exec.Call(xInput, refInput)
			require.NoError(t, err)
			assert.Equal(t, [][][]float32{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, out[0].Value())
		})

		t.Run("DynamicReshapeStaticFallback", func(t *testing.T) {
			// When operand and specs are completely static, DynamicReshape should fallback to static Reshape.
			exec, err := NewExec(backend, func(x *Node) *Node {
				// x has static shape [2, 4]
				return DynamicReshape(x, StaticDim(2), StaticDim(2), InferredDim())
			})
			require.NoError(t, err)

			xInput := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][][]float32{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, out[0].Value())
		})

		t.Run("DynamicBroadcastInDim", func(t *testing.T) {
			exec, err := NewExec(backend, func(x, bias *Node) *Node {
				// x has dynamic shape [batch, seq]
				// bias has shape [seq]
				batchDim := DynamicDimensionSize(x, 0)
				seqDim := DynamicDimensionSize(x, 1)
				bBroadcast := DynamicBroadcastInDim(bias, []int{1},
					NamedDynamicDim("batch", batchDim),
					NamedDynamicDim("seq", seqDim),
				)
				return Add(x, bBroadcast)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes(
				[]string{"batch", "seq"},
				[]string{"seq"},
			)

			xInput := [][]float32{{1, 2, 3}, {4, 5, 6}}
			biasInput := []float32{10, 20, 30}
			out, err := exec.Call(xInput, biasInput)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{11, 22, 33}, {14, 25, 36}}, out[0].Value())
		})

		t.Run("DynamicBroadcastLike", func(t *testing.T) {
			exec, err := NewExec(backend, func(x, ref *Node) *Node {
				// x is [features], ref is [batch, features]
				return DynamicBroadcastLike(x, ref)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes(
				[]string{""},
				[]string{"batch", ""},
			)

			xInput := []float32{10, 20}
			refInput := [][]float32{{0, 0}, {0, 0}, {0, 0}} // batch=3, features=2
			out, err := exec.Call(xInput, refInput)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{10, 20}, {10, 20}, {10, 20}}, out[0].Value())
		})

		t.Run("DynamicBroadcastToDims", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) *Node {
				// x is [1, features=2], broadcast to dynamic [batch, features=2]
				batchDim := Const(x.Graph(), int32(2))
				return DynamicBroadcastToDims(x, NamedDynamicDim("batch", batchDim), StaticDim(2))
			})
			require.NoError(t, err)

			xInput := [][]float32{{7, 8}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{7, 8}, {7, 8}}, out[0].Value())
		})

		t.Run("DynamicBroadcastGradient", func(t *testing.T) {
			exec, err := NewExec(backend, func(x, bias *Node) *Node {
				bBroadcast := DynamicBroadcastLike(bias, x)
				loss := ReduceAllSum(Mul(x, bBroadcast))
				grads := Gradient(loss, bias)
				return grads[0]
			})
			require.NoError(t, err)
			exec.WithDynamicAxes(
				[]string{"batch", ""},
				[]string{""},
			)

			xInput := [][]float32{{1, 2}, {3, 4}} // batch=2, features=2
			biasInput := []float32{10, 20}
			out, err := exec.Call(xInput, biasInput)
			require.NoError(t, err)
			// grad wrt bias is sum_batch(x) = [1+3, 2+4] = [4, 6]
			assert.Equal(t, []float32{4, 6}, out[0].Value())
		})

		t.Run("DynamicBroadcastToShape", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) *Node {
				targetShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
				return DynamicBroadcastToShape(x, targetShape)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			xInput := [][]float32{{1}, {2}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{1, 1, 1}, {2, 2, 2}}, out[0].Value())
		})

		t.Run("BroadcastToShape_DynamicDelegation", func(t *testing.T) {
			// Calling BroadcastToShape on a node with dynamic shape delegates to DynamicBroadcastToShape.
			exec, err := NewExec(backend, func(x *Node) *Node {
				targetShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 2}, []string{"batch", ""})
				return BroadcastToShape(x, targetShape)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			xInput := [][]float32{{5}, {6}, {7}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{5, 5}, {6, 6}, {7, 7}}, out[0].Value())
		})

		t.Run("DynamicBroadcastToShape_StaticFallback", func(t *testing.T) {
			// Calling DynamicBroadcastToShape with static shape falls back to BroadcastToShape.
			exec, err := NewExec(backend, func(x *Node) *Node {
				targetShape := shapes.Make(dtypes.Float32, 2, 3)
				return DynamicBroadcastToShape(x, targetShape)
			})
			require.NoError(t, err)

			xInput := [][]float32{{1}, {2}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{1, 1, 1}, {2, 2, 2}}, out[0].Value())
		})

		t.Run("DynamicIota", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) *Node {
				batchSize := DynamicDimensionSize(x, 0)
				return DynamicIota(x.Graph(), dtypes.Int32, 1,
					NamedDynamicDim("batch", batchSize),
					StaticDim(3),
				)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			xInput := [][]float32{{10, 20}, {30, 40}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][]int32{{0, 1, 2}, {0, 1, 2}}, out[0].Value())
		})

		t.Run("DynamicPad", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node, padStart *Node) *Node {
				fillVal := ScalarZero(x.Graph(), dtypes.Float32)
				return DynamicPad(x, fillVal,
					DynamicPadAxisSpec{StartNode: padStart, End: 1},
					DynamicPadAxis(1, 0, 0),
				)
			})
			require.NoError(t, err)

			xInput := [][]float32{{1, 2}, {3, 4}}
			padStart := int32(1)
			out, err := exec.Call(xInput, padStart)
			require.NoError(t, err)
			expected := [][]float32{
				{0, 0, 0},
				{0, 1, 2},
				{0, 3, 4},
				{0, 0, 0},
			}
			assert.Equal(t, expected, out[0].Value())
		})

		t.Run("BroadcastLike_Dynamic", func(t *testing.T) {
			exec, err := NewExec(backend, func(bias, x *Node) *Node {
				return BroadcastLike(bias, x)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes(
				[]string{""},
				[]string{"batch", ""},
			)

			biasInput := []float32{10, 20}
			xInput := [][]float32{{1, 2}, {3, 4}, {5, 6}} // batch=3
			out, err := exec.Call(biasInput, xInput)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{10, 20}, {10, 20}, {10, 20}}, out[0].Value())
		})

		t.Run("BroadcastToDims_PreserveDynamicAxis", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) *Node {
				// x is [batch, 1], broadcast to [batch, 3]
				return BroadcastToDims(x, shapes.DynamicDim, 3)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			xInput := [][]float32{{1}, {2}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{1, 1, 1}, {2, 2, 2}}, out[0].Value())
		})

		t.Run("BroadcastToDims_CannotIntroduceDynamicAxis", func(t *testing.T) {
			assert.Panics(t, func() {
				g := NewGraph(backend, "test_broadcast_to_dims_invalid")
				x := Iota(g, shapes.Make(dtypes.Float32, 1, 3), 1)
				// Cannot specify -1 for a static axis 0
				_ = BroadcastToDims(x, shapes.DynamicDim, 3)
			})
		})

		t.Run("DynamicBroadcastToShape_CannotBroadcast1ToDynamicWithoutReference", func(t *testing.T) {
			assert.Panics(t, func() {
				g := NewGraph(backend, "test_dynamic_broadcast_1_to_dynamic")
				x := Iota(g, shapes.Make(dtypes.Float32, 1, 3), 1)
				targetShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
				_ = DynamicBroadcastToShape(x, targetShape)
			})
		})

		t.Run("ExpandAxes_Dynamic", func(t *testing.T) {
			var nodeShape shapes.Shape
			exec, err := NewExec(backend, func(x *Node) *Node {
				// x is [batch, 3] -> expand at axis 1 -> [batch, 1, 3]
				out := ExpandAxes(x, 1)
				nodeShape = out.Shape()
				return out
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			xInput := [][]float32{{1, 2, 3}, {4, 5, 6}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 1, 3}, []string{"batch", "", ""}), nodeShape)
			assert.Equal(t, [][][]float32{{{1, 2, 3}}, {{4, 5, 6}}}, out[0].Value())
		})

		t.Run("InsertAxes_Dynamic", func(t *testing.T) {
			var nodeShape shapes.Shape
			exec, err := NewExec(backend, func(x *Node) *Node {
				// x is [batch, 3] -> insert axis before axis 0 -> [1, batch, 3]
				out := InsertAxes(x, 0)
				nodeShape = out.Shape()
				return out
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			xInput := [][]float32{{1, 2, 3}, {4, 5, 6}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, shapes.MakeDynamic(dtypes.Float32, []int{1, shapes.DynamicDim, 3}, []string{"", "batch", ""}), nodeShape)
			assert.Equal(t, [][][]float32{{{1, 2, 3}, {4, 5, 6}}}, out[0].Value())
		})

		t.Run("Squeeze_Dynamic", func(t *testing.T) {
			var nodeShape shapes.Shape
			exec, err := NewExec(backend, func(x *Node) *Node {
				// x is [batch, 1, 3] -> squeeze axis 1 -> [batch, 3]
				out := Squeeze(x, 1)
				nodeShape = out.Shape()
				return out
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", "", ""})

			xInput := [][][]float32{{{1, 2, 3}}, {{4, 5, 6}}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""}), nodeShape)
			assert.Equal(t, [][]float32{{1, 2, 3}, {4, 5, 6}}, out[0].Value())
		})

		t.Run("BroadcastPrefix_Dynamic", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) *Node {
				// x is [batch, 2] -> prepend prefix 2 -> [2, batch, 2]
				return BroadcastPrefix(x, 2)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			xInput := [][]float32{{1, 2}, {3, 4}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][][]float32{{{1, 2}, {3, 4}}, {{1, 2}, {3, 4}}}, out[0].Value())
		})

		t.Run("IotaLike_Dynamic", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) *Node {
				// x is [batch, 3] -> iota along axis 1 -> [batch, 3]
				return IotaLike(x, 1)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			xInput := [][]float32{{0, 0, 0}, {0, 0, 0}} // batch=2
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{0, 1, 2}, {0, 1, 2}}, out[0].Value())
		})

		t.Run("ReshapeLike_Dynamic", func(t *testing.T) {
			exec, err := NewExec(backend, func(x, ref *Node) *Node {
				// x is [batch, 4], ref is [batch, 2, 2]
				return ReshapeLike(x, ref)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes(
				[]string{"batch", ""},
				[]string{"batch", "", ""},
			)

			xInput := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
			refInput := [][][]float32{{{0, 0}, {0, 0}}, {{0, 0}, {0, 0}}}
			out, err := exec.Call(xInput, refInput)
			require.NoError(t, err)
			assert.Equal(t, [][][]float32{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, out[0].Value())
		})

		t.Run("TopK_Dynamic", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) (*Node, *Node) {
				// x is [batch, 4] -> TopK(2, axis=1) -> [batch, 2]
				return TopK(x, 2, 1)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			xInput := [][]float32{{10, 40, 20, 30}, {80, 50, 70, 60}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][]float32{{40, 30}, {80, 70}}, out[0].Value())
			assert.Equal(t, [][]int32{{1, 3}, {0, 2}}, out[1].Value())
		})

		t.Run("TopKMask_Dynamic", func(t *testing.T) {
			exec, err := NewExec(backend, func(x *Node) *Node {
				// x is [batch, 4] -> TopKMask(2, axis=1) -> [batch, 4]
				return TopKMask(x, 2, 1)
			})
			require.NoError(t, err)
			exec.WithDynamicAxes([]string{"batch", ""})

			xInput := [][]float32{{10, 40, 20, 30}, {80, 50, 70, 60}}
			out, err := exec.Call(xInput)
			require.NoError(t, err)
			assert.Equal(t, [][]bool{{false, true, false, true}, {true, false, true, false}}, out[0].Value())
		})

		t.Run("DimensionSpec_AccessorsAndMethods", func(t *testing.T) {
			sStatic := StaticDim(128)
			assert.True(t, sStatic.IsStatic())
			assert.False(t, sStatic.IsDynamic())
			assert.False(t, sStatic.IsInferred())
			assert.Equal(t, 128, sStatic.Static())
			assert.Nil(t, sStatic.Dynamic())
			assert.Equal(t, "", sStatic.AxisName())
			assert.Equal(t, "StaticDim(128)", sStatic.String())

			sNamedStatic := sStatic.WithName("channels")
			assert.Equal(t, "channels", sNamedStatic.AxisName())
			assert.Equal(t, "StaticDim(\"channels\", 128)", sNamedStatic.String())

			sInferred := InferredDim()
			assert.False(t, sInferred.IsStatic())
			assert.False(t, sInferred.IsDynamic())
			assert.True(t, sInferred.IsInferred())
			assert.Equal(t, shapes.DynamicDim, sInferred.Static())
			assert.Equal(t, "InferredDim()", sInferred.String())

			sNamedInferred := NamedInferredDim("features")
			assert.True(t, sNamedInferred.IsInferred())
			assert.Equal(t, "features", sNamedInferred.AxisName())
			assert.Equal(t, "InferredDim(\"features\")", sNamedInferred.String())

			g := NewGraph(backend, "test_spec")
			dummyNode := Const(g, int32(42))
			sDynamic := DynamicDim(dummyNode)
			assert.False(t, sDynamic.IsStatic())
			assert.True(t, sDynamic.IsDynamic())
			assert.False(t, sDynamic.IsInferred())
			assert.Equal(t, dummyNode, sDynamic.Dynamic())
			assert.Equal(t, shapes.DynamicDim, sDynamic.Static())

			sNamedDynamic := NamedDynamicDim("batch", dummyNode)
			assert.True(t, sNamedDynamic.IsDynamic())
			assert.Equal(t, "batch", sNamedDynamic.AxisName())
			assert.Contains(t, sNamedDynamic.String(), "DynamicDim(\"batch\"")
		})
	})
}


