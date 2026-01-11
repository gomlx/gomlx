// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

func TestDotGeneral_LargeShapesAndCopy(t *testing.T) {
	if _, ok := backend.(*Backend); !ok {
		fmt.Printf("Skipping test because backend is not a SimpleGo Backend\n")
	}

	// Test #1: batch axes are out-of-order.
	{
		dtype := dtypes.Float64
		sourceShape := shapes.Make(dtype, 2, 1, 3)
		contractingAxes := []int{1}
		batchAxes := []int{2, 0}
		batchSize, crossSize, contractingSize, crossDims := dgFindSizes(sourceShape, contractingAxes, batchAxes)
		require.Equal(t, 6, batchSize)
		require.Equal(t, 1, crossSize)
		require.Equal(t, 1, contractingSize)
		require.Len(t, crossDims, 0)

		// Create the source buffer.
		sourceAny, sourceFlatAny, err := backend.NewSharedBuffer(0, sourceShape)
		require.NoError(t, err)
		source := sourceAny.(*Buffer)
		sourceFlat := sourceFlatAny.([]float64)
		for i := range sourceFlat {
			sourceFlat[i] = float64(i + 1)
		}

		// Create a block shape.
		blockLog2Dim := 1 // block dim is 2^1 = 2.
		blockDim := 1 << blockLog2Dim
		be := backend.(*Backend)
		outShape := dgCreateBlockedShape(dtype, batchSize, crossSize, contractingSize, blockLog2Dim)
		// outShape = [6 1 1 2 2]
		fmt.Printf("\toutShape=%s, size=%d\n", outShape, outShape.Size())
		require.Equal(
			t,
			[]int{
				batchSize,
				(crossSize + blockDim - 1) / blockDim,
				(contractingSize + blockDim - 1) / blockDim,
				blockDim,
				blockDim,
			},
			outShape.Dimensions,
		)
		outBlocks := be.getBuffer(dtype, outShape.Size())
		outBlocks.shape = outShape
		outBlocks.Zeros()
		copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
		copyFlatToBlock(
			source,
			outBlocks,
			contractingAxes,
			batchAxes,
			batchSize,
			crossSize,
			contractingSize,
			blockLog2Dim,
		)

		outFlat := outBlocks.flat.([]float64)
		// Notice the reversal (transposition) of the batch axes:
		want := []float64{
			1, 0, 0, 0,
			4, 0, 0, 0,

			2, 0, 0, 0,
			5, 0, 0, 0,

			3, 0, 0, 0,
			6, 0, 0, 0,
		}
		require.Equal(t, want, outFlat)
	}

	{ // Test #2
		dtype := dtypes.Float32
		sourceShape := shapes.Make(dtype, 2, 3, 4, 5)
		contractingAxes := []int{1, 2}
		batchAxes := []int{0}
		batchSize, crossSize, contractingSize, crossDims := dgFindSizes(sourceShape, contractingAxes, batchAxes)
		require.Equal(t, 2, batchSize)
		require.Equal(t, 5, crossSize)
		require.Equal(t, 12, contractingSize)
		require.Equal(t, []int{5}, crossDims)

		// Create the source buffer.
		sourceAny, sourceFlatAny, err := backend.NewSharedBuffer(0, sourceShape)
		require.NoError(t, err)
		source := sourceAny.(*Buffer)
		sourceFlat := sourceFlatAny.([]float32)
		for i := range sourceFlat {
			sourceFlat[i] = float32(i + 1)
		}

		// Create a block shape.
		blockLog2Dim := 2 // block dim is 2^2 = 4.
		blockDim := 1 << blockLog2Dim
		be := backend.(*Backend)
		outShape := dgCreateBlockedShape(dtype, batchSize, crossSize, contractingSize, blockLog2Dim)
		// outShape = [2 2 3 4 4]
		fmt.Printf("\toutShape=%s, size=%d\n", outShape, outShape.Size())
		require.Equal(
			t,
			[]int{
				batchSize,
				(crossSize + blockDim - 1) / blockDim,
				(contractingSize + blockDim - 1) / blockDim,
				blockDim,
				blockDim,
			},
			outShape.Dimensions,
		)
		outBlocks := be.getBuffer(dtype, outShape.Size())
		outBlocks.shape = outShape
		outBlocks.Zeros()
		copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
		copyFlatToBlock(
			source,
			outBlocks,
			contractingAxes,
			batchAxes,
			batchSize,
			crossSize,
			contractingSize,
			blockLog2Dim,
		)

		outFlat := outBlocks.flat.([]float32)
		want := []float32{
			1, 6, 11, 16, // Row 0 of block 0: sourceIdx are {0, 0, [0-3], 0}
			2, 7, 12, 17, // Row 1 of block 0: sourceIdx are {0, 0, [0-3], 1}
			3, 8, 13, 18, 4, 9, 14, 19, // Rows 2 and 3 of block 0

			// Block 1: sourceIdx are {0, 1, [0-3], [0-3]}
			21, 26, 31, 36, 22, 27, 32, 37, 23, 28, 33, 38, 24, 29, 34, 39,

			// Block 2: sourceIdx are {0, 2, [0-3], [0-3]}
			41, 46, 51, 56, 42, 47, 52, 57, 43, 48, 53, 58, 44, 49, 54, 59,

			// Block 4: sourceIdx for row 0 are {0, 0, [0-3], 4}, and the rest is padding.
			5, 10, 15, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,

			// ...
			25, 30, 35, 40, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45, 50, 55, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61, 66, 71, 76, 62, 67, 72, 77, 63, 68, 73, 78, 64, 69, 74, 79, 81,
			86, 91, 96, 82, 87, 92, 97, 83, 88, 93, 98, 84, 89, 94, 99, 101, 106, 111, 116, 102, 107, 112, 117, 103, 108, 113, 118, 104, 109, 114, 119, 65, 70, 75, 80,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85, 90, 95, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 110, 115, 120, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		}
		require.Equal(t, want, outFlat)
	}
}

func TestDotGeneral_SmallNormalize(t *testing.T) {
	if _, ok := backend.(*Backend); !ok {
		fmt.Printf("Skipping test because backend is not a SimpleGo Backend\n")
	}

	// Test #1: batch axes are out-of-order.
	{
		dtype := dtypes.Float64
		sourceShape := shapes.Make(dtype, 2, 1, 3)
		contractingAxes := []int{1}
		batchAxes := []int{2, 0}
		batchSize, crossSize, contractingSize, crossDims := dgFindSizes(sourceShape, contractingAxes, batchAxes)
		require.Equal(t, 6, batchSize)
		require.Equal(t, 1, crossSize)
		require.Equal(t, 1, contractingSize)
		require.Len(t, crossDims, 0)

		// Create the source buffer.
		sourceIf, sourceFlatAny, err := backend.NewSharedBuffer(0, sourceShape)
		require.NoError(t, err)
		source := sourceIf.(*Buffer)
		sourceFlat := sourceFlatAny.([]float64)
		for i := range sourceFlat {
			sourceFlat[i] = float64(i + 1)
		}
		normalizeFn := dotGeneralNormalizeShapeDTypeMap.Get(dtype).(func(backend *Backend, source *Buffer, info *dgNormalizationInfo, batchSize, crossSize, contractingSize int) *Buffer)
		info := dgNormalizePrepare(source.shape, contractingAxes, batchAxes)
		output := normalizeFn(
			backend.(*Backend),
			source,
			info,
			batchSize,
			crossSize,
			contractingSize,
		)
		require.NotNil(t, output)
		require.NoError(t, output.shape.Check(dtype, batchSize, crossSize, contractingSize))
		require.Equal(t, []float64{1, 4, 2, 5, 3, 6}, output.flat.([]float64))
	}

	{ // Test #2: cross/contracting axes are inverted.
		dtype := dtypes.Float32
		sourceShape := shapes.Make(dtype, 2, 3, 4, 5)
		contractingAxes := []int{1, 2}
		batchAxes := []int{0}
		batchSize, crossSize, contractingSize, crossDims := dgFindSizes(sourceShape, contractingAxes, batchAxes)
		require.Equal(t, 2, batchSize)
		require.Equal(t, 5, crossSize)
		require.Equal(t, 12, contractingSize)
		require.Equal(t, []int{5}, crossDims)

		// Create the source buffer.
		sourceIf, sourceFlatAny, err := backend.NewSharedBuffer(0, sourceShape)
		require.NoError(t, err)
		source := sourceIf.(*Buffer)
		sourceFlat := sourceFlatAny.([]float32)
		for i := range sourceFlat {
			sourceFlat[i] = float32(i + 1)
		}
		normalizeFn := dotGeneralNormalizeShapeDTypeMap.Get(dtype).(func(backend *Backend, source *Buffer, info *dgNormalizationInfo, batchSize, crossSize, contractingSize int) *Buffer)
		info := dgNormalizePrepare(source.shape, contractingAxes, batchAxes)
		output := normalizeFn(
			backend.(*Backend),
			source,
			info,
			batchSize,
			crossSize,
			contractingSize,
		)
		require.NotNil(t, output)
		require.NoError(t, output.shape.Check(dtype, batchSize, crossSize, contractingSize))

		want := []float32{
			// Batch example 1:
			1, 6, 11, 16, 21, 26, 31, 36, 41, 46, 51, 56,
			2, 7, 12, 17, 22, 27, 32, 37, 42, 47, 52, 57,
			3, 8, 13, 18, 23, 28, 33, 38, 43, 48, 53, 58,
			4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59,
			5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60,

			// Batch example 2:
			61, 66, 71, 76, 81, 86, 91, 96, 101, 106, 111, 116,
			62, 67, 72, 77, 82, 87, 92, 97, 102, 107, 112, 117,
			63, 68, 73, 78, 83, 88, 93, 98, 103, 108, 113, 118,
			64, 69, 74, 79, 84, 89, 94, 99, 104, 109, 114, 119,
			65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 120,
		}
		require.Equal(t, want, output.flat.([]float32))
	}

	{ // Test #3: order preserved. There should be no transposition, and the output should be nil.
		dtype := dtypes.Float64
		sourceShape := shapes.Make(dtype, 2, 3, 4, 5)
		contractingAxes := []int{2, 3}
		batchAxes := []int{0}
		batchSize, crossSize, contractingSize, _ := dgFindSizes(sourceShape, contractingAxes, batchAxes)
		require.Equal(t, 2, batchSize)
		require.Equal(t, 3, crossSize)
		require.Equal(t, 20, contractingSize)
		sourceIf, _, err := backend.NewSharedBuffer(0, sourceShape)
		require.NoError(t, err)
		source := sourceIf.(*Buffer)
		normalizeFn := dotGeneralNormalizeShapeDTypeMap.Get(dtype).(func(backend *Backend, source *Buffer, info *dgNormalizationInfo, batchSize, crossSize, contractingSize int) *Buffer)
		info := dgNormalizePrepare(source.shape, contractingAxes, batchAxes)
		output := normalizeFn(
			backend.(*Backend),
			source,
			info,
			batchSize,
			crossSize,
			contractingSize,
		)
		require.Nil(t, output)

		// If we invert the contracting axes, we need the transposition, and normalizeFn must handle it.
		contractingAxes = []int{3, 2}
		info = dgNormalizePrepare(source.shape, contractingAxes, batchAxes)
		output = normalizeFn(
			backend.(*Backend),
			source,
			info,
			batchSize,
			crossSize,
			contractingSize,
		)
		require.NotNil(t, output)
		require.NoError(t, output.shape.Check(dtype, batchSize, crossSize, contractingSize))
	}
}

func TestDotGeneral_Shape(t *testing.T) {
	S := shapes.Make
	F32 := dtypes.Float32
	builder := backend.Builder("DotGeneral Test").(*Builder)
	mainFn := builder.Main().(*Function)
	lhs, err := mainFn.Parameter("lhs", S(F32, 2, 3, 4, 5), nil)
	require.NoError(t, err)
	rhs, err := mainFn.Parameter("rhs", S(F32, 5, 1, 2, 3), nil)
	require.NoError(t, err)
	gotOp, err := mainFn.DotGeneral(
		lhs, []int{1}, []int{3, 0},
		rhs, []int{3}, []int{0, 2},
	)
	require.NoError(t, err)
	got := gotOp.(*Node)
	// Batch dims: 5 , 2
	// Contracting dims: 3
	// Cross dims: 4 (lhs) and 1 (rhs)
	fmt.Printf("\tdotgeneral.shape=%s\n", got.shape)
	assert.NoError(t, got.shape.Check(F32, 5, 2, 4, 1))
}

func requireSameTensorsFloat32(t *testing.T, want, got *tensors.Tensor, delta float64) {
	// Make sure shapes are the same.
	require.True(t, got.Shape().Equal(want.Shape()))
	flatIdx := 0
	gotFlat := tensors.MustCopyFlatData[float32](got)
	wantFlat := tensors.MustCopyFlatData[float32](want)
	var mismatches int
	for indices := range got.Shape().Iter() {
		gotValue := gotFlat[flatIdx]
		wantValue := wantFlat[flatIdx]
		if math.Abs(float64(gotValue)-float64(wantValue)) > delta {
			if mismatches < 3 {
				fmt.Printf(
					"\tIndex %v (flatIdx=%d) has a mismatch: got %f, want %f\n",
					indices,
					flatIdx,
					gotValue,
					wantValue,
				)
			} else if mismatches == 4 {
				fmt.Printf("\t...\n")
			}
			mismatches++
		}
		flatIdx++
	}
	if mismatches > 0 {
		t.Fatalf("Found %d mismatches in tensors", mismatches)
	}
}

func TestDotGeneral_Exec(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		fmt.Printf("Skipping %s, it is meant only for the Go backend, instead backend is ", backend.Name())
		t.SkipNow()
		return
	}

	// Reset dotGeneralForceExecutionPath at exit to default (auto-select).
	defer func() {
		goBackend.dotGeneralForceExecutionPath = autoSelectPath
	}()

	for _, execPath := range []dotGeneralExecutionPath{normalizedPath, blockedPath, smallMatMulPath, checkPath} {
		// Force a specific execution path: so we exercise the corresponding algorithm irrespective of the actual size:
		// it may not be efficient for the size, but it should be correct in all sizes.
		goBackend.dotGeneralForceExecutionPath = execPath
		t.Run(execPath.String(), func(t *testing.T) {
			t.Run("Float32", func(t *testing.T) {
				// Larger example, with multiple axes.
				y0 := graph.MustExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
					return graph.DotGeneral(lhs, []int{1}, []int{3, 0}, rhs, []int{1}, []int{0, 2})
				},
					tensors.FromFlatDataAndDimensions(xslices.Iota(float32(1), 2*3*1*5), 2, 3, 1, 5),
					tensors.FromFlatDataAndDimensions(xslices.Iota(float32(1), 5*3*2*4), 5, 3, 2, 4),
				)
				fmt.Printf("\ty0=%s\n", y0)
				want := [][][][]float32{
					{
						{{242, 260, 278, 296}},
						{{899, 962, 1025, 1088}},
					}, {
						{{773, 794, 815, 836}},
						{{2522, 2588, 2654, 2720}},
					}, {
						{{1448, 1472, 1496, 1520}},
						{{4289, 4358, 4427, 4496}},
					}, {
						{{2267, 2294, 2321, 2348}},
						{{6200, 6272, 6344, 6416}},
					}, {
						{{3230, 3260, 3290, 3320}},
						{{8255, 8330, 8405, 8480}},
					}}
				require.Equal(t, want, y0.Value())
			})

			// Axis transposition example:
			t.Run("AxisTransposition", func(t *testing.T) {
				y1 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
					lhs := graph.MulScalar(graph.OnePlus(graph.IotaFull(g, shapes.Make(F32, 2, 1, 3))), 1)
					rhs := graph.Ones(g, shapes.Make(F32, 1, 3, 2))
					return graph.DotGeneral(lhs, []int{1}, []int{2, 0}, rhs, []int{0}, []int{1, 2})
				})
				fmt.Printf("\ty1=%s\n", y1)
				require.NoError(t, y1.Shape().Check(F32, 3, 2))
				want1 := [][]float32{{1, 4}, {2, 5}, {3, 6}}
				require.Equal(t, want1, y1.Value())
			})

			// A very large example: expected value computed using XLA.
			t.Run("VeryLarge", func(t *testing.T) {
				y3 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
					lhs := graph.MulScalar(graph.OnePlus(graph.IotaFull(g, shapes.Make(dtypes.F64, 16, 13, 384))), 1e-5)
					rhs := graph.Ones(g, shapes.Make(dtypes.F64, 384, 1536))
					out := graph.DotGeneral(
						lhs, []int{2}, nil,
						rhs, []int{0}, nil)
					return graph.Gather(out, graph.Const(g, [][]int32{{0, 0, 0}}))
				})
				fmt.Printf("\ty3=%s\n", y3)
				require.InDelta(t, 0.7392, tensors.MustCopyFlatData[float64](y3)[0], 1e-4)
			})

			// BFloat16 example.
			t.Run("BFloat16", func(t *testing.T) {
				bf16 := bfloat16.FromFloat32
				y2 := graph.MustExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
					return graph.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
				},
					[][]bfloat16.BFloat16{{bf16(1), bf16(2), bf16(3)}},
					[][]bfloat16.BFloat16{{bf16(10)}, {bf16(11)}, {bf16(12)}},
				)
				fmt.Printf("\ty2=%s\n", y2)
				require.NoError(t, y2.Shape().Check(dtypes.BFloat16, 1, 1))
				require.Equal(t, float32(10+22+36), tensors.MustCopyFlatData[bfloat16.BFloat16](y2)[0].Float32())
			})

			// Do not run the larger tests if running -test.short: they will break Github
			// tests:
			if testing.Short() {
				fmt.Printf("\tSkipping larger tests for %s in -short mode\n", execPath)
				return
			}

			// From DotGeneral parameters taken from LLM models that not working during development:
			t.Run("LLM_1-parallel-requests", func(t *testing.T) {
				lhs, err := tensors.Load("dotgeneral_test_lhs.bin")
				require.NoError(t, err)
				rhs, err := tensors.Load("dotgeneral_test_rhs.bin")
				require.NoError(t, err)
				want, err := tensors.Load("dotgeneral_test_out.bin")
				require.NoError(t, err)
				fmt.Printf("\tlhs=%s, rhs=%s\n", lhs.Shape(), rhs.Shape())
				exec := graph.MustNewExec(backend, func(lhs, rhs *graph.Node) *graph.Node {
					return graph.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{2}, []int{0})
				})
				got := exec.MustExec(lhs, rhs)[0]
				requireSameTensorsFloat32(t, want, got, 1e-3)
				fmt.Printf("\tgot=%s\n", got.Shape())
				fmt.Printf("\twant=%s\n", want.Shape())

				// Run 8 workers in parallel to see if concurrency is a problem:
				var wg sync.WaitGroup
				var numCalls atomic.Uint32
				for runnerIdx := range 16 {
					wg.Add(1)
					go func(_ int) {
						defer wg.Done()
						const numRepeats = 1000
						for range numRepeats {
							got := exec.MustExec(lhs, rhs)[0]
							numCalls.Add(1)
							requireSameTensorsFloat32(t, want, got, 1e-3)
						}
					}(runnerIdx)
				}
				wg.Wait()
				n := numCalls.Load()
				fmt.Printf("\tnumCalls=%d\n", n)
			})
			t.Run("LLM_2", func(t *testing.T) {
				lhs, err := tensors.Load("dotgeneral_test_lhs_2.bin")
				require.NoError(t, err)
				rhs, err := tensors.Load("dotgeneral_test_rhs_2.bin")
				require.NoError(t, err)
				want, err := tensors.Load("dotgeneral_test_out_2.bin")
				require.NoError(t, err)
				fmt.Printf("\tlhs=%s, rhs=%s\n", lhs.Shape(), rhs.Shape())
				got := graph.MustExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
					return graph.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{2}, []int{0})
				}, lhs, rhs)
				fmt.Printf("\tgot=%s\n", got.Shape())
				fmt.Printf("\twant=%s\n", want.Shape())
				requireSameTensorsFloat32(t, want, got, 1e-3)
			})
			t.Run("LLM_2_bfloat16", func(t *testing.T) {
				lhs, err := tensors.Load("dotgeneral_test_lhs_2.bin")
				require.NoError(t, err)
				rhs, err := tensors.Load("dotgeneral_test_rhs_2.bin")
				require.NoError(t, err)
				want, err := tensors.Load("dotgeneral_test_out_2.bin")
				require.NoError(t, err)
				fmt.Printf("\tlhs=%s, rhs=%s\n", lhs.Shape(), rhs.Shape())
				got := graph.MustExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
					lhs = graph.ConvertDType(lhs, dtypes.BFloat16)
					rhs = graph.ConvertDType(rhs, dtypes.BFloat16)
					output := graph.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{2}, []int{0})
					return graph.ConvertDType(output, dtypes.F32)
				}, lhs, rhs)
				fmt.Printf("\tgot=%s\n", got.Shape())
				fmt.Printf("\twant=%s\n", want.Shape())
				// Much larger delta, since BFloat16 loses precision.
				requireSameTensorsFloat32(t, want, got, 1e-1)
			})
		})
	}
}

func TestDotGeneral_Dot(t *testing.T) {
	exec := graph.MustNewExec(backend, graph.Dot)

	y0 := exec.MustExec([]float32{1, 2, 3}, []float32{10, 11, 12})[0]
	fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, float32(10+22+36), y0.Value())

	y1 := exec.MustExec([][]float32{{1, 2, 3}, {2, 4, 6}}, []float32{10, 11, 12})[0]
	fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.Equal(t, []float32{10 + 22 + 36, 20 + 44 + 72}, y1.Value())

	y2 := exec.MustExec([][]float32{{1, 2, 3}, {2, 4, 6}}, [][]float32{{10}, {11}, {12}})[0]
	fmt.Printf("\ty2=%s\n", y2.GoStr())
	assert.Equal(t, [][]float32{{10 + 22 + 36}, {20 + 44 + 72}}, y2.Value())
}

// TestBlockForDotGeneral_Deduplication tests that the same weight matrix
// is only blocked once when used in multiple DotGeneral operations.
func TestBlockForDotGeneral_Deduplication(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		t.Skip("Test requires SimpleGo backend")
	}

	builder := goBackend.Builder("TestDeduplication").(*Builder)
	mainFn := builder.Main().(*Function)

	// Create a parameter node (simulating weights)
	K, N := 128, 256
	weightsShape := shapes.Make(dtypes.Float32, K, N) // [K, N]
	weights, err := mainFn.Parameter("weights", weightsShape, nil)
	require.NoError(t, err)
	weightsNode := weights.(*Node)

	// Get blocked input twice - should return the same node due to deduplication
	// Using blockForDotGeneral with explicit parameters for a 2D weight matrix
	blocked1 := builder.blockForDotGeneral(weightsNode, []int{0}, []int{}, 1, N, K)
	blocked2 := builder.blockForDotGeneral(weightsNode, []int{0}, []int{}, 1, N, K)

	// Should be the exact same node (pointer equality)
	assert.Same(t, blocked1, blocked2, "Deduplication should return the same blocked node")

	// Verify the blocked shape is correct
	blockDim := 1 << DotGeneralTargetBlockLog2Dim[dtypes.Float32]
	expectedCrossBlocks := (N + blockDim - 1) / blockDim
	expectedContractBlocks := (K + blockDim - 1) / blockDim
	assert.Equal(t, []int{1, expectedCrossBlocks, expectedContractBlocks, blockDim, blockDim},
		blocked1.shape.Dimensions)

	builder.Finalize()
}

// TestBlockForDotGeneral_Execution tests that the BlockForDotGeneral operation
// correctly converts a flat tensor to blocked format.
func TestBlockForDotGeneral_Execution(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		t.Skip("Test requires SimpleGo backend")
	}

	// Use a small block size for testing
	// Create a simple 2D tensor [4, 4] with known values
	K, N := 4, 4
	dtype := dtypes.Float32

	// Create source buffer
	sourceShape := shapes.Make(dtype, K, N)
	sourceAny, sourceFlatAny, err := goBackend.NewSharedBuffer(0, sourceShape)
	require.NoError(t, err)
	source := sourceAny.(*Buffer)
	sourceFlat := sourceFlatAny.([]float32)

	// Fill with sequential values: 1, 2, 3, ..., 16
	for i := range sourceFlat {
		sourceFlat[i] = float32(i + 1)
	}

	// Create block data (simulating what blockRHSForDotGeneral would create)
	blockLog2Dim := 2 // Block dim = 4
	blockDim := 1 << blockLog2Dim
	blockedShape := dgCreateBlockedShape(dtype, 1, N, K, blockLog2Dim)

	data := &blockForDotGeneralData{
		blockLog2Dim:    blockLog2Dim,
		blockedShape:    blockedShape,
		batchSize:       1,
		crossSize:       N,
		contractingSize: K,
		contractingAxes: []int{0},
		batchAxes:       []int{},
	}

	// Create a mock node
	node := &Node{
		shape: blockedShape,
		data:  data,
	}

	// Execute the blocking operation
	output, err := execBlockForDotGeneral(goBackend, node, []*Buffer{source}, nil)
	require.NoError(t, err)

	// Verify output shape
	assert.Equal(t, blockedShape, output.shape)

	// Verify output has correct size
	expectedSize := 1 * 1 * 1 * blockDim * blockDim // [1, 1, 1, 4, 4]
	assert.Equal(t, expectedSize, len(output.flat.([]float32)))

	// The blocked output should preserve all the values (just reorganized)
	outputFlat := output.flat.([]float32)
	inputSum := float32(0)
	for _, v := range sourceFlat {
		inputSum += v
	}
	outputSum := float32(0)
	for _, v := range outputFlat {
		outputSum += v
	}
	assert.Equal(t, inputSum, outputSum, "Sum of values should be preserved after blocking")
}

// TestDotGeneral_PreBlockedCorrectness tests that DotGeneral with pre-blocked
// weights produces the same results as without pre-blocking.
func TestDotGeneral_PreBlockedCorrectness(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		t.Skip("Test requires SimpleGo backend")
	}

	// Test with matrices large enough to trigger pre-blocking
	// but small enough to run quickly
	M, K, N := 32, 128, 64

	// Create input tensors
	lhsData := make([]float32, M*K)
	rhsData := make([]float32, K*N)
	for i := range lhsData {
		lhsData[i] = float32(i%100) * 0.01
	}
	for i := range rhsData {
		rhsData[i] = float32(i%100) * 0.01
	}

	lhs := tensors.FromFlatDataAndDimensions(lhsData, M, K)
	rhs := tensors.FromFlatDataAndDimensions(rhsData, K, N)

	// First, compute with normalized path (no pre-blocking)
	goBackend.dotGeneralForceExecutionPath = normalizedPath
	wantResult := graph.MustExecOnce(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
		return graph.DotGeneral(lhs, []int{1}, nil, rhs, []int{0}, nil)
	}, lhs, rhs)

	// Now compute with blocked path (which may use pre-blocking for constant RHS)
	goBackend.dotGeneralForceExecutionPath = blockedPath
	gotResult := graph.MustExecOnce(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
		return graph.DotGeneral(lhs, []int{1}, nil, rhs, []int{0}, nil)
	}, lhs, rhs)

	// Reset to default (auto-select)
	goBackend.dotGeneralForceExecutionPath = autoSelectPath

	// Compare results
	require.True(t, gotResult.Shape().Equal(wantResult.Shape()))
	requireSameTensorsFloat32(t, wantResult, gotResult, 1e-4)
}

// TestBlockForDotGeneralData_Equal tests the Equal method for deduplication.
func TestBlockForDotGeneralData_Equal(t *testing.T) {
	base := &blockForDotGeneralData{
		blockLog2Dim:    5,
		blockedShape:    shapes.Make(dtypes.Float32, 1, 4, 4, 32, 32),
		batchSize:       1,
		crossSize:       128,
		contractingSize: 128,
		contractingAxes: []int{0},
		batchAxes:       []int{},
	}

	tests := []struct {
		name  string
		other *blockForDotGeneralData
		want  bool
	}{
		{
			name: "Identical",
			other: &blockForDotGeneralData{
				blockLog2Dim:    5,
				blockedShape:    shapes.Make(dtypes.Float32, 1, 4, 4, 32, 32),
				batchSize:       1,
				crossSize:       128,
				contractingSize: 128,
				contractingAxes: []int{0},
				batchAxes:       []int{},
			},
			want: true,
		},
		{
			name: "DifferentBlockLog2Dim",
			other: &blockForDotGeneralData{
				blockLog2Dim:    4, // Different
				blockedShape:    shapes.Make(dtypes.Float32, 1, 4, 4, 32, 32),
				batchSize:       1,
				crossSize:       128,
				contractingSize: 128,
				contractingAxes: []int{0},
				batchAxes:       []int{},
			},
			want: false,
		},
		{
			name: "DifferentContractingAxes",
			other: &blockForDotGeneralData{
				blockLog2Dim:    5,
				blockedShape:    shapes.Make(dtypes.Float32, 1, 4, 4, 32, 32),
				batchSize:       1,
				crossSize:       128,
				contractingSize: 128,
				contractingAxes: []int{1}, // Different
				batchAxes:       []int{},
			},
			want: false,
		},
		{
			name: "DifferentBatchAxes",
			other: &blockForDotGeneralData{
				blockLog2Dim:    5,
				blockedShape:    shapes.Make(dtypes.Float32, 1, 4, 4, 32, 32),
				batchSize:       1,
				crossSize:       128,
				contractingSize: 128,
				contractingAxes: []int{0},
				batchAxes:       []int{0}, // Different
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := base.EqualNodeData(tt.other)
			assert.Equal(t, tt.want, got)
		})
	}
}

// TestIsMatMulOrder tests the isMatMulOrder function for various axis configurations.
func TestIsMatMulOrder(t *testing.T) {
	if _, ok := backend.(*Backend); !ok {
		t.Skip("Test requires SimpleGo backend")
	}

	testCases := []struct {
		name               string
		lhsShape           shapes.Shape
		rhsShape           shapes.Shape
		lhsContractingAxes []int
		rhsContractingAxes []int
		lhsBatchAxes       []int
		rhsBatchAxes       []int
		want               bool
	}{
		// Standard 2D matrix multiplication: [M, K] x [K, N]
		{"2D_matmul_standard", shapes.Make(dtypes.Float32, 3, 4), shapes.Make(dtypes.Float32, 4, 5), []int{1}, []int{0}, []int{}, []int{}, true},
		// Transposed LHS: [K, M] x [K, N] - not matmul order
		{"2D_transposed_lhs", shapes.Make(dtypes.Float32, 4, 3), shapes.Make(dtypes.Float32, 4, 5), []int{0}, []int{0}, []int{}, []int{}, false},
		// Transposed RHS: [M, K] x [N, K] - not matmul order
		{"2D_transposed_rhs", shapes.Make(dtypes.Float32, 3, 4), shapes.Make(dtypes.Float32, 5, 4), []int{1}, []int{1}, []int{}, []int{}, false},
		// Matrix x Vector: [M, K] x [K]
		{"matrix_vector", shapes.Make(dtypes.Float32, 3, 4), shapes.Make(dtypes.Float32, 4), []int{1}, []int{0}, []int{}, []int{}, true},
		// Batched matrix multiplication: [B, M, K] x [B, K, N]
		{"batched_matmul", shapes.Make(dtypes.Float32, 2, 3, 4), shapes.Make(dtypes.Float32, 2, 4, 5), []int{2}, []int{1}, []int{0}, []int{0}, true},
		// Multiple contracting axes - not supported by SmallMatMul
		{"multiple_contracting", shapes.Make(dtypes.Float32, 2, 3, 4), shapes.Make(dtypes.Float32, 3, 4, 5), []int{1, 2}, []int{0, 1}, []int{}, []int{}, false},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := isMatMulOrder(tc.lhsShape, tc.rhsShape,
				tc.lhsContractingAxes, tc.rhsContractingAxes,
				tc.lhsBatchAxes, tc.rhsBatchAxes)
			assert.Equal(t, tc.want, got)
		})
	}
}

// TestDgUseSmallMatMul tests the build-time SmallMatMul path selection.
func TestDgUseSmallMatMul(t *testing.T) {
	t.Run("ThresholdBoundaries", func(t *testing.T) {
		testCases := []struct {
			name            string
			batchSize       int
			lhsCrossSize    int
			rhsCrossSize    int
			contractingSize int
			want            bool
		}{
			// At contracting threshold (128)
			{"contractingSize_at_threshold", 1, 10, 10, 128, true},
			// Over contracting threshold
			{"contractingSize_over_threshold", 1, 10, 10, 129, false},
			// Batch size at threshold (64)
			{"batchSize_at_threshold", 64, 10, 10, 32, true},
			// Batch size over threshold
			{"batchSize_over_threshold", 65, 10, 10, 32, false},
			// M=1 special case - uses higher thresholds for K and N
			{"M_equals_1_moderate_K", 1, 1, 256, 512, true},
			// M=1 with K at M1 threshold (1024) should be accepted
			{"M_equals_1_K_at_M1_threshold", 1, 1, 256, 1024, true},
			// M=1 with K over M1 threshold should be rejected
			{"M_equals_1_K_over_M1_threshold", 1, 1, 256, 1025, false},
			// M=1 with very large K should be rejected
			{"M_equals_1_very_large_K", 1, 1, 256, 2000, false},
			// M=1 with large N should still work (within M1 threshold of 4096)
			{"M_equals_1_large_N", 1, 1, 1000, 256, true},
			// M=1 with very large N should be rejected (over M1 threshold of 4096)
			{"M_equals_1_very_large_N", 1, 1, 5000, 256, false},
			// M=1 with N exactly at M1 threshold (4096) should be accepted
			{"M_equals_1_N_at_M1_threshold", 1, 1, 4096, 256, true},
			// M=1 with N just over M1 threshold should be rejected
			{"M_equals_1_N_over_M1_threshold", 1, 1, 4097, 256, false},
			// M=1 with large batch should be rejected
			{"M_equals_1_large_batch", 100, 1, 256, 512, false},
			// N (rhsCrossSize) at threshold (256)
			{"rhsCrossSize_at_threshold", 1, 10, smallMatMulMaxRhsCrossSize, 64, true},
			// N over threshold
			{"rhsCrossSize_over_threshold", 1, 10, smallMatMulMaxRhsCrossSize + 1, 64, false},
			// Combined thresholds: both K and N at their limits
			{"K_and_N_both_at_threshold", 1, 10, smallMatMulMaxRhsCrossSize, 128, true},
			// Combined thresholds: K at limit, N over
			{"K_at_threshold_N_over", 1, 10, 257, 128, false},
			// Combined thresholds: K over, N at limit
			{"K_over_N_at_threshold", 1, 10, 256, 129, false},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				lhsShape := shapes.Make(dtypes.Float32, tc.batchSize, tc.lhsCrossSize, tc.contractingSize)
				rhsShape := shapes.Make(dtypes.Float32, tc.batchSize, tc.contractingSize, tc.rhsCrossSize)

				params := &dotGeneralNodeData{
					lhsContractingAxes: []int{2},
					lhsBatchAxes:       []int{0},
					rhsContractingAxes: []int{1},
					rhsBatchAxes:       []int{0},
					batchSize:          tc.batchSize,
					lhsCrossSize:       tc.lhsCrossSize,
					rhsCrossSize:       tc.rhsCrossSize,
					contractingSize:    tc.contractingSize,
				}

				got := dgUseSmallMatMul(dtypes.Float32, lhsShape, rhsShape, params)
				assert.Equal(t, tc.want, got,
					"dgCanUseSmallMatMul with batch=%d, M=%d, N=%d, K=%d",
					tc.batchSize, tc.lhsCrossSize, tc.rhsCrossSize, tc.contractingSize)
			})
		}
	})

	t.Run("NonFloat32Rejected", func(t *testing.T) {
		params := &dotGeneralNodeData{
			lhsContractingAxes: []int{1},
			lhsBatchAxes:       []int{},
			rhsContractingAxes: []int{0},
			rhsBatchAxes:       []int{},
			batchSize:          1,
			lhsCrossSize:       4,
			rhsCrossSize:       6,
			contractingSize:    8,
		}

		// Float64 should be rejected
		lhsF64 := shapes.Make(dtypes.Float64, 4, 8)
		rhsF64 := shapes.Make(dtypes.Float64, 8, 6)
		assert.False(t, dgUseSmallMatMul(dtypes.Float64, lhsF64, rhsF64, params),
			"Should not use SmallMatMul for Float64")

		// BFloat16 should also be rejected
		lhsBF16 := shapes.Make(dtypes.BFloat16, 4, 8)
		rhsBF16 := shapes.Make(dtypes.BFloat16, 8, 6)
		assert.False(t, dgUseSmallMatMul(dtypes.BFloat16, lhsBF16, rhsBF16, params),
			"Should not use SmallMatMul for BFloat16")
	})

	t.Run("NonMatMulOrderRejected", func(t *testing.T) {
		// Test with non-standard axis order (not [M,K]×[K,N])
		lhsShape := shapes.Make(dtypes.Float32, 8, 4) // [K, M] instead of [M, K]
		rhsShape := shapes.Make(dtypes.Float32, 8, 6) // [K, N]

		params := &dotGeneralNodeData{
			lhsContractingAxes: []int{0}, // K is first, not last
			lhsBatchAxes:       []int{},
			rhsContractingAxes: []int{0},
			rhsBatchAxes:       []int{},
			batchSize:          1,
			lhsCrossSize:       4,
			rhsCrossSize:       6,
			contractingSize:    8,
		}

		assert.False(t, dgUseSmallMatMul(dtypes.Float32, lhsShape, rhsShape, params),
			"Should not use SmallMatMul with non-matmul axis order")
	})
}

// TestSmallMatMulCorrectness verifies that SmallMatMul produces correct results.
func TestSmallMatMulCorrectness(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		t.Skip("Test requires SimpleGo backend")
	}

	originalForce := goBackend.dotGeneralForceExecutionPath
	defer func() {
		goBackend.dotGeneralForceExecutionPath = originalForce
	}()

	testCases := []struct {
		name     string
		lhsDims  []int
		rhsDims  []int
		lhsContr []int
		lhsBatch []int
		rhsContr []int
		rhsBatch []int
	}{
		{"2D_matmul", []int{4, 8}, []int{8, 6}, []int{1}, []int{}, []int{0}, []int{}},
		{"matrix_vector", []int{4, 8}, []int{8}, []int{1}, []int{}, []int{0}, []int{}},
		{"M_equals_1", []int{1, 64}, []int{64, 32}, []int{1}, []int{}, []int{0}, []int{}},
		{"batched", []int{2, 4, 8}, []int{2, 8, 6}, []int{2}, []int{0}, []int{1}, []int{0}},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// Create test data
			lhsSize := 1
			for _, d := range tc.lhsDims {
				lhsSize *= d
			}
			rhsSize := 1
			for _, d := range tc.rhsDims {
				rhsSize *= d
			}

			lhsData := make([]float32, lhsSize)
			for i := range lhsData {
				lhsData[i] = float32(i+1) * 0.01
			}
			rhsData := make([]float32, rhsSize)
			for i := range rhsData {
				rhsData[i] = float32(i+1) * 0.01
			}

			lhsTensor := tensors.FromFlatDataAndDimensions(lhsData, tc.lhsDims...)
			rhsTensor := tensors.FromFlatDataAndDimensions(rhsData, tc.rhsDims...)

			// Compute with auto-select (may use SmallMatMul)
			goBackend.dotGeneralForceExecutionPath = autoSelectPath
			resultAuto := graph.MustExecOnce(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.DotGeneral(lhs, tc.lhsContr, tc.lhsBatch, rhs, tc.rhsContr, tc.rhsBatch)
			}, lhsTensor, rhsTensor)

			// Compute with forced checkPath (uses normalized path, not SmallMatMul)
			goBackend.dotGeneralForceExecutionPath = checkPath
			resultNormalized := graph.MustExecOnce(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.DotGeneral(lhs, tc.lhsContr, tc.lhsBatch, rhs, tc.rhsContr, tc.rhsBatch)
			}, lhsTensor, rhsTensor)

			// Compare results
			require.True(t, resultAuto.Shape().Equal(resultNormalized.Shape()),
				"Shapes should match")
			requireSameTensorsFloat32(t, resultNormalized, resultAuto, 1e-3)
		})
	}
}
