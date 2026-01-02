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
		normalizeFn := dotGeneralNormalizeShapeDTypeMap.Get(dtype).(func(backend *Backend, source *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize int) *Buffer)
		output := normalizeFn(
			backend.(*Backend),
			source,
			contractingAxes,
			batchAxes,
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
		normalizeFn := dotGeneralNormalizeShapeDTypeMap.Get(dtype).(func(backend *Backend, source *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize int) *Buffer)
		output := normalizeFn(
			backend.(*Backend),
			source,
			contractingAxes,
			batchAxes,
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
		normalizeFn := dotGeneralNormalizeShapeDTypeMap.Get(dtype).(func(backend *Backend, source *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize int) *Buffer)
		output := normalizeFn(
			backend.(*Backend),
			source,
			contractingAxes,
			batchAxes,
			batchSize,
			crossSize,
			contractingSize,
		)
		require.Nil(t, output)

		// If we invert the contracting axes, we need the transposition, and normalizeFn must handle it.
		contractingAxes = []int{3, 2}
		output = normalizeFn(
			backend.(*Backend),
			source,
			contractingAxes,
			batchAxes,
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
	lhs, err := builder.Parameter("lhs", S(F32, 2, 3, 4, 5), nil)
	require.NoError(t, err)
	rhs, err := builder.Parameter("rhs", S(F32, 5, 1, 2, 3), nil)
	require.NoError(t, err)
	gotOp, err := builder.DotGeneral(
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

	// Reset dotGeneralForceProblemSize at exit.
	defer func() {
		goBackend.dotGeneralForceProblemSize = unknownProblemSize
	}()

	for _, problemSize := range []dotGeneralProblemSizeType{smallProblemSize, largeProblemSize, checkProblemSize} {
		// Force a specific problem size: so we exercise the corresponding algorithm irrespective of the actual size:
		// it may not be efficient for the size, but it should be correct in all sizes.
		goBackend.dotGeneralForceProblemSize = problemSize
		var testName string
		switch problemSize {
		case smallProblemSize:
			testName = "DotGeneral_small_version"
		case largeProblemSize:
			testName = "DotGeneral_large_version"
		case checkProblemSize:
			testName = "DotGeneral_check_version"
		default:
			t.Fatalf("Unknown version for problem size: %d", problemSize)
		}
		t.Run(testName, func(t *testing.T) {
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

			// Axis transposition example:
			y1 := graph.MustExecOnce(backend, func(g *graph.Graph) *graph.Node {
				lhs := graph.MulScalar(graph.OnePlus(graph.IotaFull(g, shapes.Make(F32, 2, 1, 3))), 1)
				rhs := graph.Ones(g, shapes.Make(F32, 1, 3, 2))
				return graph.DotGeneral(lhs, []int{1}, []int{2, 0}, rhs, []int{0}, []int{1, 2})
			})
			fmt.Printf("\ty1=%s\n", y1)
			require.NoError(t, y1.Shape().Check(F32, 3, 2))
			want1 := [][]float32{{1, 4}, {2, 5}, {3, 6}}
			require.Equal(t, want1, y1.Value())

			// A very large example: expected value computed using XLA.
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
				fmt.Printf("\tSkipping larger tests for %s in -short mode\n", testName)
				return
			}

			// From DotGeneral parameters taken from LLM models that not working during development:
			t.Run("LLM_1-parallel-requests", func(t *testing.T) {
				lhs, err := tensors.Load("dotgeneral_lhs_test.bin")
				require.NoError(t, err)
				rhs, err := tensors.Load("dotgeneral_rhs_test.bin")
				require.NoError(t, err)
				want, err := tensors.Load("dotgeneral_out_test.bin")
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
				lhs, err := tensors.Load("dotgeneral_lhs_2_test.bin")
				require.NoError(t, err)
				rhs, err := tensors.Load("dotgeneral_rhs_2_test.bin")
				require.NoError(t, err)
				want, err := tensors.Load("dotgeneral_out_2_test.bin")
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
				lhs, err := tensors.Load("dotgeneral_lhs_2_test.bin")
				require.NoError(t, err)
				rhs, err := tensors.Load("dotgeneral_rhs_2_test.bin")
				require.NoError(t, err)
				want, err := tensors.Load("dotgeneral_out_2_test.bin")
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

// TestIsMatMulOrder tests the isMatMulOrder function for various axis configurations.
func TestIsMatMulOrder(t *testing.T) {
	if _, ok := backend.(*Backend); !ok {
		fmt.Printf("Skipping test because backend is not a SimpleGo Backend\n")
		t.SkipNow()
		return
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
		{
			name:               "2D_matmul_standard",
			lhsShape:           shapes.Make(dtypes.Float32, 3, 4),
			rhsShape:           shapes.Make(dtypes.Float32, 4, 5),
			lhsContractingAxes: []int{1},
			rhsContractingAxes: []int{0},
			lhsBatchAxes:       []int{},
			rhsBatchAxes:       []int{},
			want:               true,
		},
		// Transposed LHS: [K, M] x [K, N] - not matmul order
		{
			name:               "2D_transposed_lhs",
			lhsShape:           shapes.Make(dtypes.Float32, 4, 3),
			rhsShape:           shapes.Make(dtypes.Float32, 4, 5),
			lhsContractingAxes: []int{0},
			rhsContractingAxes: []int{0},
			lhsBatchAxes:       []int{},
			rhsBatchAxes:       []int{},
			want:               false,
		},
		// Transposed RHS: [M, K] x [N, K] - not matmul order
		{
			name:               "2D_transposed_rhs",
			lhsShape:           shapes.Make(dtypes.Float32, 3, 4),
			rhsShape:           shapes.Make(dtypes.Float32, 5, 4),
			lhsContractingAxes: []int{1},
			rhsContractingAxes: []int{1},
			lhsBatchAxes:       []int{},
			rhsBatchAxes:       []int{},
			want:               false,
		},
		// Matrix x Vector: [M, K] x [K]
		{
			name:               "matrix_vector",
			lhsShape:           shapes.Make(dtypes.Float32, 3, 4),
			rhsShape:           shapes.Make(dtypes.Float32, 4),
			lhsContractingAxes: []int{1},
			rhsContractingAxes: []int{0},
			lhsBatchAxes:       []int{},
			rhsBatchAxes:       []int{},
			want:               true,
		},
		// Batched matrix multiplication: [B, M, K] x [B, K, N]
		{
			name:               "batched_matmul",
			lhsShape:           shapes.Make(dtypes.Float32, 2, 3, 4),
			rhsShape:           shapes.Make(dtypes.Float32, 2, 4, 5),
			lhsContractingAxes: []int{2},
			rhsContractingAxes: []int{1},
			lhsBatchAxes:       []int{0},
			rhsBatchAxes:       []int{0},
			want:               true,
		},
		// Batched with wrong contracting axis order
		{
			name:               "batched_wrong_contracting",
			lhsShape:           shapes.Make(dtypes.Float32, 2, 4, 3),
			rhsShape:           shapes.Make(dtypes.Float32, 2, 4, 5),
			lhsContractingAxes: []int{1},
			rhsContractingAxes: []int{1},
			lhsBatchAxes:       []int{0},
			rhsBatchAxes:       []int{0},
			want:               false,
		},
		// Multi-batch: [B1, B2, M, K] x [B1, B2, K, N]
		{
			name:               "multi_batch_matmul",
			lhsShape:           shapes.Make(dtypes.Float32, 2, 3, 4, 5),
			rhsShape:           shapes.Make(dtypes.Float32, 2, 3, 5, 6),
			lhsContractingAxes: []int{3},
			rhsContractingAxes: []int{2},
			lhsBatchAxes:       []int{0, 1},
			rhsBatchAxes:       []int{0, 1},
			want:               true,
		},
		// Multi-batch with wrong batch order
		{
			name:               "multi_batch_wrong_order",
			lhsShape:           shapes.Make(dtypes.Float32, 2, 3, 4, 5),
			rhsShape:           shapes.Make(dtypes.Float32, 2, 3, 5, 6),
			lhsContractingAxes: []int{3},
			rhsContractingAxes: []int{2},
			lhsBatchAxes:       []int{1, 0}, // Wrong order
			rhsBatchAxes:       []int{0, 1},
			want:               false,
		},
		// Different ranks - not supported
		{
			name:               "different_ranks",
			lhsShape:           shapes.Make(dtypes.Float32, 2, 3, 4),
			rhsShape:           shapes.Make(dtypes.Float32, 4, 5),
			lhsContractingAxes: []int{2},
			rhsContractingAxes: []int{0},
			lhsBatchAxes:       []int{0},
			rhsBatchAxes:       []int{},
			want:               false,
		},
		// Multiple contracting axes - not supported by SmallMatMul
		{
			name:               "multiple_contracting_axes",
			lhsShape:           shapes.Make(dtypes.Float32, 2, 3, 4),
			rhsShape:           shapes.Make(dtypes.Float32, 3, 4, 5),
			lhsContractingAxes: []int{1, 2},
			rhsContractingAxes: []int{0, 1},
			lhsBatchAxes:       []int{},
			rhsBatchAxes:       []int{},
			want:               false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			got := isMatMulOrder(tc.lhsShape, tc.rhsShape,
				tc.lhsContractingAxes, tc.rhsContractingAxes,
				tc.lhsBatchAxes, tc.rhsBatchAxes)
			assert.Equal(t, tc.want, got,
				"isMatMulOrder(%s, %s, lhsC=%v, rhsC=%v, lhsB=%v, rhsB=%v)",
				tc.lhsShape, tc.rhsShape,
				tc.lhsContractingAxes, tc.rhsContractingAxes,
				tc.lhsBatchAxes, tc.rhsBatchAxes)
		})
	}
}

// TestUseSmallMatMul tests the useSmallMatMul function with various configurations.
func TestUseSmallMatMul(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		fmt.Printf("Skipping test because backend is not a SimpleGo Backend\n")
		t.SkipNow()
		return
	}

	// Save and restore the original force setting
	originalForce := goBackend.dotGeneralForceProblemSize
	defer func() {
		goBackend.dotGeneralForceProblemSize = originalForce
	}()

	t.Run("RespectsForceProblemSize", func(t *testing.T) {
		// Create test buffers in matmul order
		lhsShape := shapes.Make(dtypes.Float32, 4, 8)
		rhsShape := shapes.Make(dtypes.Float32, 8, 6)
		lhsBuf := goBackend.getBufferForShape(lhsShape)
		rhsBuf := goBackend.getBufferForShape(rhsShape)
		defer goBackend.putBuffer(lhsBuf)
		defer goBackend.putBuffer(rhsBuf)

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

		// With unknown (default), should use SmallMatMul
		goBackend.dotGeneralForceProblemSize = unknownProblemSize
		assert.True(t, useSmallMatMul(goBackend, lhsBuf, rhsBuf, params),
			"Should use SmallMatMul with unknownProblemSize")

		// With smallProblemSize forced, should NOT use SmallMatMul
		goBackend.dotGeneralForceProblemSize = smallProblemSize
		assert.False(t, useSmallMatMul(goBackend, lhsBuf, rhsBuf, params),
			"Should not use SmallMatMul with forced smallProblemSize")

		// With largeProblemSize forced, should NOT use SmallMatMul
		goBackend.dotGeneralForceProblemSize = largeProblemSize
		assert.False(t, useSmallMatMul(goBackend, lhsBuf, rhsBuf, params),
			"Should not use SmallMatMul with forced largeProblemSize")

		// Reset for subsequent tests
		goBackend.dotGeneralForceProblemSize = unknownProblemSize
	})

	t.Run("ThresholdBoundaries", func(t *testing.T) {
		goBackend.dotGeneralForceProblemSize = unknownProblemSize

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
			// RHS cross at threshold (256)
			{"rhsCrossSize_at_threshold", 1, 10, 256, 32, true},
			// RHS cross over threshold
			{"rhsCrossSize_over_threshold", 1, 10, 257, 32, false},
			// M=1 special case - should use SmallMatMul even with large contracting
			{"M_equals_1_large_K", 1, 1, 256, 512, true},
			// Very small sizes
			{"very_small", 1, 1, 1, 1, true},
		}

		for _, tc := range testCases {
			t.Run(tc.name, func(t *testing.T) {
				lhsShape := shapes.Make(dtypes.Float32, tc.batchSize, tc.lhsCrossSize, tc.contractingSize)
				rhsShape := shapes.Make(dtypes.Float32, tc.batchSize, tc.contractingSize, tc.rhsCrossSize)
				lhsBuf := goBackend.getBufferForShape(lhsShape)
				rhsBuf := goBackend.getBufferForShape(rhsShape)
				defer goBackend.putBuffer(lhsBuf)
				defer goBackend.putBuffer(rhsBuf)

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

				got := useSmallMatMul(goBackend, lhsBuf, rhsBuf, params)
				assert.Equal(t, tc.want, got,
					"useSmallMatMul with batch=%d, M=%d, N=%d, K=%d",
					tc.batchSize, tc.lhsCrossSize, tc.rhsCrossSize, tc.contractingSize)
			})
		}
	})

	t.Run("NonFloat32Rejected", func(t *testing.T) {
		goBackend.dotGeneralForceProblemSize = unknownProblemSize

		// Float64 should be rejected
		lhsShape := shapes.Make(dtypes.Float64, 4, 8)
		rhsShape := shapes.Make(dtypes.Float64, 8, 6)
		lhsBuf := goBackend.getBufferForShape(lhsShape)
		rhsBuf := goBackend.getBufferForShape(rhsShape)
		defer goBackend.putBuffer(lhsBuf)
		defer goBackend.putBuffer(rhsBuf)

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

		assert.False(t, useSmallMatMul(goBackend, lhsBuf, rhsBuf, params),
			"Should not use SmallMatMul for Float64")
	})
}

// TestSmallMatMulCorrectness verifies that SmallMatMul produces correct results
// compared to the normalized path.
func TestSmallMatMulCorrectness(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		fmt.Printf("Skipping test because backend is not a SimpleGo Backend\n")
		t.SkipNow()
		return
	}

	originalForce := goBackend.dotGeneralForceProblemSize
	defer func() {
		goBackend.dotGeneralForceProblemSize = originalForce
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
		// Standard 2D matrix multiplication
		{
			name:     "2D_matmul_4x8_8x6",
			lhsDims:  []int{4, 8},
			rhsDims:  []int{8, 6},
			lhsContr: []int{1},
			lhsBatch: []int{},
			rhsContr: []int{0},
			rhsBatch: []int{},
		},
		// Matrix x Vector
		{
			name:     "matrix_vector",
			lhsDims:  []int{4, 8},
			rhsDims:  []int{8},
			lhsContr: []int{1},
			lhsBatch: []int{},
			rhsContr: []int{0},
			rhsBatch: []int{},
		},
		// M=1 case (single row)
		{
			name:     "M_equals_1",
			lhsDims:  []int{1, 64},
			rhsDims:  []int{64, 32},
			lhsContr: []int{1},
			lhsBatch: []int{},
			rhsContr: []int{0},
			rhsBatch: []int{},
		},
		// N=1 case (single column output)
		{
			name:     "N_equals_1",
			lhsDims:  []int{8, 16},
			rhsDims:  []int{16, 1},
			lhsContr: []int{1},
			lhsBatch: []int{},
			rhsContr: []int{0},
			rhsBatch: []int{},
		},
		// Batched
		{
			name:     "batched_2x4x8_2x8x6",
			lhsDims:  []int{2, 4, 8},
			rhsDims:  []int{2, 8, 6},
			lhsContr: []int{2},
			lhsBatch: []int{0},
			rhsContr: []int{1},
			rhsBatch: []int{0},
		},
		// Multi-batch
		{
			name:     "multi_batch",
			lhsDims:  []int{2, 3, 4, 8},
			rhsDims:  []int{2, 3, 8, 6},
			lhsContr: []int{3},
			lhsBatch: []int{0, 1},
			rhsContr: []int{2},
			rhsBatch: []int{0, 1},
		},
		// At contracting threshold
		{
			name:     "at_contracting_threshold",
			lhsDims:  []int{4, 128},
			rhsDims:  []int{128, 8},
			lhsContr: []int{1},
			lhsBatch: []int{},
			rhsContr: []int{0},
			rhsBatch: []int{},
		},
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
			goBackend.dotGeneralForceProblemSize = unknownProblemSize
			resultAuto := graph.MustExecOnce(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.DotGeneral(lhs, tc.lhsContr, tc.lhsBatch, rhs, tc.rhsContr, tc.rhsBatch)
			}, lhsTensor, rhsTensor)

			// Compute with forced normalized path
			goBackend.dotGeneralForceProblemSize = smallProblemSize
			resultNormalized := graph.MustExecOnce(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.DotGeneral(lhs, tc.lhsContr, tc.lhsBatch, rhs, tc.rhsContr, tc.rhsBatch)
			}, lhsTensor, rhsTensor)

			// Compare results
			require.True(t, resultAuto.Shape().Equal(resultNormalized.Shape()),
				"Shapes should match: auto=%s, normalized=%s", resultAuto.Shape(), resultNormalized.Shape())

			// Use 1e-3 delta to account for floating-point differences between paths
			requireSameTensorsFloat32(t, resultNormalized, resultAuto, 1e-3)
		})
	}
}

// TestSmallMatMulM1SpecialCase tests the M=1 special case more thoroughly.
// This case is important for attention score computation in transformers.
func TestSmallMatMulM1SpecialCase(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		fmt.Printf("Skipping test because backend is not a SimpleGo Backend\n")
		t.SkipNow()
		return
	}

	originalForce := goBackend.dotGeneralForceProblemSize
	defer func() {
		goBackend.dotGeneralForceProblemSize = originalForce
	}()

	// Test various contracting sizes for M=1
	contractingSizes := []int{16, 64, 128, 256, 512}
	rhsCrossSizes := []int{16, 64, 128, 256}

	for _, K := range contractingSizes {
		for _, N := range rhsCrossSizes {
			t.Run(fmt.Sprintf("K%d_N%d", K, N), func(t *testing.T) {
				lhsData := make([]float32, K)
				for i := range lhsData {
					lhsData[i] = float32(i+1) * 0.001
				}
				rhsData := make([]float32, K*N)
				for i := range rhsData {
					rhsData[i] = float32(i+1) * 0.001
				}

				lhsTensor := tensors.FromFlatDataAndDimensions(lhsData, 1, K)
				rhsTensor := tensors.FromFlatDataAndDimensions(rhsData, K, N)

				// Auto path
				goBackend.dotGeneralForceProblemSize = unknownProblemSize
				resultAuto := graph.MustExecOnce(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
					return graph.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
				}, lhsTensor, rhsTensor)

				// Normalized path
				goBackend.dotGeneralForceProblemSize = smallProblemSize
				resultNormalized := graph.MustExecOnce(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
					return graph.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
				}, lhsTensor, rhsTensor)

				// Use relative tolerance since larger K values accumulate more floating-point error
				requireSameTensorsFloat32(t, resultNormalized, resultAuto, 1e-2)
			})
		}
	}
}

// TestSmallMatMulBatchVariations tests SmallMatMul with various batch sizes around the threshold.
func TestSmallMatMulBatchVariations(t *testing.T) {
	goBackend, ok := backend.(*Backend)
	if !ok {
		fmt.Printf("Skipping test because backend is not a SimpleGo Backend\n")
		t.SkipNow()
		return
	}

	originalForce := goBackend.dotGeneralForceProblemSize
	defer func() {
		goBackend.dotGeneralForceProblemSize = originalForce
	}()

	// Test batch sizes around the threshold (64)
	batchSizes := []int{1, 16, 32, 64, 65, 128}

	for _, B := range batchSizes {
		t.Run(fmt.Sprintf("batch%d", B), func(t *testing.T) {
			M, K, N := 4, 16, 8
			lhsData := make([]float32, B*M*K)
			for i := range lhsData {
				lhsData[i] = float32(i+1) * 0.001
			}
			rhsData := make([]float32, B*K*N)
			for i := range rhsData {
				rhsData[i] = float32(i+1) * 0.001
			}

			lhsTensor := tensors.FromFlatDataAndDimensions(lhsData, B, M, K)
			rhsTensor := tensors.FromFlatDataAndDimensions(rhsData, B, K, N)

			// Auto path
			goBackend.dotGeneralForceProblemSize = unknownProblemSize
			resultAuto := graph.MustExecOnce(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{1}, []int{0})
			}, lhsTensor, rhsTensor)

			// Normalized path
			goBackend.dotGeneralForceProblemSize = smallProblemSize
			resultNormalized := graph.MustExecOnce(goBackend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{1}, []int{0})
			}, lhsTensor, rhsTensor)

			requireSameTensorsFloat32(t, resultNormalized, resultAuto, 1e-3)
		})
	}
}
