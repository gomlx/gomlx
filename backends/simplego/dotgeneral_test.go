package simplego

import (
	"flag"
	"fmt"
	"math"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/dustin/go-humanize"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/muesli/termenv"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

var flagPerf = flag.Bool("perf", false, "Run performance table tests.")

func formatDurationWith2Decimals(d time.Duration) string {
	s := d.String()
	re := regexp.MustCompile(`(\d+\.?\d*)([µa-z]+)`)
	matches := re.FindStringSubmatch(s)
	if len(matches) != 3 {
		return s
	}
	num, err := strconv.ParseFloat(matches[1], 64)
	if err != nil {
		return s
	}
	return fmt.Sprintf("%.2f%s", num, matches[2])
}

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
		require.Equal(t, []int{batchSize, (crossSize + blockDim - 1) / blockDim, (contractingSize + blockDim - 1) / blockDim, blockDim, blockDim}, outShape.Dimensions)
		outBlocks := be.getBuffer(dtype, outShape.Size())
		outBlocks.shape = outShape
		outBlocks.Zeros()
		copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
		copyFlatToBlock(source, outBlocks, contractingAxes, batchAxes, batchSize, crossSize, contractingSize, blockLog2Dim)

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
		require.Equal(t, []int{batchSize, (crossSize + blockDim - 1) / blockDim, (contractingSize + blockDim - 1) / blockDim, blockDim, blockDim}, outShape.Dimensions)
		outBlocks := be.getBuffer(dtype, outShape.Size())
		outBlocks.shape = outShape
		outBlocks.Zeros()
		copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
		copyFlatToBlock(source, outBlocks, contractingAxes, batchAxes, batchSize, crossSize, contractingSize, blockLog2Dim)

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
		output := normalizeFn(backend.(*Backend), source, contractingAxes, batchAxes, batchSize, crossSize, contractingSize)
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
		output := normalizeFn(backend.(*Backend), source, contractingAxes, batchAxes, batchSize, crossSize, contractingSize)
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
		output := normalizeFn(backend.(*Backend), source, contractingAxes, batchAxes, batchSize, crossSize, contractingSize)
		require.Nil(t, output)

		// If we invert the contracting axes, we need the transposition, and normalizeFn must handle it.
		contractingAxes = []int{3, 2}
		output = normalizeFn(backend.(*Backend), source, contractingAxes, batchAxes, batchSize, crossSize, contractingSize)
		require.NotNil(t, output)
		require.NoError(t, output.shape.Check(dtype, batchSize, crossSize, contractingSize))
	}
}

func TestDotGeneral_Shape(t *testing.T) {
	S := shapes.Make
	F32 := dtypes.Float32
	builder := backend.Builder("DotGeneral Test").(*Builder)
	lhs, err := builder.Parameter("lhs", S(F32, 2, 3, 4, 5))
	require.NoError(t, err)
	rhs, err := builder.Parameter("rhs", S(F32, 5, 1, 2, 3))
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
	gotFlat := tensors.CopyFlatData[float32](got)
	wantFlat := tensors.CopyFlatData[float32](want)
	var mismatches int
	for indices := range got.Shape().Iter() {
		gotValue := gotFlat[flatIdx]
		wantValue := wantFlat[flatIdx]
		if math.Abs(float64(gotValue)-float64(wantValue)) > delta {
			if mismatches < 3 {
				fmt.Printf("\tIndex %v (flatIdx=%d) has a mismatch: got %f, want %f\n", indices, flatIdx, gotValue, wantValue)
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
			y0 := graph.ExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
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
			y1 := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
				lhs := graph.MulScalar(graph.OnePlus(graph.IotaFull(g, shapes.Make(F32, 2, 1, 3))), 1)
				rhs := graph.Ones(g, shapes.Make(F32, 1, 3, 2))
				return graph.DotGeneral(lhs, []int{1}, []int{2, 0}, rhs, []int{0}, []int{1, 2})
			})
			fmt.Printf("\ty1=%s\n", y1)
			require.NoError(t, y1.Shape().Check(F32, 3, 2))
			want1 := [][]float32{{1, 4}, {2, 5}, {3, 6}}
			require.Equal(t, want1, y1.Value())

			// A very large example: expected value computed using XLA.
			y3 := graph.ExecOnce(backend, func(g *graph.Graph) *graph.Node {
				lhs := graph.MulScalar(graph.OnePlus(graph.IotaFull(g, shapes.Make(dtypes.F64, 16, 13, 384))), 1e-5)
				rhs := graph.Ones(g, shapes.Make(dtypes.F64, 384, 1536))
				out := graph.DotGeneral(
					lhs, []int{2}, nil,
					rhs, []int{0}, nil)
				return graph.Gather(out, graph.Const(g, [][]int32{{0, 0, 0}}))
			})
			fmt.Printf("\ty3=%s\n", y3)
			require.InDelta(t, 0.7392, tensors.CopyFlatData[float64](y3)[0], 1e-4)

			// BFloat16 example.
			t.Run("BFloat16", func(t *testing.T) {
				bf16 := bfloat16.FromFloat32
				y2 := graph.ExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
					return graph.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
				},
					[][]bfloat16.BFloat16{{bf16(1), bf16(2), bf16(3)}},
					[][]bfloat16.BFloat16{{bf16(10)}, {bf16(11)}, {bf16(12)}},
				)
				fmt.Printf("\ty2=%s\n", y2)
				require.NoError(t, y2.Shape().Check(dtypes.BFloat16, 1, 1))
				require.Equal(t, float32(10+22+36), tensors.CopyFlatData[bfloat16.BFloat16](y2)[0].Float32())
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
				exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node {
					return graph.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{2}, []int{0})
				})
				got := exec.Call(lhs, rhs)[0]
				requireSameTensorsFloat32(t, want, got, 1e-3)
				fmt.Printf("\tgot=%s\n", got.Shape())
				fmt.Printf("\twant=%s\n", want.Shape())

				// Run 8 workers in parallel, to see if concurrency is a problem:
				var wg sync.WaitGroup
				var numCalls atomic.Uint32
				for runnerIdx := range 16 {
					wg.Add(1)
					go func(idx int) {
						defer wg.Done()
						const numRepeats = 1000
						for range numRepeats {
							got := exec.Call(lhs, rhs)[0]
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
				got := graph.ExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
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
				got := graph.ExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
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
	exec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node {
		return graph.Dot(lhs, rhs)
	})

	y0 := exec.Call([]float32{1, 2, 3}, []float32{10, 11, 12})[0]
	fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, float32(10+22+36), y0.Value())

	y1 := exec.Call([][]float32{{1, 2, 3}, {2, 4, 6}}, []float32{10, 11, 12})[0]
	fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.Equal(t, []float32{10 + 22 + 36, 20 + 44 + 72}, y1.Value())

	y2 := exec.Call([][]float32{{1, 2, 3}, {2, 4, 6}}, [][]float32{{10}, {11}, {12}})[0]
	fmt.Printf("\ty2=%s\n", y2.GoStr())
	assert.Equal(t, [][]float32{{10 + 22 + 36}, {20 + 44 + 72}}, y2.Value())
}

// dotGeneralBenchmarkParamsCase defines input parameters for DotGeneral to be benchmarked.
type dotGeneralBenchmarkParamsCase struct {
	name                                       string
	lhsShape, lhsContractingAxes, lhsBatchAxes []int
	rhsShape, rhsContractingAxes, rhsBatchAxes []int
}

func dimsToStr(dims []int) string {
	dimsStr := xslices.Map(dims, func(i int) string { return strconv.Itoa(i) })
	return fmt.Sprintf("{%s}", strings.Join(dimsStr, ", "))
}

func TestDotGeneral_PerformanceTable(t *testing.T) {
	if !*flagPerf {
		fmt.Printf("Skipping TestDotGeneral_PerformanceTable. Set -perf to run it.\n")
		return
	}
	// IMPORTANT: Populate this slice with the shapes and parameters of the dot-product.
	// lhsDims: [Batch, LhsCross, Contracting]
	// rhsDims: [Batch, RhsCross, Contracting]
	// Batch and Contracting dimensions must match between lhs and rhs.
	benchmarkCases := []dotGeneralBenchmarkParamsCase{
		/*
			{
				name:     "KA-Batch-16-#4",
				lhsShape: []int{16, 13, 384}, lhsContractingAxes: []int{2}, lhsBatchAxes: []int(nil),
				rhsShape: []int{384, 1536}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int(nil),
			},
			{
				name:     "LargeBatch-Medium",
				lhsShape: []int{64, 64, 128}, lhsContractingAxes: []int{2}, lhsBatchAxes: []int{0},
				rhsShape: []int{64, 64, 128}, rhsContractingAxes: []int{2}, rhsBatchAxes: []int{0},
			},
			{
				name:     "NoBatch-Large",
				lhsShape: []int{1536, 1920}, lhsContractingAxes: []int{1}, lhsBatchAxes: nil,
				rhsShape: []int{1920, 1024}, rhsContractingAxes: []int{0}, rhsBatchAxes: nil,
			},
		*/
		{
			name:     "NoBatch-Tiny",
			lhsShape: []int{128, 4}, lhsContractingAxes: []int{1}, lhsBatchAxes: []int{},
			rhsShape: []int{4, 1}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int{},
		},
		{
			name:     "NoBatch-Small",
			lhsShape: []int{16, 128}, lhsContractingAxes: []int{1}, lhsBatchAxes: nil,
			rhsShape: []int{128, 32}, rhsContractingAxes: []int{0}, rhsBatchAxes: nil,
		},
		{
			name:     "NoBatch-Medium",
			lhsShape: []int{128, 128}, lhsContractingAxes: []int{1}, lhsBatchAxes: nil,
			rhsShape: []int{128, 256}, rhsContractingAxes: []int{0}, rhsBatchAxes: nil,
		},
		{
			name:     "LargeBatch-Tiny",
			lhsShape: []int{1024, 128, 4}, lhsContractingAxes: []int{2}, lhsBatchAxes: []int{0},
			rhsShape: []int{1024, 4, 1}, rhsContractingAxes: []int{1}, rhsBatchAxes: []int{0},
		},
		{
			name:     "LargeBatch-Small",
			lhsShape: []int{256, 8, 32}, lhsContractingAxes: []int{2}, lhsBatchAxes: []int{0},
			rhsShape: []int{256, 32, 16}, rhsContractingAxes: []int{1}, rhsBatchAxes: []int{0},
		},
		{
			name:     "LargeBatch-Medium",
			lhsShape: []int{64, 64, 128}, lhsContractingAxes: []int{2}, lhsBatchAxes: []int{0},
			rhsShape: []int{64, 64, 128}, rhsContractingAxes: []int{2}, rhsBatchAxes: []int{0},
		},
		{
			name:     "NoBatch-Large",
			lhsShape: []int{1536, 1920}, lhsContractingAxes: []int{1}, lhsBatchAxes: nil,
			rhsShape: []int{1920, 1024}, rhsContractingAxes: []int{0}, rhsBatchAxes: nil,
		},

		// Shape values taken from the model https://huggingface.co/KnightsAnalytics/all-MiniLM-L6-v2
		// while running the benchmark `TestBenchRobSentencesXLA` from github.com/gomlx/onnx-gomlx/internal/benchmark
		// with batch size 16.
		{
			name:     "KA-Batch-16-#1",
			lhsShape: []int{16, 12, 13, 13}, lhsContractingAxes: []int{3}, lhsBatchAxes: []int{0, 1},
			rhsShape: []int{16, 12, 13, 32}, rhsContractingAxes: []int{2}, rhsBatchAxes: []int{0, 1},
		},
		{
			name:     "KA-Batch-16-#2",
			lhsShape: []int{16, 12, 13, 32}, lhsContractingAxes: []int{3}, lhsBatchAxes: []int{0, 1},
			rhsShape: []int{16, 12, 32, 13}, rhsContractingAxes: []int{2}, rhsBatchAxes: []int{0, 1},
		},
		{
			name:     "KA-Batch-16-#3",
			lhsShape: []int{16, 13, 1536}, lhsContractingAxes: []int{2}, lhsBatchAxes: []int(nil),
			rhsShape: []int{1536, 384}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int(nil),
		},
		{
			name:     "KA-Batch-16-#4",
			lhsShape: []int{16, 13, 384}, lhsContractingAxes: []int{2}, lhsBatchAxes: []int(nil),
			rhsShape: []int{384, 1536}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int(nil),
		},
		{
			// This case happens 4x more often than the other parameters.
			name:     "KA-Batch-16-#5",
			lhsShape: []int{16, 13, 384}, lhsContractingAxes: []int{2}, lhsBatchAxes: []int(nil),
			rhsShape: []int{384, 384}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int(nil),
		},

		// Shape values taken from training github.com/gomlx/gomlx/examples/adult/demo
		{
			name:     "adult-#1",
			lhsShape: []int{128, 4}, lhsContractingAxes: []int{1}, lhsBatchAxes: []int{},
			rhsShape: []int{4, 1}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int{}},
		{
			name:     "adult-#2",
			lhsShape: []int{128, 69}, lhsContractingAxes: []int{1}, lhsBatchAxes: []int{},
			rhsShape: []int{69, 4}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int{},
		},
		{
			name:     "adult-#3",
			lhsShape: []int{25, 4}, lhsContractingAxes: []int{1}, lhsBatchAxes: []int{},
			rhsShape: []int{4, 1}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int{},
		},
		{
			name:     "adult-#4",
			lhsShape: []int{25, 69}, lhsContractingAxes: []int{1}, lhsBatchAxes: []int{},
			rhsShape: []int{69, 4}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int{},
		},
		{
			name:     "adult-#5",
			lhsShape: []int{49, 4}, lhsContractingAxes: []int{1}, lhsBatchAxes: []int{},
			rhsShape: []int{4, 1}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int{},
		},
		{
			name:     "adult-#6",
			lhsShape: []int{49, 69}, lhsContractingAxes: []int{1}, lhsBatchAxes: []int{},
			rhsShape: []int{69, 4}, rhsContractingAxes: []int{0}, rhsBatchAxes: []int{},
		},

		// Add more test cases relevant to your models here
	}

	dtypesToTest := []dtypes.DType{dtypes.Float32, dtypes.Float64, dtypes.BFloat16}

	// Adjust for desired precision vs. test duration
	const numWarmupRuns = 2
	const minNumTimedRuns = 50
	const minTestTime = time.Second

	// Colors: tests usually run in batch and that disallows colors. We temporarily force a different profile:
	originalProfile := lipgloss.ColorProfile()      // Optional: store original
	lipgloss.SetColorProfile(termenv.ANSI256)       // Or termenv.TrueColor if you prefer
	defer lipgloss.SetColorProfile(originalProfile) // Optional: reset
	style1 := lipgloss.NewStyle()
	style2 := lipgloss.NewStyle().Background(lipgloss.ANSIColor(0))

	// Print table header
	fmt.Printf("\n--- execNormalizedDotGeneral Performance ---\n")
	header := fmt.Sprintf("| %-20s | %-20s | %-20s | %-10s | %-10s | %-12s | %-15s | %-10s |", "Test Name", "LHS Dims", "RHS Dims", "DType", "BatchSize", "Time/Run", "Num Ops", "GOps/Sec")
	fmt.Println(header)
	fmt.Println(strings.Repeat("-", len(header)))

	for benchCaseIdx, benchCase := range benchmarkCases {
		for _, dtype := range dtypesToTest {
			// Construct shapes from dimensions and current dtype
			lhsShape := shapes.Make(dtype, benchCase.lhsShape...)
			rhsShape := shapes.Make(dtype, benchCase.rhsShape...)
			batchSize, lhsCrossSize, contractingSize, _ := dgFindSizes(lhsShape, benchCase.lhsContractingAxes, benchCase.lhsBatchAxes)
			_, rhsCrossSize, _, _ := dgFindSizes(rhsShape, benchCase.rhsContractingAxes, benchCase.rhsBatchAxes)
			numOps := batchSize * lhsCrossSize * rhsCrossSize * contractingSize * 2 // 1 mult + 1 add = 2 ops

			// Create and initialize input Buffers
			lhsBuffer, lhsFlatAny, err := backend.NewSharedBuffer(0, lhsShape)
			require.NoError(t, err)
			rhsBuffer, rhsFlatAny, err := backend.NewSharedBuffer(0, rhsShape)
			require.NoError(t, err)
			switch dtype {
			case dtypes.Float32:
				lhsFlatF32 := lhsFlatAny.([]float32)
				rhsFlatF32 := rhsFlatAny.([]float32)
				for i := range lhsFlatF32 {
					lhsFlatF32[i] = float32(i%10 + 1)
				}
				for i := range rhsFlatF32 {
					rhsFlatF32[i] = float32(i%10 + 1)
				}

			case dtypes.Float64:
				lhsFlatF64 := lhsFlatAny.([]float64)
				rhsFlatF64 := rhsFlatAny.([]float64)
				for i := range lhsFlatF64 {
					lhsFlatF64[i] = float64(i%10 + 1)
				}
				for i := range rhsFlatF64 {
					rhsFlatF64[i] = float64(i%10 + 1)
				}

			case dtypes.BFloat16:
				lhsFlatBF16 := lhsFlatAny.([]bfloat16.BFloat16)
				rhsFlatBF16 := rhsFlatAny.([]bfloat16.BFloat16)
				for i := range lhsFlatBF16 {
					lhsFlatBF16[i] = bfloat16.FromFloat32(float32(i%10 + 1))
				}
				for i := range rhsFlatBF16 {
					rhsFlatBF16[i] = bfloat16.FromFloat32(float32(i%10 + 1))
				}
			}
			lhsTensor := tensors.FromBuffer(backend, lhsBuffer)
			rhsTensor := tensors.FromBuffer(backend, rhsBuffer)

			// Create the program that does the DotGeneral.
			testExec := graph.NewExec(backend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.DotGeneral(lhs, benchCase.lhsContractingAxes, benchCase.lhsBatchAxes,
					rhs, benchCase.rhsContractingAxes, benchCase.rhsBatchAxes)
			})

			// Warm-up runs
			for i := 0; i < numWarmupRuns; i++ {
				output := testExec.Call(lhsTensor, rhsTensor)[0]
				output.FinalizeAll()
			}

			// Timed runs
			startTime := time.Now()
			var numRuns int
			for numRuns < minNumTimedRuns || time.Since(startTime) < minTestTime { // i := 0; i < numTimedRuns; i++ {
				output := testExec.Call(lhsTensor, rhsTensor)[0]
				output.FinalizeAll()
				numRuns++
			}
			duration := time.Since(startTime)
			avgDurationPerRun := duration / time.Duration(numRuns)

			// Calculate the total number of multiply-add operations.
			gOpsPerSecond := float64(numOps) / avgDurationPerRun.Seconds() / 1e9 // Giga Ops

			// Print table row
			style := style1
			if benchCaseIdx%2 == 1 {
				style = style2
			}
			row := fmt.Sprintf("| %-20s | %-20s | %-20s | %-10s | %-10d | %-12s | %-15s | %-10.1f |",
				benchCase.name,
				dimsToStr(benchCase.lhsShape), dimsToStr(benchCase.rhsShape),
				dtype,
				batchSize,
				formatDurationWith2Decimals(avgDurationPerRun),
				humanize.Comma(int64(numOps)),
				gOpsPerSecond)
			fmt.Println(style.Render(row))
		}
	}
	fmt.Println(strings.Repeat("-", len(header)))
	fmt.Println()
}
