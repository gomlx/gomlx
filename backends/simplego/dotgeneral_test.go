package simplego

import (
	"flag"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
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

/* ONNX KnightsAnalytics/all-MiniLM-L6-v2 model outputs:

Backend: xla:cpu

 {{0.03656, -0.01616, 0.1682, ..., 0.05541, -0.1644, -0.2967},
  {0.7239, 0.6399, 0.1888, ..., 0.5946, 0.6206, 0.4897},
  {0.006379, 0.0203, 0.04476, ..., 0.3464, 1.317, -0.167},
  ...,
  {0.1479, -0.06426, 0.1457, ..., 0.8837, -0.3316, 0.2975},
  {0.5212, 0.6563, 0.5607, ..., -0.03989, 0.04121, -1.404},
  {1.082, 0.714, 0.3986, ..., -0.2301, 0.3243, -1.031}},
 {{0.2802, 0.1165, -0.04179, ..., 0.2711, -0.1685, -0.2961},
  {0.8729, 0.4545, -0.1091, ..., 0.1365, 0.458, -0.2042},
  {0.4752, 0.5731, 0.6304, ..., 0.6526, 0.5612, -1.327},
  ...,
  {0.6113, 0.792, -0.4685, ..., 0.08543, 1.059, -0.2983},
  {0.4115, 1.095, 0.2385, ..., 0.8984, 0.3684, -0.7333},
  {0.1374, 0.5555, 0.2678, ..., 0.5426, 0.4665, -0.5284}}}

Backend: xla:cuda

 {{0.03645, -0.01605, 0.1683, ..., 0.05549, -0.1644, -0.2968},
  {0.7241, 0.6391, 0.1888, ..., 0.5937, 0.6208, 0.4895},
  {0.006026, 0.02017, 0.04514, ..., 0.3465, 1.316, -0.167},
  ...,
  {0.1475, -0.06374, 0.1454, ..., 0.8843, -0.3317, 0.2972},
  {0.5215, 0.6562, 0.561, ..., -0.03975, 0.04077, -1.404},
  {1.082, 0.7137, 0.399, ..., -0.2294, 0.3244, -1.031}},
 {{0.2799, 0.1163, -0.04177, ..., 0.271, -0.1684, -0.2963},
  {0.8731, 0.4541, -0.1088, ..., 0.1363, 0.458, -0.2041},
  {0.4751, 0.5729, 0.6302, ..., 0.6525, 0.5609, -1.327},
  ...,
  {0.6113, 0.7916, -0.468, ..., 0.08592, 1.059, -0.2987},
  {0.4115, 1.095, 0.2389, ..., 0.8984, 0.3684, -0.7335},
  {0.1358, 0.5588, 0.27, ..., 0.5427, 0.4699, -0.5304}}}

Backend: go, original version

 {{0.03657, -0.01616, 0.1682, ..., 0.05541, -0.1644, -0.2967},
  {0.7239, 0.6399, 0.1888, ..., 0.5946, 0.6206, 0.4897},
  {0.006379, 0.02031, 0.04476, ..., 0.3464, 1.317, -0.167},
  ...,
  {0.1479, -0.06426, 0.1457, ..., 0.8837, -0.3316, 0.2975},
  {0.5212, 0.6563, 0.5607, ..., -0.03989, 0.04121, -1.404},
  {1.082, 0.714, 0.3986, ..., -0.2301, 0.3243, -1.031}},
 {{0.2802, 0.1165, -0.04179, ..., 0.2711, -0.1685, -0.2961},
  {0.8729, 0.4545, -0.1091, ..., 0.1366, 0.458, -0.2042},
  {0.4752, 0.5731, 0.6304, ..., 0.6526, 0.5612, -1.327},
  ...,
  {0.6113, 0.792, -0.4685, ..., 0.08543, 1.059, -0.2983},
  {0.4115, 1.095, 0.2385, ..., 0.8984, 0.3684, -0.7333},
  {0.1374, 0.5555, 0.2678, ..., 0.5426, 0.4665, -0.5284}}}




*/

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

func TestDotGeneral_ShapesAndCopy(t *testing.T) {
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

func TestDotGeneral_Exec(t *testing.T) {
	// Large example with batches.
	{
		lhs, err := tensors.Load("dotgeneral_lhs_test.bin")
		require.NoError(t, err)
		rhs, err := tensors.Load("dotgeneral_rhs_test.bin")
		require.NoError(t, err)
		want, err := tensors.Load("dotgeneral_out_test.bin")
		require.NoError(t, err)
		y := graph.ExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
			return graph.DotGeneral(lhs, []int{2}, []int{0}, rhs, []int{2}, []int{0})
		}, lhs, rhs)
		fmt.Printf("\ty=%s\n", y)
		fmt.Printf("\twant=%s\n", want)
		require.True(t, y.Shape().Equal(want.Shape()))
		require.InDeltaSlice(t, tensors.CopyFlatData[float32](want), tensors.CopyFlatData[float32](y), 1e-3)
	}

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
	/*
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
	*/
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
			lhsShape: []int{1024, 13, 1536}, lhsContractingAxes: []int{2}, lhsBatchAxes: []int(nil),
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

		// Large batch example
		//{},

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

	dtypesToTest := []dtypes.DType{dtypes.Float32, dtypes.Float64} // TODO: , dtypes.BFloat16}

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
	header := fmt.Sprintf("| %-15s | %-20s | %-20s | %-10s | %-12s | %-10s | %-10s |", "Test Name", "LHS Dims", "RHS Dims", "DType", "Time/Run", "Num Ops", "GOps/Sec")
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
			row := fmt.Sprintf("| %-15s | %-20s | %-20s | %-10s | %-12s | %-10d | %-10.1f |",
				benchCase.name,
				dimsToStr(benchCase.lhsShape), dimsToStr(benchCase.rhsShape),
				dtype,
				formatDurationWith2Decimals(avgDurationPerRun),
				numOps,
				gOpsPerSecond)
			fmt.Println(style.Render(row))
		}
	}
	fmt.Println(strings.Repeat("-", len(header)))
	fmt.Println()
}
