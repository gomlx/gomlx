package simplego

import (
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

func TestDotGeneral_transposeForDotGeneral(t *testing.T) {
	S := shapes.Make
	F32 := dtypes.Float32
	builder := backend.Builder("DotGeneral Test").(*Builder)
	operandOp, err := builder.Parameter("lhs", S(F32, 2, 3, 4, 5))
	require.NoError(t, err)
	operand := operandOp.(*Node)
	transposed, batchDims, crossDims, contractingDims, err :=
		builder.transposeForDotGeneral(operand, "lhs", []int{2, 1}, []int{3, 0})
	require.NoError(t, err)
	fmt.Printf("\ttransposed.shape=%s\n", transposed.shape)

	assert.NoError(t, transposed.shape.CheckDims(10, 1, 12))
	assert.Equal(t, []int{5, 2}, batchDims)
	assert.Len(t, crossDims, 0)
	assert.Equal(t, []int{4, 3}, contractingDims)
}

func TestDotGeneral_Shape(t *testing.T) {
	S := shapes.Make
	F32 := dtypes.Float32
	builder := backend.Builder("DotGeneral Test").(*Builder)
	lhs, err := builder.Parameter("lhs", S(F32, 2, 3, 4, 5))
	require.NoError(t, err)
	rhs, err := builder.Parameter("lhs", S(F32, 5, 1, 2, 3))
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
	assert.Equal(t, want, y0.Value())

	bf16 := bfloat16.FromFloat32
	y1 := graph.ExecOnce(backend, func(lhs, rhs *graph.Node) *graph.Node {
		return graph.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
	},
		[][]bfloat16.BFloat16{{bf16(1), bf16(2), bf16(3)}},
		[][]bfloat16.BFloat16{{bf16(10)}, {bf16(11)}, {bf16(12)}},
	)
	fmt.Printf("\ty1=%s\n", y1)
	assert.NoError(t, y1.Shape().Check(dtypes.BFloat16, 1, 1))
	assert.Equal(t, float32(10+22+36), tensors.CopyFlatData[bfloat16.BFloat16](y1)[0].Float32())
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

// normalizedDotGeneralBenchmarkDimCase defines the *normalized* dimensions for a benchmark run.
// Dims are: [Batch, Cross, Contracting]
type normalizedDotGeneralBenchmarkDimCase struct {
	name    string
	lhsDims []int
	rhsDims []int
}

func TestDotGeneral_NormalizedDotGeneralPerformanceTable(t *testing.T) {
	// IMPORTANT: Populate this slice with the *normalized* dimensions you want to benchmark.
	// lhsDims: [Batch, LhsCross, Contracting]
	// rhsDims: [Batch, RhsCross, Contracting]
	// Batch and Contracting dimensions must match between lhs and rhs.
	benchmarkDimCases := []normalizedDotGeneralBenchmarkDimCase{
		// Shape values taken from the model https://huggingface.co/KnightsAnalytics/all-MiniLM-L6-v2
		{
			name:    "KA-Bert#1",
			lhsDims: []int{24, 7, 7},
			rhsDims: []int{24, 32, 7},
		},
		{
			name:    "KA-Bert#2",
			lhsDims: []int{24, 7, 32},
			rhsDims: []int{24, 7, 32},
		},
		{
			name:    "KA-Bert#3", // This one happens 4x more than the others.
			lhsDims: []int{1, 14, 384},
			rhsDims: []int{1, 384, 384},
		},
		{
			name:    "KA-Bert#4",
			lhsDims: []int{1, 14, 384},
			rhsDims: []int{1, 1536, 384},
		},
		{
			name:    "KA-Bert#5",
			lhsDims: []int{1, 14, 1536},
			rhsDims: []int{1, 384, 1536},
		},
		// Add more test cases relevant to your models here
	}

	dtypesToTest := []dtypes.DType{dtypes.Float32, dtypes.Float64, dtypes.BFloat16}

	simpleGoBackend, ok := backend.(*Backend)
	if !ok {
		t.Fatalf("Global backend is not of type *simplego.Backend. It's %T. Ensure GOMLEX_BACKEND is 'go'.", backend)
	}

	// Adjust for desired precision vs. test duration
	const numWarmupRuns = 2
	const minNumTimedRuns = 50
	const minTestTime = time.Second

	dimsToStr := func(dims []int) string {
		dimsStr := xslices.Map(dims, func(i int) string { return strconv.Itoa(i) })
		return fmt.Sprintf("{%s}", strings.Join(dimsStr, ", "))
	}

	// Colors: tests usually run in batch and that disallows colors. We temporarily force a different profile:
	originalProfile := lipgloss.ColorProfile()      // Optional: store original
	lipgloss.SetColorProfile(termenv.ANSI256)       // Or termenv.TrueColor if you prefer
	defer lipgloss.SetColorProfile(originalProfile) // Optional: reset
	style1 := lipgloss.NewStyle()
	style2 := lipgloss.NewStyle().Background(lipgloss.ANSIColor(0))

	// Print table header
	fmt.Printf("\n--- execNormalizedDotGeneral Performance ---\n")
	header := fmt.Sprintf("| %-15s | %-20s | %-20s | %-10s | %-10s | %-10s |", "Test Name", "LHS Dims", "RHS Dims", "DType", "Time/Run", "GOps/Sec")
	fmt.Println(header)
	fmt.Println(strings.Repeat("-", len(header)))

	for dimCaseIdx, dimCase := range benchmarkDimCases {
		for _, dtype := range dtypesToTest {
			// Construct shapes from dimensions and current dtype
			if len(dimCase.lhsDims) != 3 || len(dimCase.rhsDims) != 3 {
				fmt.Printf("| %-15s | %-20v | %-20v | %-10s | %-10s | %-10s |\n", dimCase.name, dimCase.lhsDims, dimCase.rhsDims, dtype, "Invalid Dims", "N/A")
				continue
			}

			lhsShape := shapes.Make(dtype, dimCase.lhsDims...)
			rhsShape := shapes.Make(dtype, dimCase.rhsDims...)

			// Validate shapes
			if lhsShape.Dimensions[0] != rhsShape.Dimensions[0] || // Batch
				lhsShape.Dimensions[2] != rhsShape.Dimensions[2] { // Contracting
				errMsg := fmt.Sprintf("Mismatch B(%d!=%d) or C(%d!=%d)",
					lhsShape.Dimensions[0], rhsShape.Dimensions[0],
					lhsShape.Dimensions[2], rhsShape.Dimensions[2])
				fmt.Printf("| %-15s | %-20s | %-20s | %-10s | %-10s | %-10s |\n", dimCase.name, dimsToStr(dimCase.lhsDims), dimsToStr(dimCase.rhsDims), dtype, errMsg, "N/A")
				continue
			}

			batchSize := lhsShape.Dimensions[0]
			lhsCrossSize := lhsShape.Dimensions[1]
			rhsCrossSize := rhsShape.Dimensions[1]
			contractingSize := lhsShape.Dimensions[2]

			outputShape := shapes.Make(dtype, batchSize, lhsCrossSize, rhsCrossSize)
			node := &Node{shape: outputShape} // Minimal Node for execNormalizedDotGeneral

			// Create and initialize input Buffers
			lhsBuffer := simpleGoBackend.getBuffer(lhsShape.DType, lhsShape.Size())
			lhsBuffer.shape = lhsShape
			rhsBuffer := simpleGoBackend.getBuffer(rhsShape.DType, rhsShape.Size())
			rhsBuffer.shape = rhsShape

			switch dtype {
			case dtypes.Float32:
				lhsFlatF32 := make([]float32, lhsShape.Size())
				rhsFlatF32 := make([]float32, rhsShape.Size())
				for i := range lhsFlatF32 {
					lhsFlatF32[i] = float32(i%10 + 1)
				}
				for i := range rhsFlatF32 {
					rhsFlatF32[i] = float32(i%10 + 1)
				}
				lhsBuffer.flat = lhsFlatF32
				rhsBuffer.flat = rhsFlatF32
			case dtypes.Float64:
				lhsFlatF64 := make([]float64, lhsShape.Size())
				rhsFlatF64 := make([]float64, rhsShape.Size())
				for i := range lhsFlatF64 {
					lhsFlatF64[i] = float64(i%10 + 1)
				}
				for i := range rhsFlatF64 {
					rhsFlatF64[i] = float64(i%10 + 1)
				}
				lhsBuffer.flat = lhsFlatF64
				rhsBuffer.flat = rhsFlatF64
			case dtypes.BFloat16:
				lhsFlatBF16 := make([]bfloat16.BFloat16, lhsShape.Size())
				rhsFlatBF16 := make([]bfloat16.BFloat16, rhsShape.Size())
				for i := range lhsFlatBF16 {
					lhsFlatBF16[i] = bfloat16.FromFloat32(float32(i%10 + 1))
				}
				for i := range rhsFlatBF16 {
					rhsFlatBF16[i] = bfloat16.FromFloat32(float32(i%10 + 1))
				}
				lhsBuffer.flat = lhsFlatBF16
				rhsBuffer.flat = rhsFlatBF16
			default:
				fmt.Printf("| %-15s | %-20s | %-20s | %-10s | %-10s | %-10s |\n", dimCase.name, dimsToStr(dimCase.lhsDims), dimsToStr(dimCase.rhsDims), dtype, "Unsupported DType", "N/A")
				continue
			}
			inputs := []*Buffer{lhsBuffer, rhsBuffer}

			// Warm-up runs
			for i := 0; i < numWarmupRuns; i++ {
				_, err := execNormalizedDotGeneral(simpleGoBackend, node, inputs, nil)
				if err != nil {
					t.Errorf("Warm-up run for %s Dims %v, %s Dims %v, DType %s failed: %v", dimCase.name, dimCase.lhsDims, dimCase.name, dimCase.rhsDims, dtype, err)
					continue
				}
			}

			// Timed runs
			startTime := time.Now()
			var numRuns int
			for numRuns < minNumTimedRuns || time.Since(startTime) < minTestTime { // i := 0; i < numTimedRuns; i++ {
				_, err := execNormalizedDotGeneral(simpleGoBackend, node, inputs, nil)
				numRuns++
				if err != nil {
					t.Errorf("Timed run for %s Dims %v, %s Dims %v, DType %s failed: %v", dimCase.name, dimCase.lhsDims, dimCase.name, dimCase.rhsDims, dtype, err)
					continue
				}
			}
			duration := time.Since(startTime)
			avgDurationPerRun := duration / time.Duration(numRuns)
			//avgDurationPerRun := duration / time.Duration(numTimedRuns)

			// Calculate the total number of multiply-add operations.
			numOps := int64(batchSize) * int64(lhsCrossSize) * int64(rhsCrossSize) * int64(contractingSize) * 2 // 1 mult + 1 add = 2 ops
			gOpsPerSecond := float64(numOps) / avgDurationPerRun.Seconds() / 1e9                                // Giga Ops

			// Print table row
			style := style1
			if dimCaseIdx%2 == 1 {
				style = style2
			}
			row := fmt.Sprintf("| %-15s | %-20s | %-20s | %-10s | %-10s | %-10.1f |",
				dimCase.name,
				dimsToStr(dimCase.lhsDims), dimsToStr(dimCase.rhsDims),
				dtype,
				formatDurationWith2Decimals(avgDurationPerRun),
				gOpsPerSecond)
			fmt.Println(style.Render(row))
		}
	}
	fmt.Println(strings.Repeat("-", len(header)))
	fmt.Println()
}

// dotGeneralBenchmarkParamsCase defines input parameters for DotGeneral to be benchmarked.
type dotGeneralBenchmarkParamsCase struct {
	name                                       string
	lhsShape, lhsContractingAxes, lhsBatchAxes []int
	rhsShape, rhsContractingAxes, rhsBatchAxes []int
}

func dgFindSizes(shape shapes.Shape, contractingAxes, batchAxes []int) (batchSize, crossSize, contractingSize int) {
	rank := shape.Rank()
	axesTypes := make([]int, rank)

	// Mark axes types: 1 for contracting, 2 for batch
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
	}
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
	}

	// Calculate sizes by multiplying dimensions according to axis type
	batchSize, crossSize, contractingSize = 1, 1, 1
	for axis, axisType := range axesTypes {
		dim := shape.Dimensions[axis]
		switch axisType {
		case 0: // Cross axes (unmarked)
			crossSize *= dim
		case 1: // Contracting axes
			contractingSize *= dim
		case 2: // Batch axes
			batchSize *= dim
		}
	}
	return
}

func dimsToStr(dims []int) string {
	dimsStr := xslices.Map(dims, func(i int) string { return strconv.Itoa(i) })
	return fmt.Sprintf("{%s}", strings.Join(dimsStr, ", "))
}

func TestDotGeneral_DotGeneralPerformanceTable(t *testing.T) {
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

	simpleGoBackend, ok := backend.(*Backend)
	if !ok {
		t.Fatalf("Global backend is not of type *simplego.Backend. It's %T. Ensure GOMLEX_BACKEND is 'go'.", backend)
	}

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
	header := fmt.Sprintf("| %-15s | %-20s | %-20s | %-10s | %-10s | %-10s |", "Test Name", "LHS Dims", "RHS Dims", "DType", "Time/Run", "GOps/Sec")
	fmt.Println(header)
	fmt.Println(strings.Repeat("-", len(header)))

	for benchCaseIdx, benchCase := range benchmarkCases {
		for _, dtype := range dtypesToTest {
			// Construct shapes from dimensions and current dtype
			lhsShape := shapes.Make(dtype, benchCase.lhsShape...)
			rhsShape := shapes.Make(dtype, benchCase.rhsShape...)
			batchSize, lhsCrossSize, contractingSize := dgFindSizes(lhsShape, benchCase.lhsContractingAxes, benchCase.lhsBatchAxes)
			_, rhsCrossSize, _ := dgFindSizes(rhsShape, benchCase.rhsContractingAxes, benchCase.rhsBatchAxes)
			numOps := batchSize * lhsCrossSize * rhsCrossSize * contractingSize * 2 // 1 mult + 1 add = 2 ops

			// Create and initialize input Buffers
			lhsBuffer := simpleGoBackend.getBuffer(lhsShape.DType, lhsShape.Size())
			lhsBuffer.shape = lhsShape
			rhsBuffer := simpleGoBackend.getBuffer(rhsShape.DType, rhsShape.Size())
			rhsBuffer.shape = rhsShape
			switch dtype {
			case dtypes.Float32:
				lhsFlatF32 := lhsBuffer.flat.([]float32)
				rhsFlatF32 := rhsBuffer.flat.([]float32)
				for i := range lhsFlatF32 {
					lhsFlatF32[i] = float32(i%10 + 1)
				}
				for i := range rhsFlatF32 {
					rhsFlatF32[i] = float32(i%10 + 1)
				}

			case dtypes.Float64:
				lhsFlatF64 := lhsBuffer.flat.([]float64)
				rhsFlatF64 := rhsBuffer.flat.([]float64)
				for i := range lhsFlatF64 {
					lhsFlatF64[i] = float64(i%10 + 1)
				}
				for i := range rhsFlatF64 {
					rhsFlatF64[i] = float64(i%10 + 1)
				}

			case dtypes.BFloat16:
				lhsFlatBF16 := lhsBuffer.flat.([]bfloat16.BFloat16)
				rhsFlatBF16 := rhsBuffer.flat.([]bfloat16.BFloat16)
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
			row := fmt.Sprintf("| %-15s | %-20s | %-20s | %-10s | %-10s | %-10.1f |",
				benchCase.name,
				dimsToStr(benchCase.lhsShape), dimsToStr(benchCase.rhsShape),
				dtype,
				formatDurationWith2Decimals(avgDurationPerRun),
				gOpsPerSecond)
			fmt.Println(style.Render(row))
		}
	}
	fmt.Println(strings.Repeat("-", len(header)))
	fmt.Println()
}
