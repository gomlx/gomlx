//go:build perf

package simplego

import (
	"flag"
	"fmt"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/dustin/go-humanize"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/muesli/termenv"
	"github.com/stretchr/testify/require"
)

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

var (
	flagPerfTests = flag.String(
		"perf_names",
		"",
		"Comma-separated list of performance tests (part of TestDotGeneral_PerformanceTable) to run. If empty, it will run all the perf tests.",
	)
	flagPerfDTypes = flag.String(
		"perf_dtypes",
		"",
		"Comma-separated list of dtypes to run performance test (part of TestDotGeneral_PerformanceTable). If empty, it will run for all supported dtypes.",
	)
)

func must(err error) {
	if err != nil {
		panic(err)
	}
}

func must1[T any](value T, err error) T {
	must(err)
	return value
}

// TestDotGeneral_PerformanceTable generates a performance table for differently
// sized matrices.
//
// This is not included by default, only if using -tags perf.
//
// Examples:
//
//	$ GOMLX_BACKEND=go go test -tags=perf ./backends/simplego/ -test.run=TestDotGeneral_PerformanceTable -test.v -test.count=1
//	$ GOMLX_BACKEND=xla:cuda go test -tags=xla,perf ./backends/simplego/ -test.run=TestDotGeneral_PerformanceTable -test.v -test.count=1
//	$ GOMLX_BACKEND=stablehlo:cpu go test -tags=stablehlo,perf ./backends/simplego/ -test.run=TestDotGeneral_PerformanceTable -test.v -test.count=1
func TestDotGeneral_PerformanceTable(t *testing.T) {
	filterPerfs := *flagPerfTests != ""
	perfsToRun := sets.MakeWith(strings.Split(*flagPerfTests, ",")...)
	filterDTypes := *flagPerfDTypes != ""
	dtypesToRun := sets.MakeWith(strings.Split(*flagPerfDTypes, ",")...)

	// IMPORTANT: Populate this slice with the shapes and parameters of the dot-product.
	// lhsDims: [Batch, LhsCross, Contracting]
	// rhsDims: [Batch, RhsCross, Contracting]
	// Batch and Contracting dimensions must match between lhs and rhs.
	benchmarkCases := []dotGeneralBenchmarkParamsCase{
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
	header := fmt.Sprintf(
		"| %-20s | %-20s | %-20s | %-10s | %-10s | %-12s | %-15s | %-10s |",
		"Test Name",
		"LHS Dims",
		"RHS Dims",
		"DType",
		"BatchSize",
		"Time/Run",
		"Num Ops",
		"GOps/Sec",
	)
	fmt.Println(header)
	fmt.Println(strings.Repeat("-", len(header)))

	for benchCaseIdx, benchCase := range benchmarkCases {
		if filterPerfs && !perfsToRun.Has(benchCase.name) {
			continue
		}
		for _, dtype := range dtypesToTest {
			if filterDTypes && !dtypesToRun.Has(dtype.String()) {
				continue
			}
			// Construct shapes from dimensions and current dtype
			lhsShape := shapes.Make(dtype, benchCase.lhsShape...)
			rhsShape := shapes.Make(dtype, benchCase.rhsShape...)
			batchSize, lhsCrossSize, contractingSize, _ := dgFindSizes(
				lhsShape,
				benchCase.lhsContractingAxes,
				benchCase.lhsBatchAxes,
			)
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
			lhsTensor := must1(tensors.FromBuffer(backend, lhsBuffer))
			rhsTensor := must1(tensors.FromBuffer(backend, rhsBuffer))

			// Create the program that does the DotGeneral.
			testExec := graph.MustNewExec(backend, func(lhs, rhs *graph.Node) *graph.Node {
				return graph.DotGeneral(lhs, benchCase.lhsContractingAxes, benchCase.lhsBatchAxes,
					rhs, benchCase.rhsContractingAxes, benchCase.rhsBatchAxes)
			})

			// Warm-up runs
			for i := 0; i < numWarmupRuns; i++ {
				output := testExec.MustExec(lhsTensor, rhsTensor)[0]
				output.MustFinalizeAll()
			}

			// Timed runs
			startTime := time.Now()
			var numRuns int
			for numRuns < minNumTimedRuns || time.Since(startTime) < minTestTime { // i := 0; i < numTimedRuns; i++ {
				output := testExec.MustExec(lhsTensor, rhsTensor)[0]
				output.MustFinalizeAll()
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
				commandline.FormatDuration(avgDurationPerRun),
				humanize.Comma(int64(numOps)),
				gOpsPerSecond)
			fmt.Println(style.Render(row))
		}
	}
	fmt.Println(strings.Repeat("-", len(header)))
	fmt.Println()
}
