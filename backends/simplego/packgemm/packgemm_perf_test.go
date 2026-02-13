//  - --- go:build perf

// Copyright 2025 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package packgemm_test

import (
	"flag"
	"fmt"
	"slices"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gomlx/gomlx/backends/simplego/packgemm"
	"github.com/gomlx/gomlx/internal/workerspool"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/x448/float16"
)

var (
	flagPerfTests = flag.String("perf_names", "",
		"Comma-separated list of performance tests (part of TestPackGemm_PerformanceTable) to "+
			"run. If empty, it will run all the perf tests.")
	flagPerfDTypes = flag.String("perf_dtypes", "",
		"Comma-separated list of dtypes to run performance test (part of TestPackGemm_PerformanceTable). "+
			"If empty, it will run for all supported dtypes.")
	flagPerfDuration = flag.Duration("perf_duration", time.Second, "Duration to run each performance test.")
	flagPerfMinRuns  = flag.Int("perf_min_runs", 10, "Minimum number of runs for each performance test.")
)

// benchmarkCase defines input parameters for PackGemm to be benchmarked.
type benchmarkCase struct {
	name                      string
	lhsRows, lhsCols, rhsCols int // M, K, N
	batchSize                 int
	algorithms                []string
}

// TestPackGemm_PerformanceTable generates a performance table for differently
// sized matrices.
//
// This is not included by default, only if using -tags perf.
//
// Examples:
//
//	$ go test -tags=perf ./backends/simplego/packgemm -run=TestPackGemm_PerformanceTable -v -count=1
func TestPackGemm_PerformanceTable(t *testing.T) {
	// Define benchmark cases
	largeAlgos := []string{"hwy-16regs"}
	cases := []benchmarkCase{
		{
			name:    "NoBatch-Tiny",
			lhsRows: 4, lhsCols: 128, rhsCols: 1, // 4x128 * 128x1 -> 4x1
		},
		{
			name:    "NoBatch-Small",
			lhsRows: 128, lhsCols: 128, rhsCols: 32,
		},
		{
			name:    "NoBatch-Medium",
			lhsRows: 128, lhsCols: 128, rhsCols: 256,
		},
		{
			name:    "NoBatch-Large",
			lhsRows: 1536, lhsCols: 1920, rhsCols: 1024,
			algorithms: largeAlgos,
		},
		{
			name:      "LargeBatch-Small",
			batchSize: 128,
			lhsRows:   64, lhsCols: 64, rhsCols: 64,
		},
		{
			name:    "NoBatch-1024",
			lhsRows: 1024, lhsCols: 1024, rhsCols: 1024,
		},
		{
			name:      "Batched-Large",
			batchSize: 16,
			lhsRows:   1536, lhsCols: 1920, rhsCols: 1024,
			algorithms: largeAlgos,
		},
	}

	// Filter only selected cases, if there was a selection.
	if *flagPerfTests != "" {
		parts := strings.Split(*flagPerfTests, ",")
		parts = slices.DeleteFunc(parts, func(p string) bool { return p == "" })
		cases = slices.DeleteFunc(cases, func(c benchmarkCase) bool {
			for _, p := range parts {
				if strings.Contains(c.name, p) {
					return false
				}
			}
			return true
		})
		fmt.Printf("- Cases selected:      %q\n", xslices.Map(cases, func(c benchmarkCase) string {
			return c.name
		}))
	}

	// Enumerate dtypes to consider.
	dtypesToTest := []dtypes.DType{dtypes.Float32, dtypes.Float64, dtypes.BFloat16, dtypes.Float16}
	if *flagPerfDTypes != "" {
		dtypesToTest = dtypesToTest[:0]
		parts := strings.Split(*flagPerfDTypes, ",")
		for _, p := range parts {
			if p == "" {
				continue
			}
			dtype, err := dtypes.DTypeString(p)
			if err != nil {
				t.Fatalf("unknown dtype %q: %v", p, err)
			}
			dtypesToTest = append(dtypesToTest, dtype)
		}
		fmt.Printf("- Dtypes selected:     %q\n", dtypesToTest)
	}

	// Workers pool
	pool := workerspool.New() // Default parallelism

	// 1. Discover all algorithms across all cases/dtypes to define columns
	allAlgorithmNames := sets.Make[string]()
	for _, c := range cases {
		for _, dtype := range dtypesToTest {
			algs := packgemm.DTypeToGEMM[packgemm.DTypePair{Input: dtype, Output: dtype}]
			for _, alg := range algs {
				if len(c.algorithms) > 0 && !slices.Contains(c.algorithms, alg.Name) {
					continue
				}
				allAlgorithmNames.Insert(alg.Name)
			}
		}
	}
	if len(allAlgorithmNames) == 0 {
		t.Fatalf("no algorithms found for the selected cases (%q) and dtypes (%q)", *flagPerfTests, *flagPerfDTypes)
	}
	sortedAlgorithmNames := xslices.SortedKeys(allAlgorithmNames)
	fmt.Printf("- Algorithms selected: %q\n", sortedAlgorithmNames)

	// 2. Print Header
	fmt.Printf("\n--- PackGemm Performance ---\n")
	headerParts := []string{"| DType", "Batch", "Test Name", "LHS Dims", "RHS Dims"}
	for _, name := range sortedAlgorithmNames {
		headerParts = append(headerParts, name)
	}
	headerParts = append(headerParts, "|")
	fmt.Println(strings.Join(headerParts, " | "))

	// Markdown separator
	sepParts := []string{"| :---", ":---", ":---", ":---", ":---"}
	for range sortedAlgorithmNames {
		sepParts = append(sepParts, ":---")
	}
	sepParts = append(sepParts, "|")
	fmt.Println(strings.Join(sepParts, " | "))

	// 3. Run Benchmarks
	for _, bc := range cases {
		for _, dtype := range dtypesToTest {
			// Row Metadata
			batchSize := bc.batchSize
			if batchSize == 0 {
				batchSize = 1
			}
			lhsDims := fmt.Sprintf("[%d, %d]", bc.lhsRows, bc.lhsCols)
			rhsDims := fmt.Sprintf("[%d, %d]", bc.lhsCols, bc.rhsCols)

			rowParts := []string{
				fmt.Sprintf("| %s", dtype),
				fmt.Sprintf("%d", batchSize),
				fmt.Sprintf("`%s`", bc.name),
				lhsDims,
				rhsDims,
			}

			// Calculate Ops
			// 2 * M * N * K * BatchSize
			numOps := 2 * int64(bc.lhsRows) * int64(bc.rhsCols) * int64(bc.lhsCols) * int64(batchSize)

			// Iterate unique algorithms (columns)
			for _, algName := range sortedAlgorithmNames {
				// Find this specific algorithm implementation for this dtype
				algs := packgemm.DTypeToGEMM[packgemm.DTypePair{Input: dtype, Output: dtype}]
				var targetAlg packgemm.GEMMRegistration
				found := false
				for _, a := range algs {
					if a.Name == algName {
						targetAlg = a
						found = true
						break
					}
				}

				cellContent := "-"
				if found {
					// Run Benchmark
					gops := runBenchmark(bc, dtype, targetAlg, pool, numOps)
					cellContent = fmt.Sprintf("%.2f GFlops/s", gops)
				}

				rowParts = append(rowParts, cellContent)
			}
			rowParts = append(rowParts, "|")

			// Print Row
			rowStr := strings.Join(rowParts, " | ")
			fmt.Println(rowStr)
		}
	}
	fmt.Println()
}

func runBenchmark(bc benchmarkCase, dtype dtypes.DType, alg packgemm.GEMMRegistration, pool *workerspool.Pool, numOps int64) float64 {
	// 1. Setup Buffers
	batchSize := bc.batchSize
	if batchSize == 0 {
		batchSize = 1
	}
	lhsCrossSize := bc.lhsRows
	contractingSize := bc.lhsCols
	rhsCrossSize := bc.rhsCols

	// Calculate sizes
	lhsSize := batchSize * lhsCrossSize * contractingSize
	rhsSize := batchSize * contractingSize * rhsCrossSize
	outSize := batchSize * lhsCrossSize * rhsCrossSize

	// Helper to cast fields
	var runFn func() error

	// Prepare data
	switch dtype {
	case dtypes.Float32:
		lhs := xslices.SliceWithValue(lhsSize, float32(1.0))
		rhs := xslices.SliceWithValue(rhsSize, float32(1.0))
		out := make([]float32, outSize)
		bufPool := newTestBufferPool[float32]()

		fn := alg.GEMMFn.(func(alpha, beta float32, lhsFlat, rhsFlat []float32, batchSize,
			lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float32,
			bufAllocFn packgemm.BufAllocFn[float32], bufReleaseFn packgemm.BufReleaseFn, pool *workerspool.Pool) error)

		runFn = func() error {
			return fn(1.0, 0.0, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, out, bufPool.Alloc, bufPool.Release, pool)
		}

	case dtypes.Float64:
		lhs := xslices.SliceWithValue(lhsSize, 1.0)
		rhs := xslices.SliceWithValue(rhsSize, 1.0)
		out := make([]float64, outSize)
		bufPool := newTestBufferPool[float64]()

		fn := alg.GEMMFn.(func(alpha, beta float64, lhsFlat, rhsFlat []float64, batchSize,
			lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float64,
			bufAllocFn packgemm.BufAllocFn[float64], bufReleaseFn packgemm.BufReleaseFn, pool *workerspool.Pool) error)

		runFn = func() error {
			return fn(1.0, 0.0, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, out, bufPool.Alloc, bufPool.Release, pool)
		}

	case dtypes.Float16:
		one := float16.Fromfloat32(1.0)
		zero := float16.Fromfloat32(0.0)
		lhs := xslices.SliceWithValue(lhsSize, one)
		rhs := xslices.SliceWithValue(rhsSize, one)
		out := make([]float16.Float16, outSize)
		bufPool := newTestBufferPool[float16.Float16]()

		fn := alg.GEMMFn.(func(alpha, beta float16.Float16, lhsFlat, rhsFlat []float16.Float16, batchSize,
			lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []float16.Float16,
			bufAllocFn packgemm.BufAllocFn[float16.Float16], bufReleaseFn packgemm.BufReleaseFn, pool *workerspool.Pool) error)

		runFn = func() error {
			return fn(one, zero, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, out, bufPool.Alloc, bufPool.Release, pool)
		}

	case dtypes.BFloat16:
		one := bfloat16.FromFloat32(1.0)
		zero := bfloat16.FromFloat32(0.0)
		lhs := xslices.SliceWithValue(lhsSize, one)
		rhs := xslices.SliceWithValue(rhsSize, one)
		out := make([]bfloat16.BFloat16, outSize)
		bufPool := newTestBufferPool[bfloat16.BFloat16]()

		fn := alg.GEMMFn.(func(alpha, beta bfloat16.BFloat16, lhsFlat, rhsFlat []bfloat16.BFloat16, batchSize,
			lhsCrossSize, rhsCrossSize, contractingSize int, outputFlat []bfloat16.BFloat16,
			bufAllocFn packgemm.BufAllocFn[bfloat16.BFloat16], bufReleaseFn packgemm.BufReleaseFn, pool *workerspool.Pool) error)

		runFn = func() error {
			return fn(one, zero, lhs, rhs, batchSize, lhsCrossSize, rhsCrossSize, contractingSize, out, bufPool.Alloc, bufPool.Release, pool)
		}
	}

	// Warmup
	for i := 0; i < 2; i++ {
		_ = runFn()
	}

	// Timed Runs
	startTime := time.Now()
	var numRuns int
	for numRuns < *flagPerfMinRuns || time.Since(startTime) < *flagPerfDuration {
		_ = runFn()
		numRuns++
	}
	duration := time.Since(startTime)
	avgDurationPerRun := duration / time.Duration(numRuns)

	// GOps/s = NumOps / Duration(nanoseconds)
	// 1 GOp = 1e9 Ops
	// GOps/s = NumOps / (DurationSeconds * 1e9)
	gOpsPerSecond := float64(numOps) / avgDurationPerRun.Seconds() / 1e9
	return gOpsPerSecond
}

type testBufferPool[T any] struct {
	mu   sync.Mutex
	pool map[int][][]T
}

func newTestBufferPool[T any]() *testBufferPool[T] {
	return &testBufferPool[T]{
		pool: make(map[int][][]T),
	}
}

func (p *testBufferPool[T]) Alloc(size int) (any, []T) {
	p.mu.Lock()
	defer p.mu.Unlock()
	if buffers, ok := p.pool[size]; ok && len(buffers) > 0 {
		last := len(buffers) - 1
		buf := buffers[last]
		p.pool[size] = buffers[:last]
		return buf, buf
	}
	buf := make([]T, size)
	return buf, buf
}

func (p *testBufferPool[T]) Release(ref any) {
	p.mu.Lock()
	defer p.mu.Unlock()
	buf := ref.([]T)
	size := len(buf)
	p.pool[size] = append(p.pool[size], buf)
}
