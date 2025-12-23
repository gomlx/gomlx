package metrics

import (
	"bufio"
	"fmt"
	"math/rand/v2"
	"os"
	"slices"
	"strconv"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/require"
)

func must0(err error) {
	if err != nil {
		panic(err)
	}
}

func must1[T any](t T, err error) T {
	if err != nil {
		panic(err)
	}
	return t
}

func TestStreamingMedian(t *testing.T) {
	// Create an asymmetric sequence with known medianT:
	metric := (&StreamingMedianMetric{}).WithSampleSize(10_000)
	metric.Reset(nil)

	t.Run("Random 1/r numbers", func(t *testing.T) {
		// Sample from 0.01 < r < 1.0 randomly (so medianT r is expected to be 0.99/2 = 0.495),
		// and then feed StreamingMedian values of 1/r (so medianT is expected to be 1/0.495 = 2.0202020...).
		const numExamples = 100_001 // 100_001
		values := make([]float64, 0, numExamples)
		for range numExamples {
			r := rand.Float64()*0.99 + 0.01
			r = 1 / r
			values = append(values, r)
			sample := tensors.FromScalar(r)
			metric.UpdateGo(sample)
			sample.MustFinalizeAll()
		}
		median := tensors.ToScalar[float64](metric.ReadGo())
		slices.Sort(values)
		want := values[numExamples/2]
		fmt.Printf("\tgot medianT=%.5g, wanted medianT=%.5g\n", median, want)
		require.InDelta(t, want, median, 0.1)
	})

	const samplesFileName = "median_test_dataset.txt"
	t.Run("From samples in "+samplesFileName, func(t *testing.T) {
		samplesFile := must1(os.Open(samplesFileName))
		defer samplesFile.Close()
		var samples []float64
		scanner := bufio.NewScanner(samplesFile)
		for scanner.Scan() {
			line := scanner.Text()
			samples = append(samples, must1(strconv.ParseFloat(line, 64)))
		}
		must0(scanner.Err())

		// Reset metric and then feed numbers read.
		metric.Reset(nil)
		for _, sample := range samples {
			sampleT := tensors.FromScalar(sample)
			metric.UpdateGo(sampleT)
			sampleT.MustFinalizeAll()
		}
		median := tensors.ToScalar[float64](metric.ReadGo())
		slices.Sort(samples)
		want := samples[len(samples)/2]
		fmt.Printf("\tgot medianT=%.5g, wanted medianT=%.5g\n", median, want)
		require.InDelta(t, want, median, 0.1)
	})

	metric = metric.WithSampleSize(100)
	metric.Reset(nil)
	t.Run("Consecutive numbers from 1 to 1000", func(t *testing.T) {
		// Sample from 0.01 < r < 1.0 randomly (so medianT r is expected to be 0.99/2 = 0.495),
		// and then feed StreamingMedian values of 1/r (so medianT is expected to be 1/0.495 = 2.0202020...).
		const numExamples = 1_001
		values := make([]float64, 0, numExamples)
		for ii := range numExamples {
			r := float64(ii)
			values = append(values, r)
			sample := tensors.FromScalar(r)
			metric.UpdateGo(sample)
			sample.MustFinalizeAll()
		}
		median := tensors.ToScalar[float64](metric.ReadGo())
		slices.Sort(values)
		want := values[numExamples/2]
		fmt.Printf("\tgot medianT=%.5g, wanted medianT=%.5g\n", median, want)
		require.InDelta(t, want, median, 200)
	})
}
