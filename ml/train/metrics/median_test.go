package metrics

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/gomlx/gomlx/types/tensors"
	"github.com/stretchr/testify/require"
)

func TestStreamingMedian(t *testing.T) {
	// Create an asymmetric sequence with known medianT:
	metric := &StreamingMedianMetric{}
	metric.Reset(nil)

	// Sample from 0.01 < r < 1.0 randomly (so medianT r is expected to be 0.99/2 = 0.495),
	// and then feed StreamingMedian values of 1/r (so medianT is expected to be 1/0.495 = 2.0202020...).
	const numExamples = 100_001
	var medianT *tensors.Tensor
	values := make([]float64, 0, numExamples)
	for range numExamples {
		r := rand.Float64()*0.99 + 0.01
		r = 1 / r
		//r := float64(ii%10 + 1)
		values = append(values, r)
		sample := tensors.FromScalar(r)
		if medianT != nil {
			medianT.FinalizeAll()
		}
		medianT = metric.UpdateGo(sample)
		sample.FinalizeAll()
	}
	median := tensors.ToScalar[float64](medianT)
	slices.Sort(values)
	want := values[numExamples/2]
	fmt.Printf("\tgot medianT=%.5g, wanted medianT=%.5g\n", median, want)
	require.InDelta(t, want, median, 0.01)
}
