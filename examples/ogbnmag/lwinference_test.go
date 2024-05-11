package ogbnmag

import (
	"fmt"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/require"
	"strings"
	"testing"
)

func findSmallestDegreeSubgraph(t *testing.T) int32 {
	if testing.Short() {
		t.Skipf("Skipping TestLayerWiseInference: it requires downloading OGBN-MAG data.")
		return 0
	}
	require.NoError(t, Download(*flagDataDir), "Download")
	magSampler, err := NewSampler(*flagDataDir)
	require.NoError(t, err, "NewSampler")
	const batchSize = 1
	minSeedId, minMaxDegree := int32(-1), int32(1000)
	for seedId := range int32(10000) {
		seedsIds := tensor.FromValue([]int32{seedId}) // We take only one seed for testing.
		strategy := NewSamplerStrategy(magSampler, batchSize, seedsIds)
		ds := strategy.NewDataset("lwinference_test")

		_, inputs, _, err := ds.Yield()
		require.NoError(t, err, "Dataset.Yield")
		nameToState, _ := sampler.MapInputsToStates(strategy, inputs)
		maxDegree := int32(-1)
		for name, state := range nameToState {
			if !strings.HasSuffix(name, ".degree") {
				continue
			}
			values := state.Value.Local().FlatCopy().([]int32)
			for _, v := range values {
				if v > maxDegree {
					maxDegree = v
				}
			}
		}
		if maxDegree < minMaxDegree {
			minMaxDegree = maxDegree
			minSeedId = seedId
		}
	}
	fmt.Printf("SeedId=%d, max degree is %d\n", minSeedId, minMaxDegree)
	return minSeedId
}

// TestLayerWiseInference uses [flagDataDir] to store downloaded data, which defaults to `~/work/ogbnmag` by
// default.
func TestLayerWiseInference(t *testing.T) {
	if testing.Short() {
		t.Skipf("Skipping TestLayerWiseInference: it requires downloading OGBN-MAG data.")
		return
	}
	const batchSize = 1
	// Paper id with the least amount of degrees in its subgraph.
	// seedId := findSmallestDegreeSubgraph(t)
	const seedId = 3162
}
