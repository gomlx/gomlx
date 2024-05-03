package ogbnmag

import (
	"fmt"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/require"
	"io"
	"testing"
)

func TestDatasets(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running test.")
	}
	manager := graphtest.BuildTestManager()
	ctx := context.NewContext(manager)
	err := Download(*flagDataDir)
	require.NoError(t, err, "failed to download OGBN-MAG dataset")
	UploadOgbnMagVariables(ctx) // Uploads the Papers frozen embedding table.

	_, trainDS, validDS, testDS, err := MakeDatasets(*flagDataDir)
	require.NoError(t, err, "failed to make datasets")

	// Checks that shuffling also doesn't loses elements.
	magSampler, err := NewSampler(*flagDataDir)
	trainStrategy := MagStrategy(magSampler, BatchSize, TrainSplit)
	shuffledTrainDS := mldata.Parallel(
		trainStrategy.NewDataset("train").Epochs(1).Shuffle())

	for _, testCase := range []struct {
		Name  string
		DS    train.Dataset
		Seeds tensor.Tensor
	}{
		{"valid", validDS, ValidSplit},
		{"test", testDS, TestSplit},
		{"train", trainDS, TrainSplit},
		{"shuffled_train", shuffledTrainDS, TrainSplit},
	} {
		seedsInSplit := testCase.Seeds.Local().FlatCopy().([]int32)
		wanted := types.MakeSet[int32](len(seedsInSplit))
		seen := types.MakeSet[int32](len(seedsInSplit))
		for _, idx := range seedsInSplit {
			if wanted.Has(idx) {
				require.Falsef(t, wanted.Has(idx), "Dataset %q split has index %d more than once!?",
					testCase.Name, idx)
			}
			wanted.Insert(idx)
		}
		fmt.Printf("Evaluating %q: %d seeds\n", testCase.Name, len(seedsInSplit))
		batchNum := 0
		for {
			spec, inputs, _, err := testCase.DS.Yield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err)
			strategy := spec.(*sampler.Strategy)
			graphSample, remaining := sampler.MapInputsToStates[tensor.Tensor](strategy, inputs)
			require.Empty(t, remaining)
			seeds := graphSample["seeds"].Value.Local().FlatCopy().([]int32)
			mask := graphSample["seeds"].Mask.Local().FlatCopy().([]bool)
			for ii, idx := range seeds {
				if !mask[ii] {
					continue
				}
				if !wanted.Has(idx) {
					require.Truef(t, wanted.Has(idx), "Dataset %q yielded seed %d not originally provided as seed for this dataset, batchNum=%d",
						testCase.Name, idx, batchNum)
				}
				if seen.Has(idx) {
					require.Falsef(t, seen.Has(idx), "Dataset %q yielded seed %d more than once in the same epoch, batchNum=%d",
						testCase.Name, idx, batchNum)
				}
				seen.Insert(idx)
			}
			batchNum++
		}
		require.Equalf(t, len(wanted), len(seen), "Dataset %q yielded only %d seeds out of %d",
			testCase.Name, len(seen), len(wanted))
	}
}
