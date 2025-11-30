package ogbnmag

import (
	"fmt"
	"io"
	"testing"

	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	mldata "github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/stretchr/testify/require"
)

func TestDatasets(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping long-running test.")
	}
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	err := Download(*flagDataDir)
	require.NoError(t, err, "failed to download OGBN-MAG dataset")
	UploadOgbnMagVariables(backend, ctx) // Uploads the Papers frozen embedding table.

	_, trainDS, validDS, testDS, err := MakeDatasets(*flagDataDir)
	require.NoError(t, err, "failed to make datasets")

	// Checks that shuffling also doesn't loses elements.
	magSampler, err := NewSampler(*flagDataDir)
	trainStrategy := NewSamplerStrategy(magSampler, BatchSize, TrainSplit)
	shuffledTrainDS := mldata.Parallel(
		trainStrategy.NewDataset("train").Epochs(1).Shuffle())

	for _, testCase := range []struct {
		Name  string
		DS    train.Dataset
		Seeds *tensors.Tensor
	}{
		{"valid", validDS, ValidSplit},
		{"test", testDS, TestSplit},
		{"train", trainDS, TrainSplit},
		{"shuffled_train", shuffledTrainDS, TrainSplit},
	} {
		seedsInSplit := tensors.MustCopyFlatData[int32](testCase.Seeds)
		wanted := sets.Make[int32](len(seedsInSplit))
		seen := sets.Make[int32](len(seedsInSplit))
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
			graphSample, remaining := sampler.MapInputsToStates[*tensors.Tensor](strategy, inputs)
			require.Empty(t, remaining)
			seeds := tensors.MustCopyFlatData[int32](graphSample["seeds"].Value)
			mask := tensors.MustCopyFlatData[bool](graphSample["seeds"].Mask)
			for ii, idx := range seeds {
				if !mask[ii] {
					continue
				}
				if !wanted.Has(idx) {
					require.Truef(
						t,
						wanted.Has(idx),
						"Dataset %q yielded seed %d not originally provided as seed for this dataset, batchNum=%d",
						testCase.Name,
						idx,
						batchNum,
					)
				}
				if seen.Has(idx) {
					require.Falsef(
						t,
						seen.Has(idx),
						"Dataset %q yielded seed %d more than once in the same epoch, batchNum=%d",
						testCase.Name,
						idx,
						batchNum,
					)
				}
				seen.Insert(idx)
			}
			batchNum++
		}
		require.Equalf(t, len(wanted), len(seen), "Dataset %q yielded only %d seeds out of %d",
			testCase.Name, len(seen), len(wanted))
	}
}
