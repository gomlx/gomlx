package gnn

import (
	mag "github.com/gomlx/gomlx/examples/ogbnmag"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
)

// MagStrategy takes a sampler created by [ogbnmag.NewSampler], a desired batch size, and the set of
// seed ids to sample from ([ogbnmag.TrainSplit], [ogbnmag.ValidSplit] or [ogbnmag.TestSplit]) and
// returns a sampling strategy, that can be used to create datasets.
func MagStrategy(magSampler *sampler.Sampler, batchSize int, seedIdsCandidates tensor.Tensor) *sampler.Strategy {
	strategy := magSampler.NewStrategy()
	var seeds *sampler.Rule
	if seedIdsCandidates == nil {
		seeds = strategy.Nodes("seeds", "papers", batchSize)
	} else {
		seedIdsData := seedIdsCandidates.Local().FlatCopy().([]int32)
		seeds = strategy.NodesFromSet("seeds", "papers", batchSize, seedIdsData)
	}
	citations := seeds.FromEdges("citations", "cites", 8)

	// Authors
	seedsAuthors := seeds.FromEdges("seedsAuthors", "writtenBy", 8)
	citationsAuthors := citations.FromEdges("citationsAuthors", "writtenBy", 8)

	// Co-authored papers
	coauthoredPapers := seedsAuthors.FromEdges("coauthoredPapers", "writes", 8)
	coauthoredFromCitations := citationsAuthors.FromEdges("coauthoredFromCitations", "writes", 8)

	// Affiliations
	_ = seedsAuthors.FromEdges("authorsInstitutions", "affiliatedWith", 8)
	_ = citationsAuthors.FromEdges("citationAuthorsInstitutions", "affiliatedWith", 8)

	// Topics
	_ = seeds.FromEdges("seedsTopics", "hasTopic", 8)
	_ = coauthoredPapers.FromEdges("coauthoredTopics", "hasTopic", 8)
	_ = citations.FromEdges("citationsTopics", "hasTopic", 8)
	_ = coauthoredFromCitations.FromEdges("coauthoredFromCitationsTopics", "hasTopic", 8)

	return strategy
}

// BatchSize used for the sampler.
var BatchSize = 32

// magCreateLabels create the labels from the input seed indices.
func magCreateLabels(inputs, labels []tensor.Tensor) ([]tensor.Tensor, []tensor.Tensor) {
	seedsRef := inputs[0].Local().AcquireData()
	defer seedsRef.Release()
	seedsData := seedsRef.Flat().([]int32)

	seedsLabels := tensor.FromShape(shapes.Make(inputs[0].DType(), inputs[0].Shape().Size(), 1))
	labelsRef := seedsLabels.AcquireData()
	defer labelsRef.Release()
	labelsData := labelsRef.Flat().([]int32)

	papersLabelsRef := mag.PapersLabels.Local().AcquireData()
	defer papersLabelsRef.Release()
	papersLabelData := papersLabelsRef.Flat().([]int32)

	for ii, paperIdx := range seedsData {
		labelsData[ii] = papersLabelData[paperIdx]
	}
	return inputs, []tensor.Tensor{seedsLabels}
}

// MakeDatasets takes a directory where to store the downloaded data and return 4 datasets:
// "train", "trainEval", "validEval", "testEval".
//
// It uses the package `ogbnmag` to download the data.
func MakeDatasets(dataDir string) (trainDS, trainEvalDS, validEvalDS, testEvalDS train.Dataset, err error) {
	if err = mag.Download(dataDir); err != nil {
		return
	}
	magSampler, err := mag.NewSampler(dataDir)
	if err != nil {
		return
	}
	trainStrategy := MagStrategy(magSampler, BatchSize, mag.TrainSplit)
	validStrategy := MagStrategy(magSampler, BatchSize, mag.ValidSplit)
	testStrategy := MagStrategy(magSampler, BatchSize, mag.TestSplit)

	trainDS = trainStrategy.NewDataset("train").Infinite().Shuffle()
	trainEvalDS = trainStrategy.NewDataset("train").Epochs(1)
	validEvalDS = validStrategy.NewDataset("valid").Epochs(1)
	testEvalDS = testStrategy.NewDataset("test").Epochs(1)

	trainDS = mldata.Parallel(mldata.MapInHost(trainDS, magCreateLabels, ""))
	trainEvalDS = mldata.Parallel(mldata.MapInHost(trainEvalDS, magCreateLabels, ""))
	validEvalDS = mldata.Parallel(mldata.MapInHost(validEvalDS, magCreateLabels, ""))
	testEvalDS = mldata.Parallel(mldata.MapInHost(testEvalDS, magCreateLabels, ""))
	return
}