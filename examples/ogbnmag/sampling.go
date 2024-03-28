package ogbnmag

import (
	"fmt"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"os"
	"path"
)

var (
	// BatchSize used for the sampler: the value was taken from the TF-GNN OGBN-MAG demo colab, and it was the
	// best found with some hyperparameter tuning. It does lead to using almost 7Gb of the GPU ram ...
	// but it works fine in an Nvidia RTX 2080 Ti (with 11Gb memory).
	BatchSize = 128

	// ReuseShareableKernels will share the kernels across similar messages in the strategy tree.
	// So the authors to papers messages will be the same if it comes from authors of the seed papers,
	// or of the coauthored-papers.
	// Default is true.
	ReuseShareableKernels = true

	// KeepDegrees will also make sampler keep the degrees of the edges as separate tensors.
	// These can be used by the GNN pooling functions to multiply the sum to the actual degree.
	KeepDegrees = true
)

// NewSampler will create a [sampler.Sampler] and configure it with the OGBN-MAG graph definition.
func NewSampler(baseDir string) (*sampler.Sampler, error) {
	baseDir = mldata.ReplaceTildeInDir(baseDir) // If baseDir starts with "~", it is replaced.
	samplerPath := path.Join(baseDir, DownloadSubdir, "sampler.bin")
	s, err := sampler.Load(samplerPath)
	if err == nil {
		return s, nil
	}
	if !os.IsNotExist(err) {
		return nil, err
	}

	fmt.Println("> Creating a new Sampler for OGBN-MAG")
	s = sampler.New()
	s.AddNodeType("papers", NumPapers)
	s.AddNodeType("authors", NumAuthors)
	s.AddNodeType("institutions", NumInstitutions)
	s.AddNodeType("fields_of_study", NumFieldsOfStudy)

	s.AddEdgeType("writes", "authors", "papers", EdgesWrites /* reverse= */, false)
	s.AddEdgeType("writtenBy", "authors", "papers", EdgesWrites /* reverse= */, true)
	s.AddEdgeType("cites", "papers", "papers", EdgesCites /*reverse=*/, false)
	s.AddEdgeType("citedBy", "papers", "papers", EdgesCites /*reverse=*/, true)
	s.AddEdgeType("affiliatedWith", "authors", "institutions", EdgesAffiliatedWith /*reverse=*/, false)
	s.AddEdgeType("affiliations", "authors", "institutions", EdgesAffiliatedWith /*reverse=*/, true)
	s.AddEdgeType("hasTopic", "papers", "fields_of_study", EdgesHasTopic /*reverse=*/, false)
	s.AddEdgeType("topicHasPapers", "papers", "fields_of_study", EdgesHasTopic /*reverse=*/, true)
	if err := s.Save(samplerPath); err != nil {
		return nil, err
	}
	return s, nil
}

// MagStrategy takes a sampler created by [ogbnmag.NewSampler], a desired batch size, and the set of
// seed ids to sample from ([ogbnmag.TrainSplit], [ogbnmag.ValidSplit] or [ogbnmag.TestSplit]) and
// returns a sampling strategy, that can be used to create datasets.
func MagStrategy(magSampler *sampler.Sampler, batchSize int, seedIdsCandidates tensor.Tensor) *sampler.Strategy {
	strategy := magSampler.NewStrategy()
	strategy.KeepDegrees = KeepDegrees
	var seeds *sampler.Rule
	if seedIdsCandidates == nil {
		seeds = strategy.Nodes("seeds", "papers", batchSize)
	} else {
		seedIdsData := seedIdsCandidates.Local().FlatCopy().([]int32)
		seeds = strategy.NodesFromSet("seeds", "papers", batchSize, seedIdsData)
	}
	seedsBase := seeds.IdentitySubRule("seedsBase")
	citations := seeds.FromEdges("citations", "cites", 8)
	if ReuseShareableKernels {
		citations.UpdateKernelScopeName = seedsBase.UpdateKernelScopeName
	}

	// Authors
	const authorsCount = 8
	seedsAuthors := seedsBase.FromEdges("seedsAuthors", "writtenBy", authorsCount)
	citationsAuthors := citations.FromEdges("citationsAuthors", "writtenBy", authorsCount)
	if ReuseShareableKernels {
		citationsAuthors.WithKernelScopeName(seedsAuthors.ConvKernelScopeName)
	}

	// Other papers by authors.
	papersByAuthors := seedsAuthors.FromEdges("papersByAuthors", "writes", 8)
	papersByCitationAuthors := citationsAuthors.FromEdges("papersByCitationAuthors", "writes", 8)
	if ReuseShareableKernels {
		papersByCitationAuthors.WithKernelScopeName(papersByAuthors.ConvKernelScopeName)
	}

	// Affiliations
	authorsInstitutions := seedsAuthors.FromEdges("authorsInstitutions", "affiliatedWith", 8)
	citationAuthorsInstitutions := citationsAuthors.FromEdges("citationAuthorsInstitutions", "affiliatedWith", 8)
	if ReuseShareableKernels {
		citationAuthorsInstitutions.WithKernelScopeName(authorsInstitutions.ConvKernelScopeName)
	}

	// Topics
	const topicsCount = 8
	seedsTopics := seedsBase.FromEdges("seedsTopics", "hasTopic", topicsCount)
	papersByAuthorsTopics := papersByAuthors.FromEdges("papersByAuthorsTopics", "hasTopic", topicsCount)
	citationsTopics := citations.FromEdges("citationsTopics", "hasTopic", topicsCount)
	papersByCitationAuthorsTopics := papersByCitationAuthors.FromEdges("papersByCitationAuthorsTopics", "hasTopic", topicsCount)
	if ReuseShareableKernels {
		papersByAuthorsTopics.WithKernelScopeName(seedsTopics.ConvKernelScopeName)
		citationsTopics.WithKernelScopeName(seedsTopics.ConvKernelScopeName)
		papersByCitationAuthorsTopics.WithKernelScopeName(seedsTopics.ConvKernelScopeName)
	}
	return strategy
}

// magCreateLabels create the labels from the input seed indices.
// It returns the same inputs and the extracted labels (with mask).
func magCreateLabels(inputs, labels []tensor.Tensor) ([]tensor.Tensor, []tensor.Tensor) {
	seedsRef := inputs[0].Local().AcquireData()
	defer seedsRef.Release()
	seedsData := seedsRef.Flat().([]int32)
	seedsMask := inputs[1]

	seedsLabels := tensor.FromShape(shapes.Make(inputs[0].DType(), inputs[0].Shape().Size(), 1))
	labelsRef := seedsLabels.AcquireData()
	defer labelsRef.Release()
	labelsData := labelsRef.Flat().([]int32)

	papersLabelsRef := PapersLabels.Local().AcquireData()
	defer papersLabelsRef.Release()
	papersLabelData := papersLabelsRef.Flat().([]int32)

	for ii, paperIdx := range seedsData {
		labelsData[ii] = papersLabelData[paperIdx]
	}
	return inputs, []tensor.Tensor{seedsLabels, seedsMask}
}

// WithReplacement indicates whether the training dataset is created with replacement.
var WithReplacement = false

// MakeDatasets takes a directory where to store the downloaded data and return 4 datasets:
// "train", "trainEval", "validEval", "testEval".
//
// It uses the package `ogbnmag` to download the data.
func MakeDatasets(dataDir string) (trainDS, trainEvalDS, validEvalDS, testEvalDS train.Dataset, err error) {
	if err = Download(dataDir); err != nil {
		return
	}
	magSampler, err := NewSampler(dataDir)
	if err != nil {
		return
	}
	trainStrategy := MagStrategy(magSampler, BatchSize, TrainSplit)
	validStrategy := MagStrategy(magSampler, BatchSize, ValidSplit)
	testStrategy := MagStrategy(magSampler, BatchSize, TestSplit)

	trainDS = trainStrategy.NewDataset("train").Infinite().Shuffle()
	if WithReplacement {
		trainDS = trainStrategy.NewDataset("train").Infinite().Shuffle().WithReplacement()
	} else {
		trainDS = trainStrategy.NewDataset("train").Infinite().Shuffle()
	}
	trainEvalDS = trainStrategy.NewDataset("train").Epochs(1)
	validEvalDS = validStrategy.NewDataset("valid").Epochs(1)
	testEvalDS = testStrategy.NewDataset("test").Epochs(1)

	// We want to transform the dataset in 3 ways:
	// - Gather the labels
	// - Parallelize its generation: greatly speeds it up.
	// - Free GPU memory in between each use, since each batch may use lots of GPU memory.
	perDatasetFn := func(ds train.Dataset) train.Dataset {
		ds = mldata.Map(ds, magCreateLabels)
		ds = mldata.Parallel(ds)
		ds = mldata.Freeing(ds)
		return ds
	}

	trainDS = perDatasetFn(trainDS)
	trainEvalDS = perDatasetFn(trainEvalDS)
	validEvalDS = perDatasetFn(validEvalDS)
	testEvalDS = perDatasetFn(testEvalDS)
	return
}
