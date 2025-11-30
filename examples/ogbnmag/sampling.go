package ogbnmag

import (
	"fmt"
	"os"
	"path"

	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	mldata "github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
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

	// IdentitySubSeeds controls whether to use an IdentitySubSeed, to allow more sharing of the kernel.
	IdentitySubSeeds = true
)

// NewSampler will create a [sampler.Sampler] and configure it with the OGBN-MAG graph definition.
//
// Usually, one will want to use the [NewSamplerStrategy] instead, which will calls this. Call this instead if
// crafting a custom sampling strategy.
//
// `baseDir` is used to store a cached sampler called `sampler.bin` for faster startup.
// If empty, it will force re-creating the sampler.
func NewSampler(baseDir string) (*sampler.Sampler, error) {
	var samplerPath string
	if baseDir != "" {
		baseDir = fsutil.MustReplaceTildeInDir(baseDir) // If baseDir starts with "~", it is replaced.
		samplerPath = path.Join(baseDir, DownloadSubdir, "sampler.bin")
		s, err := sampler.Load(samplerPath)
		if err == nil {
			return s, nil
		}
		if !os.IsNotExist(err) {
			return nil, err
		}
	}
	fmt.Println("> Creating a new Sampler for OGBN-MAG")
	s := sampler.New()
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
	if samplerPath != "" {
		err := s.Save(samplerPath)
		if err != nil {
			return nil, err
		}
	}
	return s, nil
}

// NewSamplerStrategy creates a sampling strategy given the sampler, batch size and seeds candidates to sample from.
//
// Args:
// . [magSampler] should have been created with [ogbnmag.NewSampler]
// . [batchSize] is the number of seed nodes ("Papers") to sample.
// . [seedIdsCandidates] is the seed of seed nodes to sample from, typically [ogbnmag.TrainSplit], [ogbnmag.ValidSplit] or [ogbnmag.TestSplit]. If empty it will sample from all possible papers.
//
// It returns a [sampler.Strategy] for OGBN-MAG.
func NewSamplerStrategy(
	magSampler *sampler.Sampler,
	batchSize int,
	seedIdsCandidates *tensors.Tensor,
) (strategy *sampler.Strategy) {
	strategy = magSampler.NewStrategy()
	strategy.KeepDegrees = KeepDegrees
	var seeds *sampler.Rule
	if seedIdsCandidates == nil {
		seeds = strategy.Nodes("seeds", "papers", batchSize)
	} else {
		seedIdsData := tensors.MustCopyFlatData[int32](seedIdsCandidates)
		seeds = strategy.NodesFromSet("seeds", "papers", batchSize, seedIdsData)
	}
	citations := seeds.FromEdges("citations", "cites", 8)

	var seedsBase *sampler.Rule
	if IdentitySubSeeds {
		seedsBase = seeds.IdentitySubRule("seedsBase")
		if ReuseShareableKernels {
			citations.UpdateKernelScopeName = seedsBase.UpdateKernelScopeName
		}
	} else {
		seedsBase = seeds
	}

	const defaultSamplingCount = 8

	// Authors
	const authorsCount = 8
	seedsAuthors := seedsBase.FromEdges("seedsAuthors", "writtenBy", authorsCount)
	citationsAuthors := citations.FromEdges("citationsAuthors", "writtenBy", authorsCount)
	if ReuseShareableKernels {
		citationsAuthors.WithKernelScopeName(seedsAuthors.ConvKernelScopeName)
	}

	// Other papers by authors.
	papersByAuthors := seedsAuthors.FromEdges("papersByAuthors", "writes", defaultSamplingCount)
	papersByCitationAuthors := citationsAuthors.FromEdges("papersByCitationAuthors", "writes", defaultSamplingCount)
	if ReuseShareableKernels {
		papersByCitationAuthors.WithKernelScopeName(papersByAuthors.ConvKernelScopeName)
	}

	// Affiliations
	authorsInstitutions := seedsAuthors.FromEdges("authorsInstitutions", "affiliatedWith", defaultSamplingCount)
	citationAuthorsInstitutions := citationsAuthors.FromEdges(
		"citationAuthorsInstitutions",
		"affiliatedWith",
		defaultSamplingCount,
	)
	if ReuseShareableKernels {
		citationAuthorsInstitutions.WithKernelScopeName(authorsInstitutions.ConvKernelScopeName)
	}

	// Topics
	const topicsCount = 8
	seedsTopics := seedsBase.FromEdges("seedsTopics", "hasTopic", topicsCount)
	papersByAuthorsTopics := papersByAuthors.FromEdges("papersByAuthorsTopics", "hasTopic", topicsCount)
	citationsTopics := citations.FromEdges("citationsTopics", "hasTopic", topicsCount)
	papersByCitationAuthorsTopics := papersByCitationAuthors.FromEdges(
		"papersByCitationAuthorsTopics",
		"hasTopic",
		topicsCount,
	)
	if ReuseShareableKernels {
		papersByAuthorsTopics.WithKernelScopeName(seedsTopics.ConvKernelScopeName)
		citationsTopics.WithKernelScopeName(seedsTopics.ConvKernelScopeName)
		papersByCitationAuthorsTopics.WithKernelScopeName(seedsTopics.ConvKernelScopeName)
	}
	return strategy
}

// ExtractLabelsFromInput create the labels from the input seed indices.
// It returns the same inputs and the extracted labels (with mask).
func ExtractLabelsFromInput(inputs, labels []*tensors.Tensor) ([]*tensors.Tensor, []*tensors.Tensor) {
	seeds := inputs[0]
	seedsMask := inputs[1]
	seedsLabels := tensors.FromShape(shapes.Make(seeds.DType(), seeds.Shape().Size(), 1))
	tensors.MustConstFlatData[int32](seeds, func(seedsData []int32) {
		tensors.MustConstFlatData[int32](PapersLabels, func(papersLabelsData []int32) {
			tensors.MustMutableFlatData[int32](seedsLabels, func(labelsData []int32) {
				for ii, paperIdx := range seedsData {
					labelsData[ii] = papersLabelsData[paperIdx]
				}
			})
		})
	})
	return inputs, []*tensors.Tensor{seedsLabels, seedsMask}
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
	trainStrategy := NewSamplerStrategy(magSampler, BatchSize, TrainSplit)
	validStrategy := NewSamplerStrategy(magSampler, BatchSize, ValidSplit)
	testStrategy := NewSamplerStrategy(magSampler, BatchSize, TestSplit)

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
		ds = mldata.Map(ds, ExtractLabelsFromInput)
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
