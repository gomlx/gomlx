package ogbnmag

import (
	"fmt"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"os"
	"path"
)

func getLabelsGraph(indices, allLabels *Node) *Node {
	return Gather(allLabels, indices)
}

// PapersSeedDatasets returns the train, validation and test datasets (`data.InMemoryDataset`) with only the papers seed nodes.
// The datasets can be shuffled and batched as desired.
//
// The yielded values are papers indices, and the corresponding labels.
func PapersSeedDatasets(manager *Manager) (trainDS, validDS, testDS *mldata.InMemoryDataset, err error) {
	if PapersEmbeddings == nil {
		// Data is not loaded yet.
		err = errors.New("data is not loaded yet, please call ogbnmag.Download() first")
	}
	var trainLabels, validLabels, testLabels tensor.Tensor
	err = exceptions.TryCatch[error](func() {
		getLabels := NewExec(manager, getLabelsGraph)
		trainLabels = getLabels.Call(TrainSplit, PapersLabels)[0]
		validLabels = getLabels.Call(ValidSplit, PapersLabels)[0]
		testLabels = getLabels.Call(TestSplit, PapersLabels)[0]
	})
	if err != nil {
		return
	}
	trainDS, err = mldata.InMemoryFromData(manager, "seeds_train", []any{TrainSplit}, []any{trainLabels})
	if err != nil {
		return
	}
	validDS, err = mldata.InMemoryFromData(manager, "seeds_valid", []any{ValidSplit}, []any{validLabels})
	if err != nil {
		trainDS = nil
		return
	}
	testDS, err = mldata.InMemoryFromData(manager, "seeds_test", []any{TestSplit}, []any{testLabels})
	if err != nil {
		trainDS = nil
		validDS = nil
		return
	}
	return
}

var (
	//
	// OgbnMagVariables maps variable names to a reference to their values.
	// We keep a reference to the values because the actual values change during the call to `Download()`
	//
	// They will be stored under the "/ogbnmag" scope.
	OgbnMagVariables = map[string]*tensor.Tensor{
		"PapersEmbeddings":              &PapersEmbeddings,
		"PapersLabels":                  &PapersLabels,
		"EdgesAffiliatedWith":           &EdgesAffiliatedWith,
		"EdgesWrites":                   &EdgesWrites,
		"EdgesCites":                    &EdgesCites,
		"EdgesHasTopic":                 &EdgesHasTopic,
		"CountAuthorsAffiliations":      &CountAuthorsAffiliations,
		"CountInstitutionsAffiliations": &CountInstitutionsAffiliations,
		"CountPapersCites":              &CountPapersCites,
		"CountPapersIsCited":            &CountPapersIsCited,
		"CountPapersFieldsOfStudy":      &CountPapersFieldsOfStudy,
		"CountFieldsOfStudyPapers":      &CountFieldsOfStudyPapers,
		"CountAuthorsPapers":            &CountAuthorsPapers,
		"CountPapersAuthors":            &CountPapersAuthors,
	}

	// OgbnMagVariablesScope is the absolute scope where the dataset variables are stored.
	OgbnMagVariablesScope = "/ogbnmag"
)

// UploadOgbnMagVariables creates frozen variables with the various static tables of the OGBN-MAG dataset, so
// it can be used by models.
//
// They will be stored under the "ogbnmag" scope.
func UploadOgbnMagVariables(ctx *context.Context) *context.Context {
	ctxMag := ctx.InAbsPath(OgbnMagVariablesScope)
	for name, tPtr := range OgbnMagVariables {
		if *tPtr == nil {
			exceptions.Panicf("trying to upload OgbnMagVariables to context before calling Download()")
		}
		v := ctxMag.VariableWithValue(name, *tPtr)
		v.Trainable = false
	}
	return ctx
}

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
