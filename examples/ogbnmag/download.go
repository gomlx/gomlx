// Package ogbnmag provides `Download` method for the corresponding dataset, and some dataset tools
//
// See https://ogb.stanford.edu/ for all Open Graph Benchmark (OGB) datasets.
// See https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag for the `ogbn-mag` dataset description.
//
// The task is to predict the venue of publication of a paper, given its relations.
package ogbnmag

import (
	"fmt"
	"os"
	"path"
	"strconv"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/downloader"
	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	mldata "github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

var (
	ZipURL         = "http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip"
	ZipFile        = "mag.zip"
	ZipChecksum    = "2afe62ead87f2c301a7398796991d347db85b2d01c5442c95169372bf5a9fca4"
	DownloadSubdir = "downloads"
)

var (
	NumPapers        = 736389
	NumAuthors       = 1134649
	NumInstitutions  = 8740
	NumFieldsOfStudy = 59965

	// NumLabels is the number of labels for the papers. These correspond to publication venues.
	NumLabels = 349

	// PaperEmbeddingsSize is the size of the node features given.
	PaperEmbeddingsSize = 128

	// PapersEmbeddings contains the embeddings, shaped `(Float32)[NumPapers, PaperEmbeddingsSize]`
	PapersEmbeddings *tensors.Tensor

	// PapersYears for each paper, where year starts in 2000 (so 10 corresponds to 2010). Shaped `(Int16)[NumPapers, 1]`.
	PapersYears *tensors.Tensor

	// PapersLabels for each paper, values from 0 to 348 (so 349 in total). Shaped `(Int16)[NumPapers, 1]`.
	PapersLabels *tensors.Tensor

	// TrainSplit, ValidSplit, TestSplit  splits of the data.
	// These are indices to papers, values from `[0, NumPapers-1]`. Shaped `(Int32)[n, 1]
	// They have 629571, 41939 and 64879 elements each.
	TrainSplit, ValidSplit, TestSplit *tensors.Tensor

	// Various relations: their shape is always `(Int32)[num_edges, 2]`:

	// EdgesAffiliatedWith `(Int32)[1043998, 2]`, pairs with (author_id, institution_id).
	//
	// Thousands of institutions with only one affiliated author, and an exponential decreasing amount
	// of institutions with more affiliated authors, all the way to one institution that has 27K authors.
	//
	// Most authors are affiliated to 1 institution only, and an exponentially decreasing number affiliations up
	// to one author with 47 affiliations. ~300K authors with no affiliation.
	EdgesAffiliatedWith *tensors.Tensor

	// EdgesWrites `(Int32)[7145660, 2]`, pairs with (author_id, paper_id).
	//
	// Every author writes at least one paper, and every paper has at least one author.
	//
	// Most authors (~600K) wrote one paper, with a substantial tail with thousands of authors having written hundreds of
	// papers, and in the extreme one author wrote 1046 papers.
	//
	// Papers are written on average by 3 authors (140k papers), with a bell-curve distribution with a long
	// tail, with a dozen of papers written by thousands of authors (5050 authors in one case).
	EdgesWrites *tensors.Tensor

	// EdgesCites `(Int32)[5416271, 2]`, pairs with (paper_id, paper_id).
	//
	// ~120K papers don't cite anyone, 95K papers cite only one paper, and a long exponential decreasing tail,
	// in the extreme a paper cites 609 other papers.
	//
	// ~100K papers are never cited, 155K are cited once, and again a long exponential decreasing tail, in the extreme
	// one paper is cited by 4744 other papers.
	EdgesCites *tensors.Tensor

	// EdgesHasTopic `(Int32)[7505078, 2]`, pairs with (paper_id, topic_id).
	//
	// All papers have at least one "field of study" topic. Most (550K) papers have 12 or 13 topics. At most a paper has
	// 14 topics.
	//
	// All "fields of study" are associated to at least one topic. ~17K (out of ~60K) have only one paper associated.
	// ~50%+ topics have < 10 papers associated. Some ~30% have < 1000 papers associated. A handful have 10s of
	// thousands papers associated, and there is one topic that is associated to everyone.
	EdgesHasTopic *tensors.Tensor

	// Counts to the various edge types.
	// These are call shaped `(Int32)[NumElements, 1]` for each of their entities.
	CountAuthorsAffiliations, CountInstitutionsAffiliations *tensors.Tensor
	CountPapersCites, CountPapersIsCited                    *tensors.Tensor
	CountPapersFieldsOfStudy, CountFieldsOfStudyPapers      *tensors.Tensor
	CountAuthorsPapers, CountPapersAuthors                  *tensors.Tensor
)

// Download and prepares the tensors with the data into the `baseDir`.
//
// If files are already there, it's assumed they were correctly generated and nothing is done.
//
// The data files occupy ~415Mb, but to keep a copy of raw tensors (for faster start up), you'll need ~1Gb free disk.
func Download(baseDir string) error {
	if PapersEmbeddings != nil {
		// Already loaded.
		return nil
	}
	baseDir = fsutil.MustReplaceTildeInDir(baseDir) // If baseDir starts with "~", it is replaced.
	downloadDir := path.Join(baseDir, DownloadSubdir)

	if err := downloadZip(downloadDir); err != nil {
		return err
	}
	if err := parsePapersFromCSV(downloadDir); err != nil {
		return err
	}
	if err := parseSplitsFromCSV(downloadDir); err != nil {
		return err
	}
	if err := parseEdgesFromCSV(downloadDir); err != nil {
		return err
	}
	if err := allEdgesCount(downloadDir); err != nil {
		return err
	}

	return nil
}

// downloadZip downloads, uncompress and then removes the Zip file.
// RunWithMap this only if tensor files are not available.
func downloadZip(downloadDir string) error {
	if err := os.MkdirAll(downloadDir, 0777); err != nil && !os.IsExist(err) {
		return errors.Wrapf(err, "Failed to create path for downloading %q", downloadDir)
	}

	zipPath := path.Join(downloadDir, ZipFile)
	err := downloader.DownloadAndUnzipIfMissing(
		ZipURL,
		zipPath,
		downloadDir,
		path.Join(downloadDir, "mag"),
		ZipChecksum,
	)
	if err != nil {
		return err
	}
	if fsutil.MustFileExists(zipPath) {
		// Clean up zip file no longer needed, to save space.
		if err := os.Remove(zipPath); err != nil {
			return errors.Wrapf(err, "Failed to remove file %q", zipPath)
		}
	}
	return nil
}

const (
	papersEmbeddingsCSVFile = "mag/raw/node-feat/paper/node-feat.csv.gz"
	papersEmbeddingsFile    = "papers_embeddings.tensor"
	papersYearsCSVFile      = "mag/raw/node-feat/paper/node_year.csv.gz"
	papersYearsFile         = "papers_years.tensor"
	papersLabelsCSVFile     = "mag/raw/node-label/paper/node-label.csv.gz"
	papersLabelsFile        = "papers_labels.tensor"
)

func parsePapersFromCSV(downloadDir string) error {
	papersFeaturesCSVPath := path.Join(downloadDir, papersEmbeddingsCSVFile)
	papersFeaturesPath := path.Join(downloadDir, papersEmbeddingsFile)
	t, err := parseNumbersFromCSV(
		papersFeaturesCSVPath,
		papersFeaturesPath,
		NumPapers,
		PaperEmbeddingsSize,
		parseFloat32,
	)
	if err != nil {
		return err
	}
	PapersEmbeddings = t

	papersYearsCSVPath := path.Join(downloadDir, papersYearsCSVFile)
	papersYearsPath := path.Join(downloadDir, papersYearsFile)
	t, err = parseNumbersFromCSV(papersYearsCSVPath, papersYearsPath, NumPapers, 1, parseYear)
	if err != nil {
		return err
	}
	PapersYears = t

	papersLabelsCSVPath := path.Join(downloadDir, papersLabelsCSVFile)
	papersLabelsPath := path.Join(downloadDir, papersLabelsFile)
	t, err = parseNumbersFromCSV(papersLabelsCSVPath, papersLabelsPath, NumPapers, 1, parseInt32)
	if err != nil {
		return err
	}
	PapersLabels = t

	return nil
}

var (
	splitsCSVFiles = []string{"train.csv.gz", "test.csv.gz", "valid.csv.gz"}
	splitsNumRows  = []int{629571, 41939, 64879} // Total = NumPapers
	splitsFiles    = []string{"train", "test", "validation"}
	splitsStore    = []**tensors.Tensor{&TrainSplit, &TestSplit, &ValidSplit}
)

func parseSplitsFromCSV(downloadDir string) error {
	for ii, fileName := range splitsCSVFiles {
		splitCSVFilePath := path.Join(downloadDir, "mag/split/time/paper/", fileName)
		splitFile := path.Join(downloadDir, splitsFiles[ii]+"_split.tensor")
		t, err := parseNumbersFromCSV(splitCSVFilePath, splitFile, splitsNumRows[ii], 1, parseInt32)
		if err != nil {
			return errors.WithMessagef(err, "while parsing split file for %s", splitsFiles[ii])
		}
		*(splitsStore[ii]) = t
	}
	return nil
}

var (
	edgesCSVDirs = []string{"author___affiliated_with___institution",
		"author___writes___paper", "paper___cites___paper", "paper___has_topic___field_of_study"}
	edgesNumRows = []int{1043998, 7145660, 5416271, 7505078}
	edgesFiles   = []string{"affiliated_with", "writes", "cites", "has_topic"}
	edgesStore   = []**tensors.Tensor{&EdgesAffiliatedWith, &EdgesWrites, &EdgesCites, &EdgesHasTopic}
)

func parseEdgesFromCSV(downloadDir string) error {
	for ii, dirName := range edgesCSVDirs {
		edgesCSVFilePath := path.Join(downloadDir, "mag/raw/relations/", dirName, "edge.csv.gz")
		edgesFile := path.Join(downloadDir, "edges_"+edgesFiles[ii]+".tensor")
		t, err := parseNumbersFromCSV(edgesCSVFilePath, edgesFile, edgesNumRows[ii], 2, parseInt32)
		if err != nil {
			return errors.WithMessagef(err, "while parsing edges file for %s", edgesFiles[ii])
		}
		*(edgesStore[ii]) = t
	}
	return nil
}

func parseFloat32(str string) (float32, error) {
	v, err := strconv.ParseFloat(str, 32)
	if err != nil {
		return 0, err
	}
	return float32(v), nil
}

func parseYear(str string) (value uint8, err error) {
	var v int64
	v, err = strconv.ParseInt(str, 10, 64)
	if err != nil {
		return 0, errors.Wrapf(err, "Failed to parse %q to %T", str, value)
	}
	return uint8(v - 2000), nil
}

func parseInt32(str string) (int32, error) {
	v, err := strconv.ParseFloat(str, 32)
	if err != nil {
		return 0, err
	}
	return int32(v), nil
}

// parseNumbersFromCSV returns the numbers in a CSV file as a tensor.
func parseNumbersFromCSV[E dtypes.NumberNotComplex](
	inputFilePath, outputFilePath string,
	numRows, numCols int,
	parseNumberFn func(string) (E, error),
) (*tensors.Tensor, error) {
	var tensorOut *tensors.Tensor
	var err error
	if outputFilePath != "" && fsutil.MustFileExists(outputFilePath) {
		tensorOut, err = tensors.Load(outputFilePath)
		if err == nil {
			// Read from pre-saved tensor.
			return tensorOut, nil
		}
		return nil, errors.WithMessagef(
			err,
			"failed to load output file %q, you many need to remove so it can be regenerated",
			outputFilePath,
		)
	}
	fmt.Printf("Parsing %d rows from %q\n", numRows, inputFilePath)

	tensorOut = tensors.FromShape(shapes.Make(dtypes.FromGenericsType[E](), numRows, numCols))
	rowNum, rawDataPos := 0, 0
	tensorOut.MustMutableFlatData(func(flatAny any) {
		rawData := flatAny.([]E)
		err = downloader.ParseGzipCSVFile(inputFilePath, func(row []string) error {
			if len(row) != numCols {
				return errors.Errorf(
					"line %d has %d columns, we expected %q rows to have %d columns",
					rowNum+1,
					len(row),
					inputFilePath,
					numCols,
				)
			}
			if rowNum >= numRows {
				// Keep counting rows, but drop result.
				rowNum++
				return nil
			}
			for ii, cell := range row {
				value, err := parseNumberFn(cell)
				if err != nil {
					return errors.WithMessagef(err, "failed to parse data from row=%d, col=%d: %q", rowNum, ii, cell)
				}
				rawData[rawDataPos] = value
				rawDataPos++
			}
			rowNum++
			return nil
		})
	})
	if err != nil {
		return nil, err
	}
	if rowNum != numRows {
		return nil, errors.Errorf(
			"found %d rows in %1q, was expecting %d -- did file change ?",
			rowNum,
			inputFilePath,
			numRows,
		)
	}
	if outputFilePath != "" {
		fmt.Printf("> saving results to %q for faster access\n", outputFilePath)
		err = tensorOut.Save(outputFilePath)
		if err != nil {
			return nil, errors.WithMessagef(
				err,
				"parsed %q, but failed to save it to %q",
				inputFilePath,
				outputFilePath,
			)
		}
	}
	return tensorOut, nil
}

var (
	countsFileNames = []string{
		"count_authors_affiliations", "count_institutions_affiliations",
		"count_papers_cites", "count_papers_is_cited",
		"count_papers_fields_of_study", "count_fields_of_study_papers",
		"count_authors_papers", "count_papers_authors",
	}
)

func allEdgesCount(downloadDir string) error {
	var store = []**tensors.Tensor{
		&CountAuthorsAffiliations, &CountInstitutionsAffiliations,
		&CountPapersCites, &CountPapersIsCited,
		&CountPapersFieldsOfStudy, &CountFieldsOfStudyPapers,
		&CountAuthorsPapers, &CountPapersAuthors,
	}
	var numElements = []int{
		NumAuthors, NumInstitutions,
		NumPapers, NumPapers,
		NumPapers, NumFieldsOfStudy,
		NumAuthors, NumPapers,
	}
	idxTensor := 0
	for idxInput, input := range []*tensors.Tensor{EdgesAffiliatedWith, EdgesCites, EdgesHasTopic, EdgesWrites} {
		for column := 0; column < 2; column++ {
			outputFilePath := path.Join(downloadDir, countsFileNames[idxTensor]+".tensor")
			var counts *tensors.Tensor
			var err error
			if fsutil.MustFileExists(outputFilePath) {
				counts, err = tensors.Load(outputFilePath)
			} else {
				fmt.Printf("> Counting elements for edges %s[column %d]: %d entries\n", edgesFiles[idxInput], column, input.Shape().Dimensions[0])
				counts, err = edgesCount(input, column, numElements[idxTensor])
				if err == nil {
					err = counts.Save(outputFilePath)
				}
			}
			if err != nil {
				return errors.WithMessagef(
					err,
					"while counting elements for edges %s[column %d]",
					edgesFiles[idxInput],
					column,
				)
			}
			*(store[idxTensor]) = counts
			idxTensor++
		}
	}
	return nil
}

func edgesCount(input *tensors.Tensor, column, numElements int) (output *tensors.Tensor, err error) {
	if input.DType() != dtypes.Int32 || input.Rank() != 2 || input.Shape().Dimensions[1] != 2 {
		return nil, errors.Errorf("input shape is invalid, expected (Int32)[?, 2], got %s", input.Shape())
	}
	if column < 0 || column > 1 {
		return nil, errors.Errorf("column=%d given is invalid, only columsn 0 or 1 are valid", column)
	}
	if numElements <= 0 {
		return nil, errors.Errorf("invalid number of elements %d", numElements)
	}

	input.MustConstFlatData(func(flatAny any) {
		inputData := flatAny.([]int32)
		output = tensors.FromScalarAndDimensions(int32(0), numElements, 1)
		output.MustMutableFlatData(func(flatAny any) {
			outputData := flatAny.([]int32)
			numRows := input.Shape().Dimensions[0]
			for row := 0; row < numRows; row++ {
				idx := inputData[2*row+column]
				if idx < 0 || int(idx) > numElements {
					err = errors.Errorf(
						"In row=%d, col=%d, got index %d > numElements %d",
						row,
						column,
						idx,
						numElements,
					)
					return
				}
				outputData[idx]++
			}
		})
	})
	if err != nil {
		output = nil
		return
	}
	return
}

var (
	// OgbnMagVariablesRef maps variable names to a reference to their values.
	// We keep a reference to the values because the actual values change during the call to `Download()`
	//
	// They will be stored under the "/ogbnmag" scope.
	OgbnMagVariablesRef = map[string]**tensors.Tensor{
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
func UploadOgbnMagVariables(backend backends.Backend, ctx *context.Context) *context.Context {
	ctxMag := ctx.InAbsPath(OgbnMagVariablesScope)
	for name, tPtr := range OgbnMagVariablesRef {
		if *tPtr == nil {
			Panicf("trying to upload OgbnMagVariablesRef to context before calling Download()")
		}
		v := ctxMag.VariableWithValue(name, *tPtr)
		v.Trainable = false
	}
	convertPapersEmbeddings(backend, ctx) // Convert to the selected dtype.
	return ctx
}

// ExcludeOgbnMagVariablesFromSave marks the OGBN-MAG variables as not to be saved by the given `checkpoint`.
// Since they are read separately and are constant, no need to repeat them at every checkpoint.
func ExcludeOgbnMagVariablesFromSave(ctx *context.Context, checkpoint *checkpoints.Handler) {
	ctxMag := ctx.InAbsPath(OgbnMagVariablesScope)
	for name := range OgbnMagVariablesRef {
		v := ctxMag.GetVariable(name)
		if v == nil {
			Panicf("OGBN-MAG variable %q not found in context!?", name)
		}
		checkpoint.ExcludeVarsFromSaving(v)
	}
}

func getLabelsGraph(indices, allLabels *Node) *Node {
	return Gather(allLabels, indices, false)
}

// PapersSeedDatasets returns the train, validation and test datasets (`datasets.InMemoryDataset`) with only the papers seed nodes,
// to be used with FNN (Feedforward Neural Networks). See [MakeDataset] to make a dataset with sampled sub-graphs for
// GNNs.
//
// The datasets can be shuffled and batched as desired.
//
// The yielded values are papers indices, and the corresponding labels.
func PapersSeedDatasets(manager backends.Backend) (trainDS, validDS, testDS *mldata.InMemoryDataset, err error) {
	if PapersEmbeddings == nil {
		// Data is not loaded yet.
		err = errors.New("data is not loaded yet, please call ogbnmag.Download() first")
	}
	var trainLabels, validLabels, testLabels *tensors.Tensor
	err = TryCatch[error](func() {
		getLabels := MustNewExec(manager, getLabelsGraph)
		trainLabels = getLabels.MustExec(TrainSplit, PapersLabels)[0]
		validLabels = getLabels.MustExec(ValidSplit, PapersLabels)[0]
		testLabels = getLabels.MustExec(TestSplit, PapersLabels)[0]
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
