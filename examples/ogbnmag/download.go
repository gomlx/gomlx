// Package ogbnmag provides `Download` method for the corresponding dataset, and some dataset tools
//
// See https://ogb.stanford.edu/ for all Open Graph Benchmark (OGB) datasets.
// See https://ogb.stanford.edu/docs/nodeprop/#ogbn-mag for the `ogbn-mag` dataset description.
//
// The task is to predict the venue of publication of a paper, given it's relations.
package ogbnmag

import (
	"fmt"
	mldata "github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"os"
	"path"
	"strconv"
)

var (
	ZipURL         = "http://snap.stanford.edu/ogb/data/nodeproppred/mag.zip"
	ZipFile        = "mag.zip"
	ZipChecksum    = "2afe62ead87f2c301a7398796991d347db85b2d01c5442c95169372bf5a9fca4"
	DownloadSubdir = "downloads"
)

var (
	NumPapers       = 736389
	NumAuthors      = 1134649
	NumInstitutions = 8740
	NumFieldOfStudy = 59965

	// NumLabels is the number of labels for the papers. These correspond to publication venues.
	NumLabels = 349

	// PaperEmbeddingsSize is the size of the node features given.
	PaperEmbeddingsSize = 128

	// PapersEmbeddings contains the embeddings, shaped `(Float32)[NumPapers, PaperEmbeddingsSize]`
	PapersEmbeddings tensor.Tensor

	// PapersYears for each paper, where year starts in 2000 (so 10 corresponds to 2010). Shaped `(Int16)[NumPapers, 1]`.
	PapersYears tensor.Tensor

	// PapersLabels for each paper, values from 0 to 348 (so 349 in total). Shaped `(Int16)[NumPapers, 1]`.
	PapersLabels tensor.Tensor

	// TrainSplit, TestSplit, ValidationSplit splits of the data.
	// These are indices to papers, values from `[0, NumPapers-1]`. Shaped `(Int32)[n, 1]
	// They have 629571, 41939 and 64879 elements each.
	TrainSplit, TestSplit, ValidationSplit tensor.Tensor
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
	baseDir = mldata.ReplaceTildeInDir(baseDir) // If baseDir starts with "~", it is replaced.
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

	return nil
}

// downloadZip downloads, uncompress and then removes the Zip file.
// Run this only if tensor files are not available.
func downloadZip(downloadDir string) error {
	if err := os.MkdirAll(downloadDir, 0777); err != nil && !os.IsExist(err) {
		return errors.Wrapf(err, "Failed to create path for downloading %q", downloadDir)
	}

	zipPath := path.Join(downloadDir, ZipFile)
	err := mldata.DownloadAndUnzipIfMissing(ZipURL, zipPath, downloadDir, path.Join(downloadDir, "mag"), ZipChecksum)
	if err != nil {
		return err
	}
	if mldata.FileExists(zipPath) {
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
	t, err := parseNumbersFromCSV(papersFeaturesCSVPath, papersFeaturesPath, NumPapers, PaperEmbeddingsSize, parseFloat32)
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
	t, err = parseNumbersFromCSV(papersLabelsCSVPath, papersLabelsPath, NumPapers, 1, parseYear)
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
	splitsStore    = []*tensor.Tensor{&TrainSplit, &TestSplit, &ValidationSplit}
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
func parseNumbersFromCSV[E shapes.NumberNotComplex](inputFilePath, outputFilePath string, numRows, numCols int, parseNumberFn func(string) (E, error)) (*tensor.Local, error) {
	var tensorOut *tensor.Local
	var err error
	if outputFilePath != "" && mldata.FileExists(outputFilePath) {
		tensorOut, err = tensor.Load(outputFilePath)
		if err == nil {
			// Read from pre-saved tensor.
			return tensorOut, nil
		}
		return nil, errors.WithMessagef(err, "failed to load output file %q, you many need to remove so it can be regenerated", outputFilePath)
	}
	fmt.Printf("Parsing %d rows from %q\n", numRows, inputFilePath)
	tensorOut = tensor.FromShape(shapes.Make(shapes.DTypeGeneric[E](), numRows, numCols))
	dataRef := tensorOut.Local().AcquireData()
	defer dataRef.Release()
	rawData := dataRef.Flat().([]E)
	rowNum, rawDataPos := 0, 0
	err = mldata.ParseGzipCSVFile(inputFilePath, func(row []string) error {
		if len(row) != numCols {
			return errors.Errorf("line %d has %d columns, we expected %q rows to have %d columns", rowNum+1, len(row), inputFilePath, numCols)
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
	if err != nil {
		return nil, err
	}
	if rowNum != numRows {
		return nil, errors.Errorf("found %d rows in %1q, was expecting %d -- did file change ?", rowNum, inputFilePath, numRows)
	}
	if outputFilePath != "" {
		fmt.Printf("> saving results to %q for faster access\n", outputFilePath)
		err = tensorOut.Save(outputFilePath)
		if err != nil {
			return nil, errors.WithMessagef(err, "parsed %q, but failed to save it to %q", inputFilePath, outputFilePath)
		}
	}
	return tensorOut, nil
}