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
	NumPapers           = 736389 // Number of papers read, 736389
	PaperEmbeddingsSize = 128

	// PapersEmbeddings contains the embeddings, shaped `(Float32)[NumPapers, PaperEmbeddingsSize]`
	PapersEmbeddings tensor.Tensor

	// PapersYears for each paper, where year starts in 2000 (so 10 corresponds to 2010). Shaped `(int16)[NumPapers, 1]`.
	PapersYears tensor.Tensor
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
	if err := downloadZip(baseDir); err != nil {
		return err
	}

	if err := parsePapersFromCSV(baseDir); err != nil {
		return err
	}

	return nil
}

// downloadZip downloads, uncompress and then removes the Zip file.
// Run this only if tensor files are not available.
func downloadZip(baseDir string) error {
	downloadPath := path.Join(baseDir, DownloadSubdir)
	if err := os.MkdirAll(downloadPath, 0777); err != nil && !os.IsExist(err) {
		return errors.Wrapf(err, "Failed to create path for downloading %q", downloadPath)
	}

	zipPath := path.Join(downloadPath, ZipFile)
	err := mldata.DownloadAndUnzipIfMissing(ZipURL, zipPath, downloadPath, path.Join(downloadPath, "mag"), ZipChecksum)
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
)

func parsePapersFromCSV(baseDir string) error {
	papersFeaturesCSVPath := path.Join(baseDir, DownloadSubdir, papersEmbeddingsCSVFile)
	papersFeaturesPath := path.Join(baseDir, papersEmbeddingsFile)
	t, err := parseNumbersFromCSV(papersFeaturesCSVPath, papersFeaturesPath, NumPapers, PaperEmbeddingsSize, parseFloat32)
	if err != nil {
		return err
	}
	PapersEmbeddings = t

	papersYearsCSVPath := path.Join(baseDir, DownloadSubdir, papersYearsCSVFile)
	papersYearsPath := path.Join(baseDir, papersYearsFile)
	t, err = parseNumbersFromCSV(papersYearsCSVPath, papersYearsPath, NumPapers, 1, parseYear)
	if err != nil {
		return err
	}
	PapersYears = t
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
	if rowNum != NumPapers {
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
