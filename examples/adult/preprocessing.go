/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package adult

import (
	"encoding/gob"
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/tensors"
	"math/rand"
	"os"
	"path"
	"regexp"
	"sort"
	"strings"

	"github.com/go-gota/gota/dataframe"
	"github.com/go-gota/gota/series"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/pkg/errors"
)

// This file contains all functions need for preparation of the Adult dataset,
// including vocabulary and quantiles building (preprocessing).

// Various URLs and file names for Adult-UCI dataset.
const (
	AdultDatasetDataURL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
	AdultDatasetDataFile = "adult.data"
	AdultDatasetTestURL  = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test"
	AdultDatasetTestFile = "adult.test"

	AdultDatasetDataCecksum = "5b00264637dbfec36bdeaab5676b0b309ff9eb788d63554ca0a249491c86603d"
	AdultDatasetTestCecksum = "a2a9044bc167a35b2361efbabec64e89d69ce82d9790d2980119aac5fd7e9c05"
)

// AssertNoError checks that err is nil, otherwise it `panic`s with `err`.
func AssertNoError(err error) {
	if err != nil {
		panic(err)
	}
}

// DownloadDataset downloads the Adult dataset files into `dir`. It `log.Fatal` if
// it fails. Verbosity files >= 1 will print what it's doing.
//
// If files are already downloaded it does nothing -- except if `force` is set to true.
func DownloadDataset(dir string, force bool, verbosity int) {
	if verbosity >= 1 {
		fmt.Printf("Downloading datasets:\n")
	}
	dataNames := []struct{ url, path, checkSum string }{
		{AdultDatasetDataURL, AdultDatasetDataFile, AdultDatasetDataCecksum},
		{AdultDatasetTestURL, AdultDatasetTestFile, AdultDatasetTestCecksum},
	}
	for _, urlAndPath := range dataNames {
		filePath := path.Join(dir, urlAndPath.path)
		if force {
			_, err := data.Download(urlAndPath.url, filePath, true)
			AssertNoError(err)
			AssertNoError(data.ValidateChecksum(filePath, urlAndPath.checkSum))
		} else {
			AssertNoError(data.DownloadIfMissing(urlAndPath.url, filePath, urlAndPath.checkSum))
		}
		if verbosity >= 1 {
			fmt.Printf("\t%s => %s\n", urlAndPath.url, filePath)
		}
	}
}

// Column names:
const (
	WeightCol         = "fnlwgt" // "Final Weight", see adult.names file.
	LabelCol          = "label"  // That is the target prediction column.
	EducationTypeCol  = "education"
	EducationYearsCol = "education-num"
)

// Label values:
const (
	LabelTrue  = ">50K"
	LabelFalse = "<=50K"
)

var (
	// AdultFieldNames in the dataset.
	AdultFieldNames = []string{
		"age", "workclass", WeightCol, EducationTypeCol, EducationYearsCol, "marital-status", "occupation", "relationship",
		"race", "sex", "capital-gain", "capital-loss", "hours-per-week", "native-country", LabelCol,
	}

	// AdultFieldTypes maps the field (column) name to its format.
	AdultFieldTypes = map[string]series.Type{
		"age":             series.Float,
		"workclass":       series.String,
		WeightCol:         series.Float,
		EducationTypeCol:  series.String,
		EducationYearsCol: series.Float,
		"marital-status":  series.String,
		"occupation":      series.String,
		"relationship":    series.String,
		"race":            series.String,
		"sex":             series.String,
		"capital-gain":    series.Float,
		"capital-loss":    series.Float,
		"hours-per-week":  series.Float,
		"native-country":  series.String,
		LabelCol:          series.String,
	}
)

var (
	// removeSpaceAroundCommas fixes the issues of space on Adult data.
	removeSpaceAroundCommas = regexp.MustCompile(`\s*,\s*`)

	// removeTailingDot fixes the issue with adult.test, where rows end in ".".
	removeTailingDot = regexp.MustCompile(`\.\n`)

	// junkLine fixes the issue that adult.test comes with a junk first line.
	junkLine = "|1x3 Cross validator\n"
)

// LoadDataFrame and returns a DataFrame. `path` is the name of the downloaded file.
//
// Considering using LoadAndPreprocessData instead.
func LoadDataFrame(path string) dataframe.DataFrame {
	contents, err := os.ReadFile(path)
	if err != nil {
		panic(err)
	}
	contentsStr := string(contents)
	contentsStr = removeSpaceAroundCommas.ReplaceAllString(contentsStr, ",")
	contentsStr = removeTailingDot.ReplaceAllString(contentsStr, "\n")
	if strings.HasPrefix(contentsStr, junkLine) {
		contentsStr = contentsStr[len(junkLine):]
	}
	df := dataframe.ReadCSV(strings.NewReader(contentsStr), dataframe.HasHeader(false),
		dataframe.Names(AdultFieldNames...), dataframe.WithTypes(AdultFieldTypes))
	return df
}

// LoadAndPreprocessData all in one function call for data preprocessing for the Adult Dataset.
// Information and data available in the global Data.
//
// Parameters:
// - dir: where to store downloaded files. By default if they are already downloaded they will be reused.
// - numQuantiles: number of quantiles to generate for the continuous datasets. They can be used for piecewise-linear calibration.
// - forceDownload: will download data from the internet even if already downloaded.
// - verbosity: set to a value >= 1 to print out what it's doing.
//
// The results are stored in the global variable `Flat`.
//
// It panics in case of error.
func LoadAndPreprocessData(dir string, numQuantiles int, forceDownload bool, verbosity int) {
	dir = data.ReplaceTildeInDir(dir) // "~/..." -> "${HOME}/..."
	if Data.VocabulariesFeatures != nil {
		// Already loaded, nothing to do.
		return
	}

	// Make sure the data directory exists.
	if !data.FileExists(dir) {
		AssertNoError(os.MkdirAll(dir, 0777))
	}

	// Check whether we have a previously binary version of the data already preprocessed.
	if forceDownload || !LoadBinaryData(dir, numQuantiles) {
		DownloadDataset(dir, forceDownload, verbosity)
		df := LoadDataFrame(path.Join(dir, AdultDatasetDataFile))
		PopulateVocabularies(df)
		PopulateQuantiles(df, numQuantiles)
		if verbosity >= 2 {
			// Print a sample of the unprocessed data.
			PrintFeatures(df)
		}
		Data.Train = ConvertDataFrameToRawData(df)

		// Load test data.
		df = LoadDataFrame(path.Join(dir, AdultDatasetTestFile))
		Data.Test = ConvertDataFrameToRawData(df)

		// For future faster start up.
		AssertNoError(SaveBinaryData(dir, numQuantiles))
	}

	// Print sample of the processed data.
	if verbosity >= 2 {
		PrintRawData(Data.Train)
	}

}

// FileExists returns whether the given file path exists.
func FileExists(path string) (bool, error) {
	_, err := os.Stat(path)
	if err == nil {
		return true, nil
	}
	if os.IsNotExist(err) {
		return false, nil
	}
	return false, err
}

// Unknown representation for string columns -- used in Adult dataset.
const Unknown = "?"

// QuantileTable holds the quantiles of a set of values.
type QuantileTable []float32

// Data holds all the data (train and test), and the required information
// collected statically (i.e., non machine learned) from the training dataset
// (we don't look at test to generate these).
//
// It is filled out by LoadAndPreprocessData.
var Data struct {
	// VocabulariesFeatures is a list of feature names for the vocabularies stored in Vocabularies.
	VocabulariesFeatures []string

	// Vocabularies is a list of maps of value to integer. The
	// special string "Unknown" is mapped to 0.
	Vocabularies []map[string]int

	// FeatureNameToVocabIdx maps a feature name to its vocabulary index.
	FeatureNameToVocabIdx map[string]int

	// QuantilesFeatures is the ordered list of numeric features for which we have quantiles.
	QuantilesFeatures []string

	// Quantiles for features listed in QuantilesFeatures
	Quantiles []QuantileTable

	// Train dataset.
	Train *RawData

	// Test dataset.
	Test *RawData
}

// BinaryFilePath returns the name used to store the preprocessed data in binary (fast) format.
//
// The `numQuantiles` is the only preprocessing parameter that affects the result. We use
// it as part of the filename to make sure we don't re-use data generated for different
// `numQuantiles`.
//
// Considering using LoadAndPreprocessData instead.
func BinaryFilePath(dir string, numQuantiles int) string {
	return path.Join(dir, fmt.Sprintf("adult_data-%d_quantiles.bin", numQuantiles))
}

// SaveBinaryData saves the global Data structure in binary format, for faster access.
//
// Considering using LoadAndPreprocessData instead.
func SaveBinaryData(dir string, numQuantiles int) (err error) {
	filePath := BinaryFilePath(dir, numQuantiles)
	var f *os.File
	f, err = os.Create(filePath)
	if err != nil {
		return errors.Wrapf(err, "failed to create %q", filePath)
	}

	defer func() {
		cErr := f.Close()
		if err == nil && cErr != nil {
			err = errors.Wrapf(err, "failed to close file %q after writing", filePath)
		}
	}()

	gobW := gob.NewEncoder(f)
	if err := gobW.Encode(&Data); err != nil {
		return errors.Wrapf(err, "failed to write data to %q", filePath)
	}
	return nil
}

// LoadBinaryData saves the global Data structure in binary format, for faster access.
// It returns true if data was available and loaded.
//
// Considering using LoadAndPreprocessData instead.
func LoadBinaryData(dir string, numQuantiles int) (found bool) {
	filePath := BinaryFilePath(dir, numQuantiles)
	if !data.FileExists(filePath) {
		return false
	}

	f, err := os.Open(filePath)
	if err != nil {
		panic(errors.Wrapf(err, "failed to open %q", filePath))
	}

	defer func() { AssertNoError(f.Close()) }()
	gobW := gob.NewDecoder(f)
	if err := gobW.Decode(&Data); err != nil {
		panic(errors.Wrapf(err, "failed to load data from %q", filePath))
	}
	return true
}

// PopulateVocabularies goes over all string columns in the DataFrame and map their
// value to integers starting from 0. Results in Vocabularies.
func PopulateVocabularies(df dataframe.DataFrame) {
	Data.FeatureNameToVocabIdx = make(map[string]int)

	// Important: loop through AdultFieldNames to have a stable ordering.
	for _, featureName := range AdultFieldNames {
		dType := AdultFieldTypes[featureName]
		if dType != series.String {
			// Only string features.
			continue
		}
		if featureName == LabelCol {
			continue
		}
		Data.FeatureNameToVocabIdx[featureName] = len(Data.VocabulariesFeatures)
		Data.VocabulariesFeatures = append(Data.VocabulariesFeatures, featureName)

		// Create vocabulary, then sort it, so the values are always the same.
		vocabulary := make(map[string]int)
		vocabulary[Unknown] = 0
		var values []string
		for _, value := range df.Col(featureName).Records() {
			_, ok := vocabulary[value]
			if !ok {
				vocabulary[value] = 0
				values = append(values, value)
			}
		}
		sort.Strings(values)
		for ii, value := range values {
			// index 0 is reserved for Unknown.
			vocabulary[value] = ii + 1
		}
		Data.Vocabularies = append(Data.Vocabularies, vocabulary)
	}
}

// PopulateQuantiles with up to numQuantiles for each tloat column.
func PopulateQuantiles(df dataframe.DataFrame, numQuantiles int) {
	// Important: loop through AdultFieldNames to have a stable ordering.
	for _, featureName := range AdultFieldNames {
		dType := AdultFieldTypes[featureName]
		if dType != series.Float {
			continue
		}
		if featureName == WeightCol {
			// Weight is not calibrated.
			continue
		}
		Data.QuantilesFeatures = append(Data.QuantilesFeatures, featureName)
		table := make(QuantileTable, 0, numQuantiles)
		col := df.Col(featureName)
		table = append(table, float32(col.Min()))
		// We eliminate the zeros, which otherwise would dominate the quantiles.
		colNoZeroes := dataframe.New(col).Filter(
			dataframe.F{Colname: featureName, Comparator: series.Neq, Comparando: 0}).Col(featureName)
		step := 1.0 / (float64(numQuantiles) - 1.0)
		for fraction, count := step, 1; count < numQuantiles-1; fraction, count = fraction+step, count+1 {
			newValue := float32(colNoZeroes.Quantile(fraction))
			if table[len(table)-1] != newValue {
				table = append(table, newValue)
			}
		}
		// Add last element (if not yet present).
		newValue := float32(col.Max())
		if table[len(table)-1] != newValue {
			table = append(table, newValue)
		}
		Data.Quantiles = append(Data.Quantiles, table)
	}
}

// RawData holds the data stripped of all metadata: categorical converted to ints. It includes
// the whole dataset.
type RawData struct {
	NumRows, NumCategorical, NumContinuous int

	// Categorical is shaped `[NumRows, NumCategorical]` ordered as in VocabulariesFeatures.
	Categorical []int

	// Continuous is shaped `[NumRows, NumContinuous]`, ordered as in QuantilesFeatures.
	Continuous []float32 // AssertNoError match ModelDType.

	// Weights is shaped [NumRows]
	Weights []float32 // AssertNoError match ModelDType.

	// Labels is shaped [NumRows]: 1.0 (>50K) or 0.0 (<=50K)
	Labels []float32 // AssertNoError match ModelDType.
}

// ConvertDataFrameToRawData convert df to the raw data. It returns:
func ConvertDataFrameToRawData(df dataframe.DataFrame) *RawData {
	r := &RawData{
		NumRows:        df.Nrow(),
		NumCategorical: len(Data.VocabulariesFeatures),
		NumContinuous:  len(Data.Quantiles),
	}
	r.Categorical = make([]int, r.NumRows*r.NumCategorical)
	r.Continuous = make([]float32, r.NumRows*r.NumContinuous)
	r.Weights = make([]float32, r.NumRows)
	r.Labels = make([]float32, r.NumRows)

	// Convert categorical values.
	for colNum, featureName := range Data.VocabulariesFeatures {
		col := df.Col(featureName)
		vocab := Data.Vocabularies[colNum]
		for rowNum := 0; rowNum < r.NumRows; rowNum++ {
			idx := r.CategoricalIdx(rowNum, colNum)
			valueStr := col.Elem(rowNum).String()
			r.Categorical[idx] = vocab[valueStr]
		}
	}

	// Convert numerical values.
	for colNum, featureName := range Data.QuantilesFeatures {
		col := df.Col(featureName)
		for rowNum := 0; rowNum < r.NumRows; rowNum++ {
			idx := r.ContinuousIdx(rowNum, colNum)
			r.Continuous[idx] = float32(col.Elem(rowNum).Float())
		}
	}

	// Convert weights.
	for rowNum, value := range df.Col(WeightCol).Float() {
		r.Weights[rowNum] = float32(value)
	}
	for rowNum, valueStr := range df.Col(LabelCol).Records() {
		var value float32
		if valueStr == LabelTrue {
			value = 1.0
		} else if valueStr != LabelFalse {
			panic(fmt.Sprintf("Row %d: invalid label %q", rowNum, valueStr))
		}
		r.Labels[rowNum] = value
	}
	return r
}

func (r *RawData) CategoricalIdx(rowNum, colNum int) int {
	return rowNum*r.NumCategorical + colNum
}

func (r *RawData) ContinuousIdx(rowNum, colNum int) int {
	return rowNum*r.NumContinuous + colNum
}

func (r *RawData) CategoricalRow(rowNum int) []int {
	start := r.CategoricalIdx(rowNum, 0)
	return r.Categorical[start : start+r.NumCategorical]
}

func (r *RawData) ContinuousRow(rowNum int) []float32 {
	start := r.ContinuousIdx(rowNum, 0)
	return r.Continuous[start : start+r.NumContinuous]
}

// SampleWithReplacement in local memory.
func (r *RawData) SampleWithReplacement(numExamples int) *RawData {
	sampled := &RawData{
		NumRows:        numExamples,
		NumCategorical: r.NumCategorical,
		NumContinuous:  r.NumContinuous,
		Categorical:    make([]int, numExamples*r.NumCategorical),
		Continuous:     make([]float32, numExamples*r.NumContinuous),
		Weights:        make([]float32, numExamples),
		Labels:         make([]float32, numExamples),
	}
	for ii := 0; ii < numExamples; ii++ {
		choice := rand.Intn(r.NumRows)
		copy(sampled.ContinuousRow(ii), r.ContinuousRow(choice))
		copy(sampled.CategoricalRow(ii), r.CategoricalRow(choice))
		sampled.Weights[ii] = r.Weights[choice]
		sampled.Labels[ii] = r.Labels[choice]
	}
	return sampled
}

// TensorData contains a RawData converted to tensors.
type TensorData struct {
	CategoricalTensor, ContinuousTensor, WeightsTensor, LabelsTensor *tensors.Tensor
}

// CreateTensors of dataset, for faster ML interaction.
func (r *RawData) CreateTensors(backend backends.Backend) *TensorData {
	return &TensorData{
		CategoricalTensor: tensors.FromFlatDataAndDimensions(r.Categorical, r.NumRows, r.NumCategorical),
		ContinuousTensor:  tensors.FromFlatDataAndDimensions(r.Continuous, r.NumRows, r.NumContinuous),
		WeightsTensor:     tensors.FromFlatDataAndDimensions(r.Weights, r.NumRows, 1),
		LabelsTensor:      tensors.FromFlatDataAndDimensions(r.Labels, r.NumRows, 1),
	}
}

// PrintFeatures prints information on the vacabularies and quantiles about the features.
func PrintFeatures(df dataframe.DataFrame) {
	fmt.Println(df)
	fmt.Printf("Vocabularies:\n")
	for ii, featureName := range Data.VocabulariesFeatures {
		vocab := Data.Vocabularies[ii]
		fmt.Printf("\tFeature %q:\n", featureName)
		for value, idx := range vocab {
			fmt.Printf("\t\t%q:%d\n", value, idx)
		}
	}

	fmt.Printf("\nQuantiles:\n")
	for ii, featureName := range Data.QuantilesFeatures {
		table := Data.Quantiles[ii]
		fmt.Printf("\tFeature %q, %d quantiles:\n", featureName, len(table))
		fmt.Printf("\t\t%v\n", table)
	}
}

// PrintRawData prints positivity ratio and and some samples.
func PrintRawData(r *RawData) {
	var positive, positiveWeighted, totalWeight float32
	for rowNum := 0; rowNum < r.NumRows; rowNum++ {
		positive += r.Labels[rowNum]
		positiveWeighted += r.Labels[rowNum] * r.Weights[rowNum]
		totalWeight += r.Weights[rowNum]
	}
	fmt.Printf("\nSample Categorical: (%.2f%% positive ratio, %.2f%% weighted positive ratio)\n",
		100.0*positive/float32(r.NumRows), 100.0*positiveWeighted/totalWeight)
	for rowNum := 0; rowNum < 3; rowNum++ {
		fmt.Printf("\tRow %d:\t%v\n", rowNum, r.CategoricalRow(rowNum))
	}
	fmt.Println("\t...")
	for rowNum := r.NumRows - 3; rowNum < r.NumRows; rowNum++ {
		fmt.Printf("\tRow %d:\t%v\n", rowNum, r.CategoricalRow(rowNum))
	}

	fmt.Printf("\nSample Continuous:\n")
	for rowNum := 0; rowNum < 3; rowNum++ {
		fmt.Printf("\tRow %d:\t%v\n", rowNum, r.ContinuousRow(rowNum))
	}
	fmt.Println("\t...")
	for rowNum := r.NumRows - 3; rowNum < r.NumRows; rowNum++ {
		fmt.Printf("\tRow %d:\t%v\n", rowNum, r.ContinuousRow(rowNum))
	}
}
