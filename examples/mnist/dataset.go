/*
 *	Copyright 2023 Rener Castro
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

// package mnist - The MNIST database of handwritten digits.
// based on https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/image_classification/mnist.py

package mnist

import (
	"compress/gzip"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"io"
	"math/rand"
	"net/url"
	"os"
	"path"
	"sync"
	"time"

	"golang.org/x/exp/constraints"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	timage "github.com/gomlx/gomlx/types/tensors/images"
	"github.com/gomlx/gopjrt/dtypes"
)

const (
	downloadURL         = "https://storage.googleapis.com/cvdf-datasets/mnist"
	trainImagesFilename = "train-images-idx3-ubyte.gz"
	trainLabelsFilename = "train-labels-idx1-ubyte.gz"
	testImagesFilename  = "t10k-images-idx3-ubyte.gz"
	testLabelsFilename  = "t10k-labels-idx1-ubyte.gz"
	width               = 28
	height              = 28
	numClasses          = 10
	trainExamples       = 60000
	testExamples        = 10000

	imageMagic = 0x00000803
	labelMagic = 0x00000801
)

var muImage sync.Mutex
var muLabel sync.Mutex
var muYeld sync.Mutex

type fileType int

const (
	imageFileType fileType = iota
	labelFileType
)

var mnistFiles = map[string][2]string{
	"train": {trainImagesFilename, trainLabelsFilename},
	"test":  {testImagesFilename, testLabelsFilename},
}

var mnistSamples = map[string]int{
	"train": trainExamples,
	"test":  testExamples,
}

// Image represents a MNIST image. It is a array a bytes representing the color.
// 0 is black (the background) and 255 is white (the digit color).
type Image [width * height]byte

// Label is the digit label from 0 to 9.
type Label = int8

type imageFileHeader struct {
	Magic     int32
	NumImages int32
	Height    int32
	Width     int32
}

type labelFileHeader struct {
	Magic     int32
	NumLabels int32
}

// ColorModel implements the image.Image interface.
func (img Image) ColorModel() color.Model {
	return color.GrayModel
}

// Bounds implements the image.Image interface.
func (img Image) Bounds() image.Rectangle {
	return image.Rectangle{
		Min: image.Point{0, 0},
		Max: image.Point{width, height},
	}
}

// At implements the image.Image interface.
func (img Image) At(x, y int) color.Color {
	return color.Gray{Y: img[y*width+x]}
}

// Set modifies the pixel at (x,y).
func (img *Image) Set(x, y int, v byte) {
	img[y*width+x] = v
}

// Download MNIST Dataset to baseDir, unzips it
func Download(baseDir string) error {
	baseDir = data.ReplaceTildeInDir(baseDir)
	files := []string{trainImagesFilename, trainLabelsFilename, testImagesFilename, testLabelsFilename}
	for _, file := range files {
		downloadURLFile, _ := url.JoinPath(downloadURL, file)
		filePath := path.Join(baseDir, file)
		if err := data.DownloadIfMissing(downloadURLFile, filePath, ""); err != nil {
			return fmt.Errorf("data.DownloadAndUnzipIfMissing: %w", err)
		}
	}

	return nil
}

var (
	AssertDatasetIsTrainDataset *Dataset
	_                           train.Dataset = AssertDatasetIsTrainDataset

	AssertImageIsImageImage *Image
	_                       image.Image = AssertImageIsImageImage
)

// Dataset implements train.Dataset so it can be used by a train.Loop object to train/evaluate, and offers a few
// more functionality for sampling images (as opposed to tensors).
type Dataset struct {
	name       string
	baseDir    string
	imagesFile string
	labelsFile string

	//
	toTensor *timage.ToTensorConfig
	dtype    dtypes.DType

	//
	width  int
	height int

	//
	numClasses int
	size       int

	//
	batchSize int
	shuffle   *rand.Rand
	indices   []int
	position  int

	//
	images []image.Image
	labels []Label
}

// NewDataset creates a train.Dataset that yields images from Dogs vs Cats Dataset.
//
// It takes the following arguments:
//
//   - name:
//   - baseDir:
//   - mode: choose between 'train' and 'test'
func NewDataset(name, baseDir, mode string, batchSize int, shuffle *rand.Rand, dtype dtypes.DType) (ds *Dataset, err error) {
	ds = &Dataset{
		name:       name,
		baseDir:    baseDir,
		imagesFile: mnistFiles[mode][imageFileType],
		labelsFile: mnistFiles[mode][labelFileType],
		toTensor:   timage.ToTensor(dtype),
		dtype:      dtype,
		width:      width,
		height:     height,
		numClasses: numClasses,
		size:       mnistSamples[mode],
		batchSize:  batchSize,
		shuffle:    shuffle,
		position:   0,
	}
	ds.Reset() // Create first shuffle, if needed.
	ds.images, err = loadImageFile(path.Join(baseDir, ds.imagesFile))
	if err != nil {
		return nil, err
	}
	ds.labels, err = loadLabelFile(path.Join(baseDir, ds.labelsFile))
	if err != nil {
		return nil, err
	}
	if ds.shuffle != nil {
		ds.indices = ds.shuffle.Perm(len(ds.images))
	} else {
		for i := range len(ds.images) {
			ds.indices = append(ds.indices, i)
		}
	}

	return ds, nil
}

// Name implements train.Dataset.
func (ds *Dataset) Name() string { return ds.name }

// Yield implements `train.Dataset`. It returns:
//
//   - spec: not used, left as nil.
//   - the first is the images batch (shaped `[batch_size, height, width, depth==1]`) and
//   - the second holds the labels of the images as int, shaped `[batch_size]`.
func (ds *Dataset) Yield() (spec any, inputs, labels []*tensors.Tensor, err error) {
	muYeld.Lock()
	defer muYeld.Unlock()

	if ds.position >= len(ds.indices) {
		ds.position = 0
		if ds.shuffle != nil {
			ds.indices = ds.shuffle.Perm(len(ds.images))
		} else {
			for i := range len(ds.images) {
				ds.indices = append(ds.indices, i)
			}
		}
	}

	id_start := ds.position
	ds.position += ds.batchSize
	id_end := ds.position
	if ds.position >= len(ds.indices) {
		id_end = len(ds.indices)
		err = io.EOF
	}

	return ds,
		[]*tensors.Tensor{ds.toTensor.Batch(Select(ds.images, ds.indices[id_start:id_end]))},
		[]*tensors.Tensor{tensors.FromAnyValue(shapes.CastAsDType(Select(ds.labels, ds.indices[id_start:id_end]), ds.dtype))},
		err
}

// IsOwnershipTransferred tells the training loop that the dataset keeps ownership of the yielded tensors.
func (ds *Dataset) IsOwnershipTransferred() bool {
	return false
}

// Reset implements train.Dataset.
func (ds *Dataset) Reset() {
}

// LoadImageFile opens the image file, parses it, and returns the data in order.
func loadImageFile(filename string) ([]image.Image, error) {
	muImage.Lock()
	defer muImage.Unlock()
	f, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("os.Open: %w", err)
	}
	defer f.Close()

	reader, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("gzip.NewReader: %w", err)
	}
	defer reader.Close()

	header := imageFileHeader{}

	err = binary.Read(reader, binary.BigEndian, &header)
	if err != nil {
		return nil, err
	}

	if header.Magic != imageMagic ||
		header.Width != width ||
		header.Height != header.Height {
		return nil, fmt.Errorf("mnist: invalid format")
	}

	images := make([]image.Image, header.NumImages)
	for i := int32(0); i < header.NumImages; i++ {
		var img Image
		if err := binary.Read(reader, binary.BigEndian, &img); err != nil {
			return nil, fmt.Errorf("binary.Read: %w", err)
		}
		images[i] = img
	}

	return images, nil
}

// readImage reads a image from the file and returns it.
func loadLabelFile(filename string) ([]Label, error) {
	muLabel.Lock()
	defer muLabel.Unlock()
	f, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("os.Open: %w", err)
	}
	defer f.Close()

	reader, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("gzip.NewReader: %w", err)
	}
	defer reader.Close()

	header := labelFileHeader{}

	err = binary.Read(reader, binary.BigEndian, &header)
	if err != nil {
		return nil, err
	}

	if header.Magic != labelMagic {
		return nil, err
	}

	labels := make([]Label, header.NumLabels)
	for i := int32(0); i < header.NumLabels; i++ {
		err = binary.Read(reader, binary.BigEndian, &labels[i])
		if err != nil {
			return nil, err
		}
	}

	return labels, nil
}

// DatasetsConfiguration holds various parameters on how to transform the input images.
type DatasetsConfiguration struct {
	// DataDir, where downloaded and generated data is stored.
	DataDir string

	// BatchSize for training and evaluation batches.
	BatchSize, EvalBatchSize int

	// UseParallelism when using Dataset.
	UseParallelism bool

	// BufferSize used for data.ParallelDataset, to cache intermediary batches. This value is used
	// for each dataset.
	BufferSize int

	Dtype dtypes.DType
}

// CreateDatasets used for training and evaluation.
func CreateDatasets(config *DatasetsConfiguration) (trainDS, trainEvalDS, validationEvalDS train.Dataset) {
	shuffle := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))

	// Datasets created from original images:
	var err error
	trainDS, err = NewDataset("train", config.DataDir, "train", config.BatchSize, shuffle, config.Dtype)
	if err != nil {
		exceptions.Panicf("NewDataset: %v", err)
	}
	trainEvalDS, err = NewDataset("train-eval", config.DataDir, "train", config.EvalBatchSize, shuffle, config.Dtype)
	if err != nil {
		exceptions.Panicf("NewDataset: %v", err)
	}
	validationEvalDS, err = NewDataset("valid-eval", config.DataDir, "test", config.EvalBatchSize, shuffle, config.Dtype)
	if err != nil {
		exceptions.Panicf("NewDataset: %v", err)
	}

	// Read tensors in parallel:
	if config.UseParallelism {
		trainDS = data.CustomParallel(trainDS).Buffer(config.BufferSize).Start()
		trainEvalDS = data.CustomParallel(trainEvalDS).Buffer(config.BufferSize).Start()
		validationEvalDS = data.CustomParallel(validationEvalDS).Buffer(config.BufferSize).Start()
	}

	return
}

func Select[T any, I constraints.Integer](items []T, idx []I) []T {
	selItems := []T{}
	nItems := len(items)
	for _, i := range idx {
		if i < I(nItems) {
			selItems = append(selItems, items[i])
		}
	}
	return selItems
}
