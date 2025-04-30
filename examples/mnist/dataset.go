/*
 *	Copyright 2025 Rener Castro
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
	"net/url"
	"os"
	"path"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"

	timage "github.com/gomlx/gomlx/types/tensors/images"
)

const (
	DownloadURL         = "https://storage.googleapis.com/cvdf-datasets/mnist"
	TrainImagesFilename = "train-images-idx3-ubyte.gz"
	TrainLabelsFilename = "train-labels-idx1-ubyte.gz"
	TestImagesFilename  = "t10k-images-idx3-ubyte.gz"
	TestLabelsFilename  = "t10k-labels-idx1-ubyte.gz"
	Width               = 28
	Height              = 28
	Depth               = 3
	NumClasses          = 10
	TrainExamples       = 60000
	TestExamples        = 10000

	ImageMagic = 0x00000803
	LabelMagic = 0x00000801
)

type fileType int

const (
	ImageFileType fileType = iota
	LabelFileType
)

var mnistFiles = map[string][2]string{
	"train": {TrainImagesFilename, TrainLabelsFilename},
	"test":  {TestImagesFilename, TestLabelsFilename},
}

var mnistSamples = map[string]int{
	"train": TrainExamples,
	"test":  TestExamples,
}

// Image represents a MNIST image. It is a array a bytes representing the color.
// 0 is black (the background) and 255 is white (the digit color).
type Image [Width * Height]byte

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
		Max: image.Point{Width, Height},
	}
}

// At implements the image.Image interface.
func (img Image) At(x, y int) color.Color {
	return color.Gray{Y: img[y*Width+x]}
}

// Set modifies the pixel at (x,y).
func (img *Image) Set(x, y int, v byte) {
	img[y*Width+x] = v
}

// Download MNIST Dataset to baseDir, unzips it
func Download(baseDir string) error {
	baseDir = data.ReplaceTildeInDir(baseDir)
	files := []string{TrainImagesFilename, TrainLabelsFilename, TestImagesFilename, TestLabelsFilename}
	for _, file := range files {
		downloadURLFile, _ := url.JoinPath(DownloadURL, file)
		filePath := path.Join(baseDir, file)
		if err := data.DownloadIfMissing(downloadURLFile, filePath, ""); err != nil {
			return fmt.Errorf("data.DownloadAndUnzipIfMissing: %w", err)
		}
	}

	return nil
}

var (
	AssertImageIsImageImage *Image
	_                       image.Image = AssertImageIsImageImage
)

// NewDataset creates a train.Dataset that yields images from Dogs vs Cats Dataset.
//
// It takes the following arguments:
//
//   - name:
//   - baseDir:
//   - mode: choose between 'train' and 'test'
func NewDataset(backend backends.Backend, name, baseDir, mode string, dtype dtypes.DType) (ds *data.InMemoryDataset, err error) {
	imagesFile := mnistFiles[mode][ImageFileType]
	labelsFile := mnistFiles[mode][LabelFileType]

	images, err := loadImageFile(path.Join(baseDir, imagesFile))
	if err != nil {
		return nil, err
	}
	labels, err := loadLabelFile(path.Join(baseDir, labelsFile))
	if err != nil {
		return nil, err
	}

	return data.InMemoryFromData(
		backend,
		name,
		[]any{timage.ToTensor(dtype).Batch(images)},
		[]any{tensors.FromFlatDataAndDimensions(labels, len(labels), 1)},
	)
}

// LoadImageFile opens the image file, parses it, and returns the data in order.
func loadImageFile(filename string) ([]image.Image, error) {
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

	if header.Magic != ImageMagic ||
		header.Width != Width ||
		header.Height != Height {
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

	if header.Magic != LabelMagic {
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
func CreateDatasets(backend backends.Backend, config *DatasetsConfiguration) (trainDS, trainEvalDS, validationEvalDS train.Dataset) {
	// Datasets created from original images:
	var err error
	baseTrain, err := NewDataset(backend, "train", config.DataDir, "train", config.Dtype)
	if err != nil {
		exceptions.Panicf("NewDataset: %v", err)
	}
	baseTest, err := NewDataset(backend, "test", config.DataDir, "test", config.Dtype)
	if err != nil {
		exceptions.Panicf("NewDataset: %v", err)
	}

	trainDS = baseTrain.Copy().BatchSize(config.BatchSize, true).Shuffle().Infinite(true)
	trainEvalDS = baseTrain.BatchSize(config.EvalBatchSize, false)
	validationEvalDS = baseTest.BatchSize(config.EvalBatchSize, false)

	return
}
