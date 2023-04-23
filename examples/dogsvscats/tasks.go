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

package dogsvscats

import (
	"fmt"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"image"
	"log"
	"math/rand"
	"os"
	"path"
	"reflect"
	"time"
)

// This file implements high level tasks: pre-generating augment dataset.

// AssertNoError log.Fatal if err is not nil.
func AssertNoError(err error) {
	if err != nil {
		log.Fatalf("Failed: %+v", err)
	}
}

const (
	PreGeneratedTrainFileName      = "train_data.bin"
	PreGeneratedTrainEvalFileName  = "train_eval_data.bin"
	PreGeneratedValidationFileName = "validation_eval_data.bin"
)

// Configuration of the many pre-designed tasks.
type Configuration struct {
	// DataDir, where downloaded and generated data is stored.
	DataDir string

	DType shapes.DType

	// BatchSize for batches.
	BatchSize int

	// ModelImageSize is use for height and width of the generated images.
	ModelImageSize int

	// NumFolds for cross-validation.
	NumFolds int

	// Folds to use for train and validation.
	TrainFolds, ValidationFolds []int

	// FoldsSeed used when randomizing the folds assignment, so it can be done deterministically.
	FoldsSeed int32

	// AngleStdDev for angle perturbation of the image. Only active if > 0.
	AngleStdDev float64

	// FlipRandomly the image, for data augmentation. Only active if true.
	FlipRandomly bool

	// ForceOriginal will make CreateDatasets not use the pre-generated augmented datasets, even if
	// they are present.
	ForceOriginal bool

	// UseParallelism when using Dataset.
	UseParallelism bool

	// BufferSize used for data.ParallelDataset, to cache intermediary batches. This value is used
	// for each dataset.
	BufferSize int
}

var (
	DefaultConfig = &Configuration{
		DType:           shapes.Float32,
		BatchSize:       32,
		ModelImageSize:  64,
		NumFolds:        5,
		TrainFolds:      []int{0, 1, 2, 3},
		ValidationFolds: []int{4},
		FoldsSeed:       0,
		UseParallelism:  true,
		BufferSize:      32,
	} // DType used for model.

)

// PreGenerate create datasets that reads the original images, but then saves the scaled down and augmented for
// training images in binary format, for faster consumption later.
func PreGenerate(config *Configuration, numEpochsForTraining int) {
	// Notice we need an even sized batch-size, to have equal number of dogs and cats.

	// Validation data for evaluation.
	validPath := path.Join(config.DataDir, PreGeneratedValidationFileName)
	f, err := os.Create(validPath)
	AssertNoError(err)
	ds := NewDataset("valid", config.DataDir, 2, false, nil, config.NumFolds,
		config.ValidationFolds, config.FoldsSeed,
		config.ModelImageSize, config.ModelImageSize, 0, false, config.DType)
	fmt.Printf("Generating validation data for evaluation in %q...\n", validPath)
	err = ds.Save(f, 1, true)
	AssertNoError(err)
	AssertNoError(f.Close())

	// Training data for evaluation.
	trainEvalPath := path.Join(config.DataDir, PreGeneratedTrainEvalFileName)
	f, err = os.Create(trainEvalPath)
	AssertNoError(err)
	ds = NewDataset("train-eval", config.DataDir, 2, false, nil, config.NumFolds,
		config.TrainFolds, config.FoldsSeed,
		config.ModelImageSize, config.ModelImageSize, 0, false, config.DType)
	fmt.Printf("Generating training data for evaluation in %q...\n", trainEvalPath)
	err = ds.Save(f, 1, true)
	AssertNoError(err)
	AssertNoError(f.Close())

	// Training data.
	trainPath := path.Join(config.DataDir, PreGeneratedTrainFileName)
	f, err = os.Create(trainPath)
	AssertNoError(err)
	shuffle := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	ds = NewDataset("train", config.DataDir, 2, false, shuffle, config.NumFolds,
		config.TrainFolds, config.FoldsSeed,
		config.ModelImageSize, config.ModelImageSize, config.AngleStdDev, config.FlipRandomly, config.DType)
	fmt.Printf("Generating training data *with augmentation* in %q...\n", trainPath)
	err = ds.Save(f, numEpochsForTraining, true)
	AssertNoError(err)
	AssertNoError(f.Close())
}

// CreateDatasets used for training and evaluation. If the pre-generated files with augmented/scaled images
// exist use that, otherwise dynamically generate the images -- typically much slower than training, hence
// makes the training much, much slower.
func CreateDatasets(config *Configuration) (trainDS, trainEvalDS, validationEvalDS train.Dataset) {
	shuffle := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	trainPath := path.Join(config.DataDir, PreGeneratedTrainFileName)
	if _, err := os.Stat(trainPath); err == nil && !config.ForceOriginal {
		trainDS = NewPreGeneratedDataset("train", trainPath, config.BatchSize, true,
			config.ModelImageSize, config.ModelImageSize, config.DType)
	} else {
		trainDS = NewDataset("train", config.DataDir, config.BatchSize, true, shuffle,
			config.NumFolds, config.TrainFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, config.AngleStdDev, config.FlipRandomly, config.DType)
		if config.UseParallelism {
			trainDS = data.NewParallelDataset(trainDS, 0, config.BufferSize)
		}
	}

	trainEvalPath := path.Join(config.DataDir, PreGeneratedTrainEvalFileName)
	if _, err := os.Stat(trainEvalPath); err == nil && !config.ForceOriginal {
		trainEvalDS = NewPreGeneratedDataset("train-eval", trainEvalPath, config.BatchSize, false,
			config.ModelImageSize, config.ModelImageSize, config.DType)
	} else {
		trainEvalDS = NewDataset("train-eval", config.DataDir, config.BatchSize, false, nil,
			config.NumFolds, config.TrainFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, 0, false, config.DType)
		if config.UseParallelism {
			trainEvalDS = data.NewParallelDataset(trainEvalDS, 0, config.BufferSize)
		}
	}

	validPath := path.Join(config.DataDir, PreGeneratedValidationFileName)
	if _, err := os.Stat(validPath); err == nil && !config.ForceOriginal {
		validationEvalDS = NewPreGeneratedDataset("valid-eval", validPath, config.BatchSize, false,
			config.ModelImageSize, config.ModelImageSize, config.DType)
	} else {
		validationEvalDS = NewDataset("valid-eval", config.DataDir, config.BatchSize, false, nil,
			config.NumFolds, config.TrainFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, 0, false, config.DType)
		if config.UseParallelism {
			validationEvalDS = data.NewParallelDataset(validationEvalDS, 0, config.BufferSize)
		}
	}
	return
}

// TensorToGoImage converts image(s) in a tensor.Tensor to a image.Image object.
func TensorToGoImage(config *Configuration, images tensor.Tensor, exampleNum int) *image.NRGBA {
	width, height, depth := config.ModelImageSize, config.ModelImageSize, 4 // 4 channels, RGBA
	shapeDims := images.Shape().Dimensions
	if images.Rank() != 4 || shapeDims[0] <= exampleNum || shapeDims[1] != config.ModelImageSize || shapeDims[2] != config.ModelImageSize || shapeDims[3] != 4 {
		log.Fatalf("Received images tensor shaped %s, cannot access image %d of size (%d x %d x %d)",
			images.Shape(), exampleNum, config.ModelImageSize, config.ModelImageSize, 4)
	}
	img := image.NewNRGBA(image.Rect(0, 0, width, height))
	tensorData := reflect.ValueOf(images.Local().Data())
	imageSize := width * height * depth
	tensorPos := exampleNum * imageSize
	floatT := reflect.TypeOf(float32(0))
	for h := 0; h < height; h++ {
		for w := 0; w < width; w++ {
			for d := 0; d < 4; d++ {
				v := tensorData.Index(tensorPos)
				f := v.Convert(floatT).Interface().(float32)
				tensorPos++
				img.Pix[h*img.Stride+w*4+d] = uint8(f * 255)
			}
		}
	}
	return img
}
