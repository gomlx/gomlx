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
	"github.com/gomlx/gomlx/models/inceptionv3"
	"github.com/gomlx/gomlx/types/shapes"
	"log"
	"math/rand"
	"os"
	"path"
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

	// DType of the images when converted to Tensor.
	DType shapes.DType

	// BatchSize for training and evaluation batches.
	BatchSize, EvalBatchSize int

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
		EvalBatchSize:   100, // Faster evaluation with larger batches.
		ModelImageSize:  inceptionv3.MinimumImageSize,
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
//
// It will only run if files don't already exist.
func PreGenerate(config *Configuration, numEpochsForTraining int, force bool) {
	// Notice we need an even sized batch-size, to have equal number of dogs and cats.
	batchSize := 2

	// Validation data for evaluation.
	validPath := path.Join(config.DataDir, PreGeneratedValidationFileName)
	if !data.FileExists(validPath) || force {
		f, err := os.Create(validPath)
		AssertNoError(err)
		ds := NewDataset("valid", config.DataDir, batchSize, false, nil, config.NumFolds,
			config.ValidationFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, 0, false, config.DType)
		fmt.Printf("Generating validation data for evaluation in %q...\n", validPath)
		err = ds.Save(f, 1, true)
		AssertNoError(err)
		AssertNoError(f.Close())
	} else {
		fmt.Printf("Validation data for evaluation already generated in %q\n", validPath)
	}

	// Training data for evaluation.
	trainEvalPath := path.Join(config.DataDir, PreGeneratedTrainEvalFileName)
	if !data.FileExists(trainEvalPath) || force {
		f, err := os.Create(trainEvalPath)
		AssertNoError(err)
		ds := NewDataset("train-eval", config.DataDir, batchSize, false, nil, config.NumFolds,
			config.TrainFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, 0, false, config.DType)
		fmt.Printf("Generating training data for evaluation in %q...\n", trainEvalPath)
		err = ds.Save(f, 1, true)
		AssertNoError(err)
		AssertNoError(f.Close())
	} else {
		fmt.Printf("Training data for evaluation already generated in %q\n", trainEvalPath)
	}

	// Training data.
	trainPath := path.Join(config.DataDir, PreGeneratedTrainFileName)
	if !data.FileExists(trainPath) || force {
		f, err := os.Create(trainPath)
		AssertNoError(err)
		shuffle := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
		ds := NewDataset("train", config.DataDir, batchSize, false, shuffle, config.NumFolds,
			config.TrainFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, config.AngleStdDev, config.FlipRandomly, config.DType)
		fmt.Printf("Generating training data *with augmentation* in %q...\n", trainPath)
		err = ds.Save(f, numEpochsForTraining, true)
		AssertNoError(err)
		AssertNoError(f.Close())
	} else {
		fmt.Printf("Training data for training already generated in %q\n", trainPath)
	}
}

// CreateDatasets used for training and evaluation. If the pre-generated files with augmented/scaled images
// exist use that, otherwise dynamically generate the images -- typically much slower than training, hence
// makes the training much, much slower.
func CreateDatasets(config *Configuration) (trainDS, trainEvalDS, validationEvalDS train.Dataset) {
	shuffle := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	usePretrained := !config.ForceOriginal
	trainPath := path.Join(config.DataDir, PreGeneratedTrainFileName)
	trainEvalPath := path.Join(config.DataDir, PreGeneratedTrainEvalFileName)
	validPath := path.Join(config.DataDir, PreGeneratedValidationFileName)

	if usePretrained {
		// Check the pre-trained files exist:
		for _, filePath := range []string{trainPath, trainEvalPath, validPath} {
			if _, err := os.Stat(filePath); err != nil {
				usePretrained = false
				break
			}
		}
	}

	if usePretrained {
		// Build pre-trained datasets:
		trainDS = NewPreGeneratedDataset("train [Pre]", trainPath, config.BatchSize, true,
			config.ModelImageSize, config.ModelImageSize, config.DType)
		trainEvalDS = NewPreGeneratedDataset("train-eval [Pre]", trainEvalPath, config.EvalBatchSize, false,
			config.ModelImageSize, config.ModelImageSize, config.DType)
		validationEvalDS = NewPreGeneratedDataset("valid-eval [Pre]", validPath, config.EvalBatchSize, false,
			config.ModelImageSize, config.ModelImageSize, config.DType)

	} else {
		// Datasets created from original images:
		trainDS = NewDataset("train", config.DataDir, config.BatchSize, true, shuffle,
			config.NumFolds, config.TrainFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, config.AngleStdDev, config.FlipRandomly, config.DType)
		trainEvalDS = NewDataset("train-eval", config.DataDir, config.EvalBatchSize, false, nil,
			config.NumFolds, config.TrainFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, 0, false, config.DType)
		validationEvalDS = NewDataset("valid-eval", config.DataDir, config.EvalBatchSize, false, nil,
			config.NumFolds, config.TrainFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, 0, false, config.DType)

		// Read tensors in parallel:
		if config.UseParallelism {
			trainDS = data.CustomParallel(trainDS).Buffer(config.BufferSize).Start()
			trainEvalDS = data.CustomParallel(trainEvalDS).Buffer(config.BufferSize).Start()
			validationEvalDS = data.CustomParallel(validationEvalDS).Buffer(config.BufferSize).Start()
		}
	}

	return
}
