// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dogsvscats

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"path"
	"time"

	"github.com/gomlx/gomlx/examples/inceptionv3"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
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
	PreGeneratedTrainPairFileName  = "train_pair_data.bin"
	PreGeneratedTrainEvalFileName  = "train_eval_data.bin"
	PreGeneratedValidationFileName = "validation_eval_data.bin"
)

// PreprocessingConfiguration holds various parameters on how to transform the input images.
type PreprocessingConfiguration struct {
	// DataDir, where downloaded and generated data is stored.
	DataDir string

	// DType of the images when converted to Tensor.
	DType dtypes.DType

	// BatchSize for training and evaluation batches.
	BatchSize, EvalBatchSize int

	// ModelImageSize is use for height and width of the generated images.
	ModelImageSize int

	// YieldImagePairs if to yield an extra input with the paired image: same image, different random augmentation.
	// Only applies for Train dataset.
	YieldImagePairs bool

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

	// BufferSize used for datasets.ParallelDataset, to cache intermediary batches. This value is used
	// for each dataset.
	BufferSize int

	// NumSamples is the maximum number of samples the model is allowed to see. If set to -1
	// model can see all samples.
	NumSamples int
}

var (
	DefaultConfig = &PreprocessingConfiguration{
		DType:           dtypes.Float32,
		BatchSize:       16,
		EvalBatchSize:   100, // Faster evaluation with larger batches.
		ModelImageSize:  inceptionv3.MinimumImageSize,
		NumFolds:        5,
		TrainFolds:      []int{0, 1, 2, 3},
		ValidationFolds: []int{4},
		FoldsSeed:       0,
		UseParallelism:  true,
		BufferSize:      32,
		NumSamples:      -1,
	} // DType used for model.

)

// NewPreprocessingConfigurationFromContext create a preprocessing configuration based on hyperparameters
// set in the context.
//
// Notice some configuration parameters depends on the model type ("model" hyperparameter): "inception" has a
// specific size, "byol" model requires image pairs.
func NewPreprocessingConfigurationFromContext(ctx *context.Context, dataDir string) *PreprocessingConfiguration {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	modelType := context.GetParamOr(ctx, "model", "")
	config := &PreprocessingConfiguration{}
	*config = *DefaultConfig
	config.DataDir = dataDir
	config.BatchSize = context.GetParamOr(ctx, "batch_size", 0)
	config.EvalBatchSize = context.GetParamOr(ctx, "eval_batch_size", 0)
	config.AngleStdDev = context.GetParamOr(ctx, "augmentation_angle_stddev", 0.0)
	config.FlipRandomly = context.GetParamOr(ctx, "augmentation_random_flips", false)
	if modelType == "inception" {
		config.ModelImageSize = inceptionv3.MinimumImageSize
	}
	config.ForceOriginal = context.GetParamOr(ctx, "augmentation_force_original", false)
	config.UseParallelism = true
	config.BufferSize = 100
	config.YieldImagePairs = modelType == "byol"
	return config
}

// PreGenerate create datasets that reads the original images, but then saves the scaled down and augmented for
// training images in binary format, for faster consumption later.
//
// It will only run if files don't already exist.
func PreGenerate(config *PreprocessingConfiguration, numEpochsForTraining int, force bool) {
	// Notice we need an even sized batch-size, to have equal number of dogs and cats.
	batchSize := 2

	// Validation data for evaluation.
	validPath := path.Join(config.DataDir, PreGeneratedValidationFileName)
	if !fsutil.MustFileExists(validPath) || force {
		f, err := os.Create(validPath)
		AssertNoError(err)
		ds := NewDataset("valid", config.DataDir, batchSize, false, nil, config.NumFolds,
			config.ValidationFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, 0, false, config.DType)
		fmt.Printf("Generating validation data for evaluation in %q...\n", validPath)
		err = ds.Save(1, true, f)
		AssertNoError(err)
		AssertNoError(f.Close())
	} else {
		fmt.Printf("Validation data for evaluation already generated in %q\n", validPath)
	}

	// Training data for evaluation.
	trainEvalPath := path.Join(config.DataDir, PreGeneratedTrainEvalFileName)
	if !fsutil.MustFileExists(trainEvalPath) || force {
		f, err := os.Create(trainEvalPath)
		AssertNoError(err)
		ds := NewDataset("train-eval", config.DataDir, batchSize, false, nil, config.NumFolds,
			config.TrainFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, 0, false, config.DType)
		fmt.Printf("Generating training data for evaluation in %q...\n", trainEvalPath)
		err = ds.Save(1, true, f)
		AssertNoError(err)
		AssertNoError(f.Close())
	} else {
		fmt.Printf("Training data for evaluation already generated in %q\n", trainEvalPath)
	}

	// Training data.
	trainPath := path.Join(config.DataDir, PreGeneratedTrainFileName)
	trainPairPath := path.Join(config.DataDir, PreGeneratedTrainPairFileName)
	if !fsutil.MustFileExists(trainPath) || force {
		f, err := os.Create(trainPath)
		AssertNoError(err)
		f2, err := os.Create(trainPairPath)
		AssertNoError(err)

		shuffle := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
		ds := NewDataset("train", config.DataDir, batchSize, false, shuffle, config.NumFolds,
			config.TrainFolds, config.FoldsSeed,
			config.ModelImageSize, config.ModelImageSize, config.AngleStdDev, config.FlipRandomly, config.DType).
			WithImagePairs(true) // We want 2 augmented images per original image for training with BYOL model.
		fmt.Printf("Generating training data *with augmentation* in %q and %q...\n", trainPath, trainPairPath)
		err = ds.Save(numEpochsForTraining, true, f, f2)
		AssertNoError(err)
		AssertNoError(f.Close())
		AssertNoError(f2.Close())
	} else {
		fmt.Printf("Training data for training already generated in %q\n", trainPath)
	}
}

// CreateDatasets used for training and evaluation. If the pre-generated files with augmented/scaled images
// exist use that, otherwise dynamically generate the images -- typically much slower than training, hence
// makes the training much, much slower.
func CreateDatasets(config *PreprocessingConfiguration) (trainDS, trainEvalDS, validationEvalDS train.Dataset) {
	shuffle := rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	usePretrained := !config.ForceOriginal && config.NumSamples == -1
	trainPath := path.Join(config.DataDir, PreGeneratedTrainFileName)
	trainPairPath := path.Join(config.DataDir, PreGeneratedTrainPairFileName)
	trainEvalPath := path.Join(config.DataDir, PreGeneratedTrainEvalFileName)
	validPath := path.Join(config.DataDir, PreGeneratedValidationFileName)

	if usePretrained {
		// Check the pre-trained files exist:
		for _, filePath := range []string{trainPath, trainPairPath, trainEvalPath, validPath} {
			if _, err := os.Stat(filePath); err != nil {
				usePretrained = false
				break
			}
		}
	}

	if usePretrained {
		// Build pre-trained datasets:
		trainPre := NewPreGeneratedDataset("train [Pre]", trainPath, config.BatchSize, true,
			config.ModelImageSize, config.ModelImageSize, config.DType)
		if config.YieldImagePairs {
			trainPre = trainPre.WithImagePairs(trainPairPath)
		}
		trainDS = trainPre
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

	}

	// Read tensors in parallel:
	if config.UseParallelism {
		if trainDS != nil {
			trainDS = datasets.CustomParallel(trainDS).Buffer(config.BufferSize).Start()
		}
		if trainEvalDS != nil {
			trainEvalDS = datasets.CustomParallel(trainEvalDS).Buffer(config.BufferSize).Start()
		}
		if validationEvalDS != nil {
			validationEvalDS = datasets.CustomParallel(validationEvalDS).Buffer(config.BufferSize).Start()
		}
	}

	return
}
