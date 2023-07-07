package diffusion

import (
	"encoding/gob"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"os"
	"path"

	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"

	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
)

var (
	// DataDir is the directory where data is saved. This includes training data, saved models (checkpoints)
	// and intermediary data (like normalization constants).
	DataDir string // Directory where resources are stored

	// ImageSize used everywhere. Images are square, the same size is used for height and width.
	ImageSize int

	// BatchSize used for training.
	BatchSize int

	// EvalBatchSize used for evaluation. The size is only affected by the limitation of the accelerator memory.
	EvalBatchSize int

	// PartitionSeed used for the dataset splitting into train/validation.
	PartitionSeed = int64(42) // Some arbitrary number.

	// ValidationFraction where the rest is used for training. There is no test set.
	ValidationFraction = 0.2 // 20% of data.
)

// CreateInMemoryDatasets returns a train and a validation InMemoryDataset.
func CreateInMemoryDatasets() (trainDS, validationDS *data.InMemoryDataset) {
	Init()
	var err error
	trainDS, err = flowers.InMemoryDataset(manager, DataDir, ImageSize, "train", PartitionSeed, ValidationFraction, 1.0)
	AssertNoError(err)

	validationDS, err = flowers.InMemoryDataset(manager, DataDir, ImageSize, "validation", PartitionSeed, 0.0, ValidationFraction)
	AssertNoError(err)
	return
}

var (
	// Cached results for NormalizationValues.
	normalizationMean, normalizationStdDev tensor.Tensor

	// NormalizationInfoFile where NormalizationValues results are saved (and loaded from).
	NormalizationInfoFile = "normalization_data.bin"
)

// NormalizationValues for the flowers dataset -- only look at the training data.
func NormalizationValues() (mean, stddev tensor.Tensor) {
	Init()

	// Check if values have already been retrieved.
	if normalizationMean != nil && normalizationStdDev != nil {
		mean, stddev = normalizationMean, normalizationStdDev
		return
	}

	// If not try to load from file.
	fPath := path.Join(DataDir, NormalizationInfoFile)
	f, err := os.Open(fPath)
	if err == nil {
		// Load previously generated values.
		dec := gob.NewDecoder(f)
		mean = MustNoError(tensor.GobDeserialize(dec))
		stddev = MustNoError(tensor.GobDeserialize(dec))
		_ = f.Close()
		normalizationMean, normalizationStdDev = mean, stddev
		return
	}
	if !os.IsNotExist(err) {
		err = errors.Wrapf(err, "failed to read images mean/stddev from disk")
		AssertNoError(err)
	}

	trainDS, _ := CreateInMemoryDatasets()
	trainDS.BatchSize(128, false)
	ds := data.Map(manager, nil, trainDS, func(ctx *context.Context, inputs, labels []*Node) (mappedInputs, mappedLabels []*Node) {
		images := PreprocessImages(inputs[0], false)
		return []*Node{images}, labels
	})
	normalizationMean, normalizationStdDev, err = data.Normalization(manager, ds, 0, -1) // mean/stddev for each channel (axis=-1) separately.
	AssertNoError(err)

	// Save for future times.
	f, err = os.Create(fPath)
	AssertNoError(err)
	enc := gob.NewEncoder(f)
	AssertNoError(mean.Local().GobSerialize(enc))
	AssertNoError(stddev.Local().GobSerialize(enc))
	AssertNoError(f.Close())

	mean, stddev = normalizationMean, normalizationStdDev
	return
}

// PreprocessImages converts the image to the model `DType` and optionally normalizes
// it according to `NormalizationValues()` calculated on the training dataset.
func PreprocessImages(images *Node, normalize bool) *Node {
	Init()
	g := images.Graph()

	// ReduceAllMax(images).SetLogged("Max(uint8):")
	images = ConvertType(images, DType)
	if !normalize {
		return images
	}

	// Here we get the concrete value (tensors) of the mean and variance, and we convert to constants
	// to be used in the graph.
	meanT, stddevT := NormalizationValues()
	mean := Const(g, meanT)
	stddev := Const(g, stddevT)

	images = Div(
		Sub(images, mean),
		data.ReplaceZerosByOnes(stddev))
	return images
}

// DenormalizeImages revert images back to the 0 - 255 range.
// But it keeps it as float, it doesn't convert it back to bytes (== `shapes.S8` or `uint8`)
func DenormalizeImages(images *Node) *Node {
	g := images.Graph()
	meanT, stddevT := NormalizationValues()
	mean := Const(g, meanT)
	stddev := Const(g, stddevT)

	images = Add(
		Mul(images, data.ReplaceZerosByOnes(stddev)),
		mean)
	images = ClipScalar(images, 0.0, 255.0)
	return images

}

func finalize(tensors []tensor.Tensor) {
	for _, t := range tensors {
		t.FinalizeAll()
	}
}
