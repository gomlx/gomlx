package diffusion

import (
	"encoding/gob"
	"fmt"
	"os"
	"path"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/internal/must"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/nanlogger"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"

	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/datasets"

	flowers "github.com/gomlx/gomlx/examples/oxfordflowers102"
)

var (
	// PartitionSeed used for the dataset splitting into train/validation.
	PartitionSeed = int64(42) //nolint:mnd // Some fixed arbitrary number for a deterministic split.

	// ValidationFraction where the rest is used for training. There is no test set.
	ValidationFraction = 0.1 // 10% of data.
)

// Config holds a configuration for all diffusion image/data operations.
// See NewConfig.
type Config struct {
	Backend backends.Backend
	Context *context.Context // Usually, at the root scope.

	// DataDir is where the data is downloaded, and models are saved.
	DataDir string

	// ParamsSet are hyperparameters overridden, that it should not load from the checkpoint (see commandline.ParseContextSettings).
	ParamsSet []string

	DType                               dtypes.DType
	ImageSize, BatchSize, EvalBatchSize int

	// Checkpoint if one has been attached. See Config.AttachCheckpoint.
	Checkpoint *checkpoints.Handler

	// NanLogger is enabled by setting the hyperparameter "nan_logger=true".
	NanLogger *nanlogger.NanLogger
}

// NewConfig creates a configuration for most of the diffusion methods.
//
// paramsSet are hyperparameters overridden, that it should not load from the checkpoint (see commandline.ParseContextSettings).
func NewConfig(backend backends.Backend, ctx *context.Context, dataDir string, paramsSet []string) *Config {
	dataDir = fsutil.MustReplaceTildeInDir(dataDir)
	if !fsutil.MustFileExists(dataDir) {
		must.M(os.MkdirAll(dataDir, 0777))
	}
	dtype := must.M1(dtypes.DTypeString(
		context.GetParamOr(ctx, "dtype", "float32")))
	cfg := &Config{
		Backend:       backend,
		Context:       ctx,
		DataDir:       dataDir,
		ImageSize:     context.GetParamOr(ctx, "image_size", 64),
		BatchSize:     context.GetParamOr(ctx, "batch_size", 64),
		EvalBatchSize: context.GetParamOr(ctx, "eval_batch_size", 128),
		DType:         dtype,
		ParamsSet:     paramsSet,
	}
	useNanLogger := context.GetParamOr(ctx, "nan_logger", false)
	if useNanLogger {
		cfg.NanLogger = nanlogger.New()
	}
	return cfg
}

// CreateInMemoryDatasets returns a train and a validation InMemoryDataset.
func (c *Config) CreateInMemoryDatasets() (trainDS, validationDS *datasets.InMemoryDataset) {
	trainDS = must.M1(
		flowers.InMemoryDataset(c.Backend, c.DataDir, c.ImageSize, "train", PartitionSeed, ValidationFraction, 1.0))
	validationDS = must.M1(
		flowers.InMemoryDataset(
			c.Backend,
			c.DataDir,
			c.ImageSize,
			"validation",
			PartitionSeed,
			0.0,
			ValidationFraction,
		),
	)
	return
}

var (
	// Cached results for NormalizationValues.
	normalizationMean, normalizationStdDev *tensors.Tensor
)

// NormalizationValues for the flowers dataset -- only look at the training data.
func (c *Config) NormalizationValues() (mean, stddev *tensors.Tensor) {
	// Check if values have already been retrieved.
	if normalizationMean != nil && normalizationStdDev != nil {
		mean, stddev = normalizationMean, normalizationStdDev
		return
	}

	// If not try to load from file.
	fPath := path.Join(c.DataDir, fmt.Sprintf("normalization_data_%dx%d.bin", c.ImageSize, c.ImageSize))
	f, err := os.Open(fPath)
	if err == nil {
		// Load previously generated values.
		err = exceptions.TryCatch[error](func() {
			dec := gob.NewDecoder(f)
			mean = must.M1(tensors.GobDeserialize(dec))
			stddev = must.M1(tensors.GobDeserialize(dec))
			must.M(f.Close())
		})
		if err != nil {
			panic(errors.WithMessagef(err, "While loading  NormalizationValues to %q", fPath))
		}
		normalizationMean, normalizationStdDev = mean, stddev
		return
	}
	if !os.IsNotExist(err) {
		panic(errors.Wrapf(err, "failed to read images mean/stddev from disk"))
	}

	trainDS, _ := c.CreateInMemoryDatasets()
	trainDS.BatchSize(128, false)
	ds := datasets.MapWithGraphFn(
		c.Backend,
		nil,
		trainDS,
		func(ctx *context.Context, inputs, labels []*Node) (mappedInputs, mappedLabels []*Node) {
			images := c.PreprocessImages(inputs[0], false)
			return []*Node{images}, labels
		},
	)
	normalizationMean, normalizationStdDev = must.M2(
		datasets.Normalization(c.Backend, ds, 0, -1)) // mean/stddev for each channel (axis=-1) separately.
	mean, stddev = normalizationMean, normalizationStdDev

	// Save for future times.
	err = exceptions.TryCatch[error](func() {
		f = must.M1(os.Create(fPath))
		enc := gob.NewEncoder(f)
		must.M(mean.GobSerialize(enc))
		must.M(stddev.GobSerialize(enc))
		must.M(f.Close())
	})
	if err != nil {
		panic(errors.WithMessagef(err, "While saving NormalizationValues to %q", fPath))
	}
	return
}

// PreprocessImages converts the image to the model `DType` and optionally normalizes
// it according to `NormalizationValues()` calculated on the training dataset.
func (c *Config) PreprocessImages(images *Node, normalize bool) *Node {
	g := images.Graph()

	// ReduceAllMax(images).SetLogged("Max(uint8):")
	images = ConvertDType(images, dtypes.Float32)
	c.NanLogger.TraceFirstNaN(images, "PreprocessImages:input")
	if !normalize {

		return images
	}

	// Here we get the concrete value (tensors) of the mean and variance, and we convert to constants
	// to be used in the graph.
	meanT, stddevT := c.NormalizationValues()
	mean := Const(g, meanT)
	stddev := Const(g, stddevT)

	images = Div(
		Sub(images, mean),
		datasets.ReplaceZerosByOnes(stddev))
	images = ConvertDType(images, c.DType)
	c.NanLogger.TraceFirstNaN(images, "PreprocessImages:"+c.DType.String())
	return images
}

// DenormalizeImages revert images back to the 0 - 255 range.
// But it keeps it as float, it doesn't convert it back to bytes (== `shapes.S8` or `uint8`)
func (c *Config) DenormalizeImages(images *Node) *Node {
	g := images.Graph()
	images = ConvertDType(images, dtypes.Float32)
	meanT, stddevT := c.NormalizationValues()
	mean := Const(g, meanT)
	stddev := Const(g, stddevT)

	images = Add(
		Mul(images, datasets.ReplaceZerosByOnes(stddev)),
		mean)
	images = ClipScalar(images, 0.0, 255.0)
	return images
}

func finalize(tensors []*tensors.Tensor) {
	for _, t := range tensors {
		t.MustFinalizeAll()
	}
}
