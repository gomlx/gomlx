package diffusion

import (
	"flag"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/nanlogger"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/types/shapes"
	"log"
	"sync"
)

var (
	flagPlatform = flag.String("platform", "", "Platform to use, if empty uses the default one.")
	flagDataDir  = flag.String("data", "~/work/oxfordflowers102", "Directory to cache downloaded dataset and save checkpoints.")

	flagDType = flag.String("dtype", "float32", "DType to use for model.")

	flagImageSize = flag.Int("size", 64, "Size of the images to use: since we'll cache them in the accelerator machine, it "+
		"is limited by its memory. Largest value (the original image sizes) is 500.")
	flagBatchSize     = flag.Int("batch", 64, "Batch size for training")
	flagEvalBatchSize = flag.Int("eval_batch", 128, "Batch size for evaluation")
)

// AssertNoError will panic if the given error is not nil.
func AssertNoError(err error) {
	if err != nil {
		log.Fatalf("Failed: %+v", err)
	}
}

// MustNoError takes a value and an error. It returns the value is err is nil, otherwise
// it panics.
func MustNoError[T any](value T, err error) T {
	AssertNoError(err)
	return value
}

var (
	initialized sync.Once
)

// Init will parse and move flag values into global variables.
// It's idempotent, and can be called multiple times.
func Init() {
	initialized.Do(func() {
		flag.Parse()
		if manager == nil {
			manager = BuildManager().Platform(*flagPlatform).Done()
		}
		DType = MustNoError(shapes.DTypeString(*flagDType))
		DataDir = data.ReplaceTildeInDir(*flagDataDir)
		ImageSize = *flagImageSize
		BatchSize = *flagBatchSize
		EvalBatchSize = *flagEvalBatchSize
		if *flagNanLogger {
			// If enabled create the object, otherwise leave it as nil.
			nanLogger = nanlogger.New()
		}
	})
}

func GetManager() *Manager {
	Init()
	return manager
}
