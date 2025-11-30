package oxfordflowers102

import (
	"encoding/gob"
	"fmt"
	"image"
	"io"
	"math"
	"math/rand"
	"os"
	"path"
	"sync"
	"time"

	"github.com/disintegration/imaging"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	timage "github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/ml/datasets"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// Dataset implements train.Dataset, and yields one image at a time.
// It pre-transforms the image to the target `imageSize`.
type Dataset struct {
	name      string
	next      int
	imageSize int
	shuffle   []int
	toTensor  *timage.ToTensorConfig

	partitionSelection []bool // Which indices to take.

	mu sync.Mutex
}

// NewDataset returns a Dataset for one epoch that yields
// one image at time. It reads them from disk, and the parsing
// can be parallelized. See `data.NewParallelDataset`.
//
// The images are resized and cropped to `imageSize x imageSize` pixel,
// cut from the middle.
//
// It doesn't support batch, but you can use GoMLX's `datasets.Batch` for that.
func NewDataset(dtype dtypes.DType, imageSize int) *Dataset {
	return &Dataset{
		name:      "Oxford Flowers 102",
		imageSize: imageSize,
		toTensor:  timage.ToTensor(dtype),
	}
}

// Assert *Dataset implements train.Dataset
var _ train.Dataset = &Dataset{}

// Shuffle will shuffle the order of the images. This should be called before the
// start of an epoch.
//
// Once shuffled, every time the dataset is reset, it is reshuffled.
func (ds *Dataset) Shuffle() *Dataset {
	ds.mu.Lock()
	ds.shuffleLocked()
	ds.mu.Unlock()
	ds.name = ds.name + " [shuffled]"
	return ds
}

func (ds *Dataset) shuffleLocked() {
	if ds.shuffle == nil {
		ds.shuffle = xslices.Iota(0, NumExamples)
	}
	for ii := 0; ii < NumExamples; ii++ {
		swapPos := rand.Intn(NumExamples)
		ds.shuffle[ii], ds.shuffle[swapPos] = ds.shuffle[swapPos], ds.shuffle[ii]
	}
}

// Partition allows one to partition the dataset into different parts -- typically "train", "validation" and "test".
// This should be called before the start of an epoch.
//
// It takes a seed number based on which the partitions will be selected, and the range of elements specified as
// `from` and `to`: these are float values that represent the slice (from 0.0 to 1.0) of the examples that go into
// this dataset.
//
// Example:
//
//	seed := int64(42)
//	dsTrain := oxfordflowers102.NewDataset(dtypes.Float32, 75).Partition(seed, 0, 0.8)   // 80%
//	dsValid := oxfordflowers102.NewDataset(dtypes.Float32, 75).Partition(seed, 0.8, 0.9) // 10%
//	dsTest := oxfordflowers102.NewDataset(dtypes.Float32, 75).Partition(seed, 0.9, 1.0)  // 10%
func (ds *Dataset) Partition(seed int64, from, to float64) *Dataset {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	ds.name = fmt.Sprintf("%s [slice %d%%-%d%%]", ds.name, int(100.0*from), int(100.0*to))

	ds.partitionSelection = make([]bool, NumExamples)
	rng := rand.New(rand.NewSource(seed))
	for ii := range ds.partitionSelection {
		randPos := rng.Float64()
		if randPos >= from && randPos < to {
			ds.partitionSelection[ii] = true
		}
	}
	return ds
}

// nextIndex returns the next index and increments it.
// Concurrency safe.
// Returns -1 if reached the end of the dataset.
func (ds *Dataset) nextIndex() (index int) {
	ds.mu.Lock()
	defer ds.mu.Unlock()

	for {
		index = ds.next
		if ds.next < 0 {
			return
		}

		ds.next++
		if ds.next >= NumExamples {
			ds.next = -1 // Indicates the end of epoch.
		}

		if ds.shuffle != nil {
			index = ds.shuffle[index]
		}

		if len(ds.partitionSelection) > 0 {
			if !ds.partitionSelection[index] {
				// This index is not part of this partition, loop and take next index.
				continue
			}
		}

		return
	}
}

// Name implements train.Dataset interface.
func (ds *Dataset) Name() string {
	return ds.name
}

// Yield implements train.Dataset interface.
// It returns `ds` (the Dataset pointer) as spec.
//
// It yields one example at a time, each consists of:
//
//   - `inputs`: three values: the image itself and a scalar `int32` with
//     the index of the example and finally the type of flower (from 0 to `NumLabels-1`=101).
//   - `labels`: the type of flower (same as `inputs[2]`), an `int32` value from 0 to `NumLabels-1`
//     with the label.
func (ds *Dataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	spec = ds
	index := ds.nextIndex()
	if index == -1 {
		err = io.EOF
		return
	}
	img, label, err := ReadExample(index)
	if err != nil {
		err = errors.WithMessagef(err, "failed to read image/label #%d", index)
		return
	}

	// 1. Resize the smallest dimension to imageSize, preserving ratio.
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	if width < height {
		ratio := float64(width) / float64(ds.imageSize)
		width = ds.imageSize
		height = int(math.Round(float64(height) / ratio))
	} else if height < width {
		ratio := float64(height) / float64(ds.imageSize)
		height = ds.imageSize
		width = int(math.Round(float64(width) / ratio))
	} else {
		width = ds.imageSize
		height = ds.imageSize
	}
	img = imaging.Resize(img, width, height, imaging.Linear)

	// 2. Crop at center the largest dimension to imageSize.
	if width > height {
		start := (width - ds.imageSize) / 2
		img = imaging.Crop(img, image.Rect(start, 0, start+ds.imageSize, ds.imageSize))
	} else if height > width {
		start := (height - ds.imageSize) / 2
		img = imaging.Crop(img, image.Rect(0, start, ds.imageSize, start+ds.imageSize))
	}

	inputs = []*tensors.Tensor{ds.toTensor.Single(img), tensors.FromValue(index), tensors.FromValue(label)}
	labels = []*tensors.Tensor{tensors.FromValue(label)}
	return
}

// Reset implements train.Dataset interface.
func (ds *Dataset) Reset() {
	ds.mu.Lock()
	ds.next = 0
	ds.shuffleLocked()
	ds.mu.Unlock()
}

// InMemoryDataset creates a `datasets.InMemoryDataset` with the Oxford Flowers 102, of the given `imageSize` for both,
// height and width -- image is resized and then cropped at the center.
//
// A cache version is automatically saved at the `baseDir` and prefixed with `name`, if it is not empty.
// And if a cache file is found, it is used, instead of re-reading and processing all the images.
//
// It takes a partition of the data, defined by `partitionFrom` and `partitionTo`.
// They take values from 0.0 to 1.0 and represent the fraction of the dataset to take.
// They enable selection of arbitrary train/validation/test sizes.
// The `partitionSeed` can be used to generate different assignments -- the same seed should be used for the different
// partitions of the dataset.
//
// If the cache is not found, it automatically calls DownloadAndParse to download and untar the original
// images, if they are not yet downloaded.
func InMemoryDataset(backend backends.Backend, baseDir string, imageSize int, name string,
	partitionSeed int64, partitionFrom, partitionTo float64) (
	inMemoryDataset *datasets.InMemoryDataset, err error) {
	deviceNum := backends.DeviceNum(0)
	var f *os.File
	if baseDir != "" {
		baseDir = fsutil.MustReplaceTildeInDir(baseDir) // If dir starts with "~", it is replaced.
		inMemoryCacheFile := path.Join(baseDir, fmt.Sprintf("%s_cached_images_%dx%d.bin", name, imageSize, imageSize))

		defer func() {
			if err != nil {
				err = errors.WithMessagef(err, "File with cache: %q", inMemoryCacheFile)
			}
		}()

		f, err = os.Open(inMemoryCacheFile)
		if err == nil {
			// Reads from cached file.
			dec := gob.NewDecoder(f)
			inMemoryDataset, err = datasets.GobDeserializeInMemoryToDevice(backend, deviceNum, dec)
			_ = f.Close()
			return
		}
		if !os.IsNotExist(err) {
			return
		}

		// Prepare cache file for saving.
		f, err = os.Create(inMemoryCacheFile)
		if err != nil {
			return
		}
	}

	// Check whether dataset has been downloaded already.
	if NumExamples == 0 {
		err = DownloadAndParse(baseDir)
		if err != nil {
			return
		}
	}

	// Create InMemoryDataset.
	start := time.Now()
	fmt.Printf("Creating InMemoryDataset for %q with images cropped and scaled to %dx%d...\n", name, imageSize, imageSize)
	ds := NewDataset(dtypes.Uint8, imageSize).Partition(partitionSeed, partitionFrom, partitionTo)
	inMemoryDataset, err = datasets.InMemory(backend, datasets.Parallel(ds), false)
	elapsed := time.Since(start)
	fmt.Printf("\t- %s to process dataset.\n", elapsed)
	if err != nil {
		return
	}
	inMemoryDataset.SetName(name)

	// Save to cache.
	if f != nil {
		enc := gob.NewEncoder(f)
		err = inMemoryDataset.GobSerialize(enc)
		if err != nil {
			return
		}
		err = f.Close()
	}
	return
}
