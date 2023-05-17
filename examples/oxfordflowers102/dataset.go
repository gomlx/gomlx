package oxfordflowers102

import (
	"encoding/gob"
	"fmt"
	"github.com/disintegration/imaging"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"github.com/pkg/errors"
	"image"
	"io"
	"math"
	"math/rand"
	"os"
	"path"
	"sync"
	"time"
)

// Dataset implements train.Dataset, and yields one image at a time.
// It pre-transforms the image to the target size.
//
// To do batching or shuffling, use
type Dataset struct {
	next     int
	size     int
	shuffle  []int
	toTensor *timage.ToTensorConfig

	mu sync.Mutex
}

// NewDataset returns a Dataset for one epoch that yields
// one image at time. It reads them from disk, and the parsing
// can be parallelized. See `data.NewParallelDataset`.
//
// The images are resized and cropped to `size x size` pixel,
// cut from the middle.
func NewDataset(dtype shapes.DType, size int) *Dataset {
	return &Dataset{
		size:     size,
		toTensor: timage.ToTensor(dtype),
	}
}

// Assert *Dataset implements train.Dataset
var _ train.Dataset = &Dataset{}

// Shuffle will shuffle the order of the images. This should be called before the
// start of an epoch.
//
// Once shuffled, every time the dataset is reset, it is reshuffled.
func (ds *Dataset) Shuffle() {
	ds.mu.Lock()
	ds.shuffleLocked()
	ds.mu.Unlock()
}

func (ds *Dataset) shuffleLocked() {
	if ds.shuffle == nil {
		ds.shuffle = make([]int, NumExamples)
	}
	for ii := 0; ii < NumExamples; ii++ {
		newPos := rand.Intn(ii + 1)
		if newPos == ii {
			ds.shuffle[ii] = ii
		} else {
			// Swap position with the new example.
			ds.shuffle[newPos], ds.shuffle[ii] = ii, ds.shuffle[newPos]
		}
	}
}

// nextIndex returns the next index and increments it.
// Concurrency safe.
// Returns -1 if reached the end of the dataset.
func (ds *Dataset) nextIndex() (index int) {
	ds.mu.Lock()
	defer ds.mu.Unlock()
	index = ds.next
	if ds.next < 0 {
		return
	}
	ds.next++
	if ds.shuffle != nil {
		index = ds.shuffle[index]
	}
	if ds.next >= NumExamples {
		ds.next = -1 // Indicates the end of epoch.
	}
	return
}

// Name implements train.Dataset interface.
func (ds *Dataset) Name() string {
	return "Oxford Flowers 102"
}

// Yield implements train.Dataset interface.
// It returns `ds` (the Dataset pointer) as spec.
//
// It yields one example at a time, each consists of:
//
//   - `inputs`: two values: the image itself and a scalar `int32` with
//     the index of the example. The index of the example can be used,
//     for instance, to split the dataset (into training/validation/test).
//   - `labels`: a `int32` value from 0 to 101 with the label.
func (ds *Dataset) Yield() (spec any, inputs []tensor.Tensor, labels []tensor.Tensor, err error) {
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

	// 1. Resize the smallest dimension to size, preserving ratio.
	width := img.Bounds().Dx()
	height := img.Bounds().Dy()
	if width < height {
		ratio := float64(width) / float64(ds.size)
		width = ds.size
		height = int(math.Round(float64(height) / ratio))
	} else if height < width {
		ratio := float64(height) / float64(ds.size)
		height = ds.size
		width = int(math.Round(float64(width) / ratio))
	} else {
		width = ds.size
		height = ds.size
	}
	img = imaging.Resize(img, width, height, imaging.Linear)

	// 2. Crop at center the largest dimension to size.
	if width > height {
		start := (width - ds.size) / 2
		img = imaging.Crop(img, image.Rect(start, 0, start+ds.size, ds.size))
	} else if height > width {
		start := (height - ds.size) / 2
		img = imaging.Crop(img, image.Rect(0, start, ds.size, start+ds.size))
	}

	inputs = []tensor.Tensor{ds.toTensor.Single(img), tensor.FromValue(index)}
	labels = []tensor.Tensor{tensor.FromValue(label)}
	return
}

// Reset implements train.Dataset interface.
func (ds *Dataset) Reset() {
	ds.mu.Lock()
	ds.next = 0
	ds.shuffleLocked()
	ds.mu.Unlock()
}

// InMemoryDataset creates a `data.InMemoryDataset` with the Oxford Flowers 102, of the given `size` for both, height
// and width -- image is resized and then cropped at the center.
//
// A cache version is automatically saved at the baseDir, if it is not empty. And if a cache file is found, it is
// used, instead of re-reading and processing all the images.
//
// One has to first call DownloadAndParse, otherwise it will immediately fail.
func InMemoryDataset(manager *Manager, baseDir string, size int) (mds *data.InMemoryDataset, err error) {
	var f *os.File
	if baseDir != "" {
		baseDir = data.ReplaceTildeInDir(baseDir) // If dir starts with "~", it is replaced.
		inMemoryCacheFile := path.Join(baseDir, fmt.Sprintf("cached_images_%dx%d.bin", size, size))

		defer func() {
			if err != nil {
				err = errors.WithMessagef(err, "File with cache: %q", inMemoryCacheFile)
			}
		}()

		f, err = os.Open(inMemoryCacheFile)
		if err == nil {
			// Reads from cached file.
			dec := gob.NewDecoder(f)
			mds, err = data.GobDeserializeInMemory(manager, dec)
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

	if NumExamples == 0 {
		err = errors.Errorf("Oxford Flowers 102 dataset hasn't been initialized yet, please call oxfordflowers102.DownloadAndParse() first")
	}

	// Create InMemoryDataset.
	start := time.Now()
	fmt.Printf("Creating InMemoryDataset with images cropped and scaled to %dx%d...\n", size, size)
	ds := NewDataset(shapes.UInt8, size)
	mds, err = data.InMemory(manager, data.Parallel(ds), false)
	elapsed := time.Since(start)
	fmt.Printf("\t- %s to process dataset.\n", elapsed)
	if err != nil {
		return
	}

	// Save to cache.
	if f != nil {
		enc := gob.NewEncoder(f)
		err = mds.GobSerialize(enc)
		if err != nil {
			return
		}
		err = f.Close()
	}
	return
}
