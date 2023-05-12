package oxfordflowers102

import (
	"github.com/disintegration/imaging"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"image"
	"math"
	"sync"
)

// Dataset implements train.Dataset, and yields one image at a time.
// It pre-transforms the image to the target size.
//
// To do batching or shuffling, use
type Dataset struct {
	next int
	size int
	mu   sync.Mutex
}

// NewDataset returns a Dataset for one epoch that yields
// one image at time. It reads them from disk, and the parsing
// can be parallelized. See `data.NewParallelDataset`.
//
// The images are resized and cropped to `size x size` pixel,
// cut from the middle.
func NewDataset(size int) train.Dataset {
	return &Dataset{size: size}
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
	if ds.next >= NumExamples {
		ds.next = -1 // Indicates the end of epoch.
	}
	return
}

// Name implements train.Dataset interface.
func (ds *Dataset) Name() string {
	return "Oxford Flowers 102 Raw Dataset"
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

	inputs = []tensor.Tensor{tensor.FromValue(index)}
	labels = []tensor.Tensor{tensor.FromValue(label)}
	return
}

// Reset implements train.Dataset interface.
func (ds *Dataset) Reset() {
	ds.mu.Lock()
	ds.next = 0
	ds.mu.Unlock()
}
