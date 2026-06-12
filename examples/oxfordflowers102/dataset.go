// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package oxfordflowers102

import (
	"encoding/gob"
	"fmt"
	"image"
	"math"
	"math/rand"
	"os"
	"path"
	"time"

	"github.com/disintegration/imaging"
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/core/tensors"
	timage "github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/support/fsutil"
	"iter"
	"github.com/pkg/errors"
)

// Dataset implements train.Dataset, and yields one image at a time.
// It pre-transforms the image to the target `imageSize`.
type Dataset struct {
	name      string
	shuffled  bool
	imageSize int
	toTensor  *timage.ToTensorConfig

	partitionSelection []bool // Which indices to take.
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
	ds.shuffled = true
	ds.name = ds.name + " [shuffled]"
	return ds
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

// Name implements train.Dataset interface.
func (ds *Dataset) Name() string {
	return ds.name
}

// Iter implements `train.Dataset` interface.
func (ds *Dataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		var shuffle []int
		if ds.shuffled {
			shuffle = xslices.Iota(0, NumExamples)
			for ii := 0; ii < NumExamples; ii++ {
				swapPos := rand.Intn(NumExamples)
				shuffle[ii], shuffle[swapPos] = shuffle[swapPos], shuffle[ii]
			}
		}

		next := 0

		nextIndex := func() int {
			for {
				if next < 0 || next >= NumExamples {
					return -1
				}
				index := next
				next++

				if shuffle != nil {
					index = shuffle[index]
				}

				if len(ds.partitionSelection) > 0 {
					if !ds.partitionSelection[index] {
						continue
					}
				}
				return index
			}
		}

		for {
			index := nextIndex()
			if index == -1 {
				return
			}
			img, label, err := ReadExample(index)
			if err != nil {
				yield(train.Batch{}, errors.WithMessagef(err, "failed to read image/label #%d", index))
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

			inputs := []*tensors.Tensor{ds.toTensor.Single(img), tensors.FromValue(index), tensors.FromValue(label)}
			labels := []*tensors.Tensor{tensors.FromValue(label)}
			batch := train.Batch{
				Spec:   ds,
				Inputs: inputs,
				Labels: labels,
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
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
func InMemoryDataset(backend compute.Backend, baseDir string, imageSize int, name string,
	partitionSeed int64, partitionFrom, partitionTo float64) (
	inMemoryDataset *dataset.InMemoryDataset, err error) {
	deviceNum := compute.DeviceNum(0)
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
			inMemoryDataset, err = dataset.GobDeserializeInMemoryToDevice(backend, deviceNum, dec)
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
	inMemoryDataset, err = dataset.InMemory(backend, dataset.Buffer(ds), false)
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
