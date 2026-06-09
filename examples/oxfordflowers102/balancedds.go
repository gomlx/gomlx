// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package oxfordflowers102

import (
	"fmt"
	"math/rand/v2"
	"path"

	"iter"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/humanize"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/pkg/errors"
	"github.com/schollz/progressbar/v3"
)

// BalancedDataset implements a dataset that yields always one image from a different flower category.
type BalancedDataset struct {
	Backend              compute.Backend
	BaseDir              string
	Size                 int
	AllImages, AllLabels *tensors.Tensor
	PerFlowerIndices     [NumLabels][]int32
	Labels               *tensors.Tensor
}

// NewBalancedDataset creates a BalancedDataset, that yields always one image per flower category.
//
// That means the batch size is fixed in 102.
// It loops indefinitely, and it's always shuffled, hence it's meant to be used for training only.
//
// It caches the whole OxfordFlowers dataset in a large tensor, that is also yielded, along with the indices
// of the images to be used (using graph.Gather).
func NewBalancedDataset(backend compute.Backend, baseDir string, size int) (bds *BalancedDataset, err error) {
	baseDir = fsutil.MustReplaceTildeInDir(baseDir)
	imagesCachePath := fmt.Sprintf("all_images_%dx%d.tensor", size, size)
	labelsCachePath := fmt.Sprintf("all_labels_%dx%d.tensor", size, size)
	imagesCachePath = path.Join(baseDir, imagesCachePath)
	labelsCachePath = path.Join(baseDir, labelsCachePath)
	bds = &BalancedDataset{
		Backend: backend,
		BaseDir: baseDir,
		Size:    size,
	}
	if fsutil.MustFileExists(imagesCachePath) && fsutil.MustFileExists(labelsCachePath) {
		bds.AllImages, err = tensors.Load(imagesCachePath)
		if err != nil {
			err = errors.WithMessagef(err, "attempting to read cache file %q for NewBalancedDataset", imagesCachePath)
			return
		}
		bds.AllLabels, err = tensors.Load(labelsCachePath)
		if err != nil {
			err = errors.WithMessagef(err, "attempting to read cache file %q for NewBalancedDataset", labelsCachePath)
			return
		}
	} else {
		err = bds.readAllImages()
		if err != nil {
			return
		}
		err = bds.AllImages.Save(imagesCachePath)
		if err != nil {
			err = errors.WithMessagef(err, "attempting to write cache file %q for NewBalancedDataset", imagesCachePath)
			return
		}
		err = bds.AllLabels.Save(labelsCachePath)
		if err != nil {
			err = errors.WithMessagef(err, "attempting to write cache file %q for NewBalancedDataset", labelsCachePath)
			return
		}
	}
	bds.buildPerFlowerIndices()
	fmt.Printf("\timages: %s, %s\n", bds.AllImages.Shape(), humanize.Bytes(uint64(bds.AllImages.Shape().ByteSize())))
	fmt.Printf("\tlabels: %s, %s\n", bds.AllLabels.Shape(), humanize.Bytes(uint64(bds.AllLabels.Shape().ByteSize())))
	return
}

// Name implements train.Dataset.
func (bds *BalancedDataset) Name() string {
	return "train-balanced"
}

func (bds *BalancedDataset) buildPerFlowerIndices() {
	// We also build the labels of the batch, which will always be the same.
	bds.Labels = tensors.FromValue(xslices.Iota(int32(0), NumLabels))
	tensors.MustConstFlatData[int32](bds.AllLabels, func(flatAllLabels []int32) {
		for exampleIdx, label := range flatAllLabels {
			bds.PerFlowerIndices[label] = append(bds.PerFlowerIndices[label], int32(exampleIdx))
		}
	})
}

// readAllImages is called if failed to read from cache.
//
// Notice because it parallelizes the reading of the images, the order may not be the same as the order of the images
// downloaded.
func (bds *BalancedDataset) readAllImages() error {
	fmt.Println("> Generating cache for BalancedDataset.")
	err := DownloadAndParse(bds.BaseDir)
	if err != nil {
		return errors.WithMessagef(err, "NewBalancedDataset failed to download images")
	}
	dtype := dtypes.Uint8
	bds.AllImages = tensors.FromShape(shapes.Make(dtype, NumExamples, bds.Size, bds.Size, 3))
	imageSize := bds.Size * bds.Size * 3
	bds.AllLabels = tensors.FromShape(shapes.Make(dtypes.Int32, NumExamples))
	tensors.MustMutableFlatData[uint8](bds.AllImages, func(flatAllImages []uint8) {
		tensors.MustMutableFlatData[int32](bds.AllLabels, func(flatAllLabels []int32) {
			ds := dataset.Buffer(NewDataset(dtypes.Uint8, bds.Size))
			next, stop := iter.Pull2(ds.Iter())
			defer stop()
			pbar := progressbar.Default(int64(NumExamples), "Processing images")
			for exampleIdx := range NumExamples {
				batch, yieldErr, ok := next()
				if !ok {
					err = errors.Errorf("dataset ended early at example #%d", exampleIdx)
					return
				}
				if yieldErr != nil {
					err = errors.WithMessagef(yieldErr, "failed to reall all examples from normal dataset")
					return
				}
				image := batch.Inputs[0]
				label := batch.Labels[0]
				if err = image.Shape().Check(dtype, bds.Size, bds.Size, 3); err != nil {
					err = errors.WithMessagef(
						err,
						"unexpected image shape %s yielded by normal dataset, while building cache",
						image.Shape(),
					)
					return
				}
				if err = label.Shape().Check(dtypes.Int32); err != nil {
					err = errors.WithMessagef(
						err,
						"unexpected label shape %s yielded by normal dataset, while building cache",
						label.Shape(),
					)
				}
				tensors.MustConstFlatData[uint8](image, func(flatImage []uint8) {
					copy(flatAllImages[exampleIdx*imageSize:], flatImage)
				})
				flatAllLabels[exampleIdx] = tensors.ToScalar[int32](label)
				_ = batch.Finalize()
				_ = pbar.Add(1)
			}
			_ = pbar.Finish()
		})
	})

	return err
}

// Iter implements `train.Dataset` interface.
func (bds *BalancedDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for {
			examplesIdx := make([]int32, NumLabels)
			for flowerType := range NumLabels {
				choices := bds.PerFlowerIndices[flowerType]
				examplesIdx[flowerType] = choices[rand.N(len(choices))]
			}
			inputs := []*tensors.Tensor{bds.AllImages, tensors.FromValue(examplesIdx), bds.Labels}
			labels := []*tensors.Tensor{bds.Labels}
			batch := train.Batch{
				Spec:   bds,
				Inputs: inputs,
				Labels: labels,
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}
