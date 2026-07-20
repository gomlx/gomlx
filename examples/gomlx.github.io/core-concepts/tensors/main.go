// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"image"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/core/tensors"
	timages "github.com/gomlx/gomlx/core/tensors/images"
)

func main() {
	// Output to create
	fmt.Println("md:create")

	//md_start:create(-1)
	// Tensors can be created from Go values, such as multi-dimensional slices
	t := tensors.FromValue([][]float32{{1.0, 2.0}, {3.0, 4.0}})
	fmt.Printf("Tensor shape: %s\n", t.Shape())
	fmt.Printf("Tensor Go value: %v\n", t.Value())
	//md_end:create

	// Output to sync
	fmt.Println("md:sync")

	//md_start:sync(-1)
	// Tensors cache data both locally (host CPU) and on accelerator devices.
	// Transferring data between CPU and devices has a cost and is done lazily.
	fmt.Printf("Has local copy? %v\n", t.HasLocal())
	//md_end:sync

	// Output to finalize
	fmt.Println("md:finalize")

	//md_start:finalize(-1)
	// Tensors allocate memory on accelerator devices (GPU, TPU).
	// Because the Go Garbage Collector cannot track device memory,
	// you must finalize tensors that are no longer in use to prevent memory leaks.
	err := t.FinalizeAll()
	//md_end:finalize
	if err != nil {
		panic(err)
	}

	// Output to image
	fmt.Println("md:image")

	//md_start:image(-1)
	// Create two simple blank images (e.g. 100x100 RGB).
	img1 := image.NewRGBA(image.Rect(0, 0, 100, 100))
	img2 := image.NewRGBA(image.Rect(0, 0, 100, 100))

	// Load the batch of images into a Float32 tensor.
	// The resulting shape is [batch_size, height, width, channels].
	imagesTensor := timages.ToTensor(dtypes.Float32).Batch([]image.Image{img1, img2})
	fmt.Printf("Batch images shape: %s\n", imagesTensor.Shape())
	//md_end:image
}
