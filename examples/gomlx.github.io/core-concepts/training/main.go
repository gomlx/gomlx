// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"image"
	"image/color"
	_ "image/jpeg"
	"image/png"
	"log"
	"math"
	"os"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/dataset"
	"github.com/gomlx/gomlx/ml/layers/activation"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/loss"
	"github.com/gomlx/gomlx/ml/train/optimizer"
)

// denseLayer defines a simple fully-connected layer.
func denseLayer(scope *model.Scope, x *Node, outputDims int) *Node {
	g := x.Graph()
	dtype := x.DType()
	inputDims := x.Shape().Dimensions[1]

	weights := scope.VariableWithShape("weights", shapes.Make(dtype, inputDims, outputDims)).NodeValue(g)
	biases := scope.VariableWithShape("biases", shapes.Make(dtype, 1, outputDims)).NodeValue(g)

	return Add(Dot(x, weights).Product(), biases)
}

// createSyntheticImage generates a simple colorful 40x40 image in case no input image is provided.
func createSyntheticImage() image.Image {
	img := image.NewRGBA(image.Rect(0, 0, 40, 40))
	for y := range 40 {
		for x := range 40 {
			r := uint8(x * 255 / 40)
			g := uint8(y * 255 / 40)
			b := uint8((x + y) * 255 / 80)
			img.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}
	return img
}

func main() {
	imagePath := flag.String("image", "", "path to image to overfit (PNG/JPEG)")
	outputPath := flag.String("output", "reconstructed.png", "path to save the reconstructed image (PNG)")
	resolution := flag.Int("resolution", 0, "resolution in pixels of the output reconstructed image (e.g. 200). If 0, same as input image")
	trainSteps := flag.Int("steps", 5_000, "Number of steps to train for")
	flag.Parse()

	// Load input image or fallback to a synthetic one
	var img image.Image
	if *imagePath != "" {
		f, err := os.Open(*imagePath)
		if err != nil {
			log.Fatalf("failed to open image %q: %v", *imagePath, err)
		}
		img, _, err = image.Decode(f)
		f.Close()
		if err != nil {
			log.Fatalf("failed to decode image: %v", err)
		}
	} else {
		img = createSyntheticImage()
	}

	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	fmt.Printf("Image loaded: %dx%d pixels\n", width, height)

	// Output to training
	fmt.Println("md:training")

	//md_start:training
	// 1. Prepare training data: mapping (x, y) coordinates to (r, g, b) colors
	inputs := make([][]float32, 0, width*height)
	labels := make([][]float32, 0, width*height)

	for y := range height {
		for x := range width {
			// Normalize coordinates to [-1.0, 1.0] for stable neural network training
			nx := float32(x)/float32(width)*2.0 - 1.0
			ny := float32(y)/float32(height)*2.0 - 1.0
			inputs = append(inputs, []float32{nx, ny})

			r, g, b, _ := img.At(bounds.Min.X+x, bounds.Min.Y+y).RGBA()
			labels = append(labels, []float32{float32(r) / 65535.0, float32(g) / 65535.0, float32(b) / 65535.0})
		}
	}

	backend := compute.MustNew()
	store := model.NewStore()

	// 2. Create an InMemoryDataset from the prepared coordinates and pixel colors
	ds, err := dataset.InMemoryFromData(backend, "image_pixels", []any{inputs}, []any{labels})
	if err != nil {
		log.Fatalf("failed to create dataset: %v", err)
	}
	// Configure dataset to yield random batches of size 512 continuously (infinitely)
	ds.BatchSize(512, false).Shuffle().Infinite(true)

	// 3. Define the neural network model function (MLP)
	modelFn := func(scope *model.Scope, spec any, inputs []*Node) []*Node {
		x := inputs[0] // shape: [batch_size, 2]

		h := denseLayer(scope.In("layer1"), x, 64)
		h = activation.Relu(h)
		h = denseLayer(scope.In("layer2"), h, 64)
		h = activation.Relu(h)
		h = denseLayer(scope.In("layer3"), h, 64)
		h = activation.Relu(h)

		// Output RGB values mapped to [0.0, 1.0] using Sigmoid
		y := Sigmoid(denseLayer(scope.In("output"), h, 3))
		return []*Node{y}
	}

	// 4. Initialize Trainer with Adam optimizer and Mean Squared Error (MSE) loss
	trainer := train.NewTrainer(
		backend,
		store,
		modelFn,
		loss.MeanSquaredError,
		optimizer.Adam().LearningRate(0.003).Done(),
		nil, // Train step metrics (optional)
		nil, // Eval metrics (optional)
	)

	// 5. Run the training loop for -steps (default 5000) steps
	loop := train.NewLoop(trainer)
	// Register a simple callback to print loss metrics periodically
	train.EveryNSteps(loop, 1000, "log_metrics", 0, func(l *train.Loop, metrics []*tensors.Tensor) error {
		fmt.Printf("Step %5d: MSE Loss = %.6f (moving average = %.6f)\n", l.LoopStep, metrics[0].Value(), metrics[1].Value())
		return nil
	})

	fmt.Println("Starting training loop...")
	_, err = loop.RunSteps(ds, *trainSteps)
	if err != nil {
		log.Fatalf("training failed: %v", err)
	}
	fmt.Println("Training finished!")
	//md_end:training

	// 6. Predict the colors for all pixel coordinates to reconstruct the image
	predictExec := model.MustNewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
		return modelFn(scope, nil, []*Node{x})[0]
	})

	outWidth, outHeight := width, height
	if *resolution > 0 {
		outWidth = *resolution
		outHeight = *resolution
	}

	gridInputs := make([][]float32, 0, outWidth*outHeight)
	for y := range outHeight {
		for x := range outWidth {
			nx := float32(x)/float32(outWidth)*2.0 - 1.0
			ny := float32(y)/float32(outHeight)*2.0 - 1.0
			gridInputs = append(gridInputs, []float32{nx, ny})
		}
	}

	predictedTensor := predictExec.MustCall1(tensors.FromValue(gridInputs))
	predictedColors := predictedTensor.Value().([][]float32)
	predictedTensor.MustFinalizeAll()

	// Write predicted colors back to a new image file
	outImg := image.NewRGBA(image.Rect(0, 0, outWidth, outHeight))
	pixelIdx := 0
	for y := 0; y < outHeight; y++ {
		for x := 0; x < outWidth; x++ {
			c := predictedColors[pixelIdx]
			pixelIdx++
			r := uint8(math.Min(math.Max(float64(c[0]), 0), 1) * 255.0)
			g := uint8(math.Min(math.Max(float64(c[1]), 0), 1) * 255.0)
			b := uint8(math.Min(math.Max(float64(c[2]), 0), 1) * 255.0)
			outImg.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}

	outPath := *outputPath
	outFile, err := os.Create(outPath)
	if err != nil {
		log.Fatalf("failed to create output file: %v", err)
	}
	err = png.Encode(outFile, outImg)
	outFile.Close()
	if err != nil {
		log.Fatalf("failed to encode output image: %v", err)
	}
	fmt.Printf("Successfully reconstructed image saved to %s\n", outPath)
}
