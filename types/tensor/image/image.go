// Package image provides several functions to transform images back and
// forth from tensors.
package image

import (
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"image"
	"log"
	"reflect"
)

// ImagesToTensor converts a batch of images to a tensor.Local.
//
// The dtype used will be derived from the type T. The `maxValue` defines the
// maximum value (if all channels set to this value, this means white).
//
// `numChannels` can be set to 1, 3 or 4.
func ToTensor[T shapes.Number](imgs []image.Image, maxValue T, includeAlpha bool) (t *tensor.Local) {
	var zero T
	imgSize := imgs[0].Bounds().Size()
	t = tensor.FromShape(shapes.Make(shapes.DTypeForType(reflect.TypeOf(zero)), len(imgs), imgSize.Y, imgSize.X, 4))
	tensorData := t.Data().([]T)
	pos := 0

	convertToDType := func(val uint32) T {
		// color.RGBA() returns 16 bits values packaged in uint32.
		// FutureWork: it will require a specialized version for types like
		//   uint16 or uint8
		return T(val) * maxValue / T(0xFFFF)
	}

	for imgIdx, img := range imgs {
		if !img.Bounds().Size().Eq(imgSize) {
			t.Finalize()
			return tensor.LocalWithError(errors.Errorf(
				"image[%d] has size %s, but image[0] has size %s -- they must all be the same",
				imgIdx, img.Bounds().Size(), imgSize))
		}
		for y := 0; y < imgSize.X; y++ {
			for x := 0; x < imgSize.Y; x++ {
				r, g, b, a := img.At(x, y).RGBA()
				for _, channel := range []uint32{r, g, b, a} {
					tensorData[pos] = convertToDType(channel)
					pos++
				}
			}
		}
	}
	if pos != t.Shape().Size() {
		log.Fatalf("Incorrect number of values set.")
	}
	return
}
