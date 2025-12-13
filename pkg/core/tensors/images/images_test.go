package images

import (
	"fmt"
	"image"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/dtypes/bfloat16"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"github.com/x448/float16"
)

func TestGetUpSampledSizes(t *testing.T) {
	s := shapes.Make(dtypes.Float32, 2, 3, 4, 5)
	assert.Equal(t, []int{2, 6, 8, 5}, GetUpSampledSizes(s, ChannelsLast, 2))
	assert.Equal(t, []int{2, 3, 12, 15}, GetUpSampledSizes(s, ChannelsFirst, 3))
}

func testTensorToFromImageImpl[T dtypes.NumberNotComplex | float16.Float16 | bfloat16.BFloat16](
	t *testing.T, img *image.NRGBA) {
	dtype := dtypes.FromGenericsType[T]()
	tensor := ToTensor(dtype).WithAlpha().Single(img)
	fmt.Printf("\ttensor.shape=%s\n", tensor.Shape())
	require.NoError(t, tensor.Shape().Check(dtype, 2, 3, 4))
	convertedImg := ToImage().Single(tensor)
	require.Equal(t, img.Bounds(), convertedImg.Bounds())
	for y := range 2 {
		for x := range 3 {
			require.Equal(t, img.At(x, y), convertedImg.At(x, y))
			//fmt.Printf("\t\t%d,%d -> %v/%v\n", x, y, img.At(x, y), convertedImg.At(x, y))
		}
	}
}

func TestTensorToFromImage(t *testing.T) {
	img := image.NewNRGBA(image.Rect(0, 0, 3, 2))
	require.Len(t, img.Pix, 6*4)
	copy(img.Pix, []uint8{
		1, 1, 1, 255,
		3, 3, 3, 255,
		5, 5, 5, 255,
		10, 10, 10, 255,
		30, 30, 30, 255,
		50, 50, 50, 255})

	testTensorToFromImageImpl[float32](t, img)
	testTensorToFromImageImpl[float64](t, img)
	testTensorToFromImageImpl[int](t, img)
	testTensorToFromImageImpl[int8](t, img)
	testTensorToFromImageImpl[int16](t, img)
	testTensorToFromImageImpl[int32](t, img)
	testTensorToFromImageImpl[int64](t, img)
	testTensorToFromImageImpl[uint8](t, img)
	testTensorToFromImageImpl[uint16](t, img)
	testTensorToFromImageImpl[uint32](t, img)
	testTensorToFromImageImpl[uint64](t, img)

	testTensorToFromImageImpl[float16.Float16](t, img)
	testTensorToFromImageImpl[bfloat16.BFloat16](t, img)
}
