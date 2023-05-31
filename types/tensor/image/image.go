// Package image provides several functions to transform images back and
// forth from tensors.
package image

import (
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"image"
	"k8s.io/klog/v2"
	"log"
	"math"
	"reflect"
)

// ChannelsAxisConfig indicates if a tensor with an image has the channels axis
// coming last (last axis) or first (first axis after batch axis).
type ChannelsAxisConfig uint8

const (
	ChannelsFirst ChannelsAxisConfig = iota
	ChannelsLast
)

// GetChannelsAxis from a given image tensor and configuration. It assumes the
// leading axis is for the batch dimension. So it either returns 1 or
// `image.Rank()-1`.
func GetChannelsAxis(image shapes.HasShape, config ChannelsAxisConfig) int {
	switch config {
	case ChannelsFirst:
		return 1
	case ChannelsLast:
		return image.Shape().Rank() - 1
	default:
		klog.Errorf("GetChannelsAxis(image, %v): invalid ChannelsAxisConfig!?", config)
		return -1
	}
}

// GetSpatialAxes from a given image tensor and configuration. It assumes the
// leading axis is for the batch dimension.
//
// Example: if image has shape `[batch_dim, height, width, channels]`, it will
// return `[]int{1, 2}`.
func GetSpatialAxes(image shapes.HasShape, config ChannelsAxisConfig) (spatialAxes []int) {
	numSpatialDims := image.Shape().Rank() - 2
	if numSpatialDims <= 0 {
		return
	}
	switch config {
	case ChannelsFirst:
		spatialAxes = slices.IotaSlice(2, numSpatialDims)
	case ChannelsLast:
		spatialAxes = slices.IotaSlice(1, numSpatialDims)
	default:
		klog.Errorf("GetSpatialAxes(image, %v): invalid ChannelsAxisConfig!?", config)
	}
	return
}

// ToTensorConfig holds the configuration returned by the ToTensor function. Once
// configured, use Single or Batch to actually convert.
type ToTensorConfig struct {
	channels int
	maxValue float64
	dtype    shapes.DType
}

// ToTensor converts an image (or batch) a tensor.Local.
//
// It returns a configuration object that can be further configured. Once set, use Single or Batch
// methods to convert an image or a batch of images.
func ToTensor(dtype shapes.DType) *ToTensorConfig {
	tt := &ToTensorConfig{
		channels: 3,
		maxValue: 1.0,
		dtype:    dtype,
	}
	if !dtype.IsFloat() {
		// Use 255 for integer types.
		tt.maxValue = 255.0
	}
	return tt
}

// WithAlpha configures ToTensorConfig object to include the alpha channel in the conversion,
// so the converted tensor will have 4 channels. The default is dropping the alpha channel.
//
// It returns the ToTensorConfig object, so configuration calls can be cascaded.
func (tt *ToTensorConfig) WithAlpha() *ToTensorConfig {
	tt.channels = 4
	return tt
}

// MaxValue sets the MaxValue of each channel. It defaults to 1.0 for float dtypes (Float32
// and Float64) and 255 for integer types.
//
// Notice while this value is given as float64, it is converted to the corresponding dtype.
//
// It returns the ToTensorConfig object, so configuration calls can be cascaded.
func (tt *ToTensorConfig) MaxValue(v float64) *ToTensorConfig {
	tt.maxValue = v
	return tt
}

// Single converts the given img to a tensor, using the ToTensorConfig.
//
// It returns a 3D tensor, shaped as `[height, width, channels]`.
func (tt *ToTensorConfig) Single(img image.Image) (t *tensor.Local) {
	return toTensorImpl(tt, []image.Image{img}, false)
}

// Batch converts the given images to a tensor, using the ToTensorConfig.
//
// It returns a 4D tensor, shaped as `[batch_size, height, width, channels]`.
func (tt *ToTensorConfig) Batch(images []image.Image) (t *tensor.Local) {
	return toTensorImpl(tt, images, true)
}

func toTensorImpl(tt *ToTensorConfig, images []image.Image, batch bool) (t *tensor.Local) {
	if !batch && len(images) != 1 {
		log.Printf("ToTensor asked for a single image, but given %d images", len(images))
		return nil
	}
	switch tt.dtype {
	case shapes.Float32:
		t = toTensorGenericsImpl[float32](tt, images, batch)
	case shapes.Float64:
		t = toTensorGenericsImpl[float64](tt, images, batch)
	case shapes.Int32:
		t = toTensorGenericsImpl[int32](tt, images, batch)
	case shapes.Int64:
		t = toTensorGenericsImpl[int](tt, images, batch)
	case shapes.UInt8:
		t = toTensorGenericsImpl[uint8](tt, images, batch)
	default:
		log.Printf("image.ToTensor does not support dtype %s", tt.dtype)
		t = nil
	}
	return
}

func toTensorGenericsImpl[T shapes.Number](tt *ToTensorConfig, images []image.Image, batch bool) (t *tensor.Local) {
	var zero T
	if len(images) > 1 && !batch {
		return tensor.MakeLocalWithError(errors.Errorf("image.ToTensor in none-batch mode, but more than one image (%d) requested for conversion", len(images)))
	}
	imgSize := images[0].Bounds().Size()
	if batch {
		t = tensor.FromShape(shapes.Make(shapes.DTypeForType(reflect.TypeOf(zero)), len(images), imgSize.Y, imgSize.X, tt.channels))
	} else {
		t = tensor.FromShape(shapes.Make(shapes.DTypeForType(reflect.TypeOf(zero)), imgSize.Y, imgSize.X, tt.channels))
	}
	tensorData := t.Data().([]T)
	pos := 0

	convertToDType := func(val uint32) T {
		// color.RGBA() returns 16 bits values packaged in uint32.
		return T(float64(val) * tt.maxValue / float64(0xFFFF))
	}

	for imgIdx, img := range images {
		if !img.Bounds().Size().Eq(imgSize) {
			t.Finalize()
			return tensor.MakeLocalWithError(errors.Errorf(
				"image[%d] has size %s, but image[0] has size %s -- they must all be the same",
				imgIdx, img.Bounds().Size(), imgSize))
		}
		switch tt.channels {
		case 3: // No alpha channel
			for y := 0; y < imgSize.Y; y++ {
				for x := 0; x < imgSize.X; x++ {
					r, g, b, _ := img.At(x, y).RGBA()
					for _, channel := range [3]uint32{r, g, b} {
						tensorData[pos] = convertToDType(channel)
						pos++
					}
				}
			}
		case 4: // Include alpha channel
			for y := 0; y < imgSize.Y; y++ {
				for x := 0; x < imgSize.X; x++ {
					r, g, b, a := img.At(x, y).RGBA()
					for _, channel := range [4]uint32{r, g, b, a} {
						tensorData[pos] = convertToDType(channel)
						pos++
					}
				}
			}
		}
	}
	if pos != t.Shape().Size() {
		log.Fatalf("Incorrect number of values set.")
	}
	return
}

// ToImageConfig holds the configuration returned by the ToImage function. Once
// configured, use Single or Batch to actually convert a tensor to image(s).
type ToImageConfig struct {
	maxValue float64
}

// ToImage returns a configuration that can be used to convert tensors to Images.
// Use Single or Batch to convert single images or batch of images at once.
//
// For now, it only supports `*image.NRGBA` image type.
func ToImage() *ToImageConfig {
	return &ToImageConfig{}
}

// MaxValue sets the MaxValue of each channel. It defaults to 1.0 for float dtypes (Float32
// and Float64) and 255 for integer types.
//
// Notice while this value is given as float64, it is converted to the corresponding dtype.
//
// It returns the ToImageConfig object, so configuration calls can be cascaded.
func (ti *ToImageConfig) MaxValue(v float64) *ToImageConfig {
	ti.maxValue = v
	return ti
}

// Single converts the given 3D tensor shaped as `[height, width, channels]`
// with an image to a tensor, using the ToImageConfig.
func (ti *ToImageConfig) Single(t tensor.Tensor) (img image.Image, err error) {
	var images []image.Image
	images, err = toImageImpl(ti, t)
	if err != nil {
		return
	}
	if len(images) > 0 {
		img = images[0]
	}
	return
}

// Batch converts the given 4D tensor shaped as `[batch_size, height, width, channels]`
// to a collection of images to a tensor, using the ToImageConfig.
func (ti *ToImageConfig) Batch(t tensor.Tensor) (images []image.Image, err error) {
	return toImageImpl(ti, t)
}

func toImageImpl(ti *ToImageConfig, imagesTensor tensor.Tensor) (images []image.Image, err error) {
	var numImages, width, height, channels int
	switch imagesTensor.Rank() {
	case 3:
		numImages = 1
		height = imagesTensor.Shape().Dimensions[0]
		width = imagesTensor.Shape().Dimensions[1]
		channels = imagesTensor.Shape().Dimensions[2]
	case 4:
		numImages = imagesTensor.Shape().Dimensions[0]
		height = imagesTensor.Shape().Dimensions[1]
		width = imagesTensor.Shape().Dimensions[2]
		channels = imagesTensor.Shape().Dimensions[3]
	default:
		err = errors.Errorf("invalid tensor shape %s for ToImage conversion, must be either rank-3 or rank-4", imagesTensor.Shape())
		return
	}
	if channels != 3 && channels != 4 {
		err = errors.Errorf("invalid tensor shape %s, with %d channels: only images with 3 or 4 channels are supported", imagesTensor.Shape(), channels)
		return
	}
	maxValue := ti.maxValue
	if maxValue == 0 {
		if imagesTensor.DType().IsFloat() {
			maxValue = 1.0
		} else {
			maxValue = 255.0
		}
	}
	dtype := imagesTensor.DType()
	switch dtype {
	case shapes.Float32:
		images = toImageGenericsImpl[float32](imagesTensor, numImages, height, width, channels, maxValue)
	case shapes.Float64:
		images = toImageGenericsImpl[float64](imagesTensor, numImages, height, width, channels, maxValue)
	case shapes.Int32:
		images = toImageGenericsImpl[int32](imagesTensor, numImages, height, width, channels, maxValue)
	case shapes.Int64:
		images = toImageGenericsImpl[int](imagesTensor, numImages, height, width, channels, maxValue)
	case shapes.UInt8:
		images = toImageGenericsImpl[uint8](imagesTensor, numImages, height, width, channels, maxValue)
	default:
		err = errors.Errorf("cannot convert unsupported dtype %s to images.image", dtype)
		return
	}
	return
}

func toImageGenericsImpl[T shapes.Number](imagesTensor tensor.Tensor, numImages, height, width, channels int, maxValue float64) (images []image.Image) {
	images = make([]image.Image, 0, numImages)
	tensorPos := 0
	imagesRef := imagesTensor.Local().AcquireData()
	defer imagesRef.Release()
	tensorData := tensor.FlatFromRef[T](imagesRef)
	for imageIdx := 0; imageIdx < numImages; imageIdx++ {
		img := image.NewNRGBA(image.Rect(0, 0, width, height))
		for h := 0; h < height; h++ {
			for w := 0; w < width; w++ {
				for d := 0; d < channels; d++ {
					v := float64(tensorData[tensorPos])
					tensorPos++
					v = math.Round(255 * (v / maxValue))
					img.Pix[h*img.Stride+w*4+d] = uint8(v)
				}
				if channels < 4 {
					img.Pix[h*img.Stride+w*4+3] = uint8(255) // Alpha channel.
				}
			}
		}
		images = append(images, img)
	}
	return
}
