// Package images provides several functions to transform images back and
// forth from tensors.
package images

import (
	"image"
	"log"
	"math"

	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/x448/float16"
	"k8s.io/klog/v2"
)

// ChannelsAxisConfig indicates if a tensor with an image has the channel axis
// coming last (last axis) or first (first axis after batch axis).
type ChannelsAxisConfig uint8

//go:generate go tool enumer -type=ChannelsAxisConfig images.go

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
		klog.Errorf("GetChannelsAxis(image, %s): invalid ChannelsAxisConfig!?", config)
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
		spatialAxes = xslices.Iota(2, numSpatialDims)
	case ChannelsLast:
		spatialAxes = xslices.Iota(1, numSpatialDims)
	default:
		klog.Errorf("GetSpatialAxes(image, %v): invalid ChannelsAxisConfig!?", config)
	}
	return
}

// GetUpSampledSizes returns the dimensions of an image tensor with the spatial
// dimensions up-sampled by the given factors. If only one factor is given, it is
// applied to all spatial dimensions.
//
// This can be used to up-sample an image, when combined with interpolation.
//
// Example: To double the size of an image (or video)
//
//	img = Interpolate(img, GetUpSampledSizes(img, ChannelsLast, 2)...).Done()
func GetUpSampledSizes(image shapes.HasShape, config ChannelsAxisConfig, factors ...int) (dims []int) {
	dims = image.Shape().Clone().Dimensions
	if len(factors) == 0 {
		return dims
	}
	spatialAxes := GetSpatialAxes(image, config)
	for ii, axis := range spatialAxes {
		factor := factors[0]
		if ii < len(factors) {
			factor = factors[ii]
		}
		dims[axis] *= factor
	}
	return dims
}

// ToTensorConfig holds the configuration returned by the ToTensor function. Once
// configured, use Single or Batch to actually convert.
type ToTensorConfig struct {
	channels int
	maxValue float64
	dtype    dtypes.DType
}

// ToTensor converts an image (or batch) a tensor.Local.
//
// It returns a configuration object that can be further configured. Once set, use Single or Batch
// methods to convert an image or a batch of images.
func ToTensor(dtype dtypes.DType) *ToTensorConfig {
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
//
// It panics in case of error.
func (tt *ToTensorConfig) Single(img image.Image) (t *tensors.Tensor) {
	return toTensorImpl(tt, []image.Image{img}, false)
}

// Batch converts the given images to a tensor, using the ToTensorConfig.
//
// It returns a 4D tensor, shaped as `[batch_size, height, width, channels]`.
//
// It panics in case of error.
func (tt *ToTensorConfig) Batch(images []image.Image) (t *tensors.Tensor) {
	return toTensorImpl(tt, images, true)
}

func toTensorImpl(tt *ToTensorConfig, images []image.Image, batch bool) (t *tensors.Tensor) {
	if !batch && len(images) != 1 {
		log.Printf("ToTensor asked for a single image, but given %d images", len(images))
		return nil
	}
	switch tt.dtype {
	case dtypes.Float32:
		t = toTensorGenericsImpl[float32](tt, images, batch)
	case dtypes.Float64:
		t = toTensorGenericsImpl[float64](tt, images, batch)
	case dtypes.Int8:
		t = toTensorGenericsImpl[int8](tt, images, batch)
	case dtypes.Int16:
		t = toTensorGenericsImpl[int16](tt, images, batch)
	case dtypes.Int32:
		t = toTensorGenericsImpl[int32](tt, images, batch)
	case dtypes.Int64:
		t = toTensorGenericsImpl[int64](tt, images, batch)
	case dtypes.Uint8:
		t = toTensorGenericsImpl[uint8](tt, images, batch)
	case dtypes.Uint16:
		t = toTensorGenericsImpl[uint16](tt, images, batch)
	case dtypes.Uint32:
		t = toTensorGenericsImpl[uint32](tt, images, batch)
	case dtypes.Uint64:
		t = toTensorGenericsImpl[uint64](tt, images, batch)
	case dtypes.Float16:
		t = toTensorGenericsImpl[float16.Float16](tt, images, batch)
	case dtypes.BFloat16:
		t = toTensorGenericsImpl[bfloat16.BFloat16](tt, images, batch)
	default:
		log.Printf("image.ToTensor does not support dtype %s", tt.dtype)
		t = nil
	}
	return
}

func toTensorGenericsImpl[T dtypes.NumberNotComplex | float16.Float16 | bfloat16.BFloat16](
	tt *ToTensorConfig, images []image.Image, batch bool) (t *tensors.Tensor) {
	if len(images) > 1 && !batch {
		Panicf("image.ToTensor in none-batch mode, but more than one image (%d) requested for conversion", len(images))
	}
	imgSize := images[0].Bounds().Size()
	dtype := dtypes.FromGenericsType[T]()
	if batch {
		t = tensors.FromShape(shapes.Make(dtype, len(images), imgSize.Y, imgSize.X, tt.channels))
	} else {
		t = tensors.FromShape(shapes.Make(dtype, imgSize.Y, imgSize.X, tt.channels))
	}

	// convertToDType converts RGBA channel value to the given DType.
	var convertToDType func(val uint32) T
	if dtype == dtypes.Float16 {
		convertToDType = func(val uint32) T {
			// color.RGBA() returns 16 bits values packaged in uint32.
			return T(float16.Fromfloat32(float32(val) * float32(tt.maxValue) / float32(0xFFFF)))
		}
	} else if dtype == dtypes.BFloat16 {
		convertToDType = func(val uint32) T {
			// color.RGBA() returns 16 bits values packaged in uint32.
			return T(bfloat16.FromFloat32(float32(val) * float32(tt.maxValue) / float32(0xFFFF)))
		}
	} else {
		convertToDType = func(val uint32) T {
			// color.RGBA() returns 16 bits values packaged in uint32.
			return T(float64(val) * tt.maxValue / float64(0xFFFF))
		}
	}

	t.MustMutableFlatData(func(flatAny any) {
		flat := flatAny.([]T)
		pos := 0 // Position in the flat slice.
		for imgIdx, img := range images {
			if !img.Bounds().Size().Eq(imgSize) {
				t.MustFinalizeAll()
				Panicf(
					"image[%d] has size %s, but image[0] has size %s -- they must all be the same",
					imgIdx, img.Bounds().Size(), imgSize)
			}
			switch tt.channels {
			case 3: // No alpha channel
				for y := 0; y < imgSize.Y; y++ {
					for x := 0; x < imgSize.X; x++ {
						r, g, b, _ := img.At(x, y).RGBA()
						for _, channel := range [3]uint32{r, g, b} {
							flat[pos] = convertToDType(channel)
							pos++
						}
					}
				}
			case 4: // Include alpha channel
				for y := 0; y < imgSize.Y; y++ {
					for x := 0; x < imgSize.X; x++ {
						r, g, b, a := img.At(x, y).RGBA()
						for _, channel := range [4]uint32{r, g, b, a} {
							flat[pos] = convertToDType(channel)
							pos++
						}
					}
				}
			}
		}
		if pos != t.Shape().Size() {
			Panicf(
				"images.ToTensor failed to set the values for all pixels (%d written out of %d)",
				pos,
				t.Shape().Size(),
			)
		}
	})
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
//
// It panics in case of error.
func (ti *ToImageConfig) Single(t *tensors.Tensor) (img image.Image) {
	var images []image.Image
	images = toImageImpl(ti, t)
	if len(images) > 0 {
		img = images[0]
	}
	return
}

// Batch converts the given 4D tensor shaped as `[batch_size, height, width, channels]`
// to a collection of images to a tensor, using the ToImageConfig.
//
// It panics in case of error.
func (ti *ToImageConfig) Batch(t *tensors.Tensor) (images []image.Image) {
	return toImageImpl(ti, t)
}

func toImageImpl(ti *ToImageConfig, imagesTensor *tensors.Tensor) (images []image.Image) {
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
		Panicf(
			"invalid tensor shape %s for images.ToImage conversion, must be either rank-3 or rank-4",
			imagesTensor.Shape(),
		)
	}
	if channels != 3 && channels != 4 {
		Panicf(
			"images.ToImage invalid tensor shape %s, with %d channels: only images with 3 or 4 channels are supported",
			imagesTensor.Shape(),
			channels,
		)
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
	case dtypes.Float32:
		images = toImageGenericsImpl[float32](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.Float64:
		images = toImageGenericsImpl[float64](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.Int8:
		images = toImageGenericsImpl[int8](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.Int16:
		images = toImageGenericsImpl[int16](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.Int32:
		images = toImageGenericsImpl[int32](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.Int64:
		images = toImageGenericsImpl[int64](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.Uint8:
		images = toImageGenericsImpl[uint8](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.Uint16:
		images = toImageGenericsImpl[uint16](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.Uint32:
		images = toImageGenericsImpl[uint32](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.Uint64:
		images = toImageGenericsImpl[uint64](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.Float16:
		images = toImageGenericsImpl[float16.Float16](imagesTensor, numImages, height, width, channels, maxValue)
	case dtypes.BFloat16:
		images = toImageGenericsImpl[bfloat16.BFloat16](imagesTensor, numImages, height, width, channels, maxValue)
	default:
		Panicf("images.ToImage cannot convert tensor of unsupported dtype %s to Image", dtype)
		return
	}
	return
}

func toImageGenericsImpl[T dtypes.NumberNotComplex | float16.Float16 | bfloat16.BFloat16](
	imagesTensor *tensors.Tensor, numImages, height, width, channels int, maxValue float64) (images []image.Image) {
	images = make([]image.Image, 0, numImages)
	tensorPos := 0
	isFloat16 := imagesTensor.DType() == dtypes.Float16
	isBFloat16 := imagesTensor.DType() == dtypes.BFloat16
	imagesTensor.MustConstFlatData(func(flatAny any) {
		tensorData := flatAny.([]T)
		for imageIdx := 0; imageIdx < numImages; imageIdx++ {
			img := image.NewNRGBA(image.Rect(0, 0, width, height))
			for h := 0; h < height; h++ {
				for w := 0; w < width; w++ {
					for d := 0; d < channels; d++ {
						var v float64
						if isFloat16 {
							v = float64(float16.Float16(tensorData[tensorPos]).Float32())
						} else if isBFloat16 {
							v = float64(bfloat16.BFloat16(tensorData[tensorPos]).Float32())
						} else {
							v = float64(tensorData[tensorPos])
						}
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
	})
	return
}
