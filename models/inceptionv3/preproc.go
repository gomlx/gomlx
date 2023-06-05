package inceptionv3

import (
	. "github.com/gomlx/gomlx/graph"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"math"
)

// PreprocessImage makes the image in a format usable to InceptionV3 model.
//
// While the original model uses the values from -1 to 1 (see `maxValue` parameter),
// fine-tuning, or if not using the pre-trained weights work better (because of batch normalization)
// if this steps is skipped (set `maxValue=0`).
//
// It performs 2 tasks:
//
//   - It removes the alpha channel, in case it is provided.
//   - The minimum image size accepted by InceptionV3 is 75x75.
//     If any size is smaller than that, it will be resized accordingly, while preserving the aspect ratio.
//
// Input `image` must have a batch dimension (rank=4), be either 3 or 4 channels, and its
// values must be scaled from 0 to maxValue (except if it is set to -1).
func PreprocessImage(image *Node, channelsConfig timage.ChannelsAxisConfig) *Node {
	if image.Rank() != 4 {
		return image
	}

	// Remove alpha-channel, if given.
	shape := image.Shape()
	channelsAxis := timage.GetChannelsAxis(image, channelsConfig)
	if shape.Dimensions[channelsAxis] == 4 {
		axesRanges := make([]AxisRangeDef, image.Rank())
		for ii := range axesRanges {
			if ii == channelsAxis {
				axesRanges[ii] = AxisRange(0, 3)
			} else {
				axesRanges[ii] = AxisRange()
			}
		}
		image = Slice(image, axesRanges...)
	}

	// Scale to minimum size (75x75).
	spatialDims := timage.GetSpatialAxes(image, channelsConfig)
	upScale := 1.0
	for _, ii := range spatialDims {
		ratio := float64(MinimumImageSize) / float64(shape.Dimensions[ii])
		if ratio > upScale {
			upScale = ratio
		}
	}
	if upScale > 1.0 {
		newShape := image.Shape().Copy()
		for _, axis := range spatialDims {
			newSize := int(math.Round(float64(shape.Dimensions[axis]) * upScale))
			if newSize < MinimumImageSize {
				newSize = MinimumImageSize
			}
			newShape.Dimensions[axis] = newSize
		}
		image = Interpolate(image, newShape.Dimensions...).Done()
	}

	return image
}

// ScaleImageValuesKeras scales the `image` values from -1.0 to 1.0,
// assuming it is provided with values from 0.0 to `maxValue`.
//
// It doesn't work well in transfer learning.
// In particular, if not using the pre-trained weights, it seems to conflict with batch normalization.
func ScaleImageValuesKeras(image *Node, maxValue float64) *Node {
	image = MulScalar(image, 2.0/maxValue)
	image = AddScalar(image, -1.0)
	return image
}

// ScaleImageValuesTorch scales the `image` values to what, according to PyTorch version, was used
// during training.
// It assumes `image` has values from 0 to 255.0.
// It seems to work better in most cases.
//
// See:
// https://github.com/pytorch/vision/blob/6db1569c89094cf23f3bc41f79275c45e9fcb3f3/torchvision/models/inception.py#LL129C49-L129C86
func ScaleImageValuesTorch(image *Node) *Node {
	image = AddScalar(
		MulScalar(image, (0.229/0.5)),
		(0.485-0.5)/0.5)
	return image
}
