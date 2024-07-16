package inceptionv3

import (
	"github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	timage "github.com/gomlx/gomlx/types/tensor/image"
	"math"
)

// PreprocessImage makes the image in a format usable to InceptionV3 model.
//
// It performs 3 tasks:
//
//   - Scales the values from -1.0 to 1.0: this is how it was originally trained.
//     It requires `maxValue` to be carefully set to the maxValue of the images --
//     it is assumed the images are scaled from 0 to `maxValue`. Set `maxValue` to zero
//     to skip this step.
//   - It removes the alpha channel, in case it is provided.
//   - The minimum image size accepted by InceptionV3 is 75x75.
//     If any size is smaller than that, it will be resized accordingly, while preserving the aspect ratio.
//
// Input `image` must have a batch dimension (rank=4), be either 3 or 4 channels, and its
// values must be scaled from 0 to maxValue (except if it is set to -1).
func PreprocessImage(image *Node, maxValue float64, channelsConfig timage.ChannelsAxisConfig) *Node {
	if image.Rank() != 4 {
		exceptions.Panicf("inceptionv3.PreprocessImage requires image to be rank-4, got rank-%d instead", image.Rank())
	}

	// Scale image values from -1.0 to 1.0.
	if maxValue > 0 {
		image = MulScalar(image, 2.0/maxValue)
		image = AddScalar(image, -1.0)
	}

	// Remove alpha-channel, if given.
	shape := image.Shape()
	channelsAxis := timage.GetChannelsAxis(image, channelsConfig)
	if shape.Dimensions[channelsAxis] == 4 {
		axesRanges := make([]SliceAxisSpec, image.Rank())
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
		newShape := image.Shape().Clone()
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

// ScaleImageValues scales the `image` values from -1.0 to 1.0,
// assuming it is provided with values from 0.0 to `maxValue`.
//
// This is presumably how the model was trained, so one would want this if using the pre-trained weights.
// But not necessary if training from scratch.
//
// Careful with setting maxValue, setting it wrong can cause odd behavior. It's recommended checking.
func ScaleImageValues(image *Node, maxValue float64) *Node {
	return image
}
