package inceptionv3

import (
	. "github.com/gomlx/gomlx/graph"
)

// PreprocessImage takes `image` with values from 0 to `maxValue`, and it scales
// it -1.0 to 1.0, the values used in the training of the InceptionV3 model. It works
// for batches of images also.
func PreprocessImage(image *Node, maxValue float64) *Node {
	image = MulScalar(image, 1.0/127.5)
	image = AddScalar(image, -1.0)
	return image
}
