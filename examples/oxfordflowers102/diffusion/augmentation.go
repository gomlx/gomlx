package diffusion

import (
	"github.com/gomlx/gomlx/ml/context"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// AugmentImages applies random augmentations if context is set to training, otherwise it's a no-op.
// It takes as input a batch of images shaped [batchSize, height, width, channels], and it returns
// a randomly augmented batch of images with the exact same shape.
//
// Currently only random mirroring (left-to-right) is supported.
func AugmentImages(ctx *context.Context, images *Node) *Node {
	g := images.Graph()
	if !ctx.IsTraining(g) {
		// No-op if not training.
		return images
	}

	// Mirror on the horizontal axis 50% of the time.
	batchSize := images.Shape().Dim(0)
	return Where(
		ctx.RandomBernoulli(Const(g, 0.5), shapes.Make(dtypes.Bool, batchSize)), // 50% true, 50% false
		images,
		Reverse(images, 2 /* width axis */))
}
