package layers

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
)

// DropBlockConfig is created with a DropBlock.
type DropBlockConfig struct {
	ctx                *context.Context
	x                  *Node
	dropoutProbability *Node
	channelsAxisConfig images.ChannelsAxisConfig
	isFullShape        bool
	blockSizes         []int
	blockSize          int
}

var (
	// ParamDropBlockProbability is the hyperparameter that sets the default DropBlock dropout probability. Default is float64(0.0).
	ParamDropBlockProbability = "dropblock_prob"

	// ParamDropBlockSize is the hyperparameter that set the default DropBlock block size. Default is int(3).
	ParamDropBlockSize = "dropblock_size"
)

// DropBlock implements a block dropout, as described in [1].
//
// Parameters:
//   - ctx: the Context, and it's used to check if it is training, and the default dropout probability and block
//     size as a hyperparameter.
//     DropBlock is a no-op during inference (not training) or if no dropout probability was configured.
//   - x: operand to be regularized with block dropout. The default is to assume x is shaped
//     [batchSize, <...spatial axes...>, channelsAxis]. Use ChannelsAxis or FullShape to configure that.
//
// The dropout probability and the block sizes are read by default from the context hyperparameters
// (see ParamDropBlockProbability and ParamDropBlockSize). But they can be optionally configured as well.
//
// It returns a configuration object, that can be further configured.
// Call Done when finished configuring and it will return the regularized value.
//
// [1] "DropBlock: A regularization method for convolutional networks", Golnaz Ghiasi, Tsung{-}Yi Lin and Quoc V. Le, https://arxiv.org/abs/1810.12890v1
func DropBlock(ctx *context.Context, x *Node) *DropBlockConfig {
	cfg := &DropBlockConfig{
		ctx: ctx,
		x:   x,
	}
	prob := context.GetParamOr(ctx, ParamDropBlockProbability, 0.0)
	cfg.WithDropoutProbability(prob)
	blockSize := context.GetParamOr(ctx, ParamDropBlockSize, 3)
	cfg.WithBlockSize(blockSize)
	return cfg
}

// ChannelsAxis configures the axis for the channels (aka. "depth" or "features") dimension. The default is
// `images.ChannelsLast`, meaning the "channels" dimension comes last.
//
// Note: `images` refers to package `github.com/gomlx/gomlx/types/tensor/image`.
//
// If you don't want to exclude the batch size and channels from the pooling, use FullShape instead.
//
// It returns the modified Config object, so calls can be cascaded.
func (cfg *DropBlockConfig) ChannelsAxis(channelsAxisConfig images.ChannelsAxisConfig) *DropBlockConfig {
	cfg.channelsAxisConfig = channelsAxisConfig
	cfg.isFullShape = false
	return cfg
}

// FullShape configures the DropBlock operation to consider all its axes as part of the block, with no special
// considerations for the batch or channel axes.
//
// See ChannelsAxis to handle batch and channels specially, and BlockSizes to set the block size
// for each specific axis.
//
// The default is configured with ChannelsAxis(images.ChannelsLast).
func (cfg *DropBlockConfig) FullShape() *DropBlockConfig {
	cfg.isFullShape = true
	return cfg
}

// WithDropoutProbability configures the dropout probability to use.
// It is the expectation of ratio of "pixels" (or voxels if 3d) that will be dropped out in blocks
// from the image (if 2D).
//
// By default, it reads ParamDropBlockProbability hyperparameter from ctx.
func (cfg *DropBlockConfig) WithDropoutProbability(prob float64) *DropBlockConfig {
	if prob <= 0 {
		cfg.dropoutProbability = nil
		return cfg
	}
	cfg.dropoutProbability = Scalar(cfg.x.Graph(), dtypes.Float32, prob)
	return cfg
}

// WithDropoutProbabilityNode configures the dropout probability to use with a dynamic (graph.Node) probability.
// prob must be a Float32 scalar node.
// It is the expectation of ratio of "pixels" (or voxels if 3d) that will be dropped out in blocks
// from the image (if 2D).
//
// By default, it reads ParamDropBlockProbability hyperparameter from ctx and use that as a constant.
func (cfg *DropBlockConfig) WithDropoutProbabilityNode(prob *Node) *DropBlockConfig {
	if !prob.IsScalar() || prob.DType() != dtypes.Float32 {
		exceptions.Panicf("DropBlockConfig.WithDropoutProbabilityNode requires prob to be a scalar of type dtypes.Float32, got %s",
			prob.Shape())
	}
	cfg.dropoutProbability = prob
	return cfg
}

// WithBlockSizes configures a different block size per spatial dimension. The same number of values
// must be set here as the number of SpatialAxes.
//
// Default is 3 for every spatial axis. See also WithBlockSize.
func (cfg *DropBlockConfig) WithBlockSizes(blockSizes ...int) *DropBlockConfig {
	cfg.blockSizes = blockSizes
	cfg.blockSize = 0
	return cfg
}

// WithBlockSize sets the block size to be used for all spatial axes.
//
// The default is 3.
func (cfg *DropBlockConfig) WithBlockSize(blockSize int) *DropBlockConfig {
	cfg.blockSize = blockSize
	cfg.blockSizes = nil
	return cfg
}

// Done applies the configured DropBlock to the operand x, and returns the regularized value.
func (cfg *DropBlockConfig) Done() *Node {
	ctx := cfg.ctx
	x := cfg.x
	g := x.Graph()
	if !ctx.IsTraining(g) || cfg.dropoutProbability == nil {
		// No-op: either it's inference, or zero dropout was configured.
		return x
	}

	var spatialAxes []int
	channelsAxis := -1
	if cfg.isFullShape {
		spatialAxes = xslices.Iota(0, cfg.x.Rank())
	} else {
		spatialAxes = images.GetSpatialAxes(cfg.x, cfg.channelsAxisConfig)
		channelsAxis = images.GetChannelsAxis(cfg.x, cfg.channelsAxisConfig)
	}

	var blockSizes []int
	if cfg.blockSize > 0 {
		blockSizes = xslices.SliceWithValue(len(spatialAxes), cfg.blockSize)
	} else {
		blockSizes = cfg.blockSizes
	}
	if len(blockSizes) != len(spatialAxes) {
		exceptions.Panicf("DropBlock requires one BlockSize per spatial axes, but got %d block sizes and %d spatial axes -- x.shape=%s",
			len(blockSizes), len(spatialAxes), x.Shape())
	}

	// gamma is the probability calculated so that the dropout probability +/- approximates
	// the expectation of ratio of pixels that will be dropped.
	pixelsPerBlock := 1
	for _, blockDim := range blockSizes {
		pixelsPerBlock *= blockDim
	}
	gamma := DivScalar(cfg.dropoutProbability, float64(pixelsPerBlock))

	// maskShape has the same shape as X, except for channels where it is one -- it will be broadcast
	// to all channels of a "pixel" (or "voxel" if 3D, or cell is 1D).
	maskShape := x.Shape().Clone()
	maskShape.DType = dtypes.Bool
	if channelsAxis >= 0 {
		maskShape.Dimensions[channelsAxis] = 1
	}
	mask := ctx.RandomBernoulli(OneMinus(gamma), maskShape)

	// Expand masked pixels to blocks -- except if block size is 1, then it behaves like a dropout.
	if pixelsPerBlock > 1 {
		if cfg.isFullShape {
			// Simpler case, no special casing of the channels dimension.
			mask = MinPool(mask).FullShape().WindowPerAxis(blockSizes...).Strides(1).PadSame().Done()
		} else {
			mask = MinPool(mask).ChannelsAxis(cfg.channelsAxisConfig).WindowPerAxis(blockSizes...).Strides(1).PadSame().Done()
		}
	}

	// For boolean values of x, just do a logical "And" with the mask:
	if x.DType() == dtypes.Bool {
		// If input is a bool map (as mask itself):
		return LogicalAnd(x, mask)
	}

	// Apply mask by converting mask to 1/0 and multiply.
	return Mul(x, ConvertDType(mask, x.DType()))
}
