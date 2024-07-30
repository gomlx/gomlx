package layers

import (
	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
)

// ParamNormalization context hyperparameter defines the type of normalization to use
// between layers of a neural network.
//
// It is used if the model calls [NormalizeFromContext] on the embeddings in between layers.
// This is usually applied after a residual sum (but model choices varies).
//
// Valid values are "layer" for [LayerNormalization], "batch" for [BatchNormalization] or "none"/"".
//
// Notice that this won't work for special shapes setups.
// [BatchNormalization] will normalize on the batch axis (assumed to be axis-0), and
// [LayerNormalization] will normalize across the layer values, assumed to be the last.
//
// The default is `layer`.
var ParamNormalization = "normalization"

// NormalizeFromContext applies a normalization (or none) according to the hyperparameter
// ParamNormalization configured in the context.
func NormalizeFromContext(ctx *context.Context, input *Node) *Node {
	return MaskedNormalizeFromContext(ctx, input, nil)
}

// MaskedNormalizeFromContext applies a normalization (or none) according to the hyperparameter
// ParamNormalization configured in the context.
// The `mask` is actually optional, and can be set to nil if not using a mask.
func MaskedNormalizeFromContext(ctx *context.Context, input, mask *Node) *Node {
	normType := context.GetParamOr(ctx, ParamNormalization, "layer")
	switch normType {
	case "none", "":
		return input
	case "layer":
		return LayerNormalization(ctx, input, -1).Mask(mask).Done()
	case "batch":
		if mask != nil {
			Panicf("'batch' normalization set in context parameter %q does not support usage of mask yet, please open a feature request",
				ParamNormalization)
		}
		return BatchNormalization(ctx, input, -1).Done()
	default:
		Panicf("invalid normalization type %q given in context parameter %q -- valid values are 'none', 'layer' or 'batch'",
			normType, ParamNormalization)
	}
	return input
}
