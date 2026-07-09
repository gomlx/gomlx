// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package layers

import (
	"log"

	"github.com/gomlx/compute/support/xslices"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/layers/norm"
	"github.com/gomlx/gomlx/ml/model"
	. "github.com/gomlx/gomlx/support/exceptions"
)

const (
	// NormalizationLayerNorm is the value for norm.LayerNorm.
	NormalizationLayerNorm = "layer"
	// NormalizationRMSNorm is the value for norm.RMSNorm.
	NormalizationRMSNorm = "rms"
	// NormalizationBatchNorm is the value for norm.BatchNorm.
	NormalizationBatchNorm = "batch"
	// NormalizationDyT is the value for norm.DynamicTanh.
	// DyT is a drop-in replacement for layer normalization; see norm.DynamicTanh for details.
	NormalizationDyT = "dyt"
	// NormalizationNone is the value for no normalization.
	NormalizationNone = "none"
)

var (
	// KnownNormalizers is a map of normalizer string to a function that applies them
	// with the default values, with the feature axis set to -1. This will only work
	// for the most standard problems, since anything with a different shape will need
	// special feature axes configuration for each normalization technique.
	//
	// It includes "none", which is a no-op.
	//
	// Notice that some normalizers use variables, and they need to be unique
	// in their scope (`Scope.In(scope)`) -- except if one wants to deliberately share
	// normalization variables across more than one application.
	KnownNormalizers = map[string]func(scope *model.Scope, input *Node) *Node{
		NormalizationBatchNorm: func(scope *model.Scope, input *Node) *Node {
			return norm.BatchNorm(scope, input, -1).Done()
		},
		NormalizationLayerNorm: func(scope *model.Scope, input *Node) *Node {
			return norm.LayerNorm(scope, input, -1).Done()
		},
		NormalizationRMSNorm: func(scope *model.Scope, input *Node) *Node {
			return norm.RMSNorm(scope, input).Done()
		},
		NormalizationDyT: func(scope *model.Scope, input *Node) *Node {
			return norm.DynamicTanh(scope, input).Done()
		},
		NormalizationNone: func(scope *model.Scope, input *Node) *Node {
			return input
		},
	}

	// ParamNormalization scope hyperparameter defines the type of normalization to use
	// between layers of a neural network.
	//
	// It is used if the model calls NormalizeFromScope or MaskedNormalizeFromScope on the embeddings in
	// between layers.
	// This is usually applied after a residual sum (but model choices varies).
	//
	// Valid values are "layer" for LayerNormalization, "batch" for BatchNorm,
	// "rms" for RMSNorm, "dyt" for DynamicTanh, or "none".
	//
	// Notice that this won't work for special shapes setups.
	// New will normalize on the batch axis (assumed to be axis-0), and
	// LayerNormalization and RMSNorm will normalize across the layer values, assumed to be the last.
	//
	// The default is `layer`.
	ParamNormalization = "normalization"
)

// MustNormalizeByName applies the requested normalization using default parameters. If
// an invalid normalization is given, it panics with an error.
//
// This will only work for the most standard problems, since anything with a different shape will need
// special feature axes configuration for each normalization technique.
//
// It's a simple wrapper around KnownNormalizers, if one wants to handle errors, just
// check for its values. For valid values see KnownNormalizers.
//
// Some layer libraries will use this by default for you, taking the value from the scope -- e.g: fnn.New.
//
// But if not, one example use:
//
// ```
// var flagNormalization = flag.String("norm", "none",
//
//	fmt.Sprintf("Type of layer normalization to use. Valid values: %q.",
//		types.SortedKeys(layers.KnownNormalizers)))
//
// ...
//
//	func ModelGraph(...) {
//	    ...
//	    logits = MustNormalizeByName(scope, *flagNormalization, logits)
//	    ...
//	}
//
// ```
func MustNormalizeByName(scope *model.Scope, normalization string, input *Node) *Node {
	normFn, found := KnownNormalizers[normalization]
	if !found {
		log.Fatalf("Unsupported normalization %q given, valid values are %v", normalization, xslices.SortedKeys(KnownNormalizers))
	}
	return normFn(scope, input)
}

// NormalizeFromScope applies a normalization (or none) according to the hyperparameter
// ParamNormalization configured in the model.
//
// This is not recommended for images, since one may want to normalize over specific axes.
func NormalizeFromScope(scope *model.Scope, input *Node) *Node {
	return MaskedNormalizeFromScope(scope, input, nil)
}

// MaskedNormalizeFromScope applies a normalization (or none) according to the hyperparameter
// ParamNormalization configured in the model.
// The `mask` is actually optional and can be set to nil if not using a mask.
//
// This is not recommended for images, since one may want to normalize over specific axes.
func MaskedNormalizeFromScope(scope *model.Scope, input, mask *Node) *Node {
	normType := model.GetParamOr(scope, ParamNormalization, "layer")
	switch normType {
	case NormalizationNone, "":
		return input
	case NormalizationLayerNorm:
		return norm.LayerNorm(scope, input, -1).Mask(mask).Done()
	case NormalizationRMSNorm:
		return norm.RMSNorm(scope, input).Done()
	case NormalizationBatchNorm:
		if mask != nil {
			Panicf("'batch' normalization set in scope parameter %q does not support usage of mask yet, please open a feature request",
				ParamNormalization)
		}
		return norm.BatchNorm(scope, input, -1).Done()
	case NormalizationDyT:
		return norm.DynamicTanh(scope, input).Done()
	default:
		Panicf("invalid normalization type %q given in scope parameter %q -- valid values are 'none', 'layer', 'rms', 'batch', or 'dyt'",
			normType, ParamNormalization)
	}
	return input
}
