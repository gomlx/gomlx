/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package layers

import (
	. "github.com/gomlx/exceptions"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/types/xslices"
	"log"
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
	// in their scope (`Context.In(scope)`) -- except if one wants to deliberately share
	// normalization variables across more than one application.
	KnownNormalizers = map[string]func(ctx *context.Context, input *Node) *Node{
		"batch": func(ctx *context.Context, input *Node) *Node {
			return batchnorm.New(ctx, input, -1).Done()
		},
		"layer": func(ctx *context.Context, input *Node) *Node {
			return LayerNormalization(ctx, input, -1).Done()
		},
		"none": func(ctx *context.Context, input *Node) *Node {
			return input
		},
	}

	// ParamNormalization context hyperparameter defines the type of normalization to use
	// between layers of a neural network.
	//
	// It is used if the model calls NormalizeFromContext or MaskedNormalizeFromContext on the embeddings in
	// between layers.
	// This is usually applied after a residual sum (but model choices varies).
	//
	// Valid values are "layer" for [LayerNormalization], "batch" for [batchnorm.New] or "none"".
	//
	// Notice that this won't work for special shapes setups.
	// [New] will normalize on the batch axis (assumed to be axis-0), and
	// [LayerNormalization] will normalize across the layer values, assumed to be the last.
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
// Some layer libraries will use this by default for you, taking the value from the context -- e.g: fnn.New.
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
//	    logits = MustNormalizeByName(ctx, *flagNormalization, logits)
//	    ...
//	}
//
// ```
func MustNormalizeByName(ctx *context.Context, normalization string, input *Node) *Node {
	normFn, found := KnownNormalizers[normalization]
	if !found {
		log.Fatalf("Unsupported normalization %q given, valid values are %v", normalization, xslices.SortedKeys(KnownNormalizers))
	}
	return normFn(ctx, input)
}

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
		return batchnorm.New(ctx, input, -1).Done()
	default:
		Panicf("invalid normalization type %q given in context parameter %q -- valid values are 'none', 'layer' or 'batch'",
			normType, ParamNormalization)
	}
	return input
}
