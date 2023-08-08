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
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/slices"
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
			return BatchNormalization(ctx, input, -1).Done()
		},
		"norm": func(ctx *context.Context, input *Node) *Node {
			return LayerNormalization(ctx, input, -1).Done()
		},
		"none": func(ctx *context.Context, input *Node) *Node {
			return input
		},
	}
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
// Typical use:
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
		log.Fatalf("Unsupported normalization %q given, valid values are %v", normalization, slices.SortedKeys(KnownNormalizers))
	}
	return normFn(ctx, input)
}
