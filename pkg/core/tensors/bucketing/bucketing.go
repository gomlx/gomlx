/*
 *	Copyright 2025 Jan Pfeifer
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

// Package bucketing provides strategies for bucketing tensor dimensions to reduce
// the number of unique compiled graphs when input sizes vary.
//
// When training or running inference with variable-sized inputs (e.g., different batch
// sizes or sequence lengths), each unique shape requires a separate compiled graph.
// Bucketing strategies round dimensions up to reduce the number of unique shapes,
// trading off some memory for fewer graph compilations.
//
// # Available Strategies
//
//   - Pow2: Rounds to nearest power of 2 (1,2,4,8,16,32,...)
//   - Linear: Rounds to multiples of a step size (8,16,24,32,...)
//   - Exponential: Rounds to powers of a base (e.g., 1.4^n for n=1,2,3,...)
//   - None: No bucketing (each size gets its own graph)
//
// # Usage with Exec
//
// The bucketing package is designed to work with graph.Exec for pattern caching:
//
//	exec := graph.MustNewExec(backend, modelFn).
//	    SetPatternCaching(bucketing.Pow2()).
//	    SetDynamicAxes([]int{0}) // Bucket first axis (batch)
//
// # Usage with Dataset
//
// For better parallelism, bucketing can also be applied in the data pipeline:
//
//	bucketer := bucketing.Exponential(1.4)
//	for batch := range dataset {
//	    bucketedSize := bucketer.Bucket(len(batch))
//	    paddedBatch := padToSize(batch, bucketedSize)
//	    // ... use paddedBatch
//	}
//
// # Custom Strategies
//
// Implement the Strategy interface for custom bucketing logic:
//
//	type MyStrategy struct{}
//	func (MyStrategy) Bucket(dim int) int {
//	    // Custom logic
//	    return bucketedDim
//	}
package bucketing

import "math"

// Strategy defines how to bucket dimensions for cache efficiency.
// This reduces the number of unique compiled graphs for variable-sized inputs.
//
// Implementations should:
//   - Return the input unchanged for non-positive values (symbolic dimensions or zero)
//   - Return a value >= the input dimension (never shrink)
//   - Be deterministic (same input always produces same output)
type Strategy interface {
	// Bucket returns the bucketed value for a dimension.
	// Symbolic dimensions (negative) and zero are returned unchanged.
	Bucket(dim int) int
}

// Pow2Strategy rounds dimensions up to the nearest power of 2.
// This is useful for reducing cache misses when batch sizes vary.
//
// Example mappings: 1→1, 2→2, 3→4, 4→4, 5→8, 9→16, 17→32
type Pow2Strategy struct{}

// Pow2 returns a power-of-2 bucketing strategy.
func Pow2() Strategy {
	return Pow2Strategy{}
}

// Bucket implements Strategy for Pow2Strategy.
func (Pow2Strategy) Bucket(dim int) int {
	if dim <= 0 {
		return dim // Preserve symbolic or zero
	}
	if dim == 1 {
		return 1
	}
	// Round up to next power of 2
	v := uint(dim - 1)
	v |= v >> 1
	v |= v >> 2
	v |= v >> 4
	v |= v >> 8
	v |= v >> 16
	return int(v + 1)
}

// LinearStrategy rounds dimensions to multiples of a step size.
//
// Example with step=8: 1→8, 8→8, 9→16, 16→16, 17→24
type LinearStrategy struct {
	Step int
}

// Linear returns a linear bucketing strategy with the given step size.
// Dimensions are rounded up to the nearest multiple of step.
func Linear(step int) Strategy {
	if step <= 0 {
		step = 1
	}
	return LinearStrategy{Step: step}
}

// Bucket implements Strategy for LinearStrategy.
func (b LinearStrategy) Bucket(dim int) int {
	if dim <= 0 {
		return dim
	}
	return ((dim + b.Step - 1) / b.Step) * b.Step
}

// ExponentialStrategy rounds dimensions to the nearest power of a base value.
// This provides finer granularity than Pow2 for smaller dimensions while
// still providing exponential growth for larger ones.
//
// Example with base=1.4:
//
//	1→1, 2→2, 3→3, 4→4, 5→6, 7→8, 9→11, 12→15, 16→21, 22→29, ...
//
// The sequence follows ceil(base^n) for n=0,1,2,...
type ExponentialStrategy struct {
	Base float64
}

// Exponential returns an exponential bucketing strategy with the given base.
// Common values are 1.4 (fine granularity) or 1.5 (coarser).
// For base=2.0, this is equivalent to Pow2.
func Exponential(base float64) Strategy {
	if base <= 1.0 {
		base = 2.0 // Default to pow2 if invalid
	}
	return ExponentialStrategy{Base: base}
}

// Bucket implements Strategy for ExponentialStrategy.
func (b ExponentialStrategy) Bucket(dim int) int {
	if dim <= 0 {
		return dim
	}
	if dim == 1 {
		return 1
	}
	// Find smallest base^n >= dim
	// n = ceil(log_base(dim)) = ceil(ln(dim) / ln(base))
	logBase := math.Log(b.Base)
	power := math.Ceil(math.Log(float64(dim)) / logBase)
	result := int(math.Ceil(math.Pow(b.Base, power)))
	// Ensure we never return less than the input (handles edge cases with low bases)
	for result < dim {
		power++
		result = int(math.Ceil(math.Pow(b.Base, power)))
	}
	return result
}

// NoneStrategy returns dimensions unchanged.
// This is the default behavior when pattern caching is disabled.
type NoneStrategy struct{}

// None returns a no-op bucketing strategy.
func None() Strategy {
	return NoneStrategy{}
}

// Bucket implements Strategy for NoneStrategy.
func (NoneStrategy) Bucket(dim int) int {
	return dim
}
