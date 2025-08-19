package shapes

import (
	"iter"
	"slices"

	"github.com/pkg/errors"
)

// Iter iterates sequentially over all possible indices of the given shape.
//
// It yields the flat index (counter) and a slice of indices for each axis.
//
// To avoid allocating the slice of indices, the yielded indices is owned by the Iter() method:
// don't change it inside the loop.
func (s Shape) Iter() iter.Seq2[int, []int] {
	indices := make([]int, s.Rank())
	return s.IterOn(indices)
}

// IterOn iterates over all possible indices of the given shape.
//
// It yields the flat index (counter) and a slice of indices for each axis.
//
// The iteration updates the indices on the given indices slice.
// During the iteration the caller shouldn't modify the slice of indices, otherwise it will lead to undefined behavior.
//
// It expects len(indices) == s.Rank(). It will panic otherwise.
func (s Shape) IterOn(indices []int) iter.Seq2[int, []int] {
	if len(indices) != s.Rank() {
		panic(errors.Errorf("Shape.IterOn given len(indices) == %d, want it to be equal to the rank %d", len(indices), s.Rank()))
	}
	return func(yield func(int, []int) bool) {
		if !s.Ok() || s.IsTuple() {
			return // Iteration completed (vacuously true as no items were yielded)
		}

		rank := s.Rank()
		if rank == 0 {
			// Valid scalar: yield one empty index slice.
			_ = yield(0, indices)
			return
		}

		// Defensive check: if any dimension is non-positive, treat as an empty iteration.
		// shapes.Make should prevent this for validly constructed shapes.
		//
		// Also count the number of "non-trivial" axes: axes whose dimensions > 1.
		numNonTrivialAxes := 0
		for _, dimSize := range s.Dimensions {
			if dimSize <= 0 {
				return
			}
			if dimSize > 1 {
				numNonTrivialAxes++
			}
		}

		// Version 1: there are only trivial axes, there is only one value to iterate over.
		for i := range indices {
			indices[i] = 0
		}
		if numNonTrivialAxes == 0 {
			yield(0, indices)
			return
		}

		// Version 2: most axes are non-trivial, simply iterate over all of them:
		if rank > numNonTrivialAxes+2 {
			// Loop until all indices are generated.
			// This structure simulates an N-dimensional counter for the indices.
			flatIdx := 0
		v2Yielder:
			for {
				if !yield(flatIdx, indices) {
					return // Consumer requested to stop iteration.
				}
				flatIdx++

				// Increment indices to the next set of coordinates
				// (row-major order: the last index changes fastest).
				for axis := rank - 1; axis >= 0; axis-- {
					if s.Dimensions[axis] == 1 {
						// Nothing to iterate at this axis.
						continue
					}
					indices[axis]++
					if indices[axis] < s.Dimensions[axis] {
						// Successfully incremented this dimension; no carry-over needed.
						continue v2Yielder
					}
					// The current axis overflowed; reset it to 0 and
					// continue to increment the next higher-order dimension (carry-over).
					indices[axis] = 0
				}

				// If the axis is less than 0, all dimensions have been iterated through
				// (i.e., the first dimension also overflowed). Iteration is complete.
				break
			}
			return
		}

		// Version 3: many "trivial" axes (dimension == 1), create an indirection and only
		// iterate over the non-trivial axes:
		flatIdx := 0
		spatialAxes := make([]int, 0, numNonTrivialAxes)
		for axis, dim := range s.Dimensions {
			if dim > 1 {
				spatialAxes = append(spatialAxes, axis)
			}
		}
		slices.Reverse(spatialAxes) // We want to iterate over the last axis first.
	v3Yielder:
		for {
			if !yield(flatIdx, indices) {
				return // Consumer requested to stop iteration.
			}
			flatIdx++

			// Increment indices to the next set of coordinates
			// (row-major order: the last index changes fastest).
			for _, axis := range spatialAxes {
				indices[axis]++
				if indices[axis] < s.Dimensions[axis] {
					// Successfully incremented this dimension; no carry-over needed.
					continue v3Yielder
				}
				// The current axis overflowed; reset it to 0 and
				// continue to increment the next higher-order dimension (carry-over).
				indices[axis] = 0
			}

			// That was the last index.
			break
		}
	}
}
