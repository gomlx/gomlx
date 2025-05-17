package shapes

import "iter"

// Iter iterates over all possible indices of the given shape.
// To avoid allocating the slice of indices, the yielded indices is owned by the Iter() method:
// don't change it inside the loop.
func (s Shape) Iter() iter.Seq[[]int] {
	return func(yield func([]int) bool) {
		if !s.Ok() || s.IsTuple() {
			return // Iteration completed (vacuously true as no items were yielded)
		}

		rank := s.Rank()
		if rank == 0 {
			// Valid scalar: yield one empty index slice.
			_ = yield(make([]int, 0))
			return
		}

		// Defensive check: if any dimension is non-positive, treat as an empty iteration.
		// shapes.Make should prevent this for validly constructed shapes.
		for _, dimSize := range s.Dimensions {
			if dimSize <= 0 {
				return
			}
		}

		currentIndices := make([]int, rank)
		// Loop until all indices are generated.
		// This structure simulates an N-dimensional counter for the indices.
		for {
			if !yield(currentIndices) {
				return // Consumer requested to stop iteration.
			}

			// Increment currentIndices to the next set of coordinates
			// (row-major order: the last index changes fastest).
			axis := rank - 1
			for ; axis >= 0; axis-- {
				if s.Dimensions[axis] == 1 {
					// Nothing to iterate at this axis.
					continue
				}
				currentIndices[axis]++
				if currentIndices[axis] < s.Dimensions[axis] {
					// Successfully incremented this dimension; no carry-over needed.
					break
				}
				// The current axis overflowed; reset it to 0 and
				// continue to increment the next higher-order dimension (carry-over).
				currentIndices[axis] = 0
			}

			// If axis is less than 0, all dimensions have been iterated through
			// (i.e., the first dimension also overflowed). Iteration is complete.
			if axis < 0 {
				break
			}
		}
		return // Iteration completed successfully.
	}
}
