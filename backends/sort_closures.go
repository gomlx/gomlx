package backends

import "github.com/gomlx/gomlx/pkg/core/dtypes"

// SortClosureCreator is an optional interface that backends can implement
// to provide closure creation for Sort operations.
//
// Backends that implement this interface can provide optimized sort operations
// with pre-built comparators.
type SortClosureCreator interface {
	// SortComparatorAscending creates a comparator closure for ascending sort.
	// The comparator takes two scalar values of the given dtype and returns true
	// if the first should come before the second (i.e., lhs < rhs).
	SortComparatorAscending(dtype dtypes.DType) (any, error)

	// SortComparatorDescending creates a comparator closure for descending sort.
	// The comparator takes two scalar values of the given dtype and returns true
	// if the first should come before the second (i.e., lhs > rhs).
	SortComparatorDescending(dtype dtypes.DType) (any, error)

	// SortWithIndicesComparatorAscending creates a comparator for sorting values with their indices.
	// The comparator takes 4 arguments: lhs_value, lhs_index, rhs_value, rhs_index
	// and returns true if lhs_value < rhs_value.
	SortWithIndicesComparatorAscending(valueDType, indexDType dtypes.DType) (any, error)

	// SortWithIndicesComparatorDescending creates a comparator for sorting values with their indices.
	// The comparator takes 4 arguments: lhs_value, lhs_index, rhs_value, rhs_index
	// and returns true if lhs_value > rhs_value.
	SortWithIndicesComparatorDescending(valueDType, indexDType dtypes.DType) (any, error)
}

// WhileClosureCreator is an optional interface that backends can implement
// to provide closure creation for While operations.
//
// This is more complex and typically requires users to define the condition
// and body functions using the backend's specific mechanisms.
type WhileClosureCreator interface {
	// CreateWhileCondition creates a condition closure for While loops.
	// The returned closure should take the state values and return a scalar bool.
	CreateWhileCondition(stateShapes []dtypes.DType, conditionFn func(states []Op) Op) (any, error)

	// CreateWhileBody creates a body closure for While loops.
	// The returned closure should take the state values and return updated state values.
	CreateWhileBody(stateShapes []dtypes.DType, bodyFn func(states []Op) []Op) (any, error)
}
