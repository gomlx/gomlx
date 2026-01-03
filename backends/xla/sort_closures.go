package xla

import (
	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	stablehloshapes "github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/pkg/errors"
)

// Ensure Builder implements SortClosureCreator
var _ backends.SortClosureCreator = (*Builder)(nil)

// sortComparatorKey for caching sort comparators
type sortComparatorKey struct {
	dtype      dtypes.DType
	descending bool
}

// SortComparatorAscending creates a comparator closure for ascending sort.
// The comparator returns true if lhs < rhs.
func (b *Builder) SortComparatorAscending(dtype dtypes.DType) (any, error) {
	return b.getSortComparator(dtype, false)
}

// SortComparatorDescending creates a comparator closure for descending sort.
// The comparator returns true if lhs > rhs.
func (b *Builder) SortComparatorDescending(dtype dtypes.DType) (any, error) {
	return b.getSortComparator(dtype, true)
}

// getSortComparator creates or retrieves a cached sort comparator function.
func (b *Builder) getSortComparator(dtype dtypes.DType, descending bool) (*stablehlo.Function, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}

	// Create a new closure function
	comparatorFn := b.fn.Closure()

	// Create scalar inputs for comparison
	lhs, err := comparatorFn.NamedInput("lhs", stablehloshapes.Make(DTypeToXLA(dtype)))
	if err != nil {
		return nil, errors.WithMessage(err, "while creating sort comparator lhs input")
	}
	rhs, err := comparatorFn.NamedInput("rhs", stablehloshapes.Make(DTypeToXLA(dtype)))
	if err != nil {
		return nil, errors.WithMessage(err, "while creating sort comparator rhs input")
	}

	// Create the comparison operation
	var result *stablehlo.Value
	compareType := compareTypeForDType(dtype)

	if descending {
		// For descending: return lhs > rhs
		result, err = stablehlo.Compare(lhs, rhs, stablehlotypes.CompareGT, compareType)
	} else {
		// For ascending: return lhs < rhs
		result, err = stablehlo.Compare(lhs, rhs, stablehlotypes.CompareLT, compareType)
	}
	if err != nil {
		return nil, errors.WithMessage(err, "while creating sort comparator comparison")
	}

	// Set the return value
	err = comparatorFn.Return(result)
	if err != nil {
		return nil, errors.WithMessage(err, "while setting sort comparator return")
	}

	return comparatorFn, nil
}

// SortWithIndicesComparatorAscending creates a comparator for sorting values with their indices.
// The comparator takes 4 arguments: lhs_value, lhs_index, rhs_value, rhs_index
// and returns true if lhs_value < rhs_value.
func (b *Builder) SortWithIndicesComparatorAscending(valueDType, indexDType dtypes.DType) (any, error) {
	return b.getSortWithIndicesComparator(valueDType, indexDType, false)
}

// SortWithIndicesComparatorDescending creates a comparator for sorting values with their indices.
// The comparator takes 4 arguments: lhs_value, lhs_index, rhs_value, rhs_index
// and returns true if lhs_value > rhs_value.
func (b *Builder) SortWithIndicesComparatorDescending(valueDType, indexDType dtypes.DType) (any, error) {
	return b.getSortWithIndicesComparator(valueDType, indexDType, true)
}

// getSortWithIndicesComparator creates a comparator for sorting values with their indices.
// StableHLO Sort with multiple tensors expects the comparator arguments in pairs:
// for each tensor, (lhs_element, rhs_element), so for 2 tensors:
// arg0: lhs_value, arg1: rhs_value, arg2: lhs_index, arg3: rhs_index
func (b *Builder) getSortWithIndicesComparator(valueDType, indexDType dtypes.DType, descending bool) (*stablehlo.Function, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}

	// Create a new closure function
	comparatorFn := b.fn.Closure()

	// Create 4 scalar inputs in the order expected by StableHLO:
	// lhs_value, rhs_value, lhs_index, rhs_index
	lhsValue, err := comparatorFn.NamedInput("lhs_value", stablehloshapes.Make(DTypeToXLA(valueDType)))
	if err != nil {
		return nil, errors.WithMessage(err, "while creating sort comparator lhs_value input")
	}
	rhsValue, err := comparatorFn.NamedInput("rhs_value", stablehloshapes.Make(DTypeToXLA(valueDType)))
	if err != nil {
		return nil, errors.WithMessage(err, "while creating sort comparator rhs_value input")
	}
	_, err = comparatorFn.NamedInput("lhs_index", stablehloshapes.Make(DTypeToXLA(indexDType)))
	if err != nil {
		return nil, errors.WithMessage(err, "while creating sort comparator lhs_index input")
	}
	_, err = comparatorFn.NamedInput("rhs_index", stablehloshapes.Make(DTypeToXLA(indexDType)))
	if err != nil {
		return nil, errors.WithMessage(err, "while creating sort comparator rhs_index input")
	}

	// Create the comparison operation (comparing only values, ignoring indices)
	var result *stablehlo.Value
	compareType := compareTypeForDType(valueDType)

	if descending {
		// For descending: return lhs_value > rhs_value
		result, err = stablehlo.Compare(lhsValue, rhsValue, stablehlotypes.CompareGT, compareType)
	} else {
		// For ascending: return lhs_value < rhs_value
		result, err = stablehlo.Compare(lhsValue, rhsValue, stablehlotypes.CompareLT, compareType)
	}
	if err != nil {
		return nil, errors.WithMessage(err, "while creating sort comparator comparison")
	}

	// Set the return value
	err = comparatorFn.Return(result)
	if err != nil {
		return nil, errors.WithMessage(err, "while setting sort comparator return")
	}

	return comparatorFn, nil
}
