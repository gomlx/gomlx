// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapes

import (
	"fmt"
	"sort"
	"strings"

	"github.com/pkg/errors"
)

// AxisBindings maps axis names to concrete dimension values.
// Used to resolve dynamic shapes to concrete shapes at execution time.
type AxisBindings map[string]int

// Key returns a canonical string representation for map keying.
// Format: "name1=val1,name2=val2" with names sorted alphabetically.
// Returns empty string for empty or nil bindings.
func (ab AxisBindings) Key() string {
	if len(ab) == 0 {
		return ""
	}
	names := make([]string, 0, len(ab))
	for name := range ab {
		names = append(names, name)
	}
	sort.Strings(names)

	parts := make([]string, len(names))
	for i, name := range names {
		parts[i] = fmt.Sprintf("%s=%d", name, ab[name])
	}
	return strings.Join(parts, ",")
}

// Clone returns a copy of the bindings.
func (ab AxisBindings) Clone() AxisBindings {
	if ab == nil {
		return nil
	}
	clone := make(AxisBindings, len(ab))
	for k, v := range ab {
		clone[k] = v
	}
	return clone
}

// Merge combines bindings from another AxisBindings into this one.
// Returns an error if there are conflicting values for the same axis name.
func (ab AxisBindings) Merge(other AxisBindings) error {
	for name, val := range other {
		if existing, ok := ab[name]; ok && existing != val {
			return errors.Errorf("conflicting values for axis %q: %d vs %d", name, existing, val)
		}
		ab[name] = val
	}
	return nil
}

// Resolve replaces named axes with concrete values from bindings.
// Returns a new shape with all named axes resolved to their bound values.
// If a named axis has no binding, it remains dynamic (DimDynamic).
// Static dimensions are unchanged.
func (s Shape) Resolve(bindings AxisBindings) Shape {
	if !s.HasNamedAxes() || bindings == nil {
		return s.Clone()
	}

	result := s.Clone()
	for i, name := range s.AxisNames {
		if name != "" {
			if val, ok := bindings[name]; ok {
				result.Dimensions[i] = val
			}
		}
	}
	return result
}

// ExtractBindings gets axis bindings from a concrete shape matching a pattern.
// The pattern may have named dynamic axes; concrete must have actual values.
//
// Returns error if:
//   - Shapes have different ranks
//   - Shapes have different dtypes
//   - Static dimensions don't match
//   - Same axis name has conflicting values
func ExtractBindings(pattern, concrete Shape) (AxisBindings, error) {
	if pattern.Rank() != concrete.Rank() {
		return nil, errors.Errorf("rank mismatch: pattern has %d, concrete has %d",
			pattern.Rank(), concrete.Rank())
	}
	if pattern.DType != concrete.DType {
		return nil, errors.Errorf("dtype mismatch: pattern is %s, concrete is %s",
			pattern.DType, concrete.DType)
	}

	bindings := make(AxisBindings)
	for i := range pattern.Dimensions {
		name := pattern.AxisName(i)
		concreteVal := concrete.Dimensions[i]

		if name != "" {
			// Named axis: extract binding
			if existing, ok := bindings[name]; ok && existing != concreteVal {
				return nil, errors.Errorf("axis %q has conflicting values at dimension %d: %d vs %d",
					name, i, existing, concreteVal)
			}
			bindings[name] = concreteVal
		} else if pattern.Dimensions[i] != DimDynamic {
			// Static dimension: must match exactly
			if pattern.Dimensions[i] != concreteVal {
				return nil, errors.Errorf("dimension %d mismatch: pattern has %d, concrete has %d",
					i, pattern.Dimensions[i], concreteVal)
			}
		}
		// Unnamed dynamic (DimDynamic without name) accepts any value
	}
	return bindings, nil
}

// UnifyAxisName combines two axis names during shape inference.
// Used when broadcasting or combining shapes in binary operations.
//
// Rules:
//   - If either name is empty, return the other (unnamed adopts name)
//   - If both names are the same, return that name
//   - If names differ, return error (incompatible axes)
func UnifyAxisName(a, b string) (string, error) {
	if a == "" {
		return b, nil // unnamed adopts name
	}
	if b == "" {
		return a, nil // unnamed adopts name
	}
	if a == b {
		return a, nil // same name is OK
	}
	return "", errors.Errorf("incompatible axis names: %q vs %q", a, b)
}
