// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package shapes

import (
	"fmt"
	"slices"
	"sort"
	"strings"

	"github.com/pkg/errors"
)

// AxisBindings maps named axis names to their concrete dimension values.
// Used at execution time to resolve dynamic shapes.
type AxisBindings map[string]int

// Key returns a deterministic string key suitable for use as a map key.
// Axis names are sorted alphabetically and formatted as "name=value,name=value".
func (b AxisBindings) Key() string {
	if len(b) == 0 {
		return ""
	}
	names := make([]string, 0, len(b))
	for name := range b {
		names = append(names, name)
	}
	sort.Strings(names)
	parts := make([]string, len(names))
	for i, name := range names {
		parts[i] = fmt.Sprintf("%s=%d", name, b[name])
	}
	return strings.Join(parts, ",")
}

// Resolve returns a new Shape with all dynamic dimensions replaced by their bound values.
// The resolved shape retains its AxisNames for provenance/debugging.
//
// Panics if a named dynamic axis has no corresponding binding, or if the binding is non-positive.
func (s Shape) Resolve(bindings AxisBindings) Shape {
	if !s.HasDynamicDims() {
		return s
	}
	resolved := s.Clone()
	for i, dim := range resolved.Dimensions {
		if dim != DynamicDim {
			continue
		}
		name := resolved.AxisName(i)
		if name == "" {
			panic(errors.Errorf("Shape.Resolve: dynamic axis %d has no name and cannot be resolved: %s", i, s))
		}
		val, ok := bindings[name]
		if !ok {
			panic(errors.Errorf("Shape.Resolve: no binding for axis %q in shape %s", name, s))
		}
		if val <= 0 {
			panic(errors.Errorf("Shape.Resolve: binding for axis %q must be positive, got %d", name, val))
		}
		resolved.Dimensions[i] = val
	}
	return resolved
}

// ExtractBindings extracts axis bindings by comparing a template shape (with named dynamic axes)
// against a concrete shape (with all dimensions known).
//
// Returns an error if the shapes are incompatible: different ranks, different static dimensions,
// or inconsistent bindings where the same axis name maps to different concrete values.
func ExtractBindings(template, concrete Shape) (AxisBindings, error) {
	if template.Rank() != concrete.Rank() {
		return nil, errors.Errorf("ExtractBindings: rank mismatch: template %s has rank %d, concrete %s has rank %d",
			template, template.Rank(), concrete, concrete.Rank())
	}
	bindings := make(AxisBindings)
	for i := range template.Dimensions {
		templateDim := template.Dimensions[i]
		concreteDim := concrete.Dimensions[i]

		if concreteDim == DynamicDim {
			return nil, errors.Errorf("ExtractBindings: concrete shape %s has dynamic dimension at axis %d", concrete, i)
		}

		if templateDim == DynamicDim {
			name := template.AxisName(i)
			if name == "" {
				return nil, errors.Errorf("ExtractBindings: template %s has dynamic dimension at axis %d but no axis name", template, i)
			}
			if existing, ok := bindings[name]; ok && existing != concreteDim {
				return nil, errors.Errorf("ExtractBindings: axis %q has conflicting values: %d vs %d", name, existing, concreteDim)
			}
			bindings[name] = concreteDim
		} else if templateDim != concreteDim {
			return nil, errors.Errorf("ExtractBindings: dimension %d mismatch: template has %d, concrete has %d",
				i, templateDim, concreteDim)
		}
	}
	return bindings, nil
}

// MergeBindings merges multiple AxisBindings into one. Returns an error if any axis name
// has conflicting values across the inputs.
func MergeBindings(all ...AxisBindings) (AxisBindings, error) {
	merged := make(AxisBindings)
	for _, b := range all {
		for name, val := range b {
			if existing, ok := merged[name]; ok && existing != val {
				return nil, errors.Errorf("MergeBindings: axis %q has conflicting values: %d vs %d", name, existing, val)
			}
			merged[name] = val
		}
	}
	return merged, nil
}

// UnifyAxisName resolves the output axis name when combining two axes from different shapes.
//
// Rules:
//   - "" + "" = "" (both unnamed → unnamed)
//   - "name" + "" = "name" (one named → keep the name)
//   - "" + "name" = "name" (one named → keep the name)
//   - "name" + "name" = "name" (same name → keep it)
//   - "a" + "b" = error (different names → conflict)
func UnifyAxisName(name1, name2 string) (string, error) {
	if name1 == "" {
		return name2, nil
	}
	if name2 == "" {
		return name1, nil
	}
	if name1 == name2 {
		return name1, nil
	}
	return "", errors.Errorf("incompatible axis names: %q vs %q", name1, name2)
}

// UnifyAxisNames unifies axis names from two shapes of the same rank.
// Returns the unified axis names, or error on name conflicts.
// Returns nil if neither shape has axis names.
func UnifyAxisNames(s1, s2 Shape) ([]string, error) {
	if s1.AxisNames == nil && s2.AxisNames == nil {
		return nil, nil
	}
	if s1.Rank() != s2.Rank() {
		return nil, errors.Errorf("UnifyAxisNames: rank mismatch: %d vs %d", s1.Rank(), s2.Rank())
	}
	result := make([]string, s1.Rank())
	for i := range result {
		name1 := ""
		if s1.AxisNames != nil {
			name1 = s1.AxisNames[i]
		}
		name2 := ""
		if s2.AxisNames != nil {
			name2 = s2.AxisNames[i]
		}
		unified, err := UnifyAxisName(name1, name2)
		if err != nil {
			return nil, errors.Wrapf(err, "axis %d", i)
		}
		result[i] = unified
	}

	// If all names are empty, return nil for consistency.
	if slices.IndexFunc(result, func(s string) bool { return s != "" }) == -1 {
		return nil, nil
	}
	return result, nil
}
