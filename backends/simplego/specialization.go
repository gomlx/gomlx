// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// ShapeSpecialization holds a resolved-shape copy of the graph nodes
// for a specific set of axis bindings. Each specialization is created
// lazily on first execution with a given binding and cached for reuse.
//
// The resolvedNodes are shallow copies of the original Function.nodes.
// Only the shape field (and multiOutputsShapes for multi-output nodes)
// is resolved; all other fields (inputs, data, opType, etc.) are shared
// with the originals. This allows executor functions to read node.shape
// and get concrete values without any signature changes.
type ShapeSpecialization struct {
	// bindings that produced this specialization.
	bindings shapes.AxisBindings

	// resolvedNodes is a shallow copy of the original Function.nodes slice.
	// Each node has its shape field resolved (no DynamicDim values).
	// All other fields (inputs, data, opType, etc.) point to the originals.
	resolvedNodes []*Node
}

// hasDynamicParameters returns true if any parameter node has dynamic dimensions.
func hasDynamicParameters(params []*Node) bool {
	for _, p := range params {
		if p.shape.HasDynamicDims() {
			return true
		}
	}
	return false
}

// extractBindingsFromInputs extracts axis bindings by matching symbolic parameter
// shapes against concrete input buffer shapes. Returns merged bindings across all
// parameters, or an error if shapes are incompatible or bindings conflict.
func extractBindingsFromInputs(params []*Node, inputs []*Buffer) (shapes.AxisBindings, error) {
	var allBindings []shapes.AxisBindings
	for i, param := range params {
		if !param.shape.HasDynamicDims() {
			continue
		}
		b, err := shapes.ExtractBindings(param.shape, inputs[i].shape)
		if err != nil {
			paramName := ""
			if pd, ok := param.data.(*nodeParameter); ok {
				paramName = pd.name
			}
			return nil, errors.WithMessagef(err, "parameter %d %q", i, paramName)
		}
		allBindings = append(allBindings, b)
	}
	if len(allBindings) == 0 {
		return shapes.AxisBindings{}, nil
	}
	return shapes.MergeBindings(allBindings...)
}

// createSpecialization builds a ShapeSpecialization for the given bindings.
// It creates shallow copies of all nodes with shapes resolved, and rewires
// multiOutputsNodes pointers to the resolved copies.
//
// Returns an error if a DotGeneral or ConvGeneral node involves dynamic dimensions,
// as those operations require precomputed parameters that depend on concrete
// shapes (deferred to Phase 3).
func (e *Executable) createSpecialization(bindings shapes.AxisBindings) (spec *ShapeSpecialization, err error) {
	// Shape.Resolve panics on missing bindings, non-positive values, or unnamed
	// dynamic axes. Recover those panics and surface them as errors.
	defer func() {
		if r := recover(); r != nil {
			spec = nil
			if rErr, ok := r.(error); ok {
				err = errors.WithMessage(rErr, "createSpecialization")
			} else {
				err = errors.Errorf("createSpecialization: %v", r)
			}
		}
	}()

	origNodes := e.builder.mainFn.nodes
	resolved := make([]*Node, len(origNodes))

	// Guard: return error if DotGeneral or ConvGeneral have dynamic dims.
	// These ops precompute execution parameters from concrete shapes at graph
	// build time. Supporting them with dynamic shapes requires deferring that
	// computation to specialization time (Phase 3).
	// Check before resolving to give a clear error referencing the symbolic shapes.
	for _, n := range origNodes {
		if n.opType == backends.OpTypeDotGeneral || n.opType == backends.OpTypeConvGeneral {
			for _, input := range n.inputs {
				if input.shape.HasDynamicDims() {
					return nil, errors.Errorf(
						"specialization: %s with dynamic-shaped inputs is not yet supported (requires Phase 3); "+
							"input node %d has symbolic shape %s",
						n.opType, input.idx, input.shape)
				}
			}
		}
	}

	// First pass: shallow copy each node and resolve its shape.
	for i, orig := range origNodes {
		n := &Node{
			idx:                orig.idx,
			inputs:             orig.inputs,
			capturedInputs:     orig.capturedInputs,
			opType:             orig.opType,
			shape:              orig.shape.Resolve(bindings),
			builder:            orig.builder,
			function:           orig.function,
			data:               orig.data,
			isNodeSelectOutput: orig.isNodeSelectOutput,
			selectOutputIdx:    orig.selectOutputIdx,
		}

		// Resolve multi-output shapes if present.
		if orig.multiOutputsShapes != nil {
			n.multiOutputsShapes = make([]shapes.Shape, len(orig.multiOutputsShapes))
			for j, s := range orig.multiOutputsShapes {
				n.multiOutputsShapes[j] = s.Resolve(bindings)
			}
			// multiOutputsNodes will be rewired in the second pass.
			n.multiOutputsNodes = orig.multiOutputsNodes
		}

		resolved[i] = n
	}

	// Second pass: rewire multiOutputsNodes to point to resolved copies.
	for _, n := range resolved {
		if n.multiOutputsNodes != nil {
			rewired := make([]*Node, len(n.multiOutputsNodes))
			for i, sub := range n.multiOutputsNodes {
				rewired[i] = resolved[sub.idx]
			}
			n.multiOutputsNodes = rewired
		}
	}

	return &ShapeSpecialization{
		bindings:      bindings,
		resolvedNodes: resolved,
	}, nil
}

// getOrCreateSpecialization returns a cached specialization for the given bindings,
// creating one if it doesn't exist. Thread-safe via sync.Map.
func (e *Executable) getOrCreateSpecialization(bindings shapes.AxisBindings) (*ShapeSpecialization, error) {
	key := bindings.Key()
	if spec, ok := e.specializations.Load(key); ok {
		return spec.(*ShapeSpecialization), nil
	}
	newSpec, err := e.createSpecialization(bindings)
	if err != nil {
		return nil, err
	}
	actual, _ := e.specializations.LoadOrStore(key, newSpec)
	return actual.(*ShapeSpecialization), nil
}

