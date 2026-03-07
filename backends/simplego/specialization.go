// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
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
// It creates shallow copies of all nodes with shapes resolved, rewires
// multiOutputsNodes pointers to the resolved copies, and recomputes
// shape-dependent node.data for DotGeneral and ConvGeneral operations.
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

	// Third pass: recompute shape-dependent node.data for ops that need it.
	// DotGeneral and ConvGeneral store precomputed parameters (sizes, strides,
	// normalization info, exec path) in node.data at graph build time.
	// When inputs have dynamic dims, these are placeholders that must be
	// recomputed from the resolved concrete shapes.
	//
	// For DotGeneral, we must also fix the node's shape. The intermediate rank-3
	// shape [batchSize, lhsCrossSize, rhsCrossSize] may use DynamicDim for
	// product dimensions (e.g., batch*M). Shape.Resolve replaces DynamicDim
	// with the raw binding value, but the actual product (batch*M) differs.
	// After recomputation we have the correct concrete sizes.
	for i, orig := range origNodes {
		if !anyInputHasDynamicDims(orig) {
			continue // Static inputs → original data is fine
		}
		switch orig.opType {
		case backends.OpTypeDotGeneral:
			newData, recomputeErr := recomputeDotGeneralData(e.backend, resolved, orig)
			if recomputeErr != nil {
				return nil, errors.WithMessagef(recomputeErr, "specialization: recomputing DotGeneral data for node %d", i)
			}
			resolved[i].data = newData
			// Fix the intermediate shape to match the recomputed concrete sizes.
			resolved[i].shape = shapes.Make(resolved[i].shape.DType,
				newData.batchSize, newData.lhsCrossSize, newData.rhsCrossSize)
		case backends.OpTypeConvGeneral:
			newData, recomputeErr := recomputeConvGeneralData(resolved, orig)
			if recomputeErr != nil {
				return nil, errors.WithMessagef(recomputeErr, "specialization: recomputing ConvGeneral data for node %d", i)
			}
			resolved[i].data = newData
		case backends.OpTypeGather:
			newData := recomputeGatherData(resolved, orig)
			if newData != nil {
				resolved[i].data = newData
			}
		}
	}

	return &ShapeSpecialization{
		bindings:      bindings,
		resolvedNodes: resolved,
	}, nil
}

// anyInputHasDynamicDims returns true if any input of the node has dynamic dimensions.
func anyInputHasDynamicDims(n *Node) bool {
	for _, input := range n.inputs {
		if input.shape.HasDynamicDims() {
			return true
		}
	}
	return false
}

// recomputeDotGeneralData creates a new dotGeneralNodeData with all shape-dependent
// fields recomputed from the resolved (concrete) input shapes. The axis configuration
// (contracting/batch axes) is preserved from the original.
func recomputeDotGeneralData(backend *Backend, resolved []*Node, orig *Node) (*dotGeneralNodeData, error) {
	origData := orig.data.(*dotGeneralNodeData)

	// Get resolved (concrete) input shapes.
	lhsShape := resolved[orig.inputs[0].idx].shape
	rhsShape := resolved[orig.inputs[1].idx].shape

	newData := &dotGeneralNodeData{
		lhsContractingAxes: slices.Clone(origData.lhsContractingAxes),
		lhsBatchAxes:       slices.Clone(origData.lhsBatchAxes),
		rhsContractingAxes: slices.Clone(origData.rhsContractingAxes),
		rhsBatchAxes:       slices.Clone(origData.rhsBatchAxes),
	}

	// Recompute sizes from concrete shapes using the existing dgFindSizes.
	newData.batchSize, newData.lhsCrossSize, newData.contractingSize, _ =
		dgFindSizes(lhsShape, origData.lhsContractingAxes, origData.lhsBatchAxes)
	_, newData.rhsCrossSize, _, _ =
		dgFindSizes(rhsShape, origData.rhsContractingAxes, origData.rhsBatchAxes)

	// Recompute normalization info from concrete shapes.
	newData.lhsNormalization = dgNormalizePrepare(lhsShape, origData.lhsContractingAxes, origData.lhsBatchAxes)
	newData.rhsNormalization = dgNormalizePrepare(rhsShape, origData.rhsContractingAxes, origData.rhsBatchAxes)

	// Recompute blocked shapes.
	dtype := lhsShape.DType
	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	newData.lhsBlockedShape = dgCreateBlockedShape(dtype, newData.batchSize, newData.lhsCrossSize, newData.contractingSize, blockLog2Dim)
	newData.rhsBlockedShape = dgCreateBlockedShape(dtype, newData.batchSize, newData.rhsCrossSize, newData.contractingSize, blockLog2Dim)
	outputDType := dtype
	if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
		outputDType = dtypes.Float32
	}
	newData.outputBlockedShape = dgCreateBlockedShape(outputDType, newData.batchSize, newData.lhsCrossSize, newData.rhsCrossSize, blockLog2Dim)

	// Select exec path. blockedPath and checkPath require pre-blocked input nodes
	// created at graph build time, which aren't available for dynamic graphs.
	// Fall back to normalizedPath for those. packgemmPath and highwayPath work
	// directly on raw inputs and are fine for dynamic graphs.
	newData.execPath = dgSelectExecPath(backend, lhsShape, rhsShape, newData)
	if newData.execPath == blockedPath || newData.execPath == checkPath {
		newData.execPath = normalizedPath
	}

	return newData, nil
}

// recomputeGatherData updates the Gather node's sliceSizes from the resolved
// operand shape. When the operand has DynamicDim at build time, sliceSizes
// entries contain -1 (DynamicDim) which must be replaced with concrete values.
// Returns nil if no update is needed.
func recomputeGatherData(resolved []*Node, orig *Node) *gatherNode {
	origData := orig.data.(*gatherNode)
	operandShape := resolved[orig.inputs[0].idx].shape

	// Check if any sliceSize needs updating.
	needsUpdate := false
	for i, s := range origData.sliceSizes {
		if s == shapes.DynamicDim {
			if operandShape.Dimensions[i] == shapes.DynamicDim {
				// Still dynamic after resolution — shouldn't happen but guard anyway.
				continue
			}
			needsUpdate = true
			break
		}
	}
	if !needsUpdate {
		return nil
	}

	newData := &gatherNode{
		indexVectorAxis:      origData.indexVectorAxis,
		offsetOutputAxes:     origData.offsetOutputAxes,
		collapsedSlicesAxes:  origData.collapsedSlicesAxes,
		startIndexMap:        origData.startIndexMap,
		sliceSizes:           slices.Clone(origData.sliceSizes),
		indicesAreSorted:     origData.indicesAreSorted,
	}
	for i, s := range newData.sliceSizes {
		if s == shapes.DynamicDim {
			newData.sliceSizes[i] = operandShape.Dimensions[i]
		}
	}
	return newData
}

// recomputeConvGeneralData creates a new convNode with stride-dependent fields
// recomputed from the resolved (concrete) input shapes.
func recomputeConvGeneralData(resolved []*Node, orig *Node) (*convNode, error) {
	origData := orig.data.(*convNode)

	inputShape := resolved[orig.inputs[0].idx].shape
	kernelShape := resolved[orig.inputs[1].idx].shape

	newData := &convNode{
		axes:               origData.axes,
		strides:            origData.strides,
		paddings:           origData.paddings,
		inputDilations:     origData.inputDilations,
		kernelDilations:    origData.kernelDilations,
		channelGroupCount:  origData.channelGroupCount,
		batchGroupCount:    origData.batchGroupCount,
		hasInputDilations:  origData.hasInputDilations,
		hasKernelDilations: origData.hasKernelDilations,
	}

	newData.inputStrides = inputShape.Strides()
	newData.kernelStrides = kernelShape.Strides()

	spatialRank := inputShape.Rank() - 2
	newData.dilatedInputSpatialDims = make([]int, spatialRank)
	newData.inputSpatialStrides = make([]int, spatialRank)
	for spatialIdx, inputAxis := range origData.axes.InputSpatial {
		newData.inputSpatialStrides[spatialIdx] = newData.inputStrides[inputAxis]
		dim := inputShape.Dimensions[inputAxis]
		if dim > 0 {
			newData.dilatedInputSpatialDims[spatialIdx] = (dim-1)*origData.inputDilations[spatialIdx] + 1
		}
	}

	return newData, nil
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

