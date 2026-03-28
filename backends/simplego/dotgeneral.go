// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"math"
	"math/bits"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego/highway"
	"github.com/gomlx/gomlx/backends/simplego/packgemm"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

func init() {
	setNodeExecutor(backends.OpTypeDotGeneral, priorityGeneric, execDotGeneral)
}

type dotGeneralNodeData struct {
	lhsContractingAxes, lhsBatchAxes                       []int
	rhsContractingAxes, rhsBatchAxes                       []int
	batchSize, lhsCrossSize, rhsCrossSize, contractingSize int
	lhsBlockedShape, rhsBlockedShape, outputBlockedShape   shapes.Shape
	lhsNormalization, rhsNormalization                     *dgNormalizationInfo
	config                                                 backends.DotGeneralConfig

	// execPath determines which execution strategy to use. Decided at graph-build time.
	execPath dotGeneralExecutionPath
}

type dotGeneralTypePlan struct {
	inputDType             dtypes.DType
	computationDType       dtypes.DType
	resultDType            dtypes.DType
	nodeOutputDType        dtypes.DType
	convertInputsUpfrontTo dtypes.DType
}

// EqualNodeData implements nodeDataComparable for dotGeneralNodeData.
func (d *dotGeneralNodeData) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*dotGeneralNodeData)
	if d.batchSize != o.batchSize ||
		d.lhsCrossSize != o.lhsCrossSize ||
		d.rhsCrossSize != o.rhsCrossSize ||
		d.contractingSize != o.contractingSize ||
		d.config != o.config ||
		d.execPath != o.execPath {
		return false
	}
	return slices.Equal(d.lhsContractingAxes, o.lhsContractingAxes) &&
		slices.Equal(d.lhsBatchAxes, o.lhsBatchAxes) &&
		slices.Equal(d.rhsContractingAxes, o.rhsContractingAxes) &&
		slices.Equal(d.rhsBatchAxes, o.rhsBatchAxes) &&
		d.lhsBlockedShape.Equal(o.lhsBlockedShape) &&
		d.rhsBlockedShape.Equal(o.rhsBlockedShape) &&
		d.outputBlockedShape.Equal(o.outputBlockedShape)
}

// adjustAxisToRank returns a positive axis, adjusting negative numbers to the correct rank.
func adjustAxisToRank(rank, axis int) (int, error) {
	if axis < 0 {
		axis += rank
	}
	if axis < 0 || axis >= rank {
		return -1, errors.Errorf("axis %d is out of range [0, %d)", axis, rank)
	}
	return axis, nil
}

// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension and finally the 'rhs' non-contracting/non-batch dimension.
// It provides the basic means of implementing Einsum.
//
// This function implements backends.Builder interface.
//
// This is the graph building part of DotGeneral. It first transposes the operands to a normalized
// shape with rank=3 ([batchSize, crossSize, contractingSize]), and then it issues the DotGeneral
// node with normalized inputs. Finally, it reshapes back to the final result.
//
// See execDotGeneral for the implementation.
func (f *Function) DotGeneral(
	lhsOp backends.Value, lhsContractingAxes, lhsBatchAxes []int,
	rhsOp backends.Value, rhsContractingAxes, rhsBatchAxes []int,
	config backends.DotGeneralConfig) (backends.Value, error) {
	// Parse the inputs.
	inputPair, err := f.verifyAndCastValues(backends.OpTypeDotGeneral.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputPair[0], inputPair[1]
	inputDType := lhs.shape.DType
	if inputDType != rhs.shape.DType {
		return nil, errors.Errorf("DotGeneral lhs (left-hand-side) and rhs operands don't match data types: %s and %s",
			inputDType, rhs.shape.DType)
	}
	typePlan := newDotGeneralTypePlan(inputDType, config)

	// Convert inputs upfront when the requested accumulator dtype cannot be handled
	// directly by the execution kernels.
	if typePlan.convertInputsUpfrontTo != dtypes.InvalidDType {
		lhsOp, err = f.ConvertDType(lhsOp, typePlan.convertInputsUpfrontTo)
		if err != nil {
			return nil, err
		}
		rhsOp, err = f.ConvertDType(rhsOp, typePlan.convertInputsUpfrontTo)
		if err != nil {
			return nil, err
		}
		inputPair, err = f.verifyAndCastValues(backends.OpTypeDotGeneral.String(), lhsOp, rhsOp)
		if err != nil {
			return nil, err
		}
		lhs, rhs = inputPair[0], inputPair[1]
	}

	if len(lhsContractingAxes) != len(rhsContractingAxes) {
		return nil, errors.Errorf("DotGeneral number of contracting axes for lhs (%d) doesn't match rhs (%d)",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}
	if len(lhsBatchAxes) != len(rhsBatchAxes) {
		return nil, errors.Errorf("DotGeneral number of batch axes for lhs (%d) doesn't match rhs (%d)",
			len(lhsBatchAxes), len(rhsBatchAxes))
	}

	lhsRank := lhs.shape.Rank()
	rhsRank := rhs.shape.Rank()
	params := dotGeneralNodeData{
		lhsContractingAxes: lhsContractingAxes,
		lhsBatchAxes:       lhsBatchAxes,
		rhsContractingAxes: rhsContractingAxes,
		rhsBatchAxes:       rhsBatchAxes,
		config:             config,
	}

	// Validate and adjust axes.
	for ii, axis := range lhsContractingAxes {
		params.lhsContractingAxes[ii], err = adjustAxisToRank(lhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err,
				"while adjusting contractingAxes for DotGeneral(lhs=%s, lhsContractingAxes=%v)",
				lhs.shape, lhsContractingAxes)
		}
	}
	for ii, axis := range lhsBatchAxes {
		params.lhsBatchAxes[ii], err = adjustAxisToRank(lhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err,
				"while adjusting batchAxes for DotGeneral(lhs=%s, lhsBatchAxes=%v)", lhs.shape, lhsBatchAxes)
		}
	}
	for ii, axis := range rhsContractingAxes {
		params.rhsContractingAxes[ii], err = adjustAxisToRank(rhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err,
				"while adjusting contractingAxes for DotGeneral(rhs=%s, rhsContractingAxes=%v)",
				rhs.shape, rhsContractingAxes)
		}
	}
	for ii, axis := range rhsBatchAxes {
		params.rhsBatchAxes[ii], err = adjustAxisToRank(rhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err,
				"while adjusting batchAxes for DotGeneral(rhs=%s, rhsBatchAxes=%v)", rhs.shape, rhsBatchAxes)
		}
	}

	hasDynamic := lhs.shape.HasDynamicDims() || rhs.shape.HasDynamicDims()

	// Check that batch and contracting dimensions from lhs and rhs match.
	// For dynamic dims, allow DynamicDim values through and unify: if either side is
	// DynamicDim, the unified dim is DynamicDim; otherwise they must be equal.
	batchDims := make([]int, len(lhsBatchAxes))
	batchAxisNames := make([]string, len(lhsBatchAxes))
	contractingDims := make([]int, len(lhsContractingAxes))
	for ii, lhsAxis := range params.lhsContractingAxes {
		rhsAxis := params.rhsContractingAxes[ii]
		lhsDim := lhs.shape.Dimensions[lhsAxis]
		rhsDim := rhs.shape.Dimensions[rhsAxis]
		unified, err := unifyDim(lhsDim, rhsDim)
		if err != nil {
			return nil, errors.Errorf("DotGeneral contracting dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhsDim, rhsAxis, rhsDim)
		}
		contractingDims[ii] = unified
	}
	for ii, lhsAxis := range params.lhsBatchAxes {
		rhsAxis := params.rhsBatchAxes[ii]
		lhsDim := lhs.shape.Dimensions[lhsAxis]
		rhsDim := rhs.shape.Dimensions[rhsAxis]
		unified, err := unifyDim(lhsDim, rhsDim)
		if err != nil {
			return nil, errors.Errorf("DotGeneral batch dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhsDim, rhsAxis, rhsDim)
		}
		batchDims[ii] = unified
		// Propagate axis name from the side that has a name.
		name := lhs.shape.AxisName(lhsAxis)
		if name == "" {
			name = rhs.shape.AxisName(rhsAxis)
		}
		batchAxisNames[ii] = name
	}

	if hasDynamic {
		return f.dotGeneralDynamic(lhs, rhs, typePlan, &params, batchDims, batchAxisNames, contractingDims)
	}

	// --- STATIC PATH (unchanged) ---

	// Find sizes of the normalized operands ([batchSize, crossSize, contractSize]).
	var lhsCrossDims, rhsCrossDims []int
	params.batchSize, params.lhsCrossSize, params.contractingSize, lhsCrossDims = dgFindSizes(
		lhs.shape, lhsContractingAxes, lhsBatchAxes)
	_, params.rhsCrossSize, _, rhsCrossDims = dgFindSizes(rhs.shape, rhsContractingAxes, rhsBatchAxes)

	// Check that all sizes are positive
	if params.batchSize <= 0 || params.lhsCrossSize <= 0 || params.contractingSize <= 0 || params.rhsCrossSize <= 0 {
		return nil, errors.Errorf("DotGeneral sizes must be positive: lhs(batch=%d, cross=%d, contracting=%d), rhs(cross=%d)",
			params.batchSize, params.lhsCrossSize, params.contractingSize,
			params.rhsCrossSize)
	}

	params.lhsNormalization = dgNormalizePrepare(lhs.shape, params.lhsContractingAxes, params.lhsBatchAxes)
	params.rhsNormalization = dgNormalizePrepare(rhs.shape, params.rhsContractingAxes, params.rhsBatchAxes)

	blockLog2Dim := DotGeneralTargetBlockLog2Dim[typePlan.computationDType]
	params.lhsBlockedShape = dgCreateBlockedShape(
		typePlan.computationDType, params.batchSize, params.lhsCrossSize, params.contractingSize, blockLog2Dim)
	params.rhsBlockedShape = dgCreateBlockedShape(
		typePlan.computationDType, params.batchSize, params.rhsCrossSize, params.contractingSize, blockLog2Dim)
	outputDType := typePlan.computationDType
	if typePlan.computationDType == dtypes.BFloat16 || typePlan.computationDType == dtypes.Float16 {
		// For 16 bits, store the intermediary results as float32 to minimize numerical errors during accumulation.
		// Notice the blockLog2Dim must be the same, because the block dimensions much match the inputs.
		outputDType = dtypes.Float32
	}
	params.outputBlockedShape = dgCreateBlockedShape(
		outputDType, params.batchSize, params.lhsCrossSize, params.rhsCrossSize, blockLog2Dim)

	// Select execution path at build time based on problem size and matrix layout.
	// This enables proper deduplication of pre-blocked inputs via getOrCreateNode.
	params.execPath = typePlan.finalizeExecPath(dgSelectExecPath(f.builder.backend, lhs.shape, rhs.shape, &params))
	klog.V(1).Infof("DotGeneral execPath: %s\n", params.execPath)

	// For blockedPath, pre-block BOTH inputs at graph-build time.
	// This allows deduplication: if the same tensor is used in multiple DotGenerals,
	// the blocking is done once and shared.
	var lhsBlocked, rhsBlocked *Node
	if params.execPath == blockedPath || params.execPath == checkPath {
		lhsBlocked = f.blockForDotGeneral(lhs, params.lhsContractingAxes, params.lhsBatchAxes,
			params.batchSize, params.lhsCrossSize, params.contractingSize)
		rhsBlocked = f.blockForDotGeneral(rhs, params.rhsContractingAxes, params.rhsBatchAxes,
			params.batchSize, params.rhsCrossSize, params.contractingSize)
	}

	// Create dot-general node: it will generate a normalized output [batchSize, lhsCrossSize, rhsCrossSize].
	var inputs []*Node
	switch params.execPath {
	case blockedPath:
		inputs = []*Node{lhsBlocked, rhsBlocked}
	case checkPath:
		// Include inputs in both forms.
		inputs = []*Node{lhsBlocked, rhsBlocked, lhs, rhs}
	default:
		inputs = []*Node{lhs, rhs}
	}
	dotGeneral, _ := f.getOrCreateNode(
		backends.OpTypeDotGeneral,
		shapes.Make(typePlan.nodeOutputDType, params.batchSize, params.lhsCrossSize, params.rhsCrossSize),
		inputs, &params)

	// Reshape result to recover batch and cross dimensions.
	resultingDims := make([]int, 0, len(batchDims)+len(lhsCrossDims)+len(rhsCrossDims))
	resultingDims = append(resultingDims, batchDims...)
	resultingDims = append(resultingDims, lhsCrossDims...)
	resultingDims = append(resultingDims, rhsCrossDims...)
	result, err := f.Reshape(dotGeneral, resultingDims...)
	if err != nil {
		return nil, err
	}

	if typePlan.resultDType != typePlan.nodeOutputDType {
		result, err = f.ConvertDType(result, typePlan.resultDType)
		if err != nil {
			return nil, err
		}
	}
	return result, nil
}

// unifyDim unifies two dimension values: if both are concrete, they must be equal.
// If either (or both) is DynamicDim, the result is DynamicDim.
func unifyDim(a, b int) (int, error) {
	if a == shapes.DynamicDim || b == shapes.DynamicDim {
		return shapes.DynamicDim, nil
	}
	if a != b {
		return 0, errors.Errorf("dimension mismatch: %d != %d", a, b)
	}
	return a, nil
}

// dotGeneralDynamic handles the DotGeneral builder path when inputs have dynamic dims.
// It stores placeholder data in dotGeneralNodeData (to be recomputed at specialization time)
// and builds the output shape with proper axis names for dynamic dimensions.
func (f *Function) dotGeneralDynamic(
	lhs, rhs *Node, typePlan dotGeneralTypePlan, params *dotGeneralNodeData,
	batchDims []int, batchAxisNames []string, contractingDims []int,
) (backends.Value, error) {
	// Find sizes with dynamic-awareness.
	//
	// Two representations of batch axis names are used:
	//   - batchAxisNames ([]string, from DotGeneral caller): per-axis names for the
	//     output reshape, preserving individual batch dimensions.
	//   - batchAxisName (string, from dgFindSizesDynamic): single combined name for the
	//     product dimension in the rank-3 intermediate shape [batchSize, lhsCross, rhsCross].
	//     Taken from whichever side has a dynamic batch axis (preferring rhs, falling back to lhs).
	var lhsCrossDims, rhsCrossDims []int
	var lhsCrossAxisNames, rhsCrossAxisNames []string
	var batchAxisName, lhsBatchAxisName string
	params.batchSize, params.lhsCrossSize, params.contractingSize, lhsCrossDims, lhsCrossAxisNames,
		lhsBatchAxisName = dgFindSizesDynamic(lhs.shape, params.lhsContractingAxes, params.lhsBatchAxes)
	_, params.rhsCrossSize, _, rhsCrossDims, rhsCrossAxisNames,
		batchAxisName = dgFindSizesDynamic(rhs.shape, params.rhsContractingAxes, params.rhsBatchAxes)
	if batchAxisName == "" {
		batchAxisName = lhsBatchAxisName
	}

	// Check that non-dynamic sizes are positive.
	if (params.batchSize != shapes.DynamicDim && params.batchSize <= 0) ||
		(params.lhsCrossSize != shapes.DynamicDim && params.lhsCrossSize <= 0) ||
		(params.contractingSize != shapes.DynamicDim && params.contractingSize <= 0) ||
		(params.rhsCrossSize != shapes.DynamicDim && params.rhsCrossSize <= 0) {
		return nil, errors.Errorf("DotGeneral sizes must be positive: lhs(batch=%d, cross=%d, contracting=%d), rhs(cross=%d)",
			params.batchSize, params.lhsCrossSize, params.contractingSize, params.rhsCrossSize)
	}

	// Store placeholder data — will be recomputed at specialization time.
	params.lhsNormalization = nil
	params.rhsNormalization = nil
	params.lhsBlockedShape = shapes.Shape{}
	params.rhsBlockedShape = shapes.Shape{}
	params.outputBlockedShape = shapes.Shape{}
	params.execPath = autoSelectPath // sentinel: not yet selected

	// Build the intermediate DotGeneral shape: [batchSize, lhsCrossSize, rhsCrossSize].
	// Use MakeDynamic if any dimension is dynamic, otherwise use Make.
	intermediateDims := []int{params.batchSize, params.lhsCrossSize, params.rhsCrossSize}
	intermediateNames := []string{batchAxisName, dgCombinedAxisName(lhsCrossAxisNames), dgCombinedAxisName(rhsCrossAxisNames)}

	var intermediateShape shapes.Shape
	if hasDynamicDim(intermediateDims) {
		intermediateShape = shapes.MakeDynamic(typePlan.nodeOutputDType, intermediateDims, intermediateNames)
	} else {
		intermediateShape = shapes.Make(typePlan.nodeOutputDType, intermediateDims...)
	}

	// Always use 2-input layout for dynamic graphs (no pre-blocking).
	dotGeneral, _ := f.getOrCreateNode(backends.OpTypeDotGeneral, intermediateShape, []*Node{lhs, rhs}, params)

	// Build the final output shape by concatenating batch, lhsCross, and rhsCross dims.
	resultingDims := make([]int, 0, len(batchDims)+len(lhsCrossDims)+len(rhsCrossDims))
	resultingDims = append(resultingDims, batchDims...)
	resultingDims = append(resultingDims, lhsCrossDims...)
	resultingDims = append(resultingDims, rhsCrossDims...)

	resultingAxisNames := make([]string, 0, len(resultingDims))
	resultingAxisNames = append(resultingAxisNames, batchAxisNames...)
	resultingAxisNames = append(resultingAxisNames, lhsCrossAxisNames...)
	resultingAxisNames = append(resultingAxisNames, rhsCrossAxisNames...)

	// Build output shape. For dynamic case, bypass f.Reshape (which calls ReshapeOp → Size() → panic).
	// Instead create the reshape node directly with the correctly-typed output shape.
	var outputShape shapes.Shape
	if hasDynamicDim(resultingDims) {
		outputShape = shapes.MakeDynamic(typePlan.nodeOutputDType, resultingDims, resultingAxisNames)
	} else {
		outputShape = shapes.Make(typePlan.nodeOutputDType, resultingDims...)
	}

	result, _ := f.getOrCreateNode(backends.OpTypeReshape, outputShape, []*Node{dotGeneral}, nil)
	if typePlan.resultDType != typePlan.nodeOutputDType {
		converted, err := f.ConvertDType(result, typePlan.resultDType)
		if err != nil {
			return nil, err
		}
		return converted, nil
	}
	return result, nil
}

// hasDynamicDim returns true if any element is DynamicDim.
func hasDynamicDim(dims []int) bool {
	for _, d := range dims {
		if d == shapes.DynamicDim {
			return true
		}
	}
	return false
}

// dgCombinedAxisName returns the axis name for a combined (product) dimension.
// If exactly one contributing axis is dynamic, its name is used.
// If no axes are dynamic (all static), returns "" (no name needed for static dims).
// If multiple axes have names, returns the first non-empty name (the product
// of multiple named dynamic dims cannot be meaningfully represented).
func dgCombinedAxisName(names []string) string {
	for _, n := range names {
		if n != "" {
			return n
		}
	}
	return ""
}

func newDotGeneralTypePlan(inputDType dtypes.DType, config backends.DotGeneralConfig) dotGeneralTypePlan {
	plan := dotGeneralTypePlan{
		inputDType:       inputDType,
		computationDType: inputDType,
		resultDType:      inputDType,
	}
	if config.AccumulatorDType != dtypes.InvalidDType && config.AccumulatorDType != inputDType {
		// Half-precision kernels already accumulate in float32 internally, so no
		// input conversion is needed for the common AccumulatorDType(float32) case.
		if !(inputDType.IsHalfPrecision() && config.AccumulatorDType == dtypes.Float32) {
			plan.computationDType = config.AccumulatorDType
			plan.convertInputsUpfrontTo = config.AccumulatorDType
		}
	}
	if config.OutputDType != dtypes.InvalidDType {
		plan.resultDType = config.OutputDType
	}
	if plan.convertInputsUpfrontTo != dtypes.InvalidDType {
		plan.nodeOutputDType = plan.computationDType
		return plan
	}
	plan.nodeOutputDType = dgNodeOutputDType(plan.inputDType, plan.computationDType, plan.resultDType)
	return plan
}

func (p dotGeneralTypePlan) finalizeExecPath(execPath dotGeneralExecutionPath) dotGeneralExecutionPath {
	if execPath == smallMatMulPath && !p.smallMatMulSupportsOutput() {
		return normalizedPath
	}
	return execPath
}

func (p dotGeneralTypePlan) smallMatMulSupportsOutput() bool {
	return dgSmallMatMulSupportsOutput(p.kernelInputDType(), p.nodeOutputDType)
}

func (p dotGeneralTypePlan) kernelInputDType() dtypes.DType {
	if p.convertInputsUpfrontTo != dtypes.InvalidDType {
		return p.convertInputsUpfrontTo
	}
	return p.inputDType
}

func dgNodeOutputDType(inputDType, computationDType, resultDType dtypes.DType) dtypes.DType {
	// Half-precision kernels accumulate in float32 internally. When the requested
	// result dtype differs from the native half output, keep the DotGeneral node in
	// the requested dtype so the float32 accumulator converts directly to the final
	// result instead of rounding back to half first.
	if inputDType.IsHalfPrecision() && resultDType != computationDType {
		return resultDType
	}
	return computationDType
}

func dgSmallMatMulSupportsOutput(inputDType, nodeOutputDType dtypes.DType) bool {
	return !inputDType.IsHalfPrecision() || nodeOutputDType == inputDType
}

// dgFindSizes finds the combined sizes of the 3 types of axes that mather:
// batch, cross, and contracting dimensions for a DotGeneral operation
func dgFindSizes(shape shapes.Shape, contractingAxes, batchAxes []int) (
	batchSize, crossSize, contractingSize int, crossDims []int) {
	rank := shape.Rank()
	axesTypes := make([]int, rank)

	// Mark axes types: 1 for contracting, 2 for batch
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
	}
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
	}

	// Calculate sizes by multiplying dimensions according to the axis type.
	batchSize, crossSize, contractingSize = 1, 1, 1
	crossDims = make([]int, 0, rank-len(contractingAxes)-len(batchAxes))
	for axis, axisType := range axesTypes {
		dim := shape.Dimensions[axis]
		switch axisType {
		case 0: // Cross axes (unmarked)
			crossSize *= dim
			crossDims = append(crossDims, dim)
		case 1: // Contracting axes
			contractingSize *= dim
		case 2: // Batch axes
			batchSize *= dim
		}
	}
	return
}

// dgFindSizesDynamic is like dgFindSizes but handles DynamicDim values.
// If any contributing axis has DynamicDim, the running product becomes DynamicDim.
// It also returns the axis names for each cross dimension and the combined axis name
// for the batch product (used for the rank-3 intermediate shape).
func dgFindSizesDynamic(shape shapes.Shape, contractingAxes, batchAxes []int) (
	batchSize, crossSize, contractingSize int, crossDims []int, crossAxisNames []string,
	batchAxisName string) {
	rank := shape.Rank()
	axesTypes := make([]int, rank)

	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
	}
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
	}

	batchSize, crossSize, contractingSize = 1, 1, 1
	crossDims = make([]int, 0, rank-len(contractingAxes)-len(batchAxes))
	crossAxisNames = make([]string, 0, cap(crossDims))
	for axis, axisType := range axesTypes {
		dim := shape.Dimensions[axis]
		name := shape.AxisName(axis)
		switch axisType {
		case 0: // Cross axes
			crossSize = mulDynamic(crossSize, dim)
			crossDims = append(crossDims, dim)
			crossAxisNames = append(crossAxisNames, name)
		case 1: // Contracting axes
			contractingSize = mulDynamic(contractingSize, dim)
		case 2: // Batch axes
			batchSize = mulDynamic(batchSize, dim)
			if dim == shapes.DynamicDim && batchAxisName == "" {
				batchAxisName = name
			}
		}
	}
	return
}

// mulDynamic multiplies two dimension values, treating DynamicDim as absorbing:
// if either operand is DynamicDim, the result is DynamicDim.
func mulDynamic(a, b int) int {
	if a == shapes.DynamicDim || b == shapes.DynamicDim {
		return shapes.DynamicDim
	}
	return a * b
}

// dotGeneralExecutionPath indicates which execution strategy to use for DotGeneral.
// Path selection happens at graph-build time in DotGeneral(), not at execution time.
type dotGeneralExecutionPath int

const (
	// autoSelectPath means the execution path should be auto-selected based on matrix size.
	// Used as the default for backend.dotGeneralForceExecutionPath, and as a placeholder
	// sentinel in dotGeneralDynamic (replaced at specialization time by recomputeDotGeneralData).
	autoSelectPath dotGeneralExecutionPath = iota
	// normalizedPath uses the normalized transpose path (small matrices)
	normalizedPath
	// blockedPath uses execDotGeneralBlocked (cache-tiled algorithm, large matrices)
	blockedPath
	// smallMatMulPath uses the SmallMatMul fast path (small float32 matrices in standard order)
	smallMatMulPath
	// packgemmPath uses the packgemm package with a fast matmul algorithm with continuous packing of the matrices.
	packgemmPath
	// highwayPath uses the highway package (uses go-highway) with a fast matmul algorithm with continuous packing of the matrices.
	highwayPath
	// checkPath runs both paths and compares outputs (for debugging)
	checkPath
)

//go:generate go tool enumer -type dotGeneralExecutionPath -output=gen_dotgeneral_execution_path_enumer.go dotgeneral.go

// dgSelectExecPath selects the execution path based on problem size and backend configuration.
// Called at graph-build time from DotGeneral().
func dgSelectExecPath(backend *Backend, lhsShape, rhsShape shapes.Shape, params *dotGeneralNodeData) dotGeneralExecutionPath {
	dtype := lhsShape.DType
	outputDType := dtype
	if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
		// For 16 bits, store the intermediary results as float32 to minimize numerical errors during accumulation.
		// Notice the blockLog2Dim must be the same, because the block dimensions much match the inputs.
		outputDType = dtypes.Float32
	}

	// If a specific path is forced via backend config, use that.
	if backend.dotGeneralForceExecutionPath != autoSelectPath {
		// Checks whether the forced path is valid for the given problem.
		var valid bool
		switch backend.dotGeneralForceExecutionPath {
		case smallMatMulPath:
			valid = isMatMulOrder(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes,
				rhsShape, params.rhsContractingAxes, params.rhsBatchAxes)
		case packgemmPath:
			valid = backend.enablePackgemm && packgemm.HasDTypeSupport(dtype, outputDType) &&
				isMatMulOrder(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes,
					rhsShape, params.rhsContractingAxes, params.rhsBatchAxes)
		case highwayPath:
			valid = backend.enableHighway && highway.HasDTypeSupport(dtype, outputDType) &&
				isMatMulOrder(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes,
					rhsShape, params.rhsContractingAxes, params.rhsBatchAxes)
		default:
			valid = true
		}
		if valid {
			return backend.dotGeneralForceExecutionPath
		}
		klog.V(1).Infof("DotGeneral: forced path %s is invalid for problem size %s×%s\n", backend.dotGeneralForceExecutionPath, lhsShape, rhsShape)
	}

	// GEMM path:
	if backend.enablePackgemm && packgemm.HasDTypeSupport(dtype, outputDType) &&
		isMatMulOrder(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes,
			rhsShape, params.rhsContractingAxes, params.rhsBatchAxes) {
		return packgemmPath
	}

	// Highway path:
	if backend.enableHighway && highway.HasDTypeSupport(dtype, outputDType) &&
		isMatMulOrder(lhsShape, params.lhsContractingAxes, params.lhsBatchAxes,
			rhsShape, params.rhsContractingAxes, params.rhsBatchAxes) {
		return highwayPath
	}

	// Check for SmallMatMul fast path first.
	// SmallMatMul is beneficial for small float32 matrices in standard [M,K]×[K,N] order.
	if dgUseSmallMatMul(dtype, lhsShape, rhsShape, params) {
		return smallMatMulPath
	}

	// Default selection based on problem size.
	// For large matrices, the blocked path with cache-tiled algorithm is more efficient.
	crossesSize := params.rhsCrossSize * params.lhsCrossSize
	blockDim := 1 << DotGeneralTargetBlockLog2Dim[dtype]
	blockSize := blockDim * blockDim
	if crossesSize > DotGeneralBlockedPathThreshold*blockSize {
		return blockedPath
	}
	return normalizedPath
}

// execDotGeneral executes the DotGeneral operation.
// The execution path is pre-selected at graph-build time and stored in params.execPath.
// For blockedPath, inputs are already pre-blocked at build time.
func execDotGeneral(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	lhs, rhs := inputs[0], inputs[1]
	params := node.data.(*dotGeneralNodeData)
	outputShape := node.shape
	output, err := backend.getBufferForShape(outputShape)
	if err != nil {
		return nil, err
	}
	switch params.execPath {
	case autoSelectPath:
		backend.putBuffer(output)
		return nil, errors.Errorf("DotGeneral: execPath is autoSelectPath sentinel — specialization recomputation was skipped for node with shape %s", outputShape)
	case blockedPath, checkPath:
		// Inputs are pre-blocked at graph-build time. Extract block metadata from input nodes.
		lhsNode := node.inputs[0]
		rhsNode := node.inputs[1]
		_, ok := lhsNode.data.(*blockForDotGeneralData)
		if !ok {
			backend.putBuffer(output)
			return nil, errors.Errorf("blockedPath requires pre-blocked LHS input, got %T (node type: %s)",
				lhsNode.data, lhsNode.opType)
		}
		rhsBlockData, ok := rhsNode.data.(*blockForDotGeneralData)
		if !ok {
			backend.putBuffer(output)
			return nil, errors.Errorf("blockedPath requires pre-blocked RHS input, got %T (node type: %s)",
				rhsNode.data, rhsNode.opType)
		}
		hasBatch := len(rhsBlockData.batchAxes) > 0 && rhsBlockData.batchSize > 1 // batchSize is the same for lhs and rhs
		err = execDotGeneralBlocked(backend, lhs, rhs, hasBatch, params, output)
		inputDType := lhs.shape.DType

		// Now run checks against other algorithms.
		if err == nil && params.execPath == checkPath {
			// The "checkPath" is the debug path: it uses the blocked path as a reference and runs all other possible paths
			// comparing the results.
			lhsRaw, rhsRaw := inputs[2], inputs[3]
			output2, err := backend.getBufferForShape(outputShape)
			if err != nil {
				return nil, err
			}
			output2.Zeros()
			err = execDotGeneralNormalized(backend, lhsRaw, rhsRaw, params, output2)
			if err != nil {
				backend.putBuffer(output2)
				backend.putBuffer(output)
				return nil, err
			}
			err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
			if err != nil {
				backend.putBuffer(output2)
				backend.putBuffer(output)
				return nil, err
			}

			// Also verify SmallMatMul path for matrices in matmul order
			rawDType := lhsRaw.shape.DType
			if rawDType < MaxDTypes && dotGeneralSmallMatMulDTypeMap.Map[rawDType] != nil &&
				dgSmallMatMulSupportsOutput(rawDType, outputShape.DType) &&
				isMatMulOrder(lhsRaw.shape, params.lhsContractingAxes, params.lhsBatchAxes,
					rhsRaw.shape, params.rhsContractingAxes, params.rhsBatchAxes) {
				output2.Zeros()
				execSmallMatMulFnAny, err := dotGeneralSmallMatMulDTypeMap.Get(rawDType)
				if err != nil {
					return nil, err
				}
				execSmallMatMulFn := execSmallMatMulFnAny.(func(*Backend, *Buffer, *Buffer, *dotGeneralNodeData, *Buffer))
				// BFloat16/Float16 implementations accumulate in float32 internally but write to native output
				execSmallMatMulFn(backend, lhsRaw, rhsRaw, params, output2)
				err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
				if err != nil {
					backend.putBuffer(output2)
					backend.putBuffer(output)
					return nil, err
				}
			}

			// GEMM specialized executor.
			if backend.enablePackgemm && isMatMulOrder(lhsRaw.shape, params.lhsContractingAxes, params.lhsBatchAxes,
				rhsRaw.shape, params.rhsContractingAxes, params.rhsBatchAxes) &&
				packgemm.HasDTypeSupport(inputDType, inputDType) {
				err = packgemm.GEMM(float32(1), float32(0), lhsRaw.flat.([]float32), rhsRaw.flat.([]float32),
					params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize,
					output2.flat.([]float32),
					getBufAllocator[float32](backend), getBufReleaser(backend), backend.workers)
				if err == nil {
					err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
				}
				if err != nil {
					backend.putBuffer(output2)
					backend.putBuffer(output)
					return nil, err
				}
			}

			// Highway MatMul specialized executor.
			if backend.enableHighway && isMatMulOrder(lhsRaw.shape, params.lhsContractingAxes, params.lhsBatchAxes,
				rhsRaw.shape, params.rhsContractingAxes, params.rhsBatchAxes) &&
				highway.HasDTypeSupport(inputDType, inputDType) {
				err = highway.MatMulDynamic(inputDType, outputShape.DType, lhsRaw.flat, rhsRaw.flat,
					params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize,
					output2.flat,
					getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.workers)
				if err == nil {
					err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
				}
				if err != nil {
					backend.putBuffer(output2)
					backend.putBuffer(output)
					return nil, err
				}
			}

			backend.putBuffer(output2) // Discard second output, no longer needed
			return output, nil
		}

	case smallMatMulPath:
		// SmallMatMul fast path: small matrices in standard [M,K]×[K,N] order.
		// Path was selected at build time based on matrix layout and size.
		// Supports all numeric dtypes via DTypeMap registration.
		// BFloat16/Float16 implementations accumulate in float32 internally but only support native half outputs.
		dtype := lhs.shape.DType
		if !dgSmallMatMulSupportsOutput(dtype, output.shape.DType) {
			backend.putBuffer(output)
			return nil, errors.Errorf("smallMatMulPath does not support %s inputs with %s outputs",
				dtype, output.shape.DType)
		}
		execSmallMatMulFnAny, err := dotGeneralSmallMatMulDTypeMap.Get(dtype)
		if err != nil {
			return nil, err
		}
		execSmallMatMulFn := execSmallMatMulFnAny.(func(*Backend, *Buffer, *Buffer, *dotGeneralNodeData, *Buffer))
		execSmallMatMulFn(backend, lhs, rhs, params, output)
		return output, nil

	case normalizedPath:
		// Transpose-based normalized path for small matrices
		output.Zeros()
		err = execDotGeneralNormalized(backend, lhs, rhs, params, output)

	case packgemmPath:
		// Custom GEMM path for large "malmul" order.
		inputDType := lhs.shape.DType
		outputDType := output.shape.DType
		if err = packgemm.GEMMDynamic(inputDType, outputDType, 1, 0, lhs.flat.([]float32), rhs.flat.([]float32),
			params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize,
			output.flat.([]float32),
			getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.workers); err != nil {
			return nil, err
		}
		return output, nil

	case highwayPath:
		// Highway MatMul path for large "malmul" order.
		inputDType := lhs.shape.DType
		outputDType := output.shape.DType
		err = highway.MatMulDynamic(inputDType, outputDType, lhs.flat, rhs.flat,
			params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize,
			output.flat,
			getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.workers)
		return output, nil

	default:
		err = errors.Errorf("unknown execution path %d for DotGeneral", params.execPath)
	}

	if err != nil {
		backend.putBuffer(output)
		return nil, err
	}
	return output, nil
}

// log2int return the log2(x) for integer values, rounded down.
// Only defined for positive values.
func log2int(x int) int {
	return bits.Len(uint(x)) - 1
}

var dotGeneralVersionsCheckDelta = 1e-3

func dotGeneralCheckVersions(_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, outputLarge, outputSmall *Buffer) error {
	if klog.V(1).Enabled() {
		var value0 float64
		dtype := outputLarge.shape.DType
		switch dtype {
		case dtypes.Float32:
			value0 = float64(outputLarge.flat.([]float32)[0])
		case dtypes.Float64:
			value0 = outputLarge.flat.([]float64)[0]
		case dtypes.BFloat16:
			value0 = float64(outputLarge.flat.([]bfloat16.BFloat16)[0].Float32())
		}

		fmt.Printf("> %s x %s -> %s (output[...0]=%.5f)\n", lhs.shape, rhs.shape, outputLarge.shape, value0)
	}
	messages, err := dotGeneralCheckVersionsCmp(outputLarge, outputSmall)
	if err == nil {
		return nil
	}
	fmt.Printf("ERROR: dotGeneral check versions failed:\n")
	fmt.Printf("\t- lhs=%s, lhsContractingAxes=%v, lhsBatchAxes=%v\n",
		lhs.shape, params.lhsContractingAxes, params.lhsBatchAxes)
	fmt.Printf("\t- rhs=%s, rhsContractingAxes=%v, rhsBatchAxes=%v\n",
		rhs.shape, params.rhsContractingAxes, params.rhsBatchAxes)
	fmt.Printf("\t- batchSize=%d, lhsCrossSize=%d, rhsCrossAxes=%d, contractingSize=%d\n",
		params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize)
	fmt.Printf("\t- output=%s\n", outputLarge.shape)
	fmt.Printf("%s\n", strings.Join(messages, "\n"))
	return err
}

func dotGeneralCheckVersionsCmp(outputLarge, outputSmall *Buffer) (messages []string, err error) {
	// Make sure shapes are the same.
	if !outputLarge.shape.Equal(outputSmall.shape) {
		return nil, errors.Errorf("outputs have different shapes")
	}
	flatIdx := 0
	dtype := outputLarge.shape.DType
	var mismatches int
	switch dtype {
	case dtypes.Float32:
		largeFlat := outputLarge.flat.([]float32)
		smallFlat := outputSmall.flat.([]float32)
		for indices := range outputLarge.shape.Iter() {
			largeValue := largeFlat[flatIdx]
			smallValue := smallFlat[flatIdx]
			if math.Abs(float64(largeValue)-float64(smallValue)) > dotGeneralVersionsCheckDelta {
				if mismatches < 3 {
					messages = append(
						messages,
						fmt.Sprintf("\tDotGeneral: index %v (flatIdx=%d) has a mismatch on versions: large=%f, small=%f", indices, flatIdx, largeValue, smallValue))
				} else if mismatches == 4 {
					fmt.Printf("\t...")
				}
				mismatches++
			}
			flatIdx++
		}

	default:
		// Not checking other dtypes.
	}
	if mismatches > 0 {
		return messages, errors.Errorf(
			"found %d mismatches (out of %d values) between DotGeneral large and small versions", mismatches, outputLarge.shape.Size())
	}
	return
}

// getBufAllocator returns a buffer allocator for the given numeric type.
// TODO: change signature to return the error
func getBufAllocator[T dtypes.NumberNotComplex](backend *Backend) packgemm.BufAllocFn[T] {
	dtype := dtypes.FromGenericsType[T]()
	return func(size int) (ref any, data []T) {
		buf, err := backend.getBuffer(dtype, size)
		if err != nil {
			return nil, nil
		}
		return buf, buf.flat.([]T)
	}
}

// getAnyBufAllocator returns a buffer allocator for the given dtype.
// TODO: change signature to return the error
func getAnyBufAllocator(backend *Backend, dtype dtypes.DType) packgemm.BufAllocAnyFn {
	return func(size int) (ref any, data any) {
		buf, err := backend.getBuffer(dtype, size)
		if err != nil {
			return nil, nil
		}
		return buf, buf.flat
	}
}

// getBufReleaser returns a buffer releaser for the given numeric type.
func getBufReleaser(backend *Backend) packgemm.BufReleaseFn {
	return func(ref any) {
		backend.putBuffer(ref.(*Buffer))
	}
}
