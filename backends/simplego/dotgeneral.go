// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"fmt"
	"math"
	"math/bits"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/backends/simplego/highway"
	"github.com/gomlx/gomlx/backends/simplego/packgemm"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
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

	// execPath determines which execution strategy to use. Decided at graph-build time.
	execPath dotGeneralExecutionPath
}

// EqualNodeData implements nodeDataComparable for dotGeneralNodeData.
func (d *dotGeneralNodeData) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*dotGeneralNodeData)
	if d.batchSize != o.batchSize ||
		d.lhsCrossSize != o.lhsCrossSize ||
		d.rhsCrossSize != o.rhsCrossSize ||
		d.contractingSize != o.contractingSize ||
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
func (f *Function) DotGeneral(lhsOp backends.Value, lhsContractingAxes, lhsBatchAxes []int, rhsOp backends.Value, rhsContractingAxes, rhsBatchAxes []int) (backends.Value, error) {
	inputPair, err := f.verifyAndCastValues(backends.OpTypeDotGeneral.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputPair[0], inputPair[1]
	dtype := lhs.shape.DType
	if dtype != rhs.shape.DType {
		return nil, errors.Errorf("DotGeneral lhs (left-hand-side) and rhs operands don't match data types: %s and %s", dtype, rhs.shape.DType)
	}
	if len(lhsContractingAxes) != len(rhsContractingAxes) {
		return nil, errors.Errorf("DotGeneral number of contracting axes for lhs (%d) doesn't match rhs (%d)",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}
	if len(lhsBatchAxes) != len(rhsBatchAxes) {
		return nil, errors.Errorf("DotGeneral number of contracting axes for lhs (%d) doesn't match rhs (%d)",
			len(lhsContractingAxes), len(rhsContractingAxes))
	}

	lhsRank := lhs.shape.Rank()
	rhsRank := rhs.shape.Rank()
	params := dotGeneralNodeData{
		lhsContractingAxes: lhsContractingAxes,
		lhsBatchAxes:       lhsBatchAxes,
		rhsContractingAxes: rhsContractingAxes,
		rhsBatchAxes:       rhsBatchAxes,
	}

	// Validate and adjust axes.
	for ii, axis := range lhsContractingAxes {
		params.lhsContractingAxes[ii], err = adjustAxisToRank(lhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err, "while adjusting contractingAxes for DotGeneral(lhs=%s, lhsContractingAxes=%v)", lhs.shape, lhsContractingAxes)
		}
	}
	for ii, axis := range lhsBatchAxes {
		params.lhsBatchAxes[ii], err = adjustAxisToRank(lhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err, "while adjusting batchAxes for DotGeneral(lhs=%s, lhsBatchAxes=%v)", lhs.shape, lhsBatchAxes)
		}
	}
	for ii, axis := range rhsContractingAxes {
		params.rhsContractingAxes[ii], err = adjustAxisToRank(rhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err, "while adjusting contractingAxes for DotGeneral(rhs=%s, rhsContractingAxes=%v)", rhs.shape, rhsContractingAxes)
		}
	}
	for ii, axis := range rhsBatchAxes {
		params.rhsBatchAxes[ii], err = adjustAxisToRank(rhsRank, axis)
		if err != nil {
			return nil, errors.WithMessagef(err, "while adjusting batchAxes for DotGeneral(rhs=%s, rhsBatchAxes=%v)", rhs.shape, rhsBatchAxes)
		}
	}

	// Check that batch and contracting dimensions from lhs and rhs match.
	batchDims := make([]int, len(lhsBatchAxes))
	contractingDims := make([]int, len(lhsContractingAxes))
	for ii, lhsAxis := range params.lhsContractingAxes {
		rhsAxis := params.rhsContractingAxes[ii]
		if lhs.shape.Dimensions[lhsAxis] != rhs.shape.Dimensions[rhsAxis] {
			return nil, errors.Errorf("DotGeneral contracting dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhs.shape.Dimensions[lhsAxis], rhsAxis, rhs.shape.Dimensions[rhsAxis])
		}
		contractingDims[ii] = lhs.shape.Dimensions[lhsAxis]
	}
	for ii, lhsAxis := range params.lhsBatchAxes {
		rhsAxis := params.rhsBatchAxes[ii]
		if lhs.shape.Dimensions[lhsAxis] != rhs.shape.Dimensions[rhsAxis] {
			return nil, errors.Errorf("DotGeneral batch dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhs.shape.Dimensions[lhsAxis], rhsAxis, rhs.shape.Dimensions[rhsAxis])
		}
		batchDims[ii] = lhs.shape.Dimensions[lhsAxis]
	}

	// Find sizes of the normalized operands ([batchSize, crossSize, contractSize]).
	var lhsCrossDims, rhsCrossDims []int
	params.batchSize, params.lhsCrossSize, params.contractingSize, lhsCrossDims = dgFindSizes(lhs.shape, lhsContractingAxes, lhsBatchAxes)
	_, params.rhsCrossSize, _, rhsCrossDims = dgFindSizes(rhs.shape, rhsContractingAxes, rhsBatchAxes)

	// Check that all sizes are positive
	if params.batchSize <= 0 || params.lhsCrossSize <= 0 || params.contractingSize <= 0 || params.rhsCrossSize <= 0 {
		return nil, errors.Errorf("DotGeneral sizes must be positive: lhs(batch=%d, cross=%d, contracting=%d), rhs(cross=%d)",
			params.batchSize, params.lhsCrossSize, params.contractingSize,
			params.rhsCrossSize)
	}

	params.lhsNormalization = dgNormalizePrepare(lhs.shape, params.lhsContractingAxes, params.lhsBatchAxes)
	params.rhsNormalization = dgNormalizePrepare(rhs.shape, params.rhsContractingAxes, params.rhsBatchAxes)

	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	params.lhsBlockedShape = dgCreateBlockedShape(dtype, params.batchSize, params.lhsCrossSize, params.contractingSize, blockLog2Dim)
	params.rhsBlockedShape = dgCreateBlockedShape(dtype, params.batchSize, params.rhsCrossSize, params.contractingSize, blockLog2Dim)
	outputDType := dtype
	if dtype == dtypes.BFloat16 || dtype == dtypes.Float16 {
		// For 16 bits, store the intermediary results as float32 to minimize numerical errors during accumulation.
		// Notice the blockLog2Dim must be the same, because the block dimensions much match the inputs.
		outputDType = dtypes.Float32
	}
	params.outputBlockedShape = dgCreateBlockedShape(outputDType, params.batchSize, params.lhsCrossSize, params.rhsCrossSize, blockLog2Dim)

	// Select execution path at build time based on problem size and matrix layout.
	// This enables proper deduplication of pre-blocked inputs via getOrCreateNode.
	params.execPath = dgSelectExecPath(f.builder.backend, lhs.shape, rhs.shape, &params)
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
	dotGeneral, _ := f.getOrCreateNode(backends.OpTypeDotGeneral, shapes.Make(dtype, params.batchSize, params.lhsCrossSize, params.rhsCrossSize), inputs, &params)

	// Reshape result to recover batch and cross dimensions.
	resultingDims := make([]int, 0, len(batchDims)+len(lhsCrossDims)+len(rhsCrossDims))
	resultingDims = append(resultingDims, batchDims...)
	resultingDims = append(resultingDims, lhsCrossDims...)
	resultingDims = append(resultingDims, rhsCrossDims...)
	result, err := f.Reshape(dotGeneral, resultingDims...)

	if err != nil {
		return nil, err
	}
	return result, nil
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

// dotGeneralExecutionPath indicates which execution strategy to use for DotGeneral.
// Path selection happens at graph-build time in DotGeneral(), not at execution time.
type dotGeneralExecutionPath int

const (
	// autoSelectPath means the execution path should be auto-selected based on matrix size.
	// This is used only for backend.dotGeneralForceExecutionPath; never stored in params.execPath.
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
	output := backend.getBufferForShape(outputShape)

	var err error
	switch params.execPath {
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
			output2 := backend.getBufferForShape(outputShape)
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
				isMatMulOrder(lhsRaw.shape, params.lhsContractingAxes, params.lhsBatchAxes,
					rhsRaw.shape, params.rhsContractingAxes, params.rhsBatchAxes) {
				output2.Zeros()
				execSmallMatMulFn := dotGeneralSmallMatMulDTypeMap.Get(rawDType).(func(*Backend, *Buffer, *Buffer, *dotGeneralNodeData, *Buffer))
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
		// BFloat16/Float16 implementations accumulate in float32 internally but write to native output.
		dtype := lhs.shape.DType
		execSmallMatMulFn := dotGeneralSmallMatMulDTypeMap.Get(dtype).(func(*Backend, *Buffer, *Buffer, *dotGeneralNodeData, *Buffer))
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
		packgemm.GEMMDynamic(inputDType, outputDType, 1, 0, lhs.flat.([]float32), rhs.flat.([]float32),
			params.batchSize, params.lhsCrossSize, params.rhsCrossSize, params.contractingSize,
			output.flat.([]float32),
			getAnyBufAllocator(backend, inputDType), getBufReleaser(backend), backend.workers)
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

// Dot ------------------------------------------------------------------------------------------------------
// Dot implements backends.Builder interface.
//
// It is implemented using DotGeneral and Reshape.
//
// Dot returns the "dot product" operation.
// The exact semantics of this operation depend on the ranks of the operands:
// | Input | Output | Semantics |
// | vector [n] dot vector [n] | scalar | vector dot product |
// | matrix [m x k] dot vector [k] | vector [m]	matrix-vector multiplication |
// | matrix [m x k] dot matrix [k x n] | matrix [m x n] | matrix-matrix multiplication |
// The operation performs sum of products over the second dimension of x0 (or the first if it has rank 1) and
// the first dimension of x1.
// These are the "contracted" dimensions.
// The contracted dimensions of x0 and x1 must be of the same size.
// In practice, it can be used to perform dot products between vectors, vector/matrix multiplications or
// matrix/matrix multiplications.
// The op is created on the same XlaBuilder as used for x0 and x1.
func (f *Function) Dot(lhsOp, rhsOp backends.Value) (backends.Value, error) {
	inputs, err := f.verifyAndCastValues(backends.OpTypeDot.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	var output backends.Value
	switch {
	case lhs.shape.Rank() == 1 && rhs.shape.Rank() == 1:
		// Contracting both vectors.
		output, err = f.DotGeneral(lhs, []int{0}, []int{}, rhs, []int{0}, []int{})
	case lhs.shape.Rank() == 2 && rhs.shape.Rank() == 1:
		// Contract rhs vector.
		output, err = f.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
	case lhs.shape.Rank() == 2 && rhs.shape.Rank() == 2:
		// Traditional matrix multiplication:
		output, err = f.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
	default:
		return nil, errors.Errorf("Dot operands have invalid ranks: lhs=%v, rhs=%v", lhs.shape, rhs.shape)
	}
	if err != nil {
		return nil, errors.WithMessagef(err, "while building op Dot()")
	}
	return output, nil
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
		return messages, errors.Errorf("found %d mismatches (out of %d values) between DotGeneral large and small versions", mismatches, outputLarge.shape.Size())
	}
	return
}

// getBufAllocator returns a buffer allocator for the given numeric type.
func getBufAllocator[T dtypes.NumberNotComplex](backend *Backend) packgemm.BufAllocFn[T] {
	dtype := dtypes.FromGenericsType[T]()
	return func(size int) (ref any, data []T) {
		buf := backend.getBuffer(dtype, size)
		return buf, buf.flat.([]T)
	}
}

// getAnyBufAllocator returns a buffer allocator for the given dtype.
func getAnyBufAllocator(backend *Backend, dtype dtypes.DType) packgemm.BufAllocAnyFn {
	return func(size int) (ref any, data any) {
		buf := backend.getBuffer(dtype, size)
		return buf, buf.flat
	}
}

// getBufReleaser returns a buffer releaser for the given numeric type.
func getBufReleaser(backend *Backend) packgemm.BufReleaseFn {
	return func(ref any) {
		backend.putBuffer(ref.(*Buffer))
	}
}
