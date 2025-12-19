package simplego

import (
	"fmt"
	"math"
	"math/bits"
	"strings"

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
func (b *Builder) DotGeneral(lhsOp backends.Op, lhsContractingAxes, lhsBatchAxes []int, rhsOp backends.Op, rhsContractingAxes, rhsBatchAxes []int) (backends.Op, error) {
	inputs, err := b.checkOps(backends.OpTypeDotGeneral.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	lhsDType := lhs.shape.DType
	rhsDType := rhs.shape.DType

	if lhsDType != rhsDType {
		return nil, errors.Errorf("DotGeneral lhs (left-hand-side) and rhs operands don't match data types: %s and %s", lhsDType, rhsDType)
	}

	dtype := lhsDType

	// Output dtype matches input dtype (StableHLO-compliant behavior).
	// Internal accumulation uses wider types (int32 for int8/uint8, float32 for bf16/f16)
	// to prevent overflow, but the final output is converted back to the input dtype.
	nodeOutputDType := dtype
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

	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	params.lhsBlockedShape = dgCreateBlockedShape(dtype, params.batchSize, params.lhsCrossSize, params.contractingSize, blockLog2Dim)
	params.rhsBlockedShape = dgCreateBlockedShape(dtype, params.batchSize, params.rhsCrossSize, params.contractingSize, blockLog2Dim)
	// Determine the internal accumulator dtype for numerical precision.
	// This is separate from the node output dtype - we accumulate in wider types internally
	// to prevent overflow/precision loss, then convert back to the original dtype.
	accumulatorDType := nodeOutputDType
	switch dtype {
	case dtypes.BFloat16, dtypes.Float16:
		// For 16-bit floats, accumulate in float32 to minimize numerical errors.
		// Notice the blockLog2Dim must be the same, because the block dimensions must match the inputs.
		accumulatorDType = dtypes.Float32
	case dtypes.Int8, dtypes.Uint8:
		// For 8-bit integers, accumulate in int32 to prevent overflow.
		// The final output will be saturated back to int8/uint8.
		accumulatorDType = dtypes.Int32
	}
	params.outputBlockedShape = dgCreateBlockedShape(accumulatorDType, params.batchSize, params.lhsCrossSize, params.rhsCrossSize, blockLog2Dim)

	// Check if LHS should be pre-blocked for efficient execution.
	// Pre-blocking is beneficial for constant/parameter tensors that are reused.
	// The blocking node will be de-duplicated if the same input is used in multiple DotGenerals.
	lhsInput := lhs
	if shouldPreBlock(lhs, params.lhsCrossSize, params.contractingSize) {
		lhsInput = b.getOrCreateBlockedInputGeneral(lhs,
			params.lhsContractingAxes, params.lhsBatchAxes,
			params.batchSize, params.lhsCrossSize, params.contractingSize)
	}

	// Check if RHS should be pre-blocked for efficient execution.
	// Pre-blocking is beneficial for 2D weights [K, N] that are reused across batches.
	// The blocking node will be de-duplicated if the same RHS is used in multiple DotGenerals.
	rhsInput := rhs
	if shouldPreBlockRHS(rhs, params.lhsContractingAxes, params.rhsContractingAxes, params.rhsBatchAxes) {
		rhsInput = b.getOrCreateBlockedInput(rhs)
	} else if shouldPreBlock(rhs, params.rhsCrossSize, params.contractingSize) {
		// Generalized pre-blocking for non-2D RHS or non-standard layouts
		rhsInput = b.getOrCreateBlockedInputGeneral(rhs,
			params.rhsContractingAxes, params.rhsBatchAxes,
			params.batchSize, params.rhsCrossSize, params.contractingSize)
	}

	// Create dot-general node: it will generate a normalized output [batchSize, lhsCrossSize, rhsCrossSize].
	// Use nodeOutputDType for output shape (same as input dtype for float types, int32 for int8/uint8).
	dotGeneral := b.newNode(backends.OpTypeDotGeneral, shapes.Make(nodeOutputDType, params.batchSize, params.lhsCrossSize, params.rhsCrossSize), lhsInput, rhsInput)
	dotGeneral.data = &params

	// Reshape result to recover batch and cross dimensions.
	resultingDims := make([]int, 0, len(batchDims)+len(lhsCrossDims)+len(rhsCrossDims))
	resultingDims = append(resultingDims, batchDims...)
	resultingDims = append(resultingDims, lhsCrossDims...)
	resultingDims = append(resultingDims, rhsCrossDims...)
	result, err := b.Reshape(dotGeneral, resultingDims...)

	// fmt.Printf("DotGeneral(*lhs*: %s, c:%v, b:%v; *rhs*:  %s, c:%v, b:%v) -> %s\n",
	//	lhs.shape, lhsContractingAxes, lhsBatchAxes, rhs.shape, rhsContractingAxes, rhsBatchAxes,
	//	result.(*Node).shape)

	if err != nil {
		return nil, err
	}
	return result, nil
}

func dgFindSizes(shape shapes.Shape, contractingAxes, batchAxes []int) (batchSize, crossSize, contractingSize int, crossDims []int) {
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
type dotGeneralExecutionPath int

const (
	// autoSelectPath lets execDotGeneral choose based on matrix size
	autoSelectPath dotGeneralExecutionPath = iota
	// normalizedPath forces use of execDotGeneralSmallNormalized (transpose to [B,Cross,Contract])
	normalizedPath
	// blockedPath forces use of execDotGeneralBlocked (cache-tiled algorithm)
	blockedPath
	// checkPath runs both normalized and blocked, comparing results for debugging
	checkPath
)

// execDotGeneral executes the DotGeneral operation, selecting the optimal execution path.
//
// Execution paths (in order of preference):
//  1. Both LHS and RHS pre-blocked: Most efficient path, both inputs already in blocked format
//  2. Pre-blocked LHS only: LHS is pre-blocked, RHS needs blocking at runtime
//  3. Pre-blocked RHS only: RHS is pre-blocked, LHS needs blocking at runtime
//  4. SmallMatMul path: For small matrices in contract-last order, skip transpose (see execDotGeneralSmallMatMul)
//  5. Normalized path: Transpose to [B,Cross,Contract] form (see execDotGeneralSmallNormalized)
//  6. Blocked path: Cache-tiled algorithm for large matrices (see execDotGeneralBlocked)
func execDotGeneral(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	lhs, rhs := inputs[0], inputs[1]
	params := node.data.(*dotGeneralNodeData)
	outputShape := node.shape
	dtype := lhs.shape.DType
	output := backend.getBufferForShape(outputShape)
	output.Zeros()

	// Check if inputs were pre-blocked at graph build time (via BlockForDotGeneral node).
	lhsNode := node.inputs[0]
	rhsNode := node.inputs[1]
	lhsBlocked := lhsNode.opType == backends.OpTypeBlockForDotGeneral
	rhsBlocked := rhsNode.opType == backends.OpTypeBlockForDotGeneral

	// Handle pre-blocked cases
	switch {
	case lhsBlocked && rhsBlocked:
		// Both inputs are pre-blocked - most efficient path
		lhsBlockData := lhsNode.data.(*blockForDotGeneralData)
		rhsBlockData := rhsNode.data.(*blockForDotGeneralData)
		if err := execDotGeneralWithBothBlocked(backend, lhs, rhs, lhsBlockData, rhsBlockData, params, output); err != nil {
			backend.putBuffer(output)
			return nil, err
		}
		return output, nil

	case lhsBlocked:
		// Only LHS is pre-blocked
		lhsBlockData := lhsNode.data.(*blockForDotGeneralData)
		if err := execDotGeneralWithGraphBlockedLHS(backend, lhs, rhs, lhsBlockData, params, output); err != nil {
			backend.putBuffer(output)
			return nil, err
		}
		return output, nil

	case rhsBlocked:
		// Only RHS is pre-blocked
		blockData := rhsNode.data.(*blockForDotGeneralData)
		if err := execDotGeneralWithGraphBlockedRHS(backend, lhs, rhs, blockData, params, output); err != nil {
			backend.putBuffer(output)
			return nil, err
		}
		return output, nil
	}

	// Neither input is pre-blocked, try other execution paths

	// Try the direct path for small matrices in contract-last order.
	// This skips transpose but has strided RHS access, so only beneficial for small matrices.
	if execDotGeneralSmallMatMul(backend, lhs, rhs, params, output) {
		return output, nil
	}

	// Select execution path based on problem size.
	// For large matrices, the blocked (cache-tiled) algorithm is more efficient.
	// For smaller matrices, the normalized path with simple loops is sufficient.
	crossesSize := params.rhsCrossSize * params.lhsCrossSize
	blockDim := 1 << DotGeneralTargetBlockLog2Dim[dtype]
	blockSize := blockDim * blockDim
	var err error
	execPath := normalizedPath
	if crossesSize > 16*blockSize {
		execPath = blockedPath
	}
	if backend.dotGeneralForceExecutionPath != autoSelectPath {
		execPath = backend.dotGeneralForceExecutionPath
	}
	switch execPath {
	case blockedPath:
		err = execDotGeneralBlocked(backend, lhs, rhs, params, output)
	case normalizedPath:
		err = execDotGeneralSmallNormalized(backend, lhs, rhs, params, output)
	case checkPath:
		output2 := backend.getBufferForShape(outputShape)
		output2.Zeros()
		err = execDotGeneralSmallNormalized(backend, lhs, rhs, params, output2)
		if err != nil {
			return nil, err
		}
		err = execDotGeneralBlocked(backend, lhs, rhs, params, output)
		if err != nil {
			return nil, err
		}
		err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
		backend.putBuffer(output2)
	default:
		err = errors.Errorf("unknown execution path %d for DotGeneral", execPath)
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
func (b *Builder) Dot(lhsOp, rhsOp backends.Op) (backends.Op, error) {
	inputs, err := b.checkOps(backends.OpTypeDot.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]
	var output backends.Op
	switch {
	case lhs.shape.Rank() == 1 && rhs.shape.Rank() == 1:
		// Contracting both vectors.
		output, err = b.DotGeneral(lhs, []int{0}, []int{}, rhs, []int{0}, []int{})
	case lhs.shape.Rank() == 2 && rhs.shape.Rank() == 1:
		// Contract rhs vector.
		output, err = b.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
	case lhs.shape.Rank() == 2 && rhs.shape.Rank() == 2:
		// Traditional matrix multiplication:
		output, err = b.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
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
