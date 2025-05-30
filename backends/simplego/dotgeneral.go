package simplego

import (
	"fmt"
	"math"
	"math/bits"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

func init() {
	nodeExecutors[backends.OpTypeDotGeneral] = execDotGeneral
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

	// Create dot-general node: it will generate a normalized output [batchSize, lhsCrossSize, rhsCrossSize].
	dotGeneral := b.newNode(backends.OpTypeDotGeneral, shapes.Make(dtype, params.batchSize, params.lhsCrossSize, params.rhsCrossSize), lhs, rhs)
	dotGeneral.data = &params

	// Reshape result to recover batch and cross dimensions.
	resultingDims := make([]int, 0, len(batchDims)+len(lhsCrossDims)+len(rhsCrossDims))
	resultingDims = append(resultingDims, batchDims...)
	resultingDims = append(resultingDims, lhsCrossDims...)
	resultingDims = append(resultingDims, rhsCrossDims...)
	result, err := b.Reshape(dotGeneral, resultingDims...)

	//fmt.Printf("DotGeneral(*lhs*: %s, c:%v, b:%v; *rhs*:  %s, c:%v, b:%v) -> %s\n",
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

type dotGeneralProblemSizeType int

const (
	unknownProblemSize dotGeneralProblemSizeType = iota
	smallProblemSize
	largeProblemSize
	checkProblemSize
)

// execDotGeneral executes the DotGeneral by first normalizing and repackaging the tensors into blocks.
func execDotGeneral(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	lhs, rhs := inputs[0], inputs[1]
	params := node.data.(*dotGeneralNodeData)
	outputShape := node.shape
	dtype := lhs.shape.DType
	output := backend.getBufferForShape(outputShape)
	output.Zeros()

	// Problem size (per example of the batch):
	crossesSize := params.rhsCrossSize * params.lhsCrossSize
	blockDim := 1 << DotGeneralTargetBlockLog2Dim[dtype]
	blockSize := blockDim * blockDim
	var err error
	problemSize := smallProblemSize
	if crossesSize > 16*blockSize {
		problemSize = largeProblemSize
	}
	if backend.dotGeneralForceProblemSize != unknownProblemSize {
		problemSize = backend.dotGeneralForceProblemSize
	}
	switch problemSize {
	case largeProblemSize:
		err = execDotGeneralLarge(backend, lhs, rhs, params, output)
	case smallProblemSize:
		err = execDotGeneralSmall(backend, lhs, rhs, params, output)
	case checkProblemSize:
		output2 := backend.getBufferForShape(outputShape)
		output2.Zeros()
		err = execDotGeneralSmall(backend, lhs, rhs, params, output2)
		if err != nil {
			return nil, err
		}
		err = execDotGeneralLarge(backend, lhs, rhs, params, output)
		if err != nil {
			return nil, err
		}
		err = dotGeneralCheckVersions(backend, lhs, rhs, params, output, output2)
		backend.putBuffer(output2)
	default:
		err = errors.Errorf("unknown problem size %d for DotGeneral", problemSize)
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
	if lhs.shape.Rank() == 1 && rhs.shape.Rank() == 1 {
		// Contracting both vectors.
		output, err = b.DotGeneral(lhs, []int{0}, []int{}, rhs, []int{0}, []int{})
	} else if lhs.shape.Rank() == 2 && rhs.shape.Rank() == 1 {
		// Contract rhs vector.
		output, err = b.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
	} else if lhs.shape.Rank() == 2 && rhs.shape.Rank() == 2 {
		// Traditional matrix multiplication:
		output, err = b.DotGeneral(lhs, []int{1}, []int{}, rhs, []int{0}, []int{})
	} else {
		return nil, errors.Errorf("Dot operands have invalid ranks: lhs=%v, rhs=%v", lhs.shape, rhs.shape)
	}
	if err != nil {
		return nil, errors.WithMessagef(err, "while building op Dot()")
	}
	return output, nil
}

var dotGeneralVersionsCheckDelta = 1e-3

func dotGeneralCheckVersions(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, outputLarge, outputSmall *Buffer) error {
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
