package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/pkg/errors"
	"slices"
)

func init() {
	nodeExecutors[backends.OpTypeDotGeneral] = execNormalizedDotGeneral
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
// non-contracting/non-batch dimension, and finally the 'rhs' non-contracting/non-batch dimension.
// It provides the basic means of implementing Einsum.
//
// This function implements backends.Builder interface.
//
// This is the graph building part of DotGeneral. It first transposes the operands to a normalized
// shape with rank=3 ([batchSize, crossSize, contractingSize]), and then it issues the DotGeneral
// node with normalized inputs. Finally, it reshapes back to the final result.
//
// The actual generic dot multiplication happens during execution though.
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

	// Transpose operands to [batchSize, crossSize, contractingSize].
	lhsTransposed, lhsBatchDims, lhsCrossDims, lhsContractingDims, err := b.transposeForDotGeneral(lhs, "lhs", lhsContractingAxes, lhsBatchAxes)
	if err != nil {
		return nil, err
	}
	rhsTransposed, rhsBatchDims, rhsCrossDims, rhsContractingDims, err := b.transposeForDotGeneral(rhs, "rhs", rhsContractingAxes, rhsBatchAxes)
	if err != nil {
		return nil, err
	}
	if slices.Compare(lhsBatchDims, rhsBatchDims) != 0 {
		return nil, errors.Errorf("DotGeneral batch axes from lhs (left-hand-side) and rhs operands don't match dimenions: lhs.BatchDims=%v, rhs.BatchDims=%v", lhsBatchDims, rhsBatchDims)
	}
	if slices.Compare(lhsContractingDims, rhsContractingDims) != 0 {
		return nil, errors.Errorf("DotGeneral contracting axes from lhs (left-hand-side) and rhs operands don't match dimenions: lhs.ContractingDims=%v, rhs.ContractingDims=%v", lhsContractingDims, rhsContractingDims)
	}

	// DotGeneral on the normalized operands.
	batchSize := lhsTransposed.shape.Dimensions[0]
	dotGeneralShape := shapes.Make(dtype, batchSize, lhsTransposed.shape.Dimensions[1], rhsTransposed.shape.Dimensions[1])
	dotGeneral := b.newNode(backends.OpTypeDotGeneral, dotGeneralShape, lhsTransposed, rhsTransposed)

	// Reshape result to recover batch and cross dimensions.
	resultingDims := make([]int, 0, len(lhsBatchDims)+len(lhsCrossDims)+len(rhsCrossDims))
	resultingDims = append(resultingDims, lhsBatchDims...)
	resultingDims = append(resultingDims, lhsCrossDims...)
	resultingDims = append(resultingDims, rhsCrossDims...)
	result, err := b.Reshape(dotGeneral, resultingDims...)
	if err != nil {
		return nil, err
	}
	return result, nil
}

// transposeForDotGeneral transposes and reshapes the lhs or the rhs operand for the DotGeneral
// so that it is shaped as [batchSize, crossSize, contractSize].
//
// It returns the node of the transposed operand, and the dimensions of each set of axes.
func (b *Builder) transposeForDotGeneral(operand *Node, operandName string, contractingAxes, batchAxes []int) (
	transposed *Node, batchDims, crossDims, contractingDims []int, err error) {
	shape := operand.shape
	rank := shape.Rank()
	axesTypes := make([]int, rank)
	if len(contractingAxes) > 0 {
		contractingDims = make([]int, 0, len(contractingAxes))
		for _, axis := range contractingAxes {
			if axis < 0 || axis >= rank {
				err = errors.Errorf("DotGeneral operand %s has an invalid contracting axis %d (%s rank is %d)", operandName, axis, operandName, rank)
				return
			}
			if axesTypes[axis] != 0 {
				err = errors.Errorf("DotGeneral operand %s contracting axes (%v) have repeated values ", operandName, contractingAxes)
				return
			}
			axesTypes[axis] = 1
			contractingDims = append(contractingDims, shape.Dimensions[axis])
		}
	}
	if len(batchAxes) > 0 {
		batchDims = make([]int, 0, len(batchAxes))
		for _, axis := range batchAxes {
			if axis < 0 || axis >= rank {
				err = errors.Errorf("DotGeneral operand %s has an invalid batch axis %d (%s rank is %d)", operandName, axis, operandName, rank)
				return
			}
			if axesTypes[axis] != 0 {
				err = errors.Errorf("DotGeneral operand %s batch axes (%v) have repeated values (maybe with contracting axes=%v) ", operandName, batchAxes, contractingAxes)
				return
			}
			axesTypes[axis] = 2
			batchDims = append(batchDims, shape.Dimensions[axis])
		}
	}

	// Calculate the crossAxes and crossDims and each of the main resulting dimensions
	batchSize, crossSize, contractSize := 1, 1, 1
	crossAxes := make([]int, 0, rank-len(batchAxes)-len(contractingAxes))
	crossDims = make([]int, 0, rank-len(batchAxes)-len(contractingAxes))
	for axis, axisType := range axesTypes {
		dim := shape.Dimensions[axis]
		switch axisType {
		case 0:
			crossSize *= dim
			crossAxes = append(crossAxes, axis)
			crossDims = append(crossDims, dim)
		case 1:
			contractSize *= dim
		case 2:
			batchSize *= dim
		}
	}

	// Final permutations: batchAxes, crossAxes, contractingAxes
	permutations := make([]int, 0, rank)
	permutations = append(permutations, batchAxes...)
	permutations = append(permutations, crossAxes...)
	permutations = append(permutations, contractingAxes...)
	transposedOp, err := b.Transpose(operand, permutations...)
	if err != nil {
		return
	}
	transposed = transposedOp.(*Node)
	transposedOp, err = b.Reshape(transposed, batchSize, crossSize, contractSize)
	if err != nil {
		return
	}
	transposed = transposedOp.(*Node)
	return
}

// execNormalizedDotGeneral executes the DotGeneral where the inputs are already shaped [batchSize, crossSize, contractingSize].
func execNormalizedDotGeneral(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
	lhs, rhs := inputs[0], inputs[1]
	outputShape := node.shape
	output := backend.getBuffer(outputShape.DType, outputShape.Size())
	output.shape = outputShape
	dispatchDotGeneral.Dispatch(outputShape.DType, lhs, rhs, output)
	return output, nil
}

var dispatchDotGeneral = NewDTypeDispatcher("DotGeneral")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchDotGeneral -generic=execNormalizedDotGeneralGeneric -int -uint -float

// execNormalizedDotGeneralGeneric operands lhs and rhs are normalized to shape
// [batchSize, crossSize, contractingSize].
func execNormalizedDotGeneralGeneric[T PODNumericConstraints](params ...any) any {
	lhs, rhs, output := params[0].(*Buffer), params[1].(*Buffer), params[2].(*Buffer)
	lhsFlat := lhs.flat.([]T)
	rhsFlat := rhs.flat.([]T)
	outputFlat := output.flat.([]T)
	var lhsIdx, rhsIdx, outputIdx int
	batchSize := lhs.shape.Dimensions[0] // same as rhs'.
	lhsCrossSize := lhs.shape.Dimensions[1]
	rhsCrossSize := rhs.shape.Dimensions[1]
	contractingSize := lhs.shape.Dimensions[2] // same as rhs'.
	rhsBatchStride := contractingSize * rhsCrossSize
	for range batchSize {
		for range lhsCrossSize {
			rhsBatchStartIdx := rhsIdx
			for range rhsCrossSize {
				lhsRowStartIdx := lhsIdx
				for range contractingSize {
					outputFlat[outputIdx] += lhsFlat[lhsIdx] * rhsFlat[rhsIdx]
					lhsIdx++
					rhsIdx++
				}
				lhsIdx = lhsRowStartIdx
				outputIdx++
			}
			lhsIdx += contractingSize
			rhsIdx = rhsBatchStartIdx
		}
		rhsIdx += rhsBatchStride
	}
	return nil
}

// Register the BFloat16 version of DotGeneral.
func init() { dispatchDotGeneral.Register(dtypes.BFloat16, execNormalizedDotGeneralBFloat16) }

func execNormalizedDotGeneralBFloat16(params ...any) any {
	lhs, rhs, output := params[0].(*Buffer), params[1].(*Buffer), params[2].(*Buffer)
	lhsFlat := lhs.flat.([]bfloat16.BFloat16)
	rhsFlat := rhs.flat.([]bfloat16.BFloat16)
	outputFlat := output.flat.([]bfloat16.BFloat16)
	var lhsIdx, rhsIdx, outputIdx int
	batchSize := lhs.shape.Dimensions[0] // same as rhs'.
	lhsCrossSize := lhs.shape.Dimensions[1]
	rhsCrossSize := rhs.shape.Dimensions[1]
	contractingSize := lhs.shape.Dimensions[2] // same as rhs'.
	rhsBatchStride := contractingSize * rhsCrossSize
	for range batchSize {
		for range lhsCrossSize {
			rhsBatchStartIdx := rhsIdx
			for range rhsCrossSize {
				lhsRowStartIdx := lhsIdx
				// Accumulate result in float32.
				acc := outputFlat[outputIdx].Float32()
				for range contractingSize {
					acc += lhsFlat[lhsIdx].Float32() * rhsFlat[rhsIdx].Float32()
					lhsIdx++
					rhsIdx++
				}
				outputFlat[outputIdx] = bfloat16.FromFloat32(acc)
				lhsIdx = lhsRowStartIdx
				outputIdx++
			}
			lhsIdx += contractingSize
			rhsIdx = rhsBatchStartIdx
		}
		rhsIdx += rhsBatchStride
	}
	return nil
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
