package simplego

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/pkg/errors"
)

func init() {
	nodeExecutors[backends.OpTypeDotGeneral] = execDotGeneral
}

type dotGeneralNodeData struct {
	lhsContractingAxes, lhsBatchAxes                     []int
	rhsContractingAxes, rhsBatchAxes                     []int
	lhsBatchSize, lhsCrossSize, lhsContractingSize       int
	rhsBatchSize, rhsCrossSize, rhsContractingSize       int
	lhsBlockedShape, rhsBlockedShape, outputBlockedShape shapes.Shape
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
	for ii, lhsAxis := range params.lhsBatchAxes {
		rhsAxis := params.rhsBatchAxes[ii]
		if lhs.shape.Dimensions[lhsAxis] != rhs.shape.Dimensions[rhsAxis] {
			return nil, errors.Errorf("DotGeneral batch dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhs.shape.Dimensions[lhsAxis], rhsAxis, rhs.shape.Dimensions[rhsAxis])
		}
	}
	for ii, lhsAxis := range params.lhsContractingAxes {
		rhsAxis := params.rhsContractingAxes[ii]
		if lhs.shape.Dimensions[lhsAxis] != rhs.shape.Dimensions[rhsAxis] {
			return nil, errors.Errorf("DotGeneral contracting dimensions don't match: lhs[%d]=%d != rhs[%d]=%d",
				lhsAxis, lhs.shape.Dimensions[lhsAxis], rhsAxis, rhs.shape.Dimensions[rhsAxis])
		}
	}

	// Find sizes of the normalized operands ([batchSize, crossSize, contractSize]).
	params.lhsBatchSize, params.lhsCrossSize, params.lhsContractingSize = dgFindSizes(lhs.shape, lhsContractingAxes, lhsBatchAxes)
	params.rhsBatchSize, params.rhsCrossSize, params.rhsContractingSize = dgFindSizes(rhs.shape, rhsContractingAxes, rhsBatchAxes)
	if params.lhsBatchSize != params.rhsBatchSize {
		return nil, errors.Errorf("DotGeneral batch axes from lhs (left-hand-side) and rhs operands don't match dimenions: lhs.BatchDims=%v, rhs.BatchDims=%v", lhs.shape.Dimensions, rhs.shape.Dimensions)
	}
	if params.lhsContractingSize != params.rhsContractingSize {
		return nil, errors.Errorf("DotGeneral contracting axes from lhs (left-hand-side) and rhs operands don't match dimenions: lhs.ContractingDims=%v, rhs.ContractingDims=%v", lhs.shape.Dimensions, rhs.shape.Dimensions)
	}

	// Check that all sizes are positive
	if params.lhsBatchSize <= 0 || params.lhsCrossSize <= 0 || params.lhsContractingSize <= 0 ||
		params.rhsBatchSize <= 0 || params.rhsCrossSize <= 0 || params.rhsContractingSize <= 0 {
		return nil, errors.Errorf("DotGeneral sizes must be positive: lhs(batch=%d, cross=%d, contracting=%d), rhs(batch=%d, cross=%d, contracting=%d)",
			params.lhsBatchSize, params.lhsCrossSize, params.lhsContractingSize,
			params.rhsBatchSize, params.rhsCrossSize, params.rhsContractingSize)
	}

	blkLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	params.lhsBlockedShape = dgCreateBlockedShape(dtype, params.lhsBatchSize, params.lhsCrossSize, params.lhsContractingSize, blkLog2Dim)
	params.rhsBlockedShape = dgCreateBlockedShape(dtype, params.rhsBatchSize, params.rhsCrossSize, params.rhsContractingSize, blkLog2Dim)

	// Create dot-general node: it will generate a normalized output [batchSize, lhsCrossSize, rhsCrossSize].
	dotGeneral := b.newNode(backends.OpTypeDotGeneral, shapes.Make(dtype, params.lhsBatchSize, params.lhsCrossSize, params.rhsCrossSize), lhs, rhs)
	dotGeneral.data = &params

	// Reshape result to recover batch and cross dimensions.
	lhsBatchDims := xslices.Map(lhsBatchAxes, func(axis int) int { return lhs.shape.Dimensions[axis] })
	lhsCrossDims := make([]int, 0, lhs.shape.Rank()-len(lhsContractingAxes)+len(lhsBatchAxes))
	rhsCrossDims := make([]int, 0, rhs.shape.Rank()-len(rhsContractingAxes)+len(rhsBatchAxes))
	resultingDims := make([]int, 0, len(lhsBatchAxes)+len(lhsCrossDims)+len(rhsCrossDims))
	resultingDims = append(resultingDims, lhsBatchDims...)
	resultingDims = append(resultingDims, lhsCrossDims...)
	resultingDims = append(resultingDims, rhsCrossDims...)
	result, err := b.Reshape(dotGeneral, resultingDims...)

	fmt.Printf("DotGeneral(*lhs*: %s, c:%v, b:%v; *rhs*:  %s, c:%v, b:%v) -> %s\n",
		lhs.shape, lhsContractingAxes, lhsBatchAxes, rhs.shape, rhsContractingAxes, rhsBatchAxes,
		result.(*Node).shape)

	if err != nil {
		return nil, err
	}
	return result, nil
}

func dgFindSizes(shape shapes.Shape, contractingAxes, batchAxes []int) (batchSize, crossSize, contractingSize int) {
	rank := shape.Rank()
	axesTypes := make([]int, rank)

	// Mark axes types: 1 for contracting, 2 for batch
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
	}
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
	}

	// Calculate sizes by multiplying dimensions according to axis type
	batchSize, crossSize, contractingSize = 1, 1, 1
	for axis, axisType := range axesTypes {
		dim := shape.Dimensions[axis]
		switch axisType {
		case 0: // Cross axes (unmarked)
			crossSize *= dim
		case 1: // Contracting axes
			contractingSize *= dim
		case 2: // Batch axes
			batchSize *= dim
		}
	}
	return
}

var (
	// DotGeneralTargetBlockSize is hardware-specific, it should be aligned with the L1 cache size.
	// It should be the number per thread, not necessarily the number per core.
	// TODO: find out how to initialize this number in runtime.
	DotGeneralTargetBlockSize = 32 * 1024

	// DotGeneralTargetBlockLog2Dim is set per dtype, such that it is square and fits DotGeneralTargetBlockSize.
	// The block dim is 2^(DotGeneralTargetBlockLog2Dim[dtype]).
	DotGeneralTargetBlockLog2Dim [MaxDTypes]int
)

func init() {
	for _, dtype := range []dtypes.DType{dtypes.F32, dtypes.F64, dtypes.BFloat16} {
		sizePerElem := dtype.Size()
		if dtype == dtypes.BFloat16 {
			sizePerElem = 4 // Because for BF16 we store the results in F32, and only later convert to BF16.
		}
		dim := 1
		log2Dim := 0
		for dim*dim*sizePerElem < DotGeneralTargetBlockSize {
			dim *= 2
			log2Dim++
		}
		dim /= 2
		log2Dim--
		DotGeneralTargetBlockLog2Dim[dtype] = log2Dim
	}
}

// dgCreateBlockedShape returns a shape that is able to split the original shape into blocks, with extra
// padding (zero initialized) to make it fit.
//
// Input shape: [batchSize, crossSize, contractingSize]
// Output shape: [batchSize, crossBlocks * blkDim, contractBlocks * blkDim]
func dgCreateBlockedShape(dtype dtypes.DType, batchSize, crossSize, contractingSize, blkLog2Dim int) shapes.Shape {
	blkDim := 1 << blkLog2Dim
	newCrossDim := (crossSize + blkDim - 1) / blkDim
	newContractDim := (contractingSize + blkDim - 1) / blkDim
	return shapes.Make(dtype, batchSize, newCrossDim, newContractDim, blkDim, blkDim)
}

var dotGeneralFlatToBlockDTypeMap = NewDTypeMap("DotGeneralFlatToBlock")

// dgCopyDataToBlockedShape copies the data from original (with a non-normalized shape, with the contracting axes
// and batch axes given) to blocked, whose shape is normalized to [batchSize, crossSize, contractingSize] and
// is organized in blocks (packages) of shape [1, blkDim, blkDim].
//
// blkOutput is assumed to have been created with a size that is multiple of blkDim for the cross and contracting axes.
//
// source shape: any combination of batch, cross or contracting dimensions.
// blkOutput shape: [batchSize, crossBlocks * blkDim, contractBlocks * blkDim]
func dgCopyFlatToBlockShape[T interface {
	PODNumericConstraints | bfloat16.BFloat16
}](
	source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int) {
	rank := source.shape.Rank()

	// Map source axes to their types (0: cross, 1: contracting, 2: batch)
	axesTypes := make([]int, rank)
	for _, axis := range contractingAxes {
		axesTypes[axis] = 1
	}
	for _, axis := range batchAxes {
		axesTypes[axis] = 2
	}
	sourceDims := source.shape.Dimensions
	batchStride, crossStride, contractStride := 1, 1, 1
	sourceStrides := make([]int, rank)      // Stride is per type of axis.
	sourceRewindAmount := make([]int, rank) // dim-1 * stride.
	for axis := rank - 1; axis >= 0; axis-- {
		switch axesTypes[axis] {
		case 0: // Cross
			sourceStrides[axis] = crossStride
			sourceRewindAmount[axis] = crossStride * (sourceDims[axis] - 1)
			crossStride *= sourceDims[axis]
		case 1: // Contracting
			sourceStrides[axis] = contractStride
			sourceRewindAmount[axis] = contractStride * (sourceDims[axis] - 1)
			contractStride *= sourceDims[axis]
		case 2: // Batch
			sourceStrides[axis] = batchStride
			sourceRewindAmount[axis] = batchStride * (sourceDims[axis] - 1)
			batchStride *= sourceDims[axis]
		}
	}
	sourceIdx := make([]int, rank)

	// Calculate sizes
	blkDim := 1 << blkLog2Dim
	blkMask := blkDim - 1
	crossBlocks := (crossSize + blkDim - 1) / blkDim
	contractBlocks := (contractingSize + blkDim - 1) / blkDim

	// Calculate the virtual output shape as: [batchSize, outerCross, outerContracting, innerCross, innerContracting],
	// where innerCross = innerContracting = blkDim.
	outputDims := [5]int{batchSize, crossBlocks, contractBlocks, blkDim, blkDim}
	outputStrides := [5]int{1, 1, 1, 1, 1}
	for ii := 3; ii >= 0; ii-- {
		outputStrides[ii] = outputStrides[ii+1] * outputDims[ii+1]
	}
	var outputIdx [5]int
	var outputCrossIdx, outputContractIdx int

	// Pre-compute axis counters and limits
	sourceData := source.flat.([]T)
	outputData := blkOutput.flat.([]T)

	// Sequential iteration over source data
	for sourceFlatIdx := range len(sourceData) {
		// Copy over value.
		outputIdx[4] = outputContractIdx & blkMask     // Take only the innerContracting bits.
		outputIdx[2] = outputContractIdx >> blkLog2Dim // Shift innerContracting bits away.
		outputIdx[3] = outputCrossIdx & blkMask
		outputIdx[1] = outputCrossIdx >> blkLog2Dim
		outputFlatIdx := outputIdx[4] +
			outputIdx[3]*outputStrides[3] +
			outputIdx[2]*outputStrides[2] +
			outputIdx[1]*outputStrides[1] +
			outputIdx[0]*outputStrides[0]
		outputData[outputFlatIdx] = sourceData[sourceFlatIdx]
		//fmt.Printf("\toutput%v (flat %d) = source%v (flat %d)\n", outputIdx, outputFlatIdx, sourceIdx, sourceFlatIdx)

		// Increment position.
		for axis := rank - 1; axis >= 0; axis-- {
			if sourceDims[axis] == 1 {
				continue
			}

			sourceIdx[axis]++
			if sourceIdx[axis] < sourceDims[axis] {
				// Not reached end of this axis.
				switch axesTypes[axis] {
				case 0: // Cross
					outputCrossIdx += sourceStrides[axis]
				case 1: // Contracting
					outputContractIdx += sourceStrides[axis]
				case 2: // Batch
					outputIdx[0] += sourceStrides[axis]
				}
				break
			}

			// Reached the end of this axis, rewind the index to 0: both in sourceIdx and the corresponding output index.
			sourceIdx[axis] = 0
			switch axesTypes[axis] {
			case 0: // Cross
				outputCrossIdx -= sourceRewindAmount[axis]
			case 1: // Contracting
				outputContractIdx -= sourceRewindAmount[axis]
			case 2: // Batch
				outputIdx[0] -= sourceRewindAmount[axis]
			}
		}
	}
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

// execDotGeneral executes the DotGeneral by first normalizing and repackaging the tensors into blocks.
func execDotGeneral(backend *Backend, node *Node, inputs []*Buffer, _ []bool) (*Buffer, error) {
	lhs, rhs := inputs[0], inputs[1]
	outputShape := node.shape
	dtype := lhs.shape.DType
	params := node.data.(*dotGeneralNodeData)
	copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))

	// Get block buffers.
	blkLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	lhsBlocks := backend.getBuffer(dtype, params.lhsBlockedShape.Size())
	lhsBlocks.shape = params.lhsBlockedShape
	lhsBlocks.Zeros()
	copyFlatToBlock(lhs, lhsBlocks, params.lhsContractingAxes, params.lhsBatchAxes,
		params.lhsBatchSize, params.lhsCrossSize, params.lhsContractingSize, blkLog2Dim)

	rhsBlocks := backend.getBuffer(dtype, params.rhsBlockedShape.Size())
	rhsBlocks.shape = params.rhsBlockedShape
	rhsBlocks.Zeros()
	copyFlatToBlock(rhs, rhsBlocks, params.rhsContractingAxes, params.rhsBatchAxes,
		params.rhsBatchSize, params.rhsCrossSize, params.rhsContractingSize, blkLog2Dim)

	outputBlocks := backend.getBuffer(dtype, params.outputBlockedShape.Size())
	outputBlocks.shape = params.outputBlockedShape
	outputBlocks.Zeros()

	fn := dotGeneralDTypeMap.Get(dtype).(func(lhs, rhs, output *Buffer))
	fn(lhsBlocks, rhsBlocks, outputBlocks)

	// Copy over outputBlocks to the normal output.
	output := backend.getBuffer(dtype, outputShape.Size())
	output.shape = outputShape
	return output, nil
}

var dotGeneralDTypeMap = NewDTypeMap("DotGeneral")

//go:generate go run ../../internal/cmd/simplego_dispatcher -dispatcher=dispatchDotGeneral -generic=execNormalizedDotGeneralGeneric -int -uint -float

// execNormalizedDotGeneralGeneric operands lhs and rhs are normalized to shape
// [batchSize, crossSize, contractingSize].
func execNormalizedDotGeneralGeneric[T PODNumericConstraints](lhs, rhs, output *Buffer) {
	//fmt.Printf("execNormalizedDotGeneralGeneric(%s, %s) -> %s\n", lhs.shape, rhs.shape, output.shape)
	lhsFlat := lhs.flat.([]T)
	rhsFlat := rhs.flat.([]T)
	outputFlat := output.flat.([]T)

	batchSize := lhs.shape.Dimensions[0]
	lhsCrossSize := lhs.shape.Dimensions[1]
	rhsCrossSize := rhs.shape.Dimensions[1]
	contractingSize := lhs.shape.Dimensions[2]

	// Pre-compute strides to avoid repeated calculations
	lhsBatchStride := lhsCrossSize * contractingSize
	rhsBatchStride := rhsCrossSize * contractingSize
	outputBatchStride := lhsCrossSize * rhsCrossSize

	// Cache block sizes - adjust based on typical matrix sizes and CPU cache
	const blockSize = 64 // Tune this based on your typical workload and L1 cache size

	for batchIdx := 0; batchIdx < batchSize; batchIdx++ {
		lhsBaseIdx := batchIdx * lhsBatchStride
		rhsBaseIdx := batchIdx * rhsBatchStride
		outputBaseIdx := batchIdx * outputBatchStride

		// Use blocking to improve cache locality
		for idxLhsCross := 0; idxLhsCross < lhsCrossSize; idxLhsCross += blockSize {
			iEnd := min(idxLhsCross+blockSize, lhsCrossSize)

			for idxRhsCross := 0; idxRhsCross < rhsCrossSize; idxRhsCross += blockSize {
				jEnd := min(idxRhsCross+blockSize, rhsCrossSize)

				for idxContracting := 0; idxContracting < contractingSize; idxContracting += blockSize {
					kEnd := min(idxContracting+blockSize, contractingSize)

					// Process the current block
					for i := idxLhsCross; i < iEnd; i++ {
						lhsRowStartIdx := lhsBaseIdx + i*contractingSize
						outputRowStartIdx := outputBaseIdx + i*rhsCrossSize

						for j := idxRhsCross; j < jEnd; j++ {
							rhsColStartIdx := rhsBaseIdx + j*contractingSize
							sum := outputFlat[outputRowStartIdx+j]

							// Unroll the innermost loop for better vectorization
							k := idxContracting
							for ; k+7 < kEnd; k += 8 {
								sum += lhsFlat[lhsRowStartIdx+k]*rhsFlat[rhsColStartIdx+k] +
									lhsFlat[lhsRowStartIdx+k+1]*rhsFlat[rhsColStartIdx+k+1] +
									lhsFlat[lhsRowStartIdx+k+2]*rhsFlat[rhsColStartIdx+k+2] +
									lhsFlat[lhsRowStartIdx+k+3]*rhsFlat[rhsColStartIdx+k+3] +
									lhsFlat[lhsRowStartIdx+k+4]*rhsFlat[rhsColStartIdx+k+4] +
									lhsFlat[lhsRowStartIdx+k+5]*rhsFlat[rhsColStartIdx+k+5] +
									lhsFlat[lhsRowStartIdx+k+6]*rhsFlat[rhsColStartIdx+k+6] +
									lhsFlat[lhsRowStartIdx+k+7]*rhsFlat[rhsColStartIdx+k+7]
							}
							//for ; k+3 < kEnd; k += 4 {
							//	sum += lhsFlat[lhsRowStartIdx+k]*rhsFlat[rhsColStartIdx+k] +
							//		lhsFlat[lhsRowStartIdx+k+1]*rhsFlat[rhsColStartIdx+k+1] +
							//		lhsFlat[lhsRowStartIdx+k+2]*rhsFlat[rhsColStartIdx+k+2] +
							//		lhsFlat[lhsRowStartIdx+k+3]*rhsFlat[rhsColStartIdx+k+3]
							//}

							// Handle remaining elements
							for ; k < kEnd; k++ {
								sum += lhsFlat[lhsRowStartIdx+k] * rhsFlat[rhsColStartIdx+k]
							}

							outputFlat[outputRowStartIdx+j] = sum
						}
					}
				}
			}
		}
	}
}

// Register the BFloat16 version of DotGeneral.
func init() { dotGeneralDTypeMap.Register(dtypes.BFloat16, execNormalizedDotGeneralBFloat16) }

func execNormalizedDotGeneralBFloat16(lhs, rhs, output *Buffer) {
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
	return
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
