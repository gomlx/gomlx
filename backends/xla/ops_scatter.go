package xla

import (
	"github.com/gomlx/go-xla/pkg/stablehlo"
	xla_dtypes "github.com/gomlx/go-xla/pkg/types/dtypes"
	xla_shapes "github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gomlx/backends"
)

func scatterOpToReduceOp(opType backends.OpType) backends.OpType {
	switch opType {
	case backends.OpTypeScatterMax:
		return backends.OpTypeReduceMax
	case backends.OpTypeScatterMin:
		return backends.OpTypeReduceMin
	case backends.OpTypeScatterSum:
		return backends.OpTypeReduceSum
	default:
		return -1
	}
}

func (b *Builder) scatter(opType backends.OpType,
	operandOp, scatterIndicesOp, updatesOp backends.Op, indexVectorAxis int, updateWindowAxes,
	insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues(opType.String(), operandOp, scatterIndicesOp, updatesOp)
	if err != nil {
		return nil, err
	}
	operand, scatterIndices, updates := nodes[0], nodes[1], nodes[2]
	dtype := operand.shape.DType

	// Handle bounds mismatch between scatter indices and updates.
	// StableHLO requires that scatter dimensions have matching bounds.
	// If indices have smaller bounds than updates, broadcast indices to match.
	scatterIndicesValue := scatterIndices.value
	scatterIndicesValue, err = b.ensureScatterBoundsMatch(scatterIndicesValue, updates.value, indexVectorAxis)
	if err != nil {
		return nil, err
	}

	// Get update closure: it's the same as a reduction of the corresponding type.
	reductionFn, err := b.getReductionFn(dtype, scatterOpToReduceOp(opType))
	if err != nil {
		return nil, err
	}

	// Batching axes are left empty, since in GoMLX we don't support them.
	var inputBatchingAxes, scatterIndicesBatchingAxes []int

	value, err := stablehlo.Scatter(
		operand.value, scatterIndicesValue, updates.value,
		updateWindowAxes, insertedWindowAxes,
		inputBatchingAxes, scatterIndicesBatchingAxes,
		scatterAxesToOperandAxes, indexVectorAxis,
		indicesAreSorted, uniqueIndices,
		reductionFn)
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}

// ensureScatterBoundsMatch checks if scatter indices and updates have compatible bounds
// on the scatter dimensions. If indices have smaller bounds, broadcasts indices to match.
// This is required because StableHLO scatter requires matching bounds.
func (b *Builder) ensureScatterBoundsMatch(scatterIndices, updates *stablehlo.Value, indexVectorAxis int) (*stablehlo.Value, error) {
	indicesShape := scatterIndices.Shape()
	updatesShape := updates.Shape()

	// Determine the number of scatter dimensions (all except the index vector axis)
	indicesRank := indicesShape.Rank()
	updatesRank := updatesShape.Rank()

	// For scatter, the first (indicesRank - 1) dimensions of indices should match
	// the first (indicesRank - 1) dimensions of updates
	numScatterDims := indicesRank - 1
	if indexVectorAxis != indicesRank-1 {
		// If index vector is not the last axis, the calculation is different
		// For now, only handle the common case where index vector is last
		return scatterIndices, nil
	}

	if numScatterDims > updatesRank {
		return scatterIndices, nil
	}

	// Check if we need to broadcast
	needsBroadcast := false
	targetDims := make([]int, indicesRank)
	targetBounds := make([]int, indicesRank)

	for i := 0; i < indicesRank; i++ {
		targetDims[i] = indicesShape.Dimensions[i]
		// DimensionBounds removed - just use dimensions
		targetBounds[i] = indicesShape.Dimensions[i]
	}

	for i := 0; i < numScatterDims; i++ {
		indicesDim := indicesShape.Dimensions[i]
		updatesDim := updatesShape.Dimensions[i]

		// Get bounds - DimensionBounds removed, use dimensions directly
		indicesBound := indicesDim
		updatesBound := updatesDim

		// Check for mismatch
		if indicesDim < 0 && updatesDim < 0 {
			// Both dynamic - check if bounds differ
			if indicesBound > 0 && updatesBound > 0 && indicesBound != updatesBound {
				// Bounds mismatch - need to broadcast indices to match updates
				needsBroadcast = true
				targetBounds[i] = updatesBound
			} else if indicesBound == 0 && updatesBound > 0 {
				// Indices has no bound but updates does
				needsBroadcast = true
				targetBounds[i] = updatesBound
			}
		} else if indicesDim == 1 && updatesDim > 0 && updatesDim != 1 {
			// Concrete dimension mismatch - broadcast from 1 to updatesDim
			needsBroadcast = true
			targetDims[i] = updatesDim
			targetBounds[i] = updatesDim
		} else if indicesDim < 0 && updatesDim > 0 {
			// Indices is dynamic, updates is concrete
			// Set bounds to match updates
			if indicesBound == 0 || indicesBound != updatesDim {
				needsBroadcast = true
				targetBounds[i] = updatesDim
			}
		}
	}

	if !needsBroadcast {
		return scatterIndices, nil
	}

	// Build shape tensor for broadcast
	fn := b.fn // Use the builder's function for creating constants
	shapeParts := make([]*stablehlo.Value, indicesRank)
	for i := 0; i < indicesRank; i++ {
		var part *stablehlo.Value
		var err error
		if targetDims[i] < 0 {
			// Dynamic dimension - use GetDimensionSize from updates if it's a scatter dimension
			if i < numScatterDims && updatesShape.Dimensions[i] >= 0 {
				part, err = fn.ConstantFromScalar(int32(updatesShape.Dimensions[i]))
			} else if i < numScatterDims {
				part, err = stablehlo.GetDimensionSize(updates, i)
			} else {
				part, err = stablehlo.GetDimensionSize(scatterIndices, i)
			}
		} else {
			part, err = fn.ConstantFromScalar(int32(targetDims[i]))
		}
		if err != nil {
			return nil, err
		}
		// Reshape scalar to 1D tensor for concatenation
		if part.Shape().Rank() == 0 {
			targetShape := xla_shapes.Make(part.Shape().DType, 1)
			part, err = stablehlo.Reshape(part, targetShape)
			if err != nil {
				return nil, err
			}
		}
		shapeParts[i] = part
	}

	// Concatenate shape parts
	shapeTensor, err := stablehlo.Concatenate(0, shapeParts...)
	if err != nil {
		return nil, err
	}

	// Convert to int64 if needed
	if shapeTensor.Shape().DType != xla_dtypes.Int64 {
		shapeTensor, err = stablehlo.Convert(shapeTensor, xla_dtypes.Int64)
		if err != nil {
			return nil, err
		}
	}

	// Build broadcast dimensions (identity mapping)
	broadcastDims := make([]int, indicesRank)
	for i := 0; i < indicesRank; i++ {
		broadcastDims[i] = i
	}

	// XLA cannot translate dynamic_broadcast_in_dim to XLA HLO.
	// Use static BroadcastInDim to the target bounds instead.
	targetShape := xla_shapes.Make(scatterIndices.Shape().DType, targetBounds...)
	return stablehlo.BroadcastInDim(scatterIndices, targetShape, broadcastDims)
}

// ScatterMax scatter values from updates pointed by scatterIndices to operand, by taking the Max.
func (b *Builder) ScatterMax(operand, scatterIndices, updates backends.Op, indexVectorAxis int,
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (backends.Op, error) {
	return b.scatter(backends.OpTypeScatterMax,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}

// ScatterMin scatter values from updates pointed by scatterIndices to operand, by taking the Min.
func (b *Builder) ScatterMin(operand, scatterIndices, updates backends.Op, indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (backends.Op, error) {
	return b.scatter(backends.OpTypeScatterMin,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}

// ScatterSum values from updates pointed by scatterIndices to operand.
func (b *Builder) ScatterSum(operand, scatterIndices, updates backends.Op, indexVectorAxis int, updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int, indicesAreSorted, uniqueIndices bool) (backends.Op, error) {
	return b.scatter(backends.OpTypeScatterSum,
		operand, scatterIndices, updates, indexVectorAxis,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes, indicesAreSorted, uniqueIndices)
}
