package stablehlo

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo"
	"github.com/pkg/errors"
)

// AllReduce implements the collective AllReduce operation.
func (b *Builder) AllReduce(operands []backends.Op, reductionType backends.ReduceOpType,
	replicaGroups [][]int) ([]backends.Op, error) {
	nodes, err := b.verifyAndCastValues("stablehlo.AllReduce", operands...)
	if err != nil {
		return nil, err
	}

	// StableHLO/PJRT only allow AllReduce with operands of the same dtype.
	// So we need to split the operands by dtype and later re-merge them.
	// Also, the reduceFn will be per operand.
	operandsPerDType, indicesPerDType := splitOperandsByDType(nodes)
	outputs := make([]backends.Op, len(operands))
	for dtype, operandsDType := range operandsPerDType {
		opType, err := b.getReductionOp(reductionType)
		if err != nil {
			return nil, errors.WithMessage(err, "while building reduction function for AllReduce")
		}
		reduceFn, err := b.getReductionFn(dtype, opType)
		if err != nil {
			return nil, errors.WithMessage(err, "while building reduction function for AllReduce")
		}
		values, err := stablehlo.AllReduce(
			xslices.Map(operandsDType, func(node *Node) *stablehlo.Value { return node.value }),
			replicaGroups, reduceFn)
		if err != nil {
			return nil, err
		}
		outputsPerDType := xslices.Map(values, func(v *stablehlo.Value) backends.Op {
			return b.newNode(v)
		})

		// Place the "per dtype" outputs back into the original order.
		targetIndices := indicesPerDType[dtype]
		for i, output := range outputsPerDType {
			outputs[targetIndices[i]] = output
		}
	}
	return outputs, nil
}

// splitOperandsByDType splits the operands by dtype and returns a mapping of dtype to operands and their indices,
// so later the order can be reconstructed.
func splitOperandsByDType(operands []*Node) (
	operandsPerDType map[dtypes.DType][]*Node, indicesPerDType map[dtypes.DType][]int) {
	operandsPerDType = make(map[dtypes.DType][]*Node)
	indicesPerDType = make(map[dtypes.DType][]int)
	for i, operand := range operands {
		dtype := operand.shape.DType
		operandsPerDType[dtype] = append(operandsPerDType[dtype], operand)
		indicesPerDType[dtype] = append(indicesPerDType[dtype], i)
	}
	return operandsPerDType, indicesPerDType
}
