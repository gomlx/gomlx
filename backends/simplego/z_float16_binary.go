package simplego

// This file must come alphabetically after gen_exec_binary.go to ensure its init() runs later.

import (
	"math"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/x448/float16"
)

// Float16 binary operations

func execBinaryFloat16[OpFn func(a, b float32) float32](opFn OpFn, lhs, rhs []float16.Float16, output []float16.Float16,
	lhsShape, rhsShape, outputShape shapes.Shape) {
	if len(rhs) == 1 {
		// One side (rhs) is a scalar: only iterate over the lhs.
		c := rhs[0].Float32()
		for ii, input := range lhs {
			a := input.Float32()
			output[ii] = float16.Fromfloat32(opFn(a, c))
		}
		return
	} else if lhsShape.Equal(rhsShape) {
		// Case 2: Exact same shapes, no broadcasting.
		for outputIdx := range output {
			a := lhs[outputIdx].Float32()
			b := rhs[outputIdx].Float32()
			output[outputIdx] = float16.Fromfloat32(opFn(a, b))
		}
		return
	} else {
		// Case 3: with broadcasting non-scalar tensors:
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for outputIdx := range output {
			lhsIdx := lhsIter.Next()
			rhsIdx := rhsIter.Next()
			a := lhs[lhsIdx].Float32()
			b := rhs[rhsIdx].Float32()
			output[outputIdx] = float16.Fromfloat32(opFn(a, b))
		}
	}
}

func execCompareFloat16[OpFn func(a, b float32) bool](opFn OpFn, lhs, rhs []float16.Float16, output []bool,
	lhsShape, rhsShape, outputShape shapes.Shape) {
	if len(rhs) == 1 {
		c := rhs[0].Float32()
		for ii, input := range lhs {
			a := input.Float32()
			output[ii] = opFn(a, c)
		}
		return
	} else if lhsShape.Equal(rhsShape) {
		for outputIdx := range output {
			a := lhs[outputIdx].Float32()
			b := rhs[outputIdx].Float32()
			output[outputIdx] = opFn(a, b)
		}
		return
	} else {
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for outputIdx := range output {
			lhsIdx := lhsIter.Next()
			rhsIdx := rhsIter.Next()
			a := lhs[lhsIdx].Float32()
			b := rhs[rhsIdx].Float32()
			output[outputIdx] = opFn(a, b)
		}
	}
}

// Store original executors before wrapping
var (
	origExecAdd            func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecSub            func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecMul            func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecDiv            func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecMax            func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecMin            func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecPow            func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecEqual          func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecNotEqual       func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecGreaterOrEqual func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecGreaterThan    func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecLessOrEqual    func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
	origExecLessThan       func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error)
)

func makeFloat16BinaryWrapper(
	origExec func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error),
	opFn func(a, b float32) float32,
) func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error) {
	return func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
		if inputs[0].shape.DType != dtypes.Float16 {
			return origExec(backend, node, inputs, inputsOwned)
		}
		lhs, rhs, output, lhsIsScalarOr1, rhsIsScalarOr1 := binaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
		if lhsIsScalarOr1 && !rhsIsScalarOr1 {
			lhs, rhs = rhs, lhs
		}
		execBinaryFloat16(opFn, lhs.flat.([]float16.Float16), rhs.flat.([]float16.Float16),
			output.flat.([]float16.Float16), lhs.shape, rhs.shape, output.shape)
		return output, nil
	}
}

func makeFloat16CompareWrapper(
	origExec func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error),
	opFn func(a, b float32) bool,
) func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error) {
	return func(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {
		if inputs[0].shape.DType != dtypes.Float16 {
			return origExec(backend, node, inputs, inputsOwned)
		}
		lhs, rhs := inputs[0], inputs[1]
		lhsIsScalarOr1, rhsIsScalarOr1 := lhs.shape.Size() == 1, rhs.shape.Size() == 1
		output := backend.getBuffer(node.shape.DType, node.shape.Size())
		output.shape = node.shape
		if lhsIsScalarOr1 && !rhsIsScalarOr1 {
			lhs, rhs = rhs, lhs
		}
		execCompareFloat16(opFn, lhs.flat.([]float16.Float16), rhs.flat.([]float16.Float16),
			output.flat.([]bool), lhs.shape, rhs.shape, output.shape)
		return output, nil
	}
}

// Init must run after gen_exec_binary.go's init()
func init() {
	// Save original executors
	origExecAdd = nodeExecutors[backends.OpTypeAdd]
	origExecSub = nodeExecutors[backends.OpTypeSub]
	origExecMul = nodeExecutors[backends.OpTypeMul]
	origExecDiv = nodeExecutors[backends.OpTypeDiv]
	origExecMax = nodeExecutors[backends.OpTypeMax]
	origExecMin = nodeExecutors[backends.OpTypeMin]
	origExecPow = nodeExecutors[backends.OpTypePow]
	origExecEqual = nodeExecutors[backends.OpTypeEqual]
	origExecNotEqual = nodeExecutors[backends.OpTypeNotEqual]
	origExecGreaterOrEqual = nodeExecutors[backends.OpTypeGreaterOrEqual]
	origExecGreaterThan = nodeExecutors[backends.OpTypeGreaterThan]
	origExecLessOrEqual = nodeExecutors[backends.OpTypeLessOrEqual]
	origExecLessThan = nodeExecutors[backends.OpTypeLessThan]

	// Wrap with Float16 support
	nodeExecutors[backends.OpTypeAdd] = makeFloat16BinaryWrapper(origExecAdd, func(a, b float32) float32 { return a + b })
	nodeExecutors[backends.OpTypeSub] = makeFloat16BinaryWrapper(origExecSub, func(a, b float32) float32 { return a - b })
	nodeExecutors[backends.OpTypeMul] = makeFloat16BinaryWrapper(origExecMul, func(a, b float32) float32 { return a * b })
	nodeExecutors[backends.OpTypeDiv] = makeFloat16BinaryWrapper(origExecDiv, func(a, b float32) float32 { return a / b })
	nodeExecutors[backends.OpTypeMax] = makeFloat16BinaryWrapper(origExecMax, func(a, b float32) float32 {
		if a > b {
			return a
		}
		return b
	})
	nodeExecutors[backends.OpTypeMin] = makeFloat16BinaryWrapper(origExecMin, func(a, b float32) float32 {
		if a < b {
			return a
		}
		return b
	})
	nodeExecutors[backends.OpTypePow] = makeFloat16BinaryWrapper(origExecPow, func(a, b float32) float32 {
		return float32(math.Pow(float64(a), float64(b)))
	})
	nodeExecutors[backends.OpTypeEqual] = makeFloat16CompareWrapper(origExecEqual, func(a, b float32) bool { return a == b })
	nodeExecutors[backends.OpTypeNotEqual] = makeFloat16CompareWrapper(origExecNotEqual, func(a, b float32) bool { return a != b })
	nodeExecutors[backends.OpTypeGreaterOrEqual] = makeFloat16CompareWrapper(origExecGreaterOrEqual, func(a, b float32) bool { return a >= b })
	nodeExecutors[backends.OpTypeGreaterThan] = makeFloat16CompareWrapper(origExecGreaterThan, func(a, b float32) bool { return a > b })
	nodeExecutors[backends.OpTypeLessOrEqual] = makeFloat16CompareWrapper(origExecLessOrEqual, func(a, b float32) bool { return a <= b })
	nodeExecutors[backends.OpTypeLessThan] = makeFloat16CompareWrapper(origExecLessThan, func(a, b float32) bool { return a < b })
}
