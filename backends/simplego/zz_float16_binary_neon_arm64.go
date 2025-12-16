//go:build !noasm && arm64

package simplego

// This file must come alphabetically after z_float16_binary.go to ensure its init() runs later.
// It overrides the scalar FP16 binary operations with NEON-accelerated versions.

import (
	"math"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/x448/float16"
)

// makeFloat16BinaryWrapperNEON creates a wrapper that uses NEON-accelerated FP16 operations.
// opType: 0=add, 1=mul, 2=sub, 3=div, -1=use fallback
func makeFloat16BinaryWrapperNEON(
	origExec func(*Backend, *Node, []*Buffer, []bool) (*Buffer, error),
	opType int,
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

		lhsFlat := lhs.flat.([]float16.Float16)
		rhsFlat := rhs.flat.([]float16.Float16)
		outFlat := output.flat.([]float16.Float16)

		// Use NEON-accelerated version if opType is supported
		if opType >= 0 && opType <= 3 {
			execBinaryFloat16NEON(opType, lhsFlat, rhsFlat, outFlat, lhs.shape, rhs.shape, output.shape)
		} else {
			// Fallback to scalar for unsupported ops (Max, Min, Pow)
			execBinaryFloat16Scalar(opFn, lhsFlat, rhsFlat, outFlat, lhs.shape, rhs.shape, output.shape)
		}
		return output, nil
	}
}

// execBinaryFloat16Scalar is the scalar fallback for operations not yet NEON-optimized
func execBinaryFloat16Scalar(opFn func(a, b float32) float32, lhs, rhs []float16.Float16, output []float16.Float16,
	lhsShape, rhsShape, outputShape shapes.Shape) {
	if len(rhs) == 1 {
		c := rhs[0].Float32()
		for ii, input := range lhs {
			a := input.Float32()
			output[ii] = float16.Fromfloat32(opFn(a, c))
		}
		return
	} else if lhsShape.Equal(rhsShape) {
		for outputIdx := range output {
			a := lhs[outputIdx].Float32()
			b := rhs[outputIdx].Float32()
			output[outputIdx] = float16.Fromfloat32(opFn(a, b))
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
			output[outputIdx] = float16.Fromfloat32(opFn(a, b))
		}
	}
}

func init() {
	// Override the Float16 binary ops with NEON-accelerated versions
	// opType: 0=add, 1=mul, 2=sub, 3=div

	nodeExecutors[backends.OpTypeAdd] = makeFloat16BinaryWrapperNEON(origExecAdd, 0, func(a, b float32) float32 { return a + b })
	nodeExecutors[backends.OpTypeSub] = makeFloat16BinaryWrapperNEON(origExecSub, 2, func(a, b float32) float32 { return a - b })
	nodeExecutors[backends.OpTypeMul] = makeFloat16BinaryWrapperNEON(origExecMul, 1, func(a, b float32) float32 { return a * b })
	nodeExecutors[backends.OpTypeDiv] = makeFloat16BinaryWrapperNEON(origExecDiv, 3, func(a, b float32) float32 { return a / b })

	// Max, Min, Pow still use scalar fallback (opType=-1)
	nodeExecutors[backends.OpTypeMax] = makeFloat16BinaryWrapperNEON(origExecMax, -1, func(a, b float32) float32 {
		if a > b {
			return a
		}
		return b
	})
	nodeExecutors[backends.OpTypeMin] = makeFloat16BinaryWrapperNEON(origExecMin, -1, func(a, b float32) float32 {
		if a < b {
			return a
		}
		return b
	})
	nodeExecutors[backends.OpTypePow] = makeFloat16BinaryWrapperNEON(origExecPow, -1, func(a, b float32) float32 {
		return float32(math.Pow(float64(a), float64(b)))
	})
}
