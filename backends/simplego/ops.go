package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// nodeParameter data.
type nodeParameter struct {
	name     string
	inputIdx int
}

// Parameter creates an input parameter for the computation.
// During execution of the computation this value will need to be fed, in the same order it is created.
func (b *Builder) Parameter(name string, shape shapes.Shape) backends.Op {
	dtype := shape.DType
	if dtype == dtypes.InvalidDType {
		exceptions.Panicf("invalid shape %s for Parameter", shape)
	}
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		exceptions.Panicf("Parameter: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, b.backend.Name())
	}
	n := b.newNode(backends.OpTypeParameter, shape)
	n.data = &nodeParameter{
		name:     name,
		inputIdx: len(b.inputs),
	}
	b.inputs = append(b.inputs, n)
	return n
}

// Constant creates a constant in the graph with the given flat values, and the shape defined by dims.
//
// flat must be a slice of a basic type supported -- that can be converted to a DType.
//
// The value is copied into the graph. It's recommended that for very large tensors,
// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
func (b *Builder) Constant(flat any, dims ...int) backends.Op {
	_ = b.checkOps("Constant")
	dtype, flatLen := checkFlat(flat)
	if supported, ok := Capabilities.DTypes[dtype]; !ok || !supported {
		exceptions.Panicf("Constant: data type (DType) %s not supported for backend %q, try using "+
			"a different backend, or open an issue in github.com/gomlx/gomlx", dtype, b.backend.Name())
	}
	shape := shapes.Make(dtype, dims...)
	if shape.Size() != flatLen {
		exceptions.Panicf("flat ([%d]%s) and shape size (%d) mismatch for constant value",
			flatLen, dtype, shape.Size())
	}
	n := b.newNode(backends.OpTypeConstant, shape)
	n.data = &Buffer{
		shape: shape,
		flat:  flat,
	}
	return n
}

// Unary Operations:

// Neg implements backends.Builder interface.
func (b *Builder) Neg(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeNeg, operand)
}

// Sign implements backends.Builder interface.
func (b *Builder) Sign(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeSign, operand)
}

// Abs implements backends.Builder interface.
func (b *Builder) Abs(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeAbs, operand)
}

// LogicalNot implements backends.Builder interface.
func (b *Builder) LogicalNot(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeLogicalNot, operand)
}

// BitwiseNot implements backends.Builder interface.
func (b *Builder) BitwiseNot(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeBitwiseNot, operand)
}

// BitCount implements backends.Builder interface.
func (b *Builder) BitCount(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeBitCount, operand)
}

// Clz implements backends.Builder interface.
func (b *Builder) Clz(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeClz, operand)
}

// Exp implements backends.Builder interface.
func (b *Builder) Exp(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeExp, operand)
}

// Log implements backends.Builder interface.
func (b *Builder) Log(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeLog, operand)
}

// Log1p implements backends.Builder interface.
func (b *Builder) Log1p(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeLog1p, operand)
}

// Ceil implements backends.Builder interface.
func (b *Builder) Ceil(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeCeil, operand)
}

// Floor implements backends.Builder interface.
func (b *Builder) Floor(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeFloor, operand)
}

// Round implements backends.Builder interface.
func (b *Builder) Round(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeRound, operand)
}

// Rsqrt implements backends.Builder interface.
func (b *Builder) Rsqrt(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeRsqrt, operand)
}

// Sqrt implements backends.Builder interface.
func (b *Builder) Sqrt(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeSqrt, operand)
}

// Cos implements backends.Builder interface.
func (b *Builder) Cos(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeCos, operand)
}

// Sin implements backends.Builder interface.
func (b *Builder) Sin(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeSin, operand)
}

// Tanh implements backends.Builder interface.
func (b *Builder) Tanh(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeTanh, operand)
}

// IsFinite implements backends.Builder interface.
func (b *Builder) IsFinite(operandOp backends.Op) backends.Op {
	opType := backends.OpTypeIsFinite
	inputs := b.checkOps(opType.String(), operandOp)
	operand := inputs[0]
	dtype := operand.shape.DType
	if !dtype.IsFloat() && !dtype.IsComplex() {
		exceptions.Panicf("the operation IsFinite is only defined for float types (%s), cannot use it", operand.shape.DType)
	}

	// Output will have the same shape but for the dtype that is bool.
	shape := operand.shape.Clone()
	shape.DType = dtypes.Bool
	return b.newNode(opType, shape, operand)
}

// Binary Operations:

// Add implements backends.Builder interface.
func (b *Builder) Add(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeAdd, lhsOp, rhsOp)
}

// Mul implements backends.Builder interface.
func (b *Builder) Mul(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeMul, lhsOp, rhsOp)
}

// Sub implements backends.Builder interface.
func (b *Builder) Sub(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeSub, lhsOp, rhsOp)
}

// Div implements backends.Builder interface.
func (b *Builder) Div(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeDiv, lhsOp, rhsOp)
}

// Rem implements backends.Builder interface.
func (b *Builder) Rem(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeRem, lhsOp, rhsOp)
}
