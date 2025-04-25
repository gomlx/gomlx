package simplego

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xslices"
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

// Iota creates a constant of the given shape with increasing numbers (starting from 0)
// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
// returns [[0 0][1 1]].
func (b *Builder) Iota(shape shapes.Shape, iotaAxis int) backends.Op {
	_ = b.checkOps("Iota")
	if shape.Rank() == 0 {
		exceptions.Panicf("Iota: shape must have at least one dimension")
	}
	if iotaAxis < 0 || iotaAxis >= shape.Rank() {
		exceptions.Panicf("Iota: iotaAxis (%d) must be in the range [0,%d)", iotaAxis, shape.Rank()-1)
	}
	node := b.newNode(backends.OpTypeIota, shape)
	node.data = iotaAxis
	return node
}

// Identity implements backends.Identity interface.
func (b *Builder) Identity(operandOp backends.Op) backends.Op {
	operand := b.checkOps("Reshape", operandOp)[0]
	return b.newNode(backends.OpTypeIdentity, operand.shape, operand)
}

// Where implements backends.Builder interface.
func (b *Builder) Where(conditionOp, onTrueOp, onFalseOp backends.Op) backends.Op {
	inputs := b.checkOps("Where", conditionOp, onTrueOp, onFalseOp)
	condition, onTrue, onFalse := inputs[0], inputs[1], inputs[2]
	outputShape := shapeinference.WhereOp(condition.shape, onTrue.shape, onFalse.shape)
	return b.newNode(backends.OpTypeWhere, outputShape, condition, onTrue, onFalse)
}

// Reshape implements backends.Builder interface.
//
// Notice the backends.Reshape doesn't support auto-scaling dimensions (set to -1), as graph.Reshape does.
func (b *Builder) Reshape(operandOp backends.Op, dims ...int) backends.Op {
	operand := b.checkOps("Reshape", operandOp)[0]
	outputShape := shapeinference.ReshapeOp(operand.shape, dims)
	return b.newNode(backends.OpTypeReshape, outputShape, operand)
}

// Transpose axes of x.
// There must be one value in permutations for each axis in the operand.
// The output will have: output.Shape.Dimension[ii] = operand.Shape.Dimension[permutations[i]].
func (b *Builder) Transpose(operandOp backends.Op, permutations ...int) backends.Op {
	operand := b.checkOps("Transpose", operandOp)[0]
	outputShape := shapeinference.TransposeOp(operand.shape, permutations)
	node := b.newNode(backends.OpTypeTranspose, outputShape, operand)
	node.data = permutations
	return node
}

// Broadcast prefixes dimensions to an array by duplicating the data in the array.
// See BroadcastInDim for a broadcast in between the axes.
// The new dimensions dims are inserted on the left, i.e., if
// prefixDims has values `{a0, ..., aN}` and the operand shape
// has dimensions {b0, ..., bM} then the shape of the output has
// dimensions {a0, ..., aN, b0, ..., bM}.
// The new dimensions id into copies of the operand, i.e.
//
//	output[i0, ..., iN, j0, ..., jM] = operand[j0, ..., jM]
func (b *Builder) Broadcast(operandOp backends.Op, prefixDims ...int) backends.Op {
	operand := b.checkOps("Transpose", operandOp)[0]
	outputShape := shapeinference.BroadcastOp(operand.shape, prefixDims)
	node := b.newNode(backends.OpTypeBroadcast, outputShape, operand)
	node.data = prefixDims
	return node
}

type broadcastInDimNode struct {
	outputShape   shapes.Shape
	broadcastAxes []int
}

// BroadcastInDim broadcasts x to an output with the given shape.
//
//   - outputShape will be the new shape after x is broadcast.
//   - broadcastAxes maps x-axes to the corresponding outputShape axes (len(broadcastAxes) == x.Shape.Rank()),
//     the i-th axis of x is mapped to the broadcastAxes[i]-th dimension of the output.
//     broadcastAxes must be also increasing: this operation cannot be used to transpose axes,
//     it will only broadcast and introduce new axes in-between.
//     -
//
// This also requires that the i-th input axis is either 1 or is the same as the
// output dimension it's broadcasting into.
// For example, say operand `x = (s32)[2]{1, 2}`; outputShape = `(s32)[2,2]`:
//   - Specifying []int{1} as broadcastAxes will generate output
//     {{1, 2},
//     {1, 2}}
//   - On the other hand, specifying []int{0} as broadcastAxes
//     will generate output
//     {{1 , 1},
//     {2 , 2}}
func (b *Builder) BroadcastInDim(operandOp backends.Op, outputShape shapes.Shape, broadcastAxes []int) backends.Op {
	operand := b.checkOps("Transpose", operandOp)[0]
	shapeinference.BroadcastInDimOp(operand.shape, outputShape, broadcastAxes)
	node := b.newNode(backends.OpTypeBroadcast, outputShape, operand)
	node.data = &broadcastInDimNode{outputShape, broadcastAxes}
	return node
}

// ReduceMax implements backends.Builder interface.
func (b *Builder) ReduceMax(operandOp backends.Op, axis ...int) backends.Op {
	return b.reduceImpls(backends.OpTypeReduceMax, operandOp, axis...)
}

// ReduceMin implements backends.Builder interface.
func (b *Builder) ReduceMin(operandOp backends.Op, axis ...int) backends.Op {
	return b.reduceImpls(backends.OpTypeReduceMin, operandOp, axis...)
}

// ReduceSum implements backends.Builder interface.
func (b *Builder) ReduceSum(operandOp backends.Op, axis ...int) backends.Op {
	return b.reduceImpls(backends.OpTypeReduceSum, operandOp, axis...)
}

// ReduceProduct implements backends.Builder interface.
func (b *Builder) ReduceProduct(operandOp backends.Op, axis ...int) backends.Op {
	return b.reduceImpls(backends.OpTypeReduceProduct, operandOp, axis...)
}

func (b *Builder) reduceImpls(reduceOpType backends.OpType, operandOp backends.Op, axes ...int) backends.Op {
	operand := b.checkOps("ReduceOp", operandOp)[0]
	if len(axes) == 0 {
		// Default if no axes are given, is to reduce all axes.
		axes = xslices.Iota(0, operand.shape.Rank())
	}
	outputShape := shapeinference.ReduceOp(operand.shape, axes)
	outputShape.DType = operand.shape.DType
	node := b.newNode(reduceOpType, outputShape, operand)
	node.data = axes
	return node
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

// Expm1 implements backends.Builder interface. It returns e(x)-1.
func (b *Builder) Expm1(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeExpm1, operand)
}

// Log implements backends.Builder interface.
func (b *Builder) Log(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeLog, operand)
}

// Log1p implements backends.Builder interface.
func (b *Builder) Log1p(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeLog1p, operand)
}

// Logistic implements backends.Builder interface. Aka as sigmoid. It returns 1/(1+exp(-x)).
func (b *Builder) Logistic(operand backends.Op) backends.Op {
	return b.addUnaryOp(backends.OpTypeLogistic, operand)
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

// Pow implements backends.Builder interface.
func (b *Builder) Pow(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypePow, lhsOp, rhsOp)
}

// BitwiseAnd implements backends.Builder interface.
func (b *Builder) BitwiseAnd(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeBitwiseAnd, lhsOp, rhsOp)
}

// BitwiseOr implements backends.Builder interface.
func (b *Builder) BitwiseOr(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeBitwiseOr, lhsOp, rhsOp)
}

// BitwiseXor implements backends.Builder interface.
func (b *Builder) BitwiseXor(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeBitwiseXor, lhsOp, rhsOp)
}

// LogicalAnd implements backends.Builder interface.
func (b *Builder) LogicalAnd(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeLogicalAnd, lhsOp, rhsOp)
}

// LogicalOr implements backends.Builder interface.
func (b *Builder) LogicalOr(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeLogicalOr, lhsOp, rhsOp)
}

// LogicalXor implements backends.Builder interface.
func (b *Builder) LogicalXor(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeLogicalXor, lhsOp, rhsOp)
}

// Max implements backends.Builder interface.
func (b *Builder) Max(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeMax, lhsOp, rhsOp)
}

// Min implements backends.Builder interface.
func (b *Builder) Min(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addBinaryOp(backends.OpTypeMin, lhsOp, rhsOp)
}

// Equal implements backends.Builder interface.
func (b *Builder) Equal(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addComparisonOp(backends.OpTypeEqual, lhsOp, rhsOp)
}

// GreaterOrEqual implements backends.Builder interface.
func (b *Builder) GreaterOrEqual(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addComparisonOp(backends.OpTypeGreaterOrEqual, lhsOp, rhsOp)
}

// GreaterThan implements backends.Builder interface.
func (b *Builder) GreaterThan(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addComparisonOp(backends.OpTypeGreaterThan, lhsOp, rhsOp)
}

// LessOrEqual implements backends.Builder interface.
func (b *Builder) LessOrEqual(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addComparisonOp(backends.OpTypeLessOrEqual, lhsOp, rhsOp)
}

// LessThan implements backends.Builder interface.
func (b *Builder) LessThan(lhsOp, rhsOp backends.Op) backends.Op {
	return b.addComparisonOp(backends.OpTypeLessThan, lhsOp, rhsOp)
}
