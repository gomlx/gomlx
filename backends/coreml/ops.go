//go:build darwin

package coreml

import (
	"fmt"

	"github.com/gomlx/go-coreml/model"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/shapeinference"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// This file implements StandardOps interface methods on the Builder,
// mapping GoMLX operations to go-coreml MIL operations.

// Helper methods for common operation patterns

// addUnaryOp is a helper that adds a unary operation to the computation graph.
// It validates inputs, calls the MIL operation builder, and creates a new Node with the result.
func (b *Builder) addUnaryOp(
	opType backends.OpType,
	milOp func(*model.Value) *model.Value,
	x backends.Op,
) (*Node, error) {
	inputs, err := b.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.UnaryOp(opType, operand.shape)
	if err != nil {
		return nil, err
	}

	// Get the MIL Value from the operand node
	operandValue := operand.milValue

	// Call the MIL operation
	resultValue := milOp(operandValue)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// addBinaryOp is a helper that adds a binary operation to the computation graph.
// It validates inputs, calls the MIL operation builder, and creates a new Node with the result.
func (b *Builder) addBinaryOp(
	opType backends.OpType,
	milOp func(*model.Value, *model.Value) *model.Value,
	lhs, rhs backends.Op,
) (*Node, error) {
	inputs, err := b.checkOps(opType.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.BinaryOp(opType, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}

	// Get the MIL Values from the input nodes
	lhsValue := lhsNode.milValue
	rhsValue := rhsNode.milValue

	// Call the MIL operation
	resultValue := milOp(lhsValue, rhsValue)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, lhsNode, rhsNode)

	return node, nil
}

// addComparisonOp is a helper that adds a comparison operation to the computation graph.
// It validates inputs, calls the MIL operation builder, and creates a new Node with the result.
// Comparison operations return Bool dtype.
func (b *Builder) addComparisonOp(
	opType backends.OpType,
	milOp func(*model.Value, *model.Value) *model.Value,
	lhs, rhs backends.Op,
) (*Node, error) {
	inputs, err := b.checkOps(opType.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Compute output shape using shapeinference.ComparisonOp
	// which handles broadcast and sets dtype to Bool
	outputShape, err := shapeinference.ComparisonOp(opType, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}

	// Get the MIL Values from the input nodes
	lhsValue := lhsNode.milValue
	rhsValue := rhsNode.milValue

	// Call the MIL operation
	resultValue := milOp(lhsValue, rhsValue)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, lhsNode, rhsNode)

	return node, nil
}

//======================================================================================================================
// Unary Operations ----------------------------------------------------------------------------------------------------
//======================================================================================================================

// Abs implements the backends.Builder interface.
// Computes element-wise absolute value: z = |x|.
func (b *Builder) Abs(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeAbs, b.milBuilder.Abs, x)
}

// Neg implements the backends.Builder interface.
// Computes element-wise negation: z = -x.
// Note: CoreML doesn't have a direct neg operator, so we implement it as mul(x, -1).
func (b *Builder) Neg(x backends.Op) (backends.Op, error) {
	opType := backends.OpTypeNeg
	inputs, err := b.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Create a constant -1 for multiplication (scalar broadcasts)
	constName := fmt.Sprintf("neg_one_%d", b.nextConstID)
	b.nextConstID++
	negOne := b.milBuilder.Const(constName, operand.milValue.DType(), []int64{}, []float32{-1.0})

	// Multiply by -1 to negate
	resultValue := b.milBuilder.Mul(operand.milValue, negOne)

	// Create a new node with the result
	node := b.newNode(opType, operand.shape, resultValue, operand)

	return node, nil
}

// Exp implements the backends.Builder interface.
// Computes element-wise exponential: z = exp(x).
func (b *Builder) Exp(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeExp, b.milBuilder.Exp, x)
}

// Log implements the backends.Builder interface.
// Computes element-wise natural logarithm: z = log(x).
func (b *Builder) Log(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeLog, b.milBuilder.Log, x)
}

// Sqrt implements the backends.Builder interface.
// Computes element-wise square root: z = sqrt(x).
func (b *Builder) Sqrt(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeSqrt, b.milBuilder.Sqrt, x)
}

// Floor implements the backends.Builder interface.
// Computes element-wise floor: z = floor(x).
func (b *Builder) Floor(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeFloor, b.milBuilder.Floor, x)
}

// Ceil implements the backends.Builder interface.
// Computes element-wise ceiling: z = ceil(x).
func (b *Builder) Ceil(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeCeil, b.milBuilder.Ceil, x)
}

// Round implements the backends.Builder interface.
// Computes element-wise rounding: z = round(x).
func (b *Builder) Round(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeRound, b.milBuilder.Round, x)
}

// Sign implements the backends.Builder interface.
// Computes element-wise sign: z = sign(x).
// Returns -1 for negative values, 0 for zero, and 1 for positive values.
func (b *Builder) Sign(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeSign, b.milBuilder.Sign, x)
}

// Tanh implements the backends.Builder interface.
// Applies hyperbolic tangent: z = tanh(x).
func (b *Builder) Tanh(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeTanh, b.milBuilder.Tanh, x)
}

// Logistic implements the backends.Builder interface.
// Applies sigmoid activation: z = 1 / (1 + exp(-x)).
// Note: In GoMLX this is called Logistic, in CoreML it's Sigmoid.
func (b *Builder) Logistic(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeLogistic, b.milBuilder.Sigmoid, x)
}

// Cos implements the backends.Builder interface.
// Computes element-wise cosine: z = cos(x).
func (b *Builder) Cos(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeCos, b.milBuilder.Cos, x)
}

// Sin implements the backends.Builder interface.
// Computes element-wise sine: z = sin(x).
func (b *Builder) Sin(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeSin, b.milBuilder.Sin, x)
}

// Erf implements the backends.Builder interface.
// Computes element-wise error function: z = erf(x).
func (b *Builder) Erf(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeErf, b.milBuilder.Erf, x)
}

// Expm1 implements the backends.Builder interface.
// Computes element-wise exp(x) - 1.
// Implemented using Exp and Sub: exp(x) - 1.
func (b *Builder) Expm1(x backends.Op) (backends.Op, error) {
	opType := backends.OpTypeExpm1
	inputs, err := b.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape (same as input)
	outputShape, err := shapeinference.UnaryOp(opType, operand.shape)
	if err != nil {
		return nil, err
	}

	// exp(x)
	expResult := b.milBuilder.Exp(operand.milValue)

	// Create constant 1 with the same dtype as x
	constName := fmt.Sprintf("expm1_one_%d", b.nextConstID)
	b.nextConstID++
	one := b.milBuilder.Const(constName, operand.milValue.DType(), []int64{}, []float32{1.0})

	// exp(x) - 1
	resultValue := b.milBuilder.Sub(expResult, one)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// Log1p implements the backends.Builder interface.
// Computes element-wise log(1 + x).
// Implemented using Add and Log: log(x + 1).
func (b *Builder) Log1p(x backends.Op) (backends.Op, error) {
	opType := backends.OpTypeLog1p
	inputs, err := b.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape (same as input)
	outputShape, err := shapeinference.UnaryOp(opType, operand.shape)
	if err != nil {
		return nil, err
	}

	// Create constant 1 with the same dtype as x
	constName := fmt.Sprintf("log1p_one_%d", b.nextConstID)
	b.nextConstID++
	one := b.milBuilder.Const(constName, operand.milValue.DType(), []int64{}, []float32{1.0})

	// x + 1
	xPlusOne := b.milBuilder.Add(operand.milValue, one)

	// log(x + 1)
	resultValue := b.milBuilder.Log(xPlusOne)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// Rsqrt implements the backends.Builder interface.
// Computes element-wise reciprocal square root: z = 1/sqrt(x).
func (b *Builder) Rsqrt(x backends.Op) (backends.Op, error) {
	return b.addUnaryOp(backends.OpTypeRsqrt, b.milBuilder.Rsqrt, x)
}

// IsNaN implements the backends.Builder interface.
// Checks element-wise if values are NaN.
// Returns Bool dtype indicating which elements are NaN.
func (b *Builder) IsNaN(x backends.Op) (backends.Op, error) {
	inputs, err := b.checkOps(backends.OpTypeIsNaN.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	// IsNaN returns Bool dtype with same shape as input
	outputShape := shapes.Make(dtypes.Bool, operand.shape.Dimensions...)

	// Call the MIL operation
	resultValue := b.milBuilder.IsNan(operand.milValue)

	// Create a new node with the result
	node := b.newNode(backends.OpTypeIsNaN, outputShape, resultValue, operand)

	return node, nil
}

// IsFinite implements the backends.Builder interface.
// Checks element-wise if values are finite (not NaN or Inf).
// Returns Bool dtype indicating which elements are finite.
func (b *Builder) IsFinite(x backends.Op) (backends.Op, error) {
	inputs, err := b.checkOps(backends.OpTypeIsFinite.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	// IsFinite returns Bool dtype with same shape as input
	outputShape := shapes.Make(dtypes.Bool, operand.shape.Dimensions...)

	// Call the MIL operation
	resultValue := b.milBuilder.IsFinite(operand.milValue)

	// Create a new node with the result
	node := b.newNode(backends.OpTypeIsFinite, outputShape, resultValue, operand)

	return node, nil
}

//======================================================================================================================
// Binary Operations ---------------------------------------------------------------------------------------------------
//======================================================================================================================

// Add implements the backends.Builder interface.
// Performs element-wise addition: z = x + y.
func (b *Builder) Add(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeAdd, b.milBuilder.Add, lhs, rhs)
}

// Sub implements the backends.Builder interface.
// Performs element-wise subtraction: z = x - y.
func (b *Builder) Sub(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeSub, b.milBuilder.Sub, lhs, rhs)
}

// Mul implements the backends.Builder interface.
// Performs element-wise multiplication: z = x * y.
func (b *Builder) Mul(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeMul, b.milBuilder.Mul, lhs, rhs)
}

// Div implements the backends.Builder interface.
// Performs element-wise division: z = x / y.
func (b *Builder) Div(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeDiv, b.milBuilder.Div, lhs, rhs)
}

// Pow implements the backends.Builder interface.
// Performs element-wise power: z = x^y.
func (b *Builder) Pow(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypePow, b.milBuilder.Pow, lhs, rhs)
}

// Max implements the backends.Builder interface.
// Performs element-wise maximum: z = max(x, y).
func (b *Builder) Max(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeMax, b.milBuilder.Maximum, lhs, rhs)
}

// Min implements the backends.Builder interface.
// Performs element-wise minimum: z = min(x, y).
func (b *Builder) Min(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addBinaryOp(backends.OpTypeMin, b.milBuilder.Minimum, lhs, rhs)
}

//======================================================================================================================
// Comparison Operations -----------------------------------------------------------------------------------------------
//======================================================================================================================

// Equal implements the backends.Builder interface.
// Performs element-wise equality comparison: z = (x == y).
// Returns Bool dtype.
func (b *Builder) Equal(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeEqual, b.milBuilder.Equal, lhs, rhs)
}

// NotEqual implements the backends.Builder interface.
// Performs element-wise inequality comparison: z = (x != y).
// Returns Bool dtype.
func (b *Builder) NotEqual(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeNotEqual, b.milBuilder.NotEqual, lhs, rhs)
}

// LessThan implements the backends.Builder interface.
// Performs element-wise less-than comparison: z = (x < y).
// Returns Bool dtype.
func (b *Builder) LessThan(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeLessThan, b.milBuilder.Less, lhs, rhs)
}

// LessOrEqual implements the backends.Builder interface.
// Performs element-wise less-than-or-equal comparison: z = (x <= y).
// Returns Bool dtype.
func (b *Builder) LessOrEqual(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeLessOrEqual, b.milBuilder.LessEqual, lhs, rhs)
}

// GreaterThan implements the backends.Builder interface.
// Performs element-wise greater-than comparison: z = (x > y).
// Returns Bool dtype.
func (b *Builder) GreaterThan(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeGreaterThan, b.milBuilder.Greater, lhs, rhs)
}

// GreaterOrEqual implements the backends.Builder interface.
// Performs element-wise greater-than-or-equal comparison: z = (x >= y).
// Returns Bool dtype.
func (b *Builder) GreaterOrEqual(lhs, rhs backends.Op) (backends.Op, error) {
	return b.addComparisonOp(backends.OpTypeGreaterOrEqual, b.milBuilder.GreaterEqual, lhs, rhs)
}

//======================================================================================================================
// Logical Operations --------------------------------------------------------------------------------------------------
//======================================================================================================================

// LogicalAnd implements the backends.Builder interface.
// Performs element-wise logical AND: z = x && y.
// Both inputs must have Bool dtype. Returns Bool dtype.
func (b *Builder) LogicalAnd(lhs, rhs backends.Op) (backends.Op, error) {
	inputs, err := b.checkOps(backends.OpTypeLogicalAnd.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Validate that both inputs are Bool
	if lhsNode.shape.DType != dtypes.Bool || rhsNode.shape.DType != dtypes.Bool {
		return nil, errors.Errorf("LogicalAnd: both inputs must have Bool dtype, got %s and %s",
			lhsNode.shape.DType, rhsNode.shape.DType)
	}

	// Compute output shape (broadcast and Bool dtype)
	outputShape, err := shapeinference.BinaryOp(backends.OpTypeLogicalAnd, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}

	// Call the MIL operation
	resultValue := b.milBuilder.LogicalAnd(lhsNode.milValue, rhsNode.milValue)

	// Create a new node with the result
	node := b.newNode(backends.OpTypeLogicalAnd, outputShape, resultValue, lhsNode, rhsNode)

	return node, nil
}

// LogicalOr implements the backends.Builder interface.
// Performs element-wise logical OR: z = x || y.
// Both inputs must have Bool dtype. Returns Bool dtype.
func (b *Builder) LogicalOr(lhs, rhs backends.Op) (backends.Op, error) {
	inputs, err := b.checkOps(backends.OpTypeLogicalOr.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Validate that both inputs are Bool
	if lhsNode.shape.DType != dtypes.Bool || rhsNode.shape.DType != dtypes.Bool {
		return nil, errors.Errorf("LogicalOr: both inputs must have Bool dtype, got %s and %s",
			lhsNode.shape.DType, rhsNode.shape.DType)
	}

	// Compute output shape (broadcast and Bool dtype)
	outputShape, err := shapeinference.BinaryOp(backends.OpTypeLogicalOr, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}

	// Call the MIL operation
	resultValue := b.milBuilder.LogicalOr(lhsNode.milValue, rhsNode.milValue)

	// Create a new node with the result
	node := b.newNode(backends.OpTypeLogicalOr, outputShape, resultValue, lhsNode, rhsNode)

	return node, nil
}

// LogicalNot implements the backends.Builder interface.
// Performs element-wise logical NOT: z = !x.
// Input must have Bool dtype. Returns Bool dtype.
func (b *Builder) LogicalNot(x backends.Op) (backends.Op, error) {
	inputs, err := b.checkOps(backends.OpTypeLogicalNot.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Validate that input is Bool
	if operand.shape.DType != dtypes.Bool {
		return nil, errors.Errorf("LogicalNot: input must have Bool dtype, got %s", operand.shape.DType)
	}

	// Output shape is same as input
	outputShape := operand.shape

	// Call the MIL operation
	resultValue := b.milBuilder.LogicalNot(operand.milValue)

	// Create a new node with the result
	node := b.newNode(backends.OpTypeLogicalNot, outputShape, resultValue, operand)

	return node, nil
}

// LogicalXor implements the backends.Builder interface.
// Performs element-wise logical XOR: z = x ^ y.
// Both inputs must have Bool dtype. Returns Bool dtype.
func (b *Builder) LogicalXor(lhs, rhs backends.Op) (backends.Op, error) {
	inputs, err := b.checkOps(backends.OpTypeLogicalXor.String(), lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode, rhsNode := inputs[0], inputs[1]

	// Validate that both inputs are Bool
	if lhsNode.shape.DType != dtypes.Bool || rhsNode.shape.DType != dtypes.Bool {
		return nil, errors.Errorf("LogicalXor: both inputs must have Bool dtype, got %s and %s",
			lhsNode.shape.DType, rhsNode.shape.DType)
	}

	// Compute output shape (broadcast and Bool dtype)
	outputShape, err := shapeinference.BinaryOp(backends.OpTypeLogicalXor, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, err
	}

	// Call the MIL operation
	resultValue := b.milBuilder.LogicalXor(lhsNode.milValue, rhsNode.milValue)

	// Create a new node with the result
	node := b.newNode(backends.OpTypeLogicalXor, outputShape, resultValue, lhsNode, rhsNode)

	return node, nil
}

//======================================================================================================================
// Activation Functions ------------------------------------------------------------------------------------------------
//======================================================================================================================

// Note: Relu is not a standard GoMLX backend operation.
// It can be implemented using Max(x, Constant(0)) if needed.

//======================================================================================================================
// Reshape Operations --------------------------------------------------------------------------------------------------
//======================================================================================================================

// Reshape implements the backends.Builder interface.
// Changes the shape of a tensor without changing its data.
func (b *Builder) Reshape(operandOp backends.Op, dims ...int) (backends.Op, error) {
	opType := backends.OpTypeReshape
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ReshapeOp(operand.shape, dims)
	if err != nil {
		return nil, err
	}

	// Convert dims to int64 for CoreML
	dimsInt64 := make([]int64, len(dims))
	for i, d := range dims {
		dimsInt64[i] = int64(d)
	}

	// Call the MIL Reshape operation
	resultValue := b.milBuilder.Reshape(operand.milValue, dimsInt64)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// Transpose implements the backends.Builder interface.
// Permutes the dimensions of a tensor.
func (b *Builder) Transpose(operandOp backends.Op, permutations ...int) (backends.Op, error) {
	opType := backends.OpTypeTranspose
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.TransposeOp(operand.shape, permutations)
	if err != nil {
		return nil, err
	}

	// Convert permutations to int64 for CoreML
	permInt64 := make([]int64, len(permutations))
	for i, p := range permutations {
		permInt64[i] = int64(p)
	}

	// Call the MIL Transpose operation
	resultValue := b.milBuilder.Transpose(operand.milValue, permInt64)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

//======================================================================================================================
// Reduction Operations ------------------------------------------------------------------------------------------------
//======================================================================================================================

// ReduceSum implements the backends.Builder interface.
// Computes sum along specified axes.
func (b *Builder) ReduceSum(operandOp backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceSum
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// If no axes specified, reduce all axes
	if len(axes) == 0 {
		axes = make([]int, operand.shape.Rank())
		for i := 0; i < operand.shape.Rank(); i++ {
			axes[i] = i
		}
	}

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ReduceOp(operand.shape, axes)
	if err != nil {
		return nil, err
	}
	outputShape.DType = operand.shape.DType

	// Convert axes to int64 for CoreML
	axesInt64 := make([]int64, len(axes))
	for i, a := range axes {
		axesInt64[i] = int64(a)
	}

	// CoreML ReduceSum uses keepDims=false by default to match GoMLX behavior
	resultValue := b.milBuilder.ReduceSum(operand.milValue, axesInt64, false)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// ReduceMax implements the backends.Builder interface.
// Computes max along specified axes.
func (b *Builder) ReduceMax(operandOp backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceMax
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// If no axes specified, reduce all axes
	if len(axes) == 0 {
		axes = make([]int, operand.shape.Rank())
		for i := 0; i < operand.shape.Rank(); i++ {
			axes[i] = i
		}
	}

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ReduceOp(operand.shape, axes)
	if err != nil {
		return nil, err
	}
	outputShape.DType = operand.shape.DType

	// Convert axes to int64 for CoreML
	axesInt64 := make([]int64, len(axes))
	for i, a := range axes {
		axesInt64[i] = int64(a)
	}

	// CoreML ReduceMax uses keepDims=false by default to match GoMLX behavior
	resultValue := b.milBuilder.ReduceMax(operand.milValue, axesInt64, false)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// ReduceMin implements the backends.Builder interface.
// Computes min along specified axes.
func (b *Builder) ReduceMin(operandOp backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceMin
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// If no axes specified, reduce all axes
	if len(axes) == 0 {
		axes = make([]int, operand.shape.Rank())
		for i := 0; i < operand.shape.Rank(); i++ {
			axes[i] = i
		}
	}

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ReduceOp(operand.shape, axes)
	if err != nil {
		return nil, err
	}
	outputShape.DType = operand.shape.DType

	// Convert axes to int64 for CoreML
	axesInt64 := make([]int64, len(axes))
	for i, a := range axes {
		axesInt64[i] = int64(a)
	}

	// CoreML ReduceMin uses keepDims=false by default to match GoMLX behavior
	resultValue := b.milBuilder.ReduceMin(operand.milValue, axesInt64, false)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// ReduceProduct implements the backends.Builder interface.
// Computes product along specified axes.
func (b *Builder) ReduceProduct(operandOp backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReduceProduct
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// If no axes specified, reduce all axes
	if len(axes) == 0 {
		axes = make([]int, operand.shape.Rank())
		for i := 0; i < operand.shape.Rank(); i++ {
			axes[i] = i
		}
	}

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ReduceOp(operand.shape, axes)
	if err != nil {
		return nil, err
	}
	outputShape.DType = operand.shape.DType

	// Convert axes to int64 for CoreML
	axesInt64 := make([]int64, len(axes))
	for i, a := range axes {
		axesInt64[i] = int64(a)
	}

	// CoreML ReduceProd uses keepDims=false by default to match GoMLX behavior
	resultValue := b.milBuilder.ReduceProd(operand.milValue, axesInt64, false)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// ReduceWindow implements the backends.Builder interface.
// Maps ReduceWindow operations to CoreML pooling operations (MaxPool/AvgPool) for supported cases.
//
// Supported configurations:
//   - ReduceOpMax maps to MaxPool
//   - ReduceOpSum with uniform window maps to AvgPool (with appropriate scaling)
//
// Unsupported:
//   - ReduceOpMin (no direct CoreML equivalent for MinPool)
//   - ReduceOpProduct (no direct CoreML equivalent)
//   - Non-uniform window dilations
//   - Base dilations (dilated input)
func (b *Builder) ReduceWindow(
	operandOp backends.Op,
	reductionType backends.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int,
	paddings [][2]int,
) (backends.Op, error) {
	opType := backends.OpTypeReduceWindow
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Use shapeinference to compute output shape and validate parameters
	outputShape, err := shapeinference.ReduceWindowOp(
		operand.shape,
		windowDimensions, strides, baseDilations, windowDilations, paddings,
	)
	if err != nil {
		return nil, err
	}

	// Check for unsupported operations
	if reductionType == backends.ReduceOpMin {
		return nil, errors.Errorf("ReduceWindow: ReduceOpMin is not supported in CoreML backend (no MinPool operation)")
	}
	if reductionType == backends.ReduceOpProduct {
		return nil, errors.Errorf("ReduceWindow: ReduceOpProduct is not supported in CoreML backend")
	}

	// Check for base dilations (dilated input)
	if len(baseDilations) > 0 {
		for _, d := range baseDilations {
			if d != 1 {
				return nil, errors.Errorf("ReduceWindow: base dilations not supported in CoreML backend, got %v", baseDilations)
			}
		}
	}

	// Validate input rank - pooling requires at least rank 3 (batch, channel, spatial...)
	rank := operand.shape.Rank()
	if rank < 3 {
		return nil, errors.Errorf("ReduceWindow: CoreML pooling requires rank >= 3 (NCHW format), got rank %d", rank)
	}

	// CoreML pooling only operates on spatial dimensions (assumes NCHW layout)
	// Window and stride sizes should only apply to spatial dimensions
	numSpatialDims := rank - 2

	// Validate that window/stride are specified for all dimensions or just spatial dimensions
	if len(windowDimensions) != rank && len(windowDimensions) != numSpatialDims {
		return nil, errors.Errorf("ReduceWindow: windowDimensions length %d must match rank %d or spatial dims %d",
			len(windowDimensions), rank, numSpatialDims)
	}

	// If windowDimensions is for all dims, check that batch and channel dims have window size 1
	var spatialWindowDims, spatialStrides []int64
	var spatialWindowDilations []int64
	var spatialPadBefore, spatialPadAfter []int64

	if len(windowDimensions) == rank {
		// Window specified for all dimensions - batch and channel should be 1
		if windowDimensions[0] != 1 || windowDimensions[1] != 1 {
			return nil, errors.Errorf("ReduceWindow: window dimensions for batch and channel axes must be 1, got %v", windowDimensions)
		}
		// Extract spatial dimensions
		spatialWindowDims = make([]int64, numSpatialDims)
		for i := 0; i < numSpatialDims; i++ {
			spatialWindowDims[i] = int64(windowDimensions[i+2])
		}
	} else {
		// Window specified only for spatial dimensions
		spatialWindowDims = make([]int64, numSpatialDims)
		for i := 0; i < numSpatialDims; i++ {
			spatialWindowDims[i] = int64(windowDimensions[i])
		}
	}

	// Handle strides
	if len(strides) == 0 {
		// Default: strides equal to window dimensions
		spatialStrides = make([]int64, len(spatialWindowDims))
		copy(spatialStrides, spatialWindowDims)
	} else if len(strides) == rank {
		// Strides for all dimensions
		if strides[0] != 1 || strides[1] != 1 {
			return nil, errors.Errorf("ReduceWindow: strides for batch and channel axes must be 1, got %v", strides)
		}
		spatialStrides = make([]int64, numSpatialDims)
		for i := 0; i < numSpatialDims; i++ {
			spatialStrides[i] = int64(strides[i+2])
		}
	} else if len(strides) == numSpatialDims {
		spatialStrides = make([]int64, numSpatialDims)
		for i := 0; i < numSpatialDims; i++ {
			spatialStrides[i] = int64(strides[i])
		}
	} else {
		return nil, errors.Errorf("ReduceWindow: strides length %d must match rank %d or spatial dims %d",
			len(strides), rank, numSpatialDims)
	}

	// Handle window dilations
	if len(windowDilations) == 0 {
		// Default: no dilation (all 1s)
		spatialWindowDilations = make([]int64, numSpatialDims)
		for i := range spatialWindowDilations {
			spatialWindowDilations[i] = 1
		}
	} else if len(windowDilations) == rank {
		// Dilations for all dimensions - batch and channel should be 1
		if windowDilations[0] != 1 || windowDilations[1] != 1 {
			return nil, errors.Errorf("ReduceWindow: window dilations for batch and channel axes must be 1, got %v", windowDilations)
		}
		spatialWindowDilations = make([]int64, numSpatialDims)
		for i := 0; i < numSpatialDims; i++ {
			spatialWindowDilations[i] = int64(windowDilations[i+2])
		}
	} else if len(windowDilations) == numSpatialDims {
		spatialWindowDilations = make([]int64, numSpatialDims)
		for i := 0; i < numSpatialDims; i++ {
			spatialWindowDilations[i] = int64(windowDilations[i])
		}
	} else {
		return nil, errors.Errorf("ReduceWindow: windowDilations length %d must match rank %d or spatial dims %d",
			len(windowDilations), rank, numSpatialDims)
	}

	// CoreML doesn't support window dilations for pooling
	for _, d := range spatialWindowDilations {
		if d != 1 {
			return nil, errors.Errorf("ReduceWindow: window dilations not supported in CoreML pooling, got %v", windowDilations)
		}
	}

	// Handle paddings
	spatialPadBefore = make([]int64, numSpatialDims)
	spatialPadAfter = make([]int64, numSpatialDims)

	if len(paddings) == 0 {
		// Default: no padding (all zeros)
		// Already initialized to zero
	} else if len(paddings) == rank {
		// Paddings for all dimensions - batch and channel should be [0, 0]
		if paddings[0][0] != 0 || paddings[0][1] != 0 || paddings[1][0] != 0 || paddings[1][1] != 0 {
			return nil, errors.Errorf("ReduceWindow: padding for batch and channel axes must be [0,0], got %v", paddings)
		}
		for i := 0; i < numSpatialDims; i++ {
			spatialPadBefore[i] = int64(paddings[i+2][0])
			spatialPadAfter[i] = int64(paddings[i+2][1])
		}
	} else if len(paddings) == numSpatialDims {
		for i := 0; i < numSpatialDims; i++ {
			spatialPadBefore[i] = int64(paddings[i][0])
			spatialPadAfter[i] = int64(paddings[i][1])
		}
	} else {
		return nil, errors.Errorf("ReduceWindow: paddings length %d must match rank %d or spatial dims %d",
			len(paddings), rank, numSpatialDims)
	}

	// Determine padding type for CoreML
	var padType model.ConvPadType
	hasNonZeroPadding := false
	for i := 0; i < numSpatialDims; i++ {
		if spatialPadBefore[i] != 0 || spatialPadAfter[i] != 0 {
			hasNonZeroPadding = true
			break
		}
	}

	if hasNonZeroPadding {
		padType = model.ConvPadCustom
	} else {
		padType = model.ConvPadValid
	}

	// Map reduction type to pooling operation
	var resultValue *model.Value

	switch reductionType {
	case backends.ReduceOpMax:
		// MaxPool
		resultValue = b.milBuilder.MaxPool(
			operand.milValue,
			spatialWindowDims,
			spatialStrides,
			padType,
			spatialPadBefore,
			spatialPadAfter,
		)

	case backends.ReduceOpSum:
		// For ReduceOpSum, we use AvgPool and then multiply by the window size
		// AvgPool computes: sum(window) / window_size
		// To get sum, we multiply by window_size after pooling

		avgPoolResult := b.milBuilder.AvgPool(
			operand.milValue,
			spatialWindowDims,
			spatialStrides,
			padType,
			spatialPadBefore,
			spatialPadAfter,
			false, // exclude_padding_from_average=false to match ReduceSum semantics
		)

		// Compute window size (product of all window dimensions)
		windowSize := int64(1)
		for _, dim := range spatialWindowDims {
			windowSize *= dim
		}

		// Create constant for window size
		windowSizeConstName := fmt.Sprintf("reduce_window_scale_%d", b.nextConstID)
		b.nextConstID++
		windowSizeConst := b.milBuilder.Const(
			windowSizeConstName,
			model.Float32,
			[]int64{}, // scalar
			[]float32{float32(windowSize)},
		)

		// Multiply avgPool result by window size to get sum
		resultValue = b.milBuilder.Mul(avgPoolResult, windowSizeConst)

	default:
		return nil, errors.Errorf("ReduceWindow: unsupported reduction type %s for CoreML backend", reductionType)
	}

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// ArgMinMax implements the backends.Builder interface.
// Calculates the "argmin" or "argmax" across an axis of the given input array x.
func (b *Builder) ArgMinMax(x backends.Op, axis int, outputDType dtypes.DType, isMin bool) (backends.Op, error) {
	opType := backends.OpTypeArgMinMax
	inputs, err := b.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ArgMinMaxOp(operand.shape, axis, outputDType)
	if err != nil {
		return nil, err
	}

	// Call the appropriate MIL operation based on isMin
	var resultValue *model.Value
	if isMin {
		resultValue = b.milBuilder.ArgMin(operand.milValue, int64(axis), false)
	} else {
		resultValue = b.milBuilder.ArgMax(operand.milValue, int64(axis), false)
	}

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

//======================================================================================================================
// Clamping Operations -------------------------------------------------------------------------------------------------
//======================================================================================================================

// Clamp implements the backends.Builder interface.
// Clamps values to the range [minVal, maxVal].
// Note: GoMLX order is (min, x, max), CoreML order is (x, min, max).
func (b *Builder) Clamp(minVal, x, maxVal backends.Op) (backends.Op, error) {
	opType := backends.OpTypeClamp
	inputs, err := b.checkOps(opType.String(), minVal, x, maxVal)
	if err != nil {
		return nil, err
	}
	minNode, xNode, maxNode := inputs[0], inputs[1], inputs[2]

	// Compute output shape - use BinaryOp twice for broadcasting
	// First broadcast x with min, then result with max
	tempShape, err := shapeinference.BinaryOp(backends.OpTypeMin, xNode.shape, minNode.shape)
	if err != nil {
		return nil, err
	}
	outputShape, err := shapeinference.BinaryOp(backends.OpTypeMin, tempShape, maxNode.shape)
	if err != nil {
		return nil, err
	}

	// Call the MIL Clip operation (CoreML order: x, min, max)
	resultValue := b.milBuilder.Clip(xNode.milValue, minNode.milValue, maxNode.milValue)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, minNode, xNode, maxNode)

	return node, nil
}

//======================================================================================================================
// Type Conversion Operations ------------------------------------------------------------------------------------------
//======================================================================================================================

// ConvertDType implements the backends.Builder interface.
// Converts tensor to a different dtype.
func (b *Builder) ConvertDType(x backends.Op, dtype dtypes.DType) (backends.Op, error) {
	opType := backends.OpTypeConvertDType
	inputs, err := b.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// If dtype is already the same, no-op
	if operand.shape.DType == dtype {
		return operand, nil
	}

	// Compute output shape (same dimensions, different dtype)
	outputShape := operand.shape.Clone()
	outputShape.DType = dtype

	// Convert GoMLX dtype to CoreML DType
	coremlDType, err := gomlxDTypeToMIL(dtype)
	if err != nil {
		return nil, errors.WithMessagef(err, "ConvertDType: unsupported dtype %s", dtype)
	}

	// Call the MIL Cast operation
	resultValue := b.milBuilder.Cast(operand.milValue, coremlDType)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

//======================================================================================================================
// Matrix Operations ---------------------------------------------------------------------------------------------------
//======================================================================================================================

// DotGeneral implements the backends.Builder interface.
// Performs a general matrix multiplication (generalized Einstein summation).
//
// For simple matrix multiplication (C = A @ B), this would be called with:
//   - lhsContractingAxes: [1] (last axis of lhs)
//   - rhsContractingAxes: [0] (first axis of rhs)
//   - lhsBatchAxes, rhsBatchAxes: [] (no batch dimensions)
//
// Enhanced to support:
//   - Batch dimensions: [B, M, K] @ [B, K, N] -> [B, M, N]
//   - Arbitrary contracting axes via transpose normalization
func (b *Builder) DotGeneral(
	lhsOp backends.Op,
	lhsContractingAxes, lhsBatchAxes []int,
	rhsOp backends.Op,
	rhsContractingAxes, rhsBatchAxes []int,
) (backends.Op, error) {
	opType := backends.OpTypeDotGeneral
	inputs, err := b.checkOps(opType.String(), lhsOp, rhsOp)
	if err != nil {
		return nil, err
	}
	lhs, rhs := inputs[0], inputs[1]

	// Validate that shapes match
	if lhs.shape.DType != rhs.shape.DType {
		return nil, errors.Errorf("DotGeneral: lhs and rhs must have the same dtype, got %s and %s",
			lhs.shape.DType, rhs.shape.DType)
	}

	lhsRank := lhs.shape.Rank()
	rhsRank := rhs.shape.Rank()

	if lhsRank < 2 || rhsRank < 2 {
		return nil, errors.Errorf("DotGeneral: operands must have rank >= 2, got lhs rank %d, rhs rank %d",
			lhsRank, rhsRank)
	}

	// Validate contracting axes
	if len(lhsContractingAxes) != 1 || len(rhsContractingAxes) != 1 {
		return nil, errors.Errorf("DotGeneral: CoreML backend supports exactly one contracting axis, "+
			"got lhsContractingAxes=%v, rhsContractingAxes=%v",
			lhsContractingAxes, rhsContractingAxes)
	}

	// Validate batch axes match
	if len(lhsBatchAxes) != len(rhsBatchAxes) {
		return nil, errors.Errorf("DotGeneral: batch axes must have same length, got lhsBatchAxes=%v, rhsBatchAxes=%v",
			lhsBatchAxes, rhsBatchAxes)
	}

	// Normalize contracting axes (handle negative indices)
	lhsContractAxis := lhsContractingAxes[0]
	if lhsContractAxis < 0 {
		lhsContractAxis += lhsRank
	}
	rhsContractAxis := rhsContractingAxes[0]
	if rhsContractAxis < 0 {
		rhsContractAxis += rhsRank
	}

	// Normalize batch axes
	normalizedLhsBatch := make([]int, len(lhsBatchAxes))
	for i, axis := range lhsBatchAxes {
		if axis < 0 {
			normalizedLhsBatch[i] = axis + lhsRank
		} else {
			normalizedLhsBatch[i] = axis
		}
	}
	normalizedRhsBatch := make([]int, len(rhsBatchAxes))
	for i, axis := range rhsBatchAxes {
		if axis < 0 {
			normalizedRhsBatch[i] = axis + rhsRank
		} else {
			normalizedRhsBatch[i] = axis
		}
	}

	// Strategy: Transpose inputs to move axes to standard positions
	// Standard matmul layout: [batch..., M, K] @ [batch..., K, N] -> [batch..., M, N]
	// where batch dimensions come first, M and K are the last two dims

	lhsTransposed := lhsOp
	rhsTransposed := rhsOp

	// Build permutation for lhs: [batch axes, non-contracting non-batch axes, contracting axis]
	lhsPerm := make([]int, lhsRank)
	lhsIdx := 0

	// First, batch axes
	for _, batchAxis := range normalizedLhsBatch {
		lhsPerm[lhsIdx] = batchAxis
		lhsIdx++
	}

	// Then, non-batch non-contracting axes (the M dimension)
	for i := 0; i < lhsRank; i++ {
		isBatch := false
		for _, batchAxis := range normalizedLhsBatch {
			if i == batchAxis {
				isBatch = true
				break
			}
		}
		if !isBatch && i != lhsContractAxis {
			lhsPerm[lhsIdx] = i
			lhsIdx++
		}
	}

	// Finally, contracting axis (K dimension)
	lhsPerm[lhsIdx] = lhsContractAxis

	// Build permutation for rhs: [batch axes, contracting axis, non-contracting non-batch axes]
	rhsPerm := make([]int, rhsRank)
	rhsIdx := 0

	// First, batch axes
	for _, batchAxis := range normalizedRhsBatch {
		rhsPerm[rhsIdx] = batchAxis
		rhsIdx++
	}

	// Then, contracting axis (K dimension)
	rhsPerm[rhsIdx] = rhsContractAxis
	rhsIdx++

	// Finally, non-batch non-contracting axes (the N dimension)
	for i := 0; i < rhsRank; i++ {
		isBatch := false
		for _, batchAxis := range normalizedRhsBatch {
			if i == batchAxis {
				isBatch = true
				break
			}
		}
		if !isBatch && i != rhsContractAxis {
			rhsPerm[rhsIdx] = i
			rhsIdx++
		}
	}

	// Check if we need to transpose lhs
	needsLhsTranspose := false
	for i, p := range lhsPerm {
		if p != i {
			needsLhsTranspose = true
			break
		}
	}

	if needsLhsTranspose {
		lhsTransposed, err = b.Transpose(lhsOp, lhsPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: transpose lhs")
		}
	}

	// Check if we need to transpose rhs
	needsRhsTranspose := false
	for i, p := range rhsPerm {
		if p != i {
			needsRhsTranspose = true
			break
		}
	}

	if needsRhsTranspose {
		rhsTransposed, err = b.Transpose(rhsOp, rhsPerm...)
		if err != nil {
			return nil, errors.Wrap(err, "DotGeneral: transpose rhs")
		}
	}

	// Now both inputs are in standard layout: [batch..., M, K] @ [batch..., K, N]
	// CoreML's matmul handles this natively with transposeX=false, transposeY=false
	lhsNode := lhsTransposed.(*Node)
	rhsNode := rhsTransposed.(*Node)

	// Use CoreML's MatMul operation (it supports batch dimensions natively)
	resultValue := b.milBuilder.MatMulTranspose(lhsNode.milValue, rhsNode.milValue, false, false)

	// Compute output shape
	// Output: [batch..., M, N]
	outputDims := make([]int, len(normalizedLhsBatch)+2)

	// Copy batch dimensions from lhs
	for i, batchAxis := range normalizedLhsBatch {
		outputDims[i] = lhs.shape.Dimensions[batchAxis]
	}

	// M dimension (non-batch, non-contracting from lhs)
	mDim := 0
	for i := 0; i < lhsRank; i++ {
		isBatch := false
		for _, batchAxis := range normalizedLhsBatch {
			if i == batchAxis {
				isBatch = true
				break
			}
		}
		if !isBatch && i != lhsContractAxis {
			mDim = lhs.shape.Dimensions[i]
			break
		}
	}
	outputDims[len(normalizedLhsBatch)] = mDim

	// N dimension (non-batch, non-contracting from rhs)
	nDim := 0
	for i := 0; i < rhsRank; i++ {
		isBatch := false
		for _, batchAxis := range normalizedRhsBatch {
			if i == batchAxis {
				isBatch = true
				break
			}
		}
		if !isBatch && i != rhsContractAxis {
			nDim = rhs.shape.Dimensions[i]
			break
		}
	}
	outputDims[len(normalizedLhsBatch)+1] = nDim

	outputShape := shapes.Make(lhs.shape.DType, outputDims...)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, lhsNode, rhsNode)

	return node, nil
}

//======================================================================================================================
// Conditional Operations ---------------------------------------------------------------------------------------------
//======================================================================================================================

// Where implements the backends.Builder interface.
// Performs element-wise selection: returns onTrue where condition is true, onFalse otherwise.
func (b *Builder) Where(condition, onTrue, onFalse backends.Op) (backends.Op, error) {
	opType := backends.OpTypeWhere
	inputs, err := b.checkOps(opType.String(), condition, onTrue, onFalse)
	if err != nil {
		return nil, err
	}
	condNode, onTrueNode, onFalseNode := inputs[0], inputs[1], inputs[2]

	// Validate condition is Bool
	if condNode.shape.DType != dtypes.Bool {
		return nil, errors.Errorf("Where: condition must have Bool dtype, got %s", condNode.shape.DType)
	}

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.WhereOp(condNode.shape, onTrueNode.shape, onFalseNode.shape)
	if err != nil {
		return nil, err
	}

	// Call the MIL Select operation
	resultValue := b.milBuilder.Select(condNode.milValue, onTrueNode.milValue, onFalseNode.milValue)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, condNode, onTrueNode, onFalseNode)

	return node, nil
}

//======================================================================================================================
// Slice Operations ----------------------------------------------------------------------------------------------------
//======================================================================================================================

// Slice implements the backends.Builder interface.
// Extracts a sub-tensor using start indices and limits.
// GoMLX uses (starts, limits, strides) where limits are exclusive end indices.
func (b *Builder) Slice(x backends.Op, starts, limits, strides []int) (backends.Op, error) {
	opType := backends.OpTypeSlice
	inputs, err := b.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.SliceOp(operand.shape, starts, limits, strides)
	if err != nil {
		return nil, err
	}

	// Convert from int to int64 for CoreML
	begins := make([]int64, len(starts))
	ends := make([]int64, len(limits))
	stridesInt64 := make([]int64, len(strides))
	for i := range starts {
		begins[i] = int64(starts[i])
		ends[i] = int64(limits[i])
		stridesInt64[i] = int64(strides[i])
	}

	// Call the MIL SliceByIndex operation
	resultValue := b.milBuilder.SliceByIndex(operand.milValue, begins, ends, stridesInt64)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

// DynamicSlice implements the backends.Builder interface.
// Extracts a slice from the operand at the startIndices position with the given sliceSizes.
//
// The startIndices are runtime values (not compile-time constants), allowing for dynamic slicing.
// The slice sizes (sliceSizes) must be compile-time constants.
//
// Start indices are clamped to valid ranges:
//
//	adjustedStartIndices[i] = clamp(0, startIndices[i], operand.Dimensions[i] - sliceSizes[i])
func (b *Builder) DynamicSlice(operand backends.Op, startIndices []backends.Op, sliceSizes []int) (backends.Op, error) {
	opType := backends.OpTypeDynamicSlice

	// Validate and cast operand
	operandInputs, err := b.checkOps(opType.String(), operand)
	if err != nil {
		return nil, err
	}
	operandNode := operandInputs[0]

	// Validate and cast startIndices
	if len(startIndices) != operandNode.shape.Rank() {
		return nil, errors.Errorf("DynamicSlice: len(startIndices)=%d must equal operand.Rank()=%d",
			len(startIndices), operandNode.shape.Rank())
	}

	indexNodes, err := b.checkOps(opType.String()+" indices", startIndices...)
	if err != nil {
		return nil, err
	}

	// Validate sliceSizes
	if len(sliceSizes) != operandNode.shape.Rank() {
		return nil, errors.Errorf("DynamicSlice: len(sliceSizes)=%d must equal operand.Rank()=%d",
			len(sliceSizes), operandNode.shape.Rank())
	}

	// Compute output shape
	outputShape := shapes.Make(operandNode.shape.DType, sliceSizes...)

	// CoreML requires startIndices to be concatenated into a single 1D tensor [rank]
	// Strategy: Use Concat to combine all scalar indices into a 1D vector
	indexValues := make([]*model.Value, len(indexNodes))
	for i, node := range indexNodes {
		indexValues[i] = node.milValue
	}

	// Concatenate indices along axis 0
	beginIndices := b.milBuilder.Concat(indexValues, 0)

	// Convert sliceSizes to int64
	sliceSizesInt64 := make([]int64, len(sliceSizes))
	for i, dim := range sliceSizes {
		sliceSizesInt64[i] = int64(dim)
	}

	// Call CoreML's SliceBySize operation (dynamic start, fixed size)
	resultValue := b.milBuilder.SliceBySize(operandNode.milValue, beginIndices, sliceSizesInt64)

	// Create a new node with the result
	// Include all index nodes as dependencies
	allDeps := append([]*Node{operandNode}, indexNodes...)
	node := b.newNode(opType, outputShape, resultValue, allDeps...)

	return node, nil
}

// DynamicUpdateSlice implements the backends.Builder interface.
// Updates the operand with the values given in update, at the position given by startIndices.
//
// The startIndices are runtime values (not compile-time constants), allowing for dynamic updates.
//
// Implementation uses CoreML's scatter_nd operation with dynamically generated indices.
// This implementation supports all ranks and update shapes, but may have performance
// considerations for large updates due to index generation overhead.
func (b *Builder) DynamicUpdateSlice(operand, update backends.Op, startIndices []backends.Op) (backends.Op, error) {
	opType := backends.OpTypeDynamicUpdateSlice

	// Validate and cast inputs
	inputs, err := b.checkOps(opType.String(), operand, update)
	if err != nil {
		return nil, err
	}
	operandNode, updateNode := inputs[0], inputs[1]

	// Validate and cast startIndices
	if len(startIndices) != operandNode.shape.Rank() {
		return nil, errors.Errorf("DynamicUpdateSlice: len(startIndices)=%d must equal operand.Rank()=%d",
			len(startIndices), operandNode.shape.Rank())
	}

	// Check each start index individually
	indexNodes := make([]*Node, len(startIndices))
	for i, idx := range startIndices {
		nodes, err := b.checkOps(opType.String()+" index", idx)
		if err != nil {
			return nil, err
		}
		indexNodes[i] = nodes[0]
	}

	// Validate update shape
	if updateNode.shape.Rank() != operandNode.shape.Rank() {
		return nil, errors.Errorf("DynamicUpdateSlice: update.Rank()=%d must equal operand.Rank()=%d",
			updateNode.shape.Rank(), operandNode.shape.Rank())
	}

	// Validate that update fits within operand
	for i := 0; i < updateNode.shape.Rank(); i++ {
		if updateNode.shape.Dimensions[i] > operandNode.shape.Dimensions[i] {
			return nil, errors.Errorf("DynamicUpdateSlice: update dimension %d (%d) exceeds operand dimension %d (%d)",
				i, updateNode.shape.Dimensions[i], i, operandNode.shape.Dimensions[i])
		}
	}

	// Strategy: Use scatter_nd to update the slice
	// We need to generate indices for each element in the update tensor
	// indices shape: [update.size, rank] where each row is a multi-dimensional index
	//
	// For a 2D example: operand [10, 20], update [3, 4], startIndices [2, 5]
	// We need indices like:
	//   [[2, 5], [2, 6], [2, 7], [2, 8],   <- first row of update
	//    [3, 5], [3, 6], [3, 7], [3, 8],   <- second row
	//    [4, 5], [4, 6], [4, 7], [4, 8]]   <- third row

	// Concatenate startIndices into a single 1D tensor [rank]
	// Ensure each index is at least rank 1 before concatenating
	indexValues := make([]*model.Value, len(indexNodes))
	for i, node := range indexNodes {
		// Reshape scalar indices to [1] for concatenation
		idxShape := node.milValue.Shape()
		if len(idxShape) == 0 {
			indexValues[i] = b.milBuilder.Reshape(node.milValue, []int64{1})
		} else {
			indexValues[i] = node.milValue
		}
	}
	startIndicesConcat := b.milBuilder.Concat(indexValues, 0)

	// Generate indices for scatter_nd
	// We'll use a combination of Range1D and broadcasting to create the index grid
	rank := operandNode.shape.Rank()
	updateDims := updateNode.shape.Dimensions

	// Create range tensors for each dimension of the update
	// Then broadcast and add to startIndices
	var allIndices []*model.Value

	// For each dimension, create a range tensor and broadcast it
	for dim := 0; dim < rank; dim++ {
		// Create range [0, 1, 2, ..., updateDims[dim]-1]
		dimSize := updateDims[dim]
		start := b.milBuilder.Const(fmt.Sprintf("start_%d", dim), model.Int32, []int64{}, []int32{0})
		end := b.milBuilder.Const(fmt.Sprintf("end_%d", dim), model.Int32, []int64{}, []int32{int32(dimSize)})
		step := b.milBuilder.Const(fmt.Sprintf("step_%d", dim), model.Int32, []int64{}, []int32{1})
		dimRange := b.milBuilder.Range1D(start, end, step)

		// Compute the broadcast shape: [1, 1, ..., dimSize, ..., 1]
		// where dimSize is at position dim
		broadcastShape := make([]int64, rank)
		for i := 0; i < rank; i++ {
			broadcastShape[i] = 1
		}
		broadcastShape[dim] = int64(dimSize)

		// Reshape dimRange to broadcast shape
		dimRangeBroadcast := b.milBuilder.Reshape(dimRange, broadcastShape)

		// Tile to match full update shape
		tileCounts := make([]int64, rank)
		for i := 0; i < rank; i++ {
			if i == dim {
				tileCounts[i] = 1
			} else {
				tileCounts[i] = int64(updateDims[i])
			}
		}
		dimRangeTiled := b.milBuilder.Tile(dimRangeBroadcast, tileCounts)

		// Extract the start index for this dimension
		dimStartIdx := b.milBuilder.SliceByIndex(startIndicesConcat,
			[]int64{int64(dim)},
			[]int64{int64(dim + 1)},
			[]int64{1})

		// Reshape to scalar for broadcasting
		dimStartIdxScalar := b.milBuilder.Reshape(dimStartIdx, []int64{})

		// Add start index offset
		dimIndices := b.milBuilder.Add(dimRangeTiled,
			b.milBuilder.Reshape(dimStartIdxScalar, []int64{1}))

		allIndices = append(allIndices, dimIndices)
	}

	// Stack all dimension indices to create the final indices tensor
	// Shape: [updateDims..., rank]
	// First, reshape each dimension's indices to flatten, then stack
	totalUpdateSize := int64(1)
	for _, dim := range updateDims {
		totalUpdateSize *= int64(dim)
	}

	flattenedIndices := make([]*model.Value, rank)
	for i := 0; i < rank; i++ {
		// Reshape to [totalUpdateSize, 1] for stacking
		flattenedIndices[i] = b.milBuilder.Reshape(allIndices[i], []int64{totalUpdateSize, 1})
	}

	// Concatenate along axis 1 to create [totalUpdateSize, rank] indices tensor
	indices := b.milBuilder.Concat(flattenedIndices, 1)

	// Flatten update tensor to [totalUpdateSize]
	updatesFlat := b.milBuilder.Reshape(updateNode.milValue, []int64{totalUpdateSize})

	// Use ScatterND to perform the update
	resultValue := b.milBuilder.ScatterND(operandNode.milValue, indices, updatesFlat, "update")

	// Create output node
	allDeps := append([]*Node{operandNode, updateNode}, indexNodes...)
	node := b.newNode(opType, operandNode.shape, resultValue, allDeps...)

	return node, nil
}

// Gather implements the backends.Builder interface.
// Gathers values from operand using indices along a specified axis.
//
// GoMLX's Gather is XLA-style with many parameters, but CoreML's gather is simpler.
// This implementation handles a subset of cases that map to CoreML's gather operation.
//
// Supported cases:
//   - Simple gather along a single axis where indices are scalar or 1D
//   - indexVectorAxis must be the last dimension of startIndices
//   - collapsedSliceAxes contains the gather axis
//   - sliceSizes are all 1 for the gather axis
//
// For full XLA Gather semantics, consider using the XLA backend.
func (b *Builder) Gather(
	operandOp, startIndicesOp backends.Op,
	indexVectorAxis int,
	offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
	indicesAreSorted bool,
) (backends.Op, error) {
	opType := backends.OpTypeGather
	inputs, err := b.checkOps(opType.String(), operandOp, startIndicesOp)
	if err != nil {
		return nil, err
	}
	operand, startIndices := inputs[0], inputs[1]

	// For simplicity, we currently support only the most common case:
	// - Single axis gather (len(startIndexMap) == 1)
	// - indexVectorAxis is the last dimension of startIndices
	// - The gather axis is in collapsedSliceAxes
	// - sliceSizes[gatherAxis] == 1
	//
	// This maps to CoreML's gather(x, indices, axis) operation.

	// Check if this is a simple gather case
	if len(startIndexMap) != 1 {
		return nil, errors.Errorf("Gather: CoreML backend currently supports only single-axis gather, "+
			"got startIndexMap with %d elements: %v", len(startIndexMap), startIndexMap)
	}

	gatherAxis := startIndexMap[0]

	// Check that the gather axis is collapsed
	isCollapsed := false
	for _, axis := range collapsedSliceAxes {
		if axis == gatherAxis {
			isCollapsed = true
			break
		}
	}
	if !isCollapsed {
		return nil, errors.Errorf("Gather: CoreML backend requires the gather axis (%d) to be in collapsedSliceAxes, "+
			"got collapsedSliceAxes=%v", gatherAxis, collapsedSliceAxes)
	}

	// Check that sliceSizes[gatherAxis] == 1
	if gatherAxis >= len(sliceSizes) || sliceSizes[gatherAxis] != 1 {
		return nil, errors.Errorf("Gather: CoreML backend requires sliceSizes[%d] == 1, got sliceSizes=%v",
			gatherAxis, sliceSizes)
	}

	// Check that indexVectorAxis is the last dimension
	if indexVectorAxis != startIndices.shape.Rank() {
		return nil, errors.Errorf("Gather: CoreML backend requires indexVectorAxis to be the last dimension "+
			"(rank=%d), got indexVectorAxis=%d", startIndices.shape.Rank(), indexVectorAxis)
	}

	// Use shapeinference to compute the output shape
	outputShape, err := shapeinference.Gather(
		operand.shape,
		startIndices.shape,
		indexVectorAxis,
		offsetOutputAxes,
		collapsedSliceAxes,
		startIndexMap,
		sliceSizes,
		indicesAreSorted,
	)
	if err != nil {
		return nil, err
	}

	// Call CoreML's Gather operation
	// CoreML's gather takes (x, indices, axis) where:
	// - x: the operand tensor
	// - indices: the indices tensor
	// - axis: the axis along which to gather
	resultValue := b.milBuilder.Gather(operand.milValue, startIndices.milValue, int64(gatherAxis))

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand, startIndices)

	return node, nil
}

//======================================================================================================================
// Padding Operations --------------------------------------------------------------------------------------------------
//======================================================================================================================

// Pad implements the backends.Builder interface.
// Injects padding on the start, end, or interior (in between each element) of the given operand.
func (b *Builder) Pad(x, fillValue backends.Op, axesConfig ...backends.PadAxis) (backends.Op, error) {
	opType := backends.OpTypePad
	inputs, err := b.checkOps(opType.String(), x, fillValue)
	if err != nil {
		return nil, err
	}
	operand, fill := inputs[0], inputs[1]

	// Validate fillValue is a scalar
	if !fill.shape.IsScalar() {
		return nil, errors.Errorf("Pad: fillValue must be a scalar, got shape %s", fill.shape)
	}

	// Handle scalar input (no padding needed)
	if operand.shape.IsScalar() {
		return x, nil
	}

	rank := operand.shape.Rank()
	if len(axesConfig) > rank {
		return nil, errors.Errorf("Pad: too many axesConfig values: %d > x.Rank()=%d", len(axesConfig), rank)
	}

	// Check for interior padding (not supported by CoreML)
	for i, axisConfig := range axesConfig {
		if axisConfig.Interior != 0 {
			return nil, errors.Errorf("Pad: CoreML backend does not support interior padding, "+
				"got Interior=%d for axis %d", axisConfig.Interior, i)
		}
	}

	// Build padding arrays for CoreML
	padBefore := make([]int64, rank)
	padAfter := make([]int64, rank)
	for i, axisConfig := range axesConfig {
		padBefore[i] = int64(axisConfig.Start)
		padAfter[i] = int64(axisConfig.End)
	}

	// Compute output shape
	// Output dimensions are: inputDim + padBefore + padAfter
	outputDims := make([]int, rank)
	for i := 0; i < rank; i++ {
		outputDims[i] = operand.shape.Dimensions[i] + int(padBefore[i]) + int(padAfter[i])
	}
	outputShape := shapes.Make(operand.shape.DType, outputDims...)

	// Extract constant value from fillValue
	// For simplicity, we assume fillValue is 0 for now
	// TODO: Support non-zero constant values by inspecting the fillValue node
	constantValue := float32(0.0)

	// Call CoreML's Pad operation
	resultValue := b.milBuilder.Pad(operand.milValue, padBefore, padAfter, model.PadConstant, constantValue)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand, fill)

	return node, nil
}

//======================================================================================================================
// Reverse Operations --------------------------------------------------------------------------------------------------
//======================================================================================================================

// Reverse implements the backends.Builder interface.
// Returns x with the values for the given dimensions reversed.
func (b *Builder) Reverse(x backends.Op, axes ...int) (backends.Op, error) {
	opType := backends.OpTypeReverse
	inputs, err := b.checkOps(opType.String(), x)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Validate axes
	rank := operand.shape.Rank()
	for _, axis := range axes {
		if axis < 0 || axis >= rank {
			return nil, errors.Errorf("Reverse: axis %d out of range for rank %d", axis, rank)
		}
	}

	// Output shape is the same as input shape
	outputShape := operand.shape

	// Convert axes to int64 for CoreML
	axesInt64 := make([]int64, len(axes))
	for i, axis := range axes {
		axesInt64[i] = int64(axis)
	}

	// Call CoreML's Reverse operation
	resultValue := b.milBuilder.Reverse(operand.milValue, axesInt64)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

//======================================================================================================================
// Convolution Operations -------------------------------------------------------------------------------------------------
//======================================================================================================================

// ConvGeneral implements the backends.Builder interface.
// Performs a general convolution operation with support for strides, padding, dilations, and groups.
//
// Note: This implementation currently supports the common case where:
//   - Input layout: [batch, channels, spatial...] (NCHW format)
//   - Kernel layout: [outputChannels, inputChannels/groups, spatial...]
//   - Output layout: [batch, channels, spatial...]
//
// For other axis configurations, consider using the XLA backend.
func (b *Builder) ConvGeneral(
	input, kernel backends.Op,
	axes backends.ConvolveAxesConfig,
	strides []int,
	paddings [][2]int,
	inputDilations, kernelDilations []int,
	channelGroupCount, batchGroupCount int,
) (backends.Op, error) {
	opType := backends.OpTypeConvGeneral
	inputs, err := b.checkOps(opType.String(), input, kernel)
	if err != nil {
		return nil, err
	}
	inputNode, kernelNode := inputs[0], inputs[1]

	// Use shapeinference to compute the output shape and validate parameters
	outputShape, err := shapeinference.ConvGeneralOp(
		inputNode.shape, kernelNode.shape, axes,
		strides, paddings, inputDilations, kernelDilations,
		channelGroupCount, batchGroupCount,
	)
	if err != nil {
		return nil, err
	}

	// Check for unsupported features in CoreML
	if batchGroupCount > 1 {
		return nil, errors.Errorf("ConvGeneral: CoreML backend does not support batchGroupCount > 1, got %d", batchGroupCount)
	}

	if len(inputDilations) > 0 {
		for _, d := range inputDilations {
			if d != 1 {
				return nil, errors.Errorf("ConvGeneral: CoreML backend does not support input dilations, got %v", inputDilations)
			}
		}
	}

	// Verify standard axis layout (NCHW for input, OIHW for kernel)
	rank := inputNode.shape.Rank()
	spatialRank := rank - 2

	// Check input axes: expect batch=0, channels=1, spatial=[2,3,...]
	if axes.InputBatch != 0 || axes.InputChannels != 1 {
		return nil, errors.Errorf("ConvGeneral: CoreML backend requires input axes batch=0, channels=1, got batch=%d, channels=%d",
			axes.InputBatch, axes.InputChannels)
	}
	for i, axis := range axes.InputSpatial {
		if axis != i+2 {
			return nil, errors.Errorf("ConvGeneral: CoreML backend requires input spatial axes [2,3,...], got %v",
				axes.InputSpatial)
		}
	}

	// Check kernel axes: expect outputChannels=0, inputChannels=1, spatial=[2,3,...]
	if axes.KernelOutputChannels != 0 || axes.KernelInputChannels != 1 {
		return nil, errors.Errorf("ConvGeneral: CoreML backend requires kernel axes outputChannels=0, inputChannels=1, got outputChannels=%d, inputChannels=%d",
			axes.KernelOutputChannels, axes.KernelInputChannels)
	}
	for i, axis := range axes.KernelSpatial {
		if axis != i+2 {
			return nil, errors.Errorf("ConvGeneral: CoreML backend requires kernel spatial axes [2,3,...], got %v",
				axes.KernelSpatial)
		}
	}

	// Convert parameters to int64 for CoreML
	stridesInt64 := make([]int64, spatialRank)
	dilationsInt64 := make([]int64, spatialRank)
	for i := 0; i < spatialRank; i++ {
		if i < len(strides) {
			stridesInt64[i] = int64(strides[i])
		} else {
			stridesInt64[i] = 1
		}
		if i < len(kernelDilations) {
			dilationsInt64[i] = int64(kernelDilations[i])
		} else {
			dilationsInt64[i] = 1
		}
	}

	// Determine padding type and explicit padding values
	// Note: We always use ConvPadCustom to ensure the pad parameter is set
	// CoreML seems to require it even for zero padding
	var padType model.ConvPadType
	var padBefore, padAfter []int64

	padType = model.ConvPadCustom
	padBefore = make([]int64, spatialRank)
	padAfter = make([]int64, spatialRank)

	if len(paddings) > 0 && !allZeroPadding(paddings) {
		// Custom padding
		for i := 0; i < spatialRank && i < len(paddings); i++ {
			padBefore[i] = int64(paddings[i][0])
			padAfter[i] = int64(paddings[i][1])
		}
	}
	// Otherwise, padBefore and padAfter remain as zero-initialized arrays

	// Call CoreML's Conv operation
	resultValue := b.milBuilder.Conv(
		inputNode.milValue,
		kernelNode.milValue,
		stridesInt64,
		dilationsInt64,
		padType,
		padBefore,
		padAfter,
		int64(channelGroupCount),
	)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, inputNode, kernelNode)

	return node, nil
}

// allZeroPadding checks if all padding values are zero
func allZeroPadding(paddings [][2]int) bool {
	for _, p := range paddings {
		if p[0] != 0 || p[1] != 0 {
			return false
		}
	}
	return true
}

//======================================================================================================================
// Tensor Generation Operations ----------------------------------------------------------------------------------------
//======================================================================================================================

// Iota implements the backends.Builder interface.
// Creates a tensor of the given shape with values along iotaDim being 0, 1, 2, ...
// Other dimensions broadcast the iota values.
//
// Example: Iota(shape=[3,4], iotaDim=1) produces:
//
//	[[0, 1, 2, 3],
//	 [0, 1, 2, 3],
//	 [0, 1, 2, 3]]
func (b *Builder) Iota(shape shapes.Shape, iotaDim int) (backends.Op, error) {
	opType := backends.OpTypeIota

	// Validate iotaDim
	if iotaDim < 0 || iotaDim >= shape.Rank() {
		return nil, errors.Errorf("Iota: iotaDim %d out of range for rank %d", iotaDim, shape.Rank())
	}

	// Strategy:
	// 1. Create a 1D iota using Range1D(0, shape.Dimensions[iotaDim], 1)
	// 2. Reshape to have size 1 in all other dimensions
	// 3. Use Tile to broadcast to the full shape

	iotaSize := shape.Dimensions[iotaDim]

	// Step 1: Create 1D range [0, 1, 2, ..., iotaSize-1]
	// Create scalar constants for Range1D
	startName := fmt.Sprintf("iota_start_%d", b.nextConstID)
	b.nextConstID++
	start := b.milBuilder.Const(startName, model.Int32, []int64{}, []int32{0})

	endName := fmt.Sprintf("iota_end_%d", b.nextConstID)
	b.nextConstID++
	end := b.milBuilder.Const(endName, model.Int32, []int64{}, []int32{int32(iotaSize)})

	stepName := fmt.Sprintf("iota_step_%d", b.nextConstID)
	b.nextConstID++
	step := b.milBuilder.Const(stepName, model.Int32, []int64{}, []int32{1})

	// Generate 1D iota
	range1D := b.milBuilder.Range1D(start, end, step)

	// Convert to the target dtype if needed
	var iotaValue *model.Value
	targetDType, err := gomlxDTypeToMIL(shape.DType)
	if err != nil {
		return nil, errors.Wrap(err, "Iota")
	}
	if targetDType != model.Int32 {
		// Cast to target dtype
		// Note: CoreML doesn't have a direct cast operation in all cases
		// For now, we'll work with Int32 and assume conversion happens naturally
		// In a full implementation, you'd add a cast operation here
		iotaValue = range1D
	} else {
		iotaValue = range1D
	}

	// Step 2: Reshape to broadcast shape
	// Create a shape with 1 in all dimensions except iotaDim
	broadcastShape := make([]int64, shape.Rank())
	for i := range broadcastShape {
		if i == iotaDim {
			broadcastShape[i] = int64(iotaSize)
		} else {
			broadcastShape[i] = 1
		}
	}
	reshaped := b.milBuilder.Reshape(iotaValue, broadcastShape)

	// Step 3: Tile to full shape
	reps := make([]int64, shape.Rank())
	for i := range reps {
		if i == iotaDim {
			reps[i] = 1 // Already the correct size
		} else {
			reps[i] = int64(shape.Dimensions[i])
		}
	}
	tiled := b.milBuilder.Tile(reshaped, reps)

	// Create a new node with the result
	node := b.newNode(opType, shape, tiled)

	return node, nil
}

//======================================================================================================================
// Broadcast Operations ------------------------------------------------------------------------------------------------
//======================================================================================================================

// Broadcast implements the backends.Builder interface.
// Broadcasts a tensor to a new shape by adding dimensions and replicating values.
func (b *Builder) Broadcast(operandOp backends.Op, shape shapes.Shape) (backends.Op, error) {
	opType := backends.OpTypeBroadcast
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Validate that dtypes match
	if operand.shape.DType != shape.DType {
		return nil, errors.Errorf("Broadcast: operand dtype %s does not match output dtype %s",
			operand.shape.DType, shape.DType)
	}

	// Handle scalar operand
	if operand.shape.IsScalar() {
		// For scalar, we need to create a constant filled with the scalar value
		// However, since we don't have access to the constant value here,
		// we'll use ExpandDims + Tile approach
		// First expand to the right rank, then tile to the right shape
		rank := shape.Rank()
		if rank == 0 {
			// Scalar to scalar, no change
			return operandOp, nil
		}

		// Expand scalar to rank with all dimensions = 1
		axes := make([]int64, rank)
		for i := 0; i < rank; i++ {
			axes[i] = int64(i)
		}
		expandedValue := b.milBuilder.ExpandDims(operand.milValue, axes)

		// Now tile to target shape
		reps := make([]int64, rank)
		for i := 0; i < rank; i++ {
			reps[i] = int64(shape.Dimensions[i])
		}
		resultValue := b.milBuilder.Tile(expandedValue, reps)

		// Create a new node with the result
		node := b.newNode(opType, shape, resultValue, operand)
		return node, nil
	}

	// For non-scalar operands, validate rank compatibility
	operandRank := operand.shape.Rank()
	outputRank := shape.Rank()

	if operandRank > outputRank {
		return nil, errors.Errorf("Broadcast: operand rank %d cannot be greater than output rank %d",
			operandRank, outputRank)
	}

	// Step 1: ExpandDims to add leading dimensions
	numNewDims := outputRank - operandRank
	var currentValue *model.Value
	if numNewDims > 0 {
		axes := make([]int64, numNewDims)
		for i := 0; i < numNewDims; i++ {
			axes[i] = int64(i)
		}
		currentValue = b.milBuilder.ExpandDims(operand.milValue, axes)
	} else {
		currentValue = operand.milValue
	}

	// Step 2: Tile to replicate to target shape
	reps := make([]int64, outputRank)
	for i := 0; i < outputRank; i++ {
		if i < numNewDims {
			// New dimension, needs to be tiled from size 1
			reps[i] = int64(shape.Dimensions[i])
		} else {
			// Existing dimension
			operandDim := operand.shape.Dimensions[i-numNewDims]
			targetDim := shape.Dimensions[i]
			if operandDim == 1 {
				// Broadcast this dimension
				reps[i] = int64(targetDim)
			} else if operandDim == targetDim {
				// No broadcast needed
				reps[i] = 1
			} else {
				return nil, errors.Errorf("Broadcast: incompatible shapes - operand dimension %d has size %d, output has size %d",
					i-numNewDims, operandDim, targetDim)
			}
		}
	}

	resultValue := b.milBuilder.Tile(currentValue, reps)

	// Create a new node with the result
	node := b.newNode(opType, shape, resultValue, operand)

	return node, nil
}

// BroadcastInDim implements the backends.Builder interface.
// Broadcasts a tensor to a new shape with explicit dimension mapping.
func (b *Builder) BroadcastInDim(operandOp backends.Op, outputShape shapes.Shape, broadcastDimensions []int) (backends.Op, error) {
	opType := backends.OpTypeBroadcastInDim
	inputs, err := b.checkOps(opType.String(), operandOp)
	if err != nil {
		return nil, err
	}
	operand := inputs[0]

	// Validate using shapeinference
	err = shapeinference.BroadcastInDimOp(operand.shape, outputShape, broadcastDimensions)
	if err != nil {
		return nil, err
	}

	// Handle scalar operand
	if operand.shape.IsScalar() {
		// Expand scalar to output rank with all dimensions = 1
		rank := outputShape.Rank()
		if rank == 0 {
			// Scalar to scalar
			return operandOp, nil
		}

		axes := make([]int64, rank)
		for i := 0; i < rank; i++ {
			axes[i] = int64(i)
		}
		expandedValue := b.milBuilder.ExpandDims(operand.milValue, axes)

		// Tile to target shape
		reps := make([]int64, rank)
		for i := 0; i < rank; i++ {
			reps[i] = int64(outputShape.Dimensions[i])
		}
		resultValue := b.milBuilder.Tile(expandedValue, reps)

		node := b.newNode(opType, outputShape, resultValue, operand)
		return node, nil
	}

	// For non-scalar operands:
	// Strategy: First reshape to insert size-1 dims at the right positions,
	// then tile to expand to target shape

	_ = operand.shape.Rank() // operandRank not needed but kept for documentation
	outputRank := outputShape.Rank()

	// Step 1: Build intermediate shape by inserting size-1 dims
	// We need to figure out which axes in the output are NOT in broadcastDimensions
	broadcastSet := make(map[int]bool)
	for _, axis := range broadcastDimensions {
		broadcastSet[axis] = true
	}

	// Find axes to expand (those not in broadcastDimensions)
	var axesToExpand []int64
	for i := 0; i < outputRank; i++ {
		if !broadcastSet[i] {
			axesToExpand = append(axesToExpand, int64(i))
		}
	}

	// Expand dimensions
	var currentValue *model.Value
	if len(axesToExpand) > 0 {
		currentValue = b.milBuilder.ExpandDims(operand.milValue, axesToExpand)
	} else {
		currentValue = operand.milValue
	}

	// Step 2: Tile to expand to target shape
	reps := make([]int64, outputRank)
	operandIdx := 0
	for i := 0; i < outputRank; i++ {
		if broadcastSet[i] {
			// This dimension comes from operand
			operandDim := operand.shape.Dimensions[operandIdx]
			targetDim := outputShape.Dimensions[i]
			if operandDim == 1 {
				reps[i] = int64(targetDim)
			} else {
				reps[i] = 1
			}
			operandIdx++
		} else {
			// This is a new dimension (size 1 after expand)
			reps[i] = int64(outputShape.Dimensions[i])
		}
	}

	resultValue := b.milBuilder.Tile(currentValue, reps)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, operand)

	return node, nil
}

//======================================================================================================================
// Normalization Operations --------------------------------------------------------------------------------------------
//======================================================================================================================

// BatchNormForInference implements the backends.Builder interface.
// Performs batch normalization for inference mode.
func (b *Builder) BatchNormForInference(input, scale, offset, mean, variance backends.Op, epsilon float32, featureAxis int) (backends.Op, error) {
	opType := backends.OpTypeBatchNormForInference

	// Verify and cast operands
	inputs, err := b.checkOps(opType.String(), input, scale, offset, mean, variance)
	if err != nil {
		return nil, err
	}
	inputNode := inputs[0]
	scaleNode := inputs[1]
	offsetNode := inputs[2]
	meanNode := inputs[3]
	varianceNode := inputs[4]

	// Validate feature axis
	rank := inputNode.shape.Rank()
	if featureAxis < 0 || featureAxis >= rank {
		return nil, errors.Errorf("BatchNormForInference: featureAxis %d out of range for rank %d", featureAxis, rank)
	}

	// CoreML BatchNorm expects the feature axis to be axis 1 (channels)
	// For other layouts, we would need to transpose
	if featureAxis != 1 && rank > 2 {
		return nil, errors.Errorf("BatchNormForInference: CoreML backend requires featureAxis=1 (NCHW layout) for rank > 2, got featureAxis=%d", featureAxis)
	}

	// Output shape is the same as input
	outputShape := inputNode.shape

	// Call CoreML's BatchNorm operation
	// CoreML BatchNorm signature: BatchNorm(x, mean, variance, gamma, beta, epsilon)
	// Note: gamma = scale, beta = offset
	resultValue := b.milBuilder.BatchNorm(
		inputNode.milValue,
		meanNode.milValue,
		varianceNode.milValue,
		scaleNode.milValue,  // gamma = scale
		offsetNode.milValue, // beta = offset
		epsilon,
	)

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, inputs...)

	return node, nil
}

//======================================================================================================================
// Concatenation Operations --------------------------------------------------------------------------------------------
//======================================================================================================================

// Concatenate implements the backends.Builder interface.
// Concatenates multiple tensors along a specified axis.
func (b *Builder) Concatenate(axis int, operands ...backends.Op) (backends.Op, error) {
	opType := backends.OpTypeConcatenate

	// Validate operands
	if len(operands) == 0 {
		return nil, errors.Errorf("Concatenate: requires at least one operand")
	}

	inputs, err := b.checkOps(opType.String(), operands...)
	if err != nil {
		return nil, err
	}

	// Build shape array for shapeinference
	inputShapes := make([]shapes.Shape, len(inputs))
	for i, node := range inputs {
		inputShapes[i] = node.shape
	}

	// Compute output shape using shapeinference
	outputShape, err := shapeinference.ConcatenateOp(inputShapes, axis)
	if err != nil {
		return nil, err
	}

	// Get MIL Values from input nodes
	milValues := make([]*model.Value, len(inputs))
	for i, node := range inputs {
		milValues[i] = node.milValue
	}

	// Call CoreML's Concat operation
	resultValue := b.milBuilder.Concat(milValues, int64(axis))

	// Create a new node with the result
	node := b.newNode(opType, outputShape, resultValue, inputs...)

	return node, nil
}

//======================================================================================================================
// Specialized CoreML Operations ---------------------------------------------------------------------------------------
//======================================================================================================================

// Einsum is a specialized operation for CoreML backend that performs tensor multiplication
// using Einstein summation notation. This is not part of the standard backends.Builder interface
// but is available as a CoreML-specific extension.
//
// CoreML MIL einsum (available in iOS 15+) supports a limited set of equation patterns,
// specifically for multiplying matrices on dimensions -1 and -3, treating other dimensions as batch.
// Broadcasting is supported along batch dimensions.
//
// Supported equation patterns:
//
// Rank 4 inputs:
//   - Equation: "nchw,nwhu->nchu" (and equivalent variations)
//   - Input 1: [B, C, H, W1]
//   - Input 2: [B, W1, H, W2]
//   - Output:  [B, C, H, W2]
//   - Broadcasting: If B or H is 1 in one input, it broadcasts to match the other
//
// Rank 3 inputs:
//   - Equation: "chw,whr->chr" (and equivalent variations)
//   - Input 1: [C, H, W1]
//   - Input 2: [W1, H, W2]
//   - Output:  [C, H, W2]
//   - Broadcasting: If H is 1 in one input, it broadcasts to match the other
//
// Usage:
//
//	// Cast builder to CoreML-specific type to access Einsum
//	coremlBuilder, ok := builder.(*coreml.Builder)
//	if !ok {
//	    return errors.New("backend is not CoreML")
//	}
//	result, err := coremlBuilder.Einsum("nchw,nwhu->nchu", x, y)
//
// equation: Einstein summation notation string (e.g., "nchw,nwhu->nchu")
// operands: Exactly two input tensors (rank 3 or 4)
//
// Returns: Result tensor with shape determined by the equation
func (b *Builder) Einsum(equation string, operands ...backends.Op) (backends.Op, error) {
	if len(operands) != 2 {
		return nil, errors.Errorf("Einsum: requires exactly 2 operands, got %d", len(operands))
	}

	// Since there's no OpTypeEinsum, we use OpTypeInvalid with a special marker
	// This is a specialized operation that doesn't fit the standard OpType enum
	// opType := backends.OpTypeInvalid // Marker for specialized ops

	inputs, err := b.checkOps("Einsum", operands...)
	if err != nil {
		return nil, err
	}

	if len(inputs) != 2 {
		return nil, errors.Errorf("Einsum: requires exactly 2 inputs after validation")
	}

	x := inputs[0]
	y := inputs[1]

	// Validate rank (must be 3 or 4)
	xRank := x.shape.Rank()
	yRank := y.shape.Rank()
	if xRank != yRank {
		return nil, errors.Errorf("Einsum: input ranks must match, got %d and %d", xRank, yRank)
	}
	if xRank != 3 && xRank != 4 {
		return nil, errors.Errorf("Einsum: inputs must be rank 3 or 4, got rank %d", xRank)
	}

	// Validate dtypes match
	if x.shape.DType != y.shape.DType {
		return nil, errors.Errorf("Einsum: input dtypes must match, got %s and %s",
			x.shape.DType, y.shape.DType)
	}

	// Only support fp16 and fp32
	if x.shape.DType != dtypes.Float32 && x.shape.DType != dtypes.Float16 {
		return nil, errors.Errorf("Einsum: only supports fp32 and fp16, got %s", x.shape.DType)
	}

	// Compute output shape based on einsum equation and input shapes
	outputShape := computeEinsumOutputShapeGoMLX(equation, x.shape, y.shape)

	// Get MIL Values from input nodes
	milValues := []*model.Value{x.milValue, y.milValue}

	// Call CoreML's Einsum operation
	resultValue := b.milBuilder.Einsum(equation, milValues)

	// Create a new node with the result
	// We use OpTypeDot as a placeholder since this is similar to batched matrix multiplication
	node := b.newNode(backends.OpTypeDot, outputShape, resultValue, inputs...)

	return node, nil
}

// computeEinsumOutputShapeGoMLX computes the output shape for einsum operation
// using GoMLX shapes.Shape type.
func computeEinsumOutputShapeGoMLX(equation string, xShape, yShape shapes.Shape) shapes.Shape {
	rank := xShape.Rank()
	dimensions := make([]int, rank)

	if rank == 4 {
		// Rank 4: [B, C, H, W1] x [B, W1, H, W2] -> [B, C, H, W2]
		// Broadcast batch dimension
		if xShape.Dimensions[0] == 1 {
			dimensions[0] = yShape.Dimensions[0]
		} else {
			dimensions[0] = xShape.Dimensions[0]
		}
		// C from first input
		dimensions[1] = xShape.Dimensions[1]
		// H with broadcasting
		if xShape.Dimensions[2] == 1 {
			dimensions[2] = yShape.Dimensions[2]
		} else {
			dimensions[2] = xShape.Dimensions[2]
		}
		// W2 from second input
		dimensions[3] = yShape.Dimensions[3]
	} else if rank == 3 {
		// Rank 3: [C, H, W1] x [W1, H, W2] -> [C, H, W2]
		// C from first input
		dimensions[0] = xShape.Dimensions[0]
		// H with broadcasting
		if xShape.Dimensions[1] == 1 {
			dimensions[1] = yShape.Dimensions[1]
		} else {
			dimensions[1] = xShape.Dimensions[1]
		}
		// W2 from second input
		dimensions[2] = yShape.Dimensions[2]
	}

	return shapes.Make(xShape.DType, dimensions...)
}
