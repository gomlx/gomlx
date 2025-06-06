/***** File generated by ./internal/cmd/backends_generator, based on github.com/gomlx/gopjrt. Don't edit it directly. *****/

package backends

// OpType is an enum of all generic operations that can be supported by a Backend.Builder.
//
// Notice: nothing precludes a specialized backend Builder to support other ops not included here.
// It requires some careful casting of interfaces by the caller (presumably in package
// github.com/gomlx/gomlx/graph) and fallback to backends that don't support the specialized op.
type OpType int

const (
	OpTypeInvalid OpType = iota
	OpTypeParameter
	OpTypeConstant
	OpTypeIdentity
	OpTypeReduceWindow
	OpTypeRngBitGenerator
	OpTypeBatchNormForInference
	OpTypeBatchNormForTraining
	OpTypeBatchNormGradient
	OpTypeBitCount

	OpTypeAbs
	OpTypeAdd
	OpTypeArgMinMax
	OpTypeBitcast
	OpTypeBitwiseAnd
	OpTypeBitwiseNot
	OpTypeBitwiseOr
	OpTypeBitwiseXor
	OpTypeBroadcast
	OpTypeBroadcastInDim
	OpTypeCeil
	OpTypeClz
	OpTypeComplex
	OpTypeConcatenate
	OpTypeConj
	OpTypeConvGeneralDilated
	OpTypeConvertDType
	OpTypeCos
	OpTypeDiv
	OpTypeDot
	OpTypeDotGeneral
	OpTypeDynamicSlice
	OpTypeDynamicUpdateSlice
	OpTypeEqual
	OpTypeEqualTotalOrder
	OpTypeErf
	OpTypeExp
	OpTypeExpm1
	OpTypeFFT
	OpTypeFloor
	OpTypeGather
	OpTypeGreaterOrEqual
	OpTypeGreaterOrEqualTotalOrder
	OpTypeGreaterThan
	OpTypeGreaterThanTotalOrder
	OpTypeImag
	OpTypeIota
	OpTypeIsFinite
	OpTypeLessOrEqual
	OpTypeLessOrEqualTotalOrder
	OpTypeLessThan
	OpTypeLessThanTotalOrder
	OpTypeLog
	OpTypeLog1p
	OpTypeLogicalAnd
	OpTypeLogicalNot
	OpTypeLogicalOr
	OpTypeLogicalXor
	OpTypeLogistic
	OpTypeMax
	OpTypeMin
	OpTypeMul
	OpTypeNeg
	OpTypeNotEqual
	OpTypeNotEqualTotalOrder
	OpTypePad
	OpTypePow
	OpTypeReal
	OpTypeReduceBitwiseAnd
	OpTypeReduceBitwiseOr
	OpTypeReduceBitwiseXor
	OpTypeReduceLogicalAnd
	OpTypeReduceLogicalOr
	OpTypeReduceLogicalXor
	OpTypeReduceMax
	OpTypeReduceMin
	OpTypeReduceProduct
	OpTypeReduceSum
	OpTypeRem
	OpTypeReshape
	OpTypeReverse
	OpTypeRound
	OpTypeRsqrt
	OpTypeScatterMax
	OpTypeScatterMin
	OpTypeScatterSum
	OpTypeSelectAndScatterMax
	OpTypeSelectAndScatterMin
	OpTypeSelectAndScatterSum
	OpTypeShiftLeft
	OpTypeShiftRightArithmetic
	OpTypeShiftRightLogical
	OpTypeSign
	OpTypeSin
	OpTypeSlice
	OpTypeSqrt
	OpTypeSub
	OpTypeTanh
	OpTypeTranspose
	OpTypeWhere

	// OpTypeLast should always be kept the last, it is used as a counter/marker for OpType.
	OpTypeLast
)
