package backends

// OpType is an enum of all generic operations that can be supported by a Backend.Builder.
//
// Notice: nothing precludes a specialized backend Builder to support other ops not included here.
// It requires some careful casting of interfaces by the caller (presumably in package
// github.com/gomlx/gomlx/pkg/core/graph) and fallback to backends that don't support the specialized op.
type OpType int

//go:generate go tool enumer -type=OpType -trimprefix=OpType -output=gen_optype_enumer.go optype.go

const (
	OpTypeInvalid OpType = iota
	OpTypeParameter
	OpTypeConstant
	OpTypeIdentity
	OpTypeReduceWindow
	OpTypeRNGBitGenerator
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
	OpTypeClamp
	OpTypeCeil
	OpTypeClz
	OpTypeComplex
	OpTypeConcatenate
	OpTypeConj
	OpTypeConvGeneral
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
	OpTypeIsNaN
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

	// Collective (distributed across devices) operations

	OpTypeAllReduce
	OpTypeCollectiveBroadcast
	OpTypeAllGather

	// OpTypeLast should always be kept the last, it is used as a counter/marker for OpType.
	OpTypeLast
)
