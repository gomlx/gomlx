package backends

import (
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// StandardOps lists the bulk of the operations that a backends.Builder must support.
type StandardOps interface {

	// Abs returns the Op that represents the output of the corresponding operation.
	Abs(x Op) (Op, error)

	// Add returns the element-wise sum of the two values.
	// Standard broadcasting rules apply (see documentation).
	Add(lhs, rhs Op) (Op, error)

	// ArgMinMax calculates the "argmin" or "argmax" across an axis of the given input array x.
	//
	// outputDType defines the output of the argmin/argmax, it doesn't need to be the same as the input.
	// It's a form of reduction on the given axis, and that axis goes away.
	// So the rank of the result is one less than the rank of x.
	//
	// If there is a NaN in the slice being examined, it is chosen for ArgMinMax -- this is inline with Jax, TensorFlow, and PyTorch.
	//
	// Examples:
	//
	//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=1, isMin=true) -> {1, 0}  // (it chooses the 0 and the -3)
	//	ArgMinMax(x={{2, 0, 7}, {-3, 4, 2}}, axis=0, isMin=false) -> {0, 1, 0} // (it chooses the 2, 4, and 7)
	ArgMinMax(x Op, axis int, outputDType dtypes.DType, isMin bool) (Op, error)

	// BatchNormForInference implements batch normalization for inference.
	//
	// See details in https://www.tensorflow.org/xla/operation_semantics#batchnorminference.
	//
	// Based on the paper "Batch Normalization: Accelerating Deep Network Training by Reducing
	// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
	BatchNormForInference(operand, scale, offset, mean, variance Op, epsilon float32, featureAxis int) (Op, error)

	// BatchNormForTraining implements batch normalization for training.
	//
	// See details in https://www.tensorflow.org/xla/operation_semantics#batchnormtraining.
	//
	// It returns the normalized tensor, the batchMean, and the batchVariance.
	//
	// Based on the paper "Batch Normalization: Accelerating Deep Network Training by Reducing
	// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
	BatchNormForTraining(
		operand, scale, offset Op,
		epsilon float32,
		featureAxis int,
	) (normalized Op, batchMean Op, batchVariance Op, err error)

	// BatchNormGradient calculates the batch normalization gradients with respect to the input, scale, and offset.
	//
	// See details in https://openxla.org/xla/operation_semantics#batchnormgrad
	//
	// The gradOutput is the adjoint gradient (the "V" in "VJP"), that is, the gradient with respect to the output of the
	// batch normalization.
	//
	// Based on the paper "Batch Normalization: Accelerating Deep Network Training by Reducing
	// Internal Covariate Shift" (Sergey Ioffe, Christian Szegedy), https://arxiv.org/abs/1502.03167.
	BatchNormGradient(
		operand, scale, mean, variance, gradOutput Op,
		epsilon float32,
		featureAxis int,
	) (gradOperand Op, gradScale Op, gradOffset Op, err error)

	// Bitcast performs an elementwise bit-cast operation from a dtype to another dtype.
	//
	// The Bitcast doesn't "convert", rather it just reinterprets the bits from x.DType() to the targetDType.
	//
	// If x.DType() and targetDType use the same number of bytes (targetDType.Size() == x.DType().Size()),
	// the dimensions are not changed, simply the dtype is changed.
	//
	// If targetDType.Size() > x.DType().Size(), it requires x last axis to have a dimension of
	// targetDType.Size() / x.DType().Size(), and the returned shape will trim the last axis.
	//
	// If targetDType.Size() < x.DType().Size(), the returned shape will have an extra axis in the end, with dimension of
	// x.DType().Size() / targetDType.Size().
	//
	// E.g: Bitcast([1]uint32{0xdeadbeef}, dtypes.UInt16) -> [1][2]uint16{{0xbeef, 0xdead}} // Little-endian encoding.
	Bitcast(x Op, targetDType dtypes.DType) (Op, error)

	// BitCount returns the number of bits that are set to one.
	// Also known as Population Count ("Popcnt") or Hamming Weight.
	BitCount(operand Op) (Op, error)

	// BitwiseAnd returns the element-wise bitwise AND operation.
	BitwiseAnd(lhs, rhs Op) (Op, error)

	// BitwiseNot returns the element-wise bitwise AND operation.
	BitwiseNot(x Op) (Op, error)

	// BitwiseOr returns the element-wise bitwise OR operation.
	BitwiseOr(lhs, rhs Op) (Op, error)

	// BitwiseXor returns the element-wise bitwise XOR operator.
	BitwiseXor(lhs, rhs Op) (Op, error)

	// BroadcastInDim broadcasts x to an output with the given shape.
	// broadcastAxes has an output axes value for each x axes (len(broadcastAxes) == x.Shape.Rank()).
	// The i-th axis of x is mapped to the broadcastAxes[i]-th dimension of the output.
	// broadcastAxes must be also increasing: this operation cannot be used to transpose axes, it will only
	// broadcast and introduce new axes in-between.
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
	BroadcastInDim(x Op, outputShape shapes.Shape, broadcastAxes []int) (Op, error)

	// Ceil returns the Op that represents the output of the corresponding operation.
	Ceil(x Op) (Op, error)

	// Clamp returns the element-wise clamping operation.
	//
	// The values max and min can either be a scalar or have the same shape as x.
	Clamp(min, x, max Op) (Op, error)

	// Clz returns element-wise the "count leading zeros" bits of input node x -- for integer values.
	Clz(x Op) (Op, error)

	// Complex returns the complex number taking x0 as the real part and x1 as the imaginary part.
	// The real (x0) and imaginary (x1) must have the same dtype, and they must be either `dtypes.Float32` or
	// `dtypes.Float64`.
	// The output will be either `dtypes.Complex64` or `dtypes.Complex128`, depending on x0 and x1 dtypes.
	// The shapes of `real` or `imaginary` must be the same, or one must be a scalar, in which case
	// the value is broadcast to every other value.
	Complex(lhs, rhs Op) (Op, error)

	// Concatenate operands on the given axis.
	//
	// All axes that are not being concatenated must match dimensions, except on the axes being concatenated.
	// It doesn't work with scalars -- use ExpandAxes.
	// If there is only one operand, it is returned and this is a no-op.
	Concatenate(axis int, operands ...Op) (Op, error)

	// Conj returns the conjugate of a complex number. E.g: Conj(1+3i) = 1-3i
	Conj(x Op) (Op, error)

	// ConvGeneral is a generic Convolution operation with support for:
	// - Arbitrary number of spatial axes.
	// - Arbitrary transposition of axes.
	// - Strides and padding.
	// - Dilations of the input.
	// - Dilations of the kernel, aka. atrous convolution.
	// - Channels grouping (on the input channels).
	// - Batch grouping.
	// Some details in https://www.tensorflow.org/xla/operation_semantics#convwithgeneralpadding_convolution.
	// There operand and filter are called lhs and rhs.
	// (XLA documentation is unfortunately poor, much is guess-work).
	// Also useful, https://arxiv.org/pdf/1603.07285v1.pdf.
	// Note:
	//   - Another common term for "channels" is "features".
	//   - "Kernel" is also commonly called "weights" or "filters".
	ConvGeneral(
		input, kernel Op,
		axes ConvolveAxesConfig,
		strides []int,
		paddings [][2]int,
		inputDilations, kernelDilations []int,
		channelGroupCount, batchGroupCount int,
	) (Op, error)

	// ConvertDType of x to dtype.
	ConvertDType(x Op, dtype dtypes.DType) (Op, error)

	// Cos returns the Op that represents the output of the corresponding operation.
	Cos(x Op) (Op, error)

	// Div returns the element-wise division of the two values.
	// Standard broadcasting rules apply (see documentation).
	Div(lhs, rhs Op) (Op, error)

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
	// In practice, it can be used to perform dot products between vectors, vector/matrix multiplications, or
	// matrix/matrix multiplications.
	Dot(lhs, rhs Op) (Op, error)

	// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
	// for a general vector product -- a generalized "Einsum". Each axis can be:
	//
	//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
	//     must match in lhs and rhs.
	//   - Crossed (default), in which case the output is the combination (concatenation) of the
	//     dimensions.
	//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
	//     those dimensions.
	//
	// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
	// non-contracting/non-batch dimension, and finally the 'rhs' non-contracting/non-batch dimension.
	// It provides the basic means of implementing Einsum.
	DotGeneral(
		lhs Op,
		lhsContractingAxes, lhsBatchAxes []int,
		rhs Op,
		rhsContractingAxes, rhsBatchAxes []int,
	) (Op, error)

	// DynamicSlice extracts a slice from the operand at the startIndices position and the given sliceSizes.
	//
	// - operand: tensor from where to take the slice.
	// - startIndices: scalar tensors, one per axis of operand: len(startIndices) == operand.Rank().
	// - sliceSizes: static values and fixed to keep the shape of the output static.
	//
	// The startIndices are adjusted as follows:
	//
	//	adjustedStartIndices[i] = clamp(0, StartIndices[i], operand.Dimensions[i] - sliceSizes[i])
	//
	// See description in https://openxla.org/xla/operation_semantics#dynamicslice
	DynamicSlice(operand Op, startIndices []Op, sliceDims []int) (Op, error)

	// DynamicUpdateSlice updates the operand with the values given in update, at the position given by startIndices.
	//
	// - operand: original value that to be updated.
	// - update: values to "paste" on top of operand, at position startIndices.
	// - startIndices: scalar tensors, one per axis of operand: len(startIndices) == operand.Rank().
	// - sliceSizes: static values and fixed to keep the shape of the output static.
	//
	// It returns a value with the same shape as the operand, with the values updated.
	//
	// The startIndices are adjusted as follows:
	//
	//	adjustedStartIndices[i] = clamp(0, StartIndices[i], operand.Dimensions[i] - update.Dimensions[i])
	DynamicUpdateSlice(operand, update Op, startIndices []Op) (Op, error)

	// Equal performs element-wise equality check, returns boolean results with the same dimensions as input.
	Equal(lhs, rhs Op) (Op, error)

	// EqualTotalOrder returns the element-wise operation.
	// Standard broadcasting rules apply (see documentation).
	// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
	EqualTotalOrder(lhs, rhs Op) (Op, error)

	// Erf returns the "error function", defined as erf(x) = 2/Pi * \int_{0}^{x}{e^{-t^2}dt}.
	Erf(x Op) (Op, error)

	// Exp returns the Op that represents the output of the corresponding operation.
	Exp(x Op) (Op, error)

	// Expm1 returns the Op that represents the output of the corresponding operation.
	Expm1(x Op) (Op, error)

	// FFT calls the XLA FFT operation, which implements {Forward, Inverse} x {Complex, Real} versions.
	// See documentation in https://www.tensorflow.org/xla/operation_semantics.
	// Underlying, CPU FFT is backed by Eigen's TensorFFT, and GPU FFT uses cuFFT.
	FFT(operand Op, fftType FFTType, fftLength []int) (Op, error)

	// Floor returns the Op that represents the output of the corresponding operation.
	Floor(x Op) (Op, error)

	// Gather is a powerful but cumbersome Gather operation offered by XLA.
	// Full details in https://www.tensorflow.org/xla/operation_semantics#gather or
	// in https://openxla.org/stablehlo/spec#gather (StableHLO also adds batch axes).
	//
	// The output of Gather has the same DType of the operand, from where we are pulling the data.
	//
	// Its output shape will be composed of 2 parts:
	//
	//   - Batch axes: they come from the axes of startIndices, except the "indexVectorAxis" (usually the last)
	//     that is used as the indices into the operand. (*)
	//   - "Offset axes": these are axes that come from the operand, the sizes given by sliceSizes.
	//     Notice that if sliceSizes for an axis is 1, and that axis is present in the collapsedSliceAxes list, this
	//     axis gets omitted in the output.
	//
	// So in general output.Rank() = startIndices.Rank() - 1 + len(offsetAxes).
	//
	// (*) One exception is if indexVectorAxis == startIndices.Rank(), in which case we assume there is an
	// extra implicit axis in startIndices of size 1, in which case output.Rank() = startIndices.Rank() + len(offsetAxes).
	//
	// Arguments:
	//   - operand: the values from where we are gathering. The output DType will follow the operand one.
	//   - startIndices: are the indices we want to gather. The axis pointed by indexVector
	//     lists the indices of the slice to be gathered in the operand array (their values are mapped to the axis
	//     in the operand according to startIndexMap).
	//     All other axes are "batch dimensions" and they will have equivalent axes (same dimensions) in the output.
	//   - indexVectorAxis: which of the axis in startIndices is collected and used as the start index for slices
	//     to be gathered in the operand.
	//     It is typically the last axis of startIndices, so startIndices.Shape.Rank()-1.
	//     There is a special case where indexVectorAxis == startIndices.Rank() in which case we assume there is an
	//     extra virtual axis in startIndices of size 1, in which case output.Rank() = startIndices.Rank() + len(offsetAxes).
	//   - offsetOutputAxes: _output_ axes (not the operand's) that will hold the "offset slices", slices that are not
	//     collapsed. It points in which position (axis) in the output these slices should show up.
	//     The len(offsetOutputAxes) must match the dimension of indexVectorAxis (== startIndices.Dimensions[indexVectorAxis]).
	//     Notice all axes in the operand will either become an "offset axis" in the output,
	//     of optionally collapsed (or "squeezed") in the output, if included in collapsedSliceAxes.
	//     The axes in the output (given in offsetAxes) to the axes in the operand (the axes not present in collapsedSliceAxes) sequentially.
	//     One must have Rank(operand) == len(collapsedSliceAxes) + len(offsetAxes).
	//   - collapsedSliceAxes: _operand_ axes (for which sliceSizes are 1) not to be included in the output.
	//     One must have sliceSizes[collapsedSliceAxes[i]] == 1 for all i.
	//     Also, one must have Rank(operand) == len(collapsedSliceAxes) + len(offsetOutputAxes).
	//   - startIndexMap: this maps which value in startIndices is used for which axis in the operand, select the slice to be gathered.
	//     Notice len(startIndexMap) must match the startIndices.Dimensions[indexVectorAxis].
	//     Also, len(startIndexMap) == len(offsetOutputAxes) -- offsetOutputAxes maps the same axes in the output.
	//     E.g.: if startIndices.shape=(2, 3), indexVectorAxis=1, and operand.rank=4 and startIndexMap=[]int{0, 1, 2},
	//     this means each row of the startIndices will point to the first 3 axes (0,1 and 2) in the operand.
	//     In many cases this is [0, 1, 2, ..., operand.Shape.Rank()-1], that is, each "index vector" fully defines
	//     an element on the operand. In some this is only a prefix of the operand's rank.
	//     For those axes in the operand not explicitly set (so if len(startIndexMap) < operand.Rank()), the corresponding
	//     axis start index is considered to be 0, and one sets the sliceSizes to take the slice one wants (typically the
	//     full slice).
	//   - sliceSizes: a size for each operand's axis, so len(sliceSize) = operand.Rank().
	//     once the start index from where to gather is resolved, this defines how much data in each axis
	//     to gather.
	//     Constraints: sliceSizes[collapsedSliceAxes[i]] == 1, for all i.
	//   - indicesAreSorted: can be set to true if it's guaranteed that startIndices are sorted (in ascending order,
	//     after scattering its values according to start_index_map) by the user. This allows for some optimizations
	//     in some platforms.
	//
	// Out-of-bound (and negative) indices <i> are adjusted with max(min(<i>, axisDimension-1), 0), meaning they
	// are taken from the border of the axes.
	// TODO: Add batch support: operandBatchingAxes and startIndicesBatchingAxes.
	Gather(
		operand, startIndices Op,
		indexVectorAxis int,
		offsetOutputAxes, collapsedSliceAxes, startIndexMap, sliceSizes []int,
		indicesAreSorted bool,
	) (Op, error)

	// GreaterOrEqual performs element-wise comparison, returns boolean results with the same dimensions as input.
	GreaterOrEqual(lhs, rhs Op) (Op, error)

	// GreaterOrEqualTotalOrder returns the element-wise operation.
	// Standard broadcasting rules apply (see documentation).
	// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
	GreaterOrEqualTotalOrder(lhs, rhs Op) (Op, error)

	// GreaterThan performs element-wise comparison, returns boolean results with the same dimensions as input.
	GreaterThan(lhs, rhs Op) (Op, error)

	// GreaterThanTotalOrder returns the element-wise operation.
	// Standard broadcasting rules apply (see documentation).
	// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
	GreaterThanTotalOrder(lhs, rhs Op) (Op, error)

	// Identity returns an Op whose output is the same as its input.
	// It's a no-op that can serve as a place-holder.
	Identity(x Op) (Op, error)

	// Imag returns the imaginary part of a complex number. It returns 0 if the x is a float number.
	Imag(x Op) (Op, error)

	// Iota creates a constant of the given shape with increasing numbers (starting from 0)
	// on the given axis. So Iota([2,2], 1) returns [[0 1][0 1]], while Iota([2,2], 0)
	// returns [[0 0][1 1]].
	Iota(shape shapes.Shape, iotaAxis int) (Op, error)

	// IsFinite tests whether each element of operand is finite, i.e., if it is not positive nor negative infinity, and it is not NaN.
	// It returns the same shape as the input, but with boolean values where each element is true if and only if
	// the corresponding input element is finite.
	IsFinite(x Op) (Op, error)

	// IsNaN tests whether each element of operand is NaN, i.e., if it is not a finite number.
	IsNaN(x Op) (Op, error)

	// LessOrEqual performs element-wise comparison, returns boolean results with the same dimensions as input.
	LessOrEqual(lhs, rhs Op) (Op, error)

	// LessOrEqualTotalOrder returns the element-wise operation.
	// Standard broadcasting rules apply (see documentation).
	// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
	LessOrEqualTotalOrder(lhs, rhs Op) (Op, error)

	// LessThan performs element-wise comparison, returns boolean results with the same dimensions as input.
	LessThan(lhs, rhs Op) (Op, error)

	// LessThanTotalOrder returns the element-wise operation.
	// Standard broadcasting rules apply (see documentation).
	// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
	LessThanTotalOrder(lhs, rhs Op) (Op, error)

	// Log returns the Op that represents the output of the corresponding operation.
	Log(x Op) (Op, error)

	// Log1p returns the expression log(x+1).
	Log1p(x Op) (Op, error)

	// LogicalAnd returns the element-wise logical AND operation.
	LogicalAnd(lhs, rhs Op) (Op, error)

	// LogicalNot returns the Op that represents the output of the corresponding operation.
	LogicalNot(x Op) (Op, error)

	// LogicalOr returns the element-wise logical OR operation.
	LogicalOr(lhs, rhs Op) (Op, error)

	// LogicalXor returns the element-wise logical XOR operator.
	LogicalXor(lhs, rhs Op) (Op, error)

	// Logistic returns the element-wise expression 1/(1+exp(-x)). Also known as the Sigmoid function.
	Logistic(x Op) (Op, error)

	// Max returns the element-wise highest value among the two.
	Max(lhs, rhs Op) (Op, error)

	// Min returns the element-wise smallest value among the two.
	Min(lhs, rhs Op) (Op, error)

	// Mul returns the element-wise multiplication of the two values.
	// Standard broadcasting rules apply (see documentation).
	Mul(lhs, rhs Op) (Op, error)

	// Neg returns the Op that represents the output of the corresponding operation.
	Neg(x Op) (Op, error)

	// NotEqual performs element-wise inequality check, returns boolean results with the same dimensions as input.
	NotEqual(lhs, rhs Op) (Op, error)

	// NotEqualTotalOrder returns the element-wise operation.
	// Standard broadcasting rules apply (see documentation).
	// The "TotalOrder" version of the operation enforces `-NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN`.
	NotEqualTotalOrder(lhs, rhs Op) (Op, error)

	// Pad injects padding on the start, end, or interior (in between each element) of the given operand.
	// There must be at most `operand.Rank()` axesConfig values. Missing PadAxis are assumed to be zeros,
	// that is, no padding for those axes.
	Pad(x, fillValue Op, axesConfig ...PadAxis) (Op, error)

	// Pow returns the Op that represents the output of the corresponding operation.
	Pow(lhs, rhs Op) (Op, error)

	// Real return the real part of a complex number. It returns x if the x is a float number.
	Real(x Op) (Op, error)

	// ReduceBitwiseAnd reduces x over the axes selected, performing a BitwiseAnd on the slices reduced.
	//
	// The returned result rank is decreased by len(axes).
	//
	// If no axes are given, it reduces the full array.
	ReduceBitwiseAnd(x Op, axes ...int) (Op, error)

	// ReduceBitwiseOr reduces x over the axes selected, performing a BitwiseOr on the slices reduced.
	//
	// The returned result rank is decreased by len(axes).
	//
	// If no axes are given, it reduces the full array.
	ReduceBitwiseOr(x Op, axes ...int) (Op, error)

	// ReduceBitwiseXor reduces x over the axes selected, performing a BitwiseXor on the slices reduced.
	//
	// The returned result rank is decreased by len(axes).
	//
	// If no axes are given, it reduces the full array.
	ReduceBitwiseXor(x Op, axes ...int) (Op, error)

	// ReduceLogicalAnd reduces x over the axes selected, performing a LogicalAnd on the slices reduced.
	//
	// The returned result rank is decreased by len(axes).
	//
	// If no axes are given, it reduces the full array.
	ReduceLogicalAnd(x Op, axes ...int) (Op, error)

	// ReduceLogicalOr reduces x over the axes selected, performing a LogicalOr on the slices reduced.
	//
	// The returned result rank is decreased by len(axes).
	//
	// If no axes are given, it reduces the full array.
	ReduceLogicalOr(x Op, axes ...int) (Op, error)

	// ReduceLogicalXor reduces x over the axes selected, performing a LogicalXor on the slices reduced.
	//
	// The returned result rank is decreased by len(axes).
	//
	// If no axes are given, it reduces the full array.
	ReduceLogicalXor(x Op, axes ...int) (Op, error)

	// ReduceMax reduces x over the axes selected, taking the Max value of the slices reduced.
	//
	// The returned result rank is decreased by len(axes).
	//
	// If no axes are given, it reduces the full array.
	ReduceMax(x Op, axes ...int) (Op, error)

	// ReduceMin reduces x over the axes selected, taking the Min value of the slices reduced.
	//
	// The returned result rank is decreased by len(axes).
	//
	// If no axes are given, it reduces the full array.
	ReduceMin(x Op, axes ...int) (Op, error)

	// ReduceProduct reduces x over the axes selected, taking the product of the slices reduced.
	//
	// The returned result rank is decreased by len(axes).
	//
	// If no axes are given, it reduces the full array.
	ReduceProduct(x Op, axes ...int) (Op, error)

	// ReduceSum reduces x over the axes selected, taking the sum of the slices reduced.
	//
	// The returned result rank is decreased by len(axes).
	//
	// If no axes are given, it reduces the full array.
	ReduceSum(x Op, axes ...int) (Op, error)

	// ReduceWindow runs a reduction function of the type given by reductionType,
	// it can be either ReduceMaxNode, ReduceSumNode, or ReduceMultiplyNode.
	//
	// The parameter windowDimensions must be set and have a value for each axis.
	// If strides is nil, it's assumed to be the same as windowDimensions -- that is, the strides jump a window at a time.
	// If baseDilations, windowDilations are nil, they are assumed to be 1 (no dilation).
	// If paddings is nil, they are assumed to be 0.
	ReduceWindow(
		x Op,
		reductionType ReduceOpType,
		windowDimensions, strides, baseDilations, windowDilations []int,
		paddings [][2]int,
	) (Op, error)

	// Rem returns the remainder operation, also known as modulo (or Mod for short).
	// Notice despite the name XLA implements Mod not IEEE754 Remainder operation.
	Rem(lhs, rhs Op) (Op, error)

	// Reshape reshapes x to the new dimensions.
	// Total size cannot change, it's just a "reinterpretation" of the same flat data.
	// The dtype remains the same, see ConvertDType to actually convert the values.
	Reshape(x Op, dimensions ...int) (Op, error)

	// Reverse returns x with the values for the given dimensions reversed, that is,
	// the value indexed at `i` will be swapped with the value at indexed `(dimension_size - 1 - i)`.
	// The shape remains the same.
	Reverse(x Op, axes ...int) (Op, error)

	// RNGBitGenerator generates the given shape filled with random bits.
	//
	// It takes as input a state (usually [3]uint64) and returns the updated state and the generated values (with random bits).
	//
	// Currently, the backend only supports the Philox algorithm. See https://dl.acm.org/doi/10.1145/2063384.2063405
	RNGBitGenerator(state Op, shape shapes.Shape) (newState Op, values Op, err error)

	// Round returns the Op that represents the output of the corresponding operation.
	// This operation rounds to the nearest even.
	Round(x Op) (Op, error)

	// Rsqrt returns the element-wise reciprocal of square root operation 1/sqrt(x).
	Rsqrt(x Op) (Op, error)

	// ScatterMax scatter values from updates pointed by scatterIndices to operand, by taking the Max.
	ScatterMax(
		operand, scatterIndices, updates Op,
		indexVectorAxis int,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
		indicesAreSorted, uniqueIndices bool,
	) (Op, error)

	// ScatterMin scatter values from updates pointed by scatterIndices to operand, by taking the Min.
	ScatterMin(
		operand, scatterIndices, updates Op,
		indexVectorAxis int,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
		indicesAreSorted, uniqueIndices bool,
	) (Op, error)

	// ScatterSum values from updates pointed by scatterIndices to operand.
	ScatterSum(
		operand, scatterIndices, updates Op,
		indexVectorAxis int,
		updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int,
		indicesAreSorted, uniqueIndices bool,
	) (Op, error)

	// SelectAndScatterMax runs windows (similar to ReduceWindow) over the operand, selects values to update the output (like ScatterAdd)
	// It selects the values in the window such that it works as reverse for a PoolMax operation.
	// See details in https://openxla.org/xla/operation_semantics#selectandscatter
	SelectAndScatterMax(operand, source Op, windowDimensions, windowStrides []int, paddings [][2]int) (Op, error)

	// SelectAndScatterMin runs windows (similar to ReduceWindow) over the operand, selects values to update the output (like ScatterAdd)
	// It selects the values in the window such that it works as reverse for a PoolMin operation.
	// See details in https://openxla.org/xla/operation_semantics#selectandscatter
	SelectAndScatterMin(operand, source Op, windowDimensions, windowStrides []int, paddings [][2]int) (Op, error)

	// ShiftLeft n bits. It implicitly preserves the sign bit if there is no overflow. So ShiftLeft(-1, 1) = -2.
	ShiftLeft(lhs, rhs Op) (Op, error)

	// ShiftRightArithmetic shifts right by n bits, preserving the sign bit. So ShiftRight(-2, 1) = -1.
	ShiftRightArithmetic(lhs, rhs Op) (Op, error)

	// ShiftRightLogical shifts right by n bits, destroying the sign bit.
	ShiftRightLogical(lhs, rhs Op) (Op, error)

	// Sign returns element-wise +1, +/-0 or -1 depending on the sign of x. It returns NaN if the input is NaN.
	Sign(x Op) (Op, error)

	// Sin returns the Op that represents the output of the corresponding operation.
	Sin(x Op) (Op, error)

	// Slice extracts a subarray from the input array.
	// The subarray is of the same rank as the input and contains the values inside a bounding box within the input array
	// where the dimensions and indices of the bounding box are given as arguments to the slice operation.
	// The strides set the input stride of the slice in each axis and must be >= 1.
	// It is optional, and if missing, it is assumed to be 1 for every dimension.
	// Examples:
	// 	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={4}, strides=nil) -> {2, 3}
	// 	Slice(x={0, 1, 2, 3, 4}, starts={2}, limits={5}, strides={2}) -> {2, 4}
	Slice(x Op, starts, limits, strides []int) (Op, error)

	// Sqrt returns the Op that represents the output of the corresponding operation.
	Sqrt(x Op) (Op, error)

	// Sub returns the element-wise subtraction of the two values.
	// Standard broadcasting rules apply (see documentation).
	Sub(lhs, rhs Op) (Op, error)

	// Tanh returns the Op that represents the output of the corresponding operation.
	Tanh(x Op) (Op, error)

	// Transpose axes of x.
	// There should be one value in permutations for each axis in x.
	// The output will have: output.Shape.Dimension[ii] = x.Shape.Dimension[permutations[i]].
	Transpose(x Op, permutation ...int) (Op, error)

	// Where takes element-wise values from onTrue or onFalse depending on the value of the condition (must be boolean).
	//
	// The condition must be boolean, and onTrue and onFalse must have the same dtype.
	//
	// If either condition, onTrue or onFalse is a scalar, it will be broadcasted to the shape of the other operands.
	Where(condition, onTrue, onFalse Op) (Op, error)
}
