package notimplemented

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Function implements backends.Function and returns NotImplementedError for every operation.
type Function struct {
	// ErrFn is called to generate the error returned, if not nil.
	// Otherwise NotImplementedError is returned directly.
	ErrFn func(op backends.OpType) error
}

var _ backends.Function = Function{}

// baseErrFn returns the error corresponding to the op.
// It falls back to Function.ErrFn if it is defined.
func (f Function) baseErrFn(op backends.OpType) error {
	if f.ErrFn == nil {
		return NotImplementedError
	}
	return f.ErrFn(op)
}

func (f Function) Parameter(name string, shape shapes.Shape, spec *backends.ShardingSpec) (backends.Op, error) {
	return nil, f.baseErrFn(backends.OpTypeParameter)
}

func (f Function) Constant(flat any, dims ...int) (backends.Op, error) {
	return nil, f.baseErrFn(backends.OpTypeConstant)
}

func (f Function) Return(outputs []backends.Op, shardings []*backends.ShardingSpec) error {
	return errors.Wrapf(NotImplementedError, "in Return()")
}

func (f Function) Identity(x backends.Op) (backends.Op, error) {
	return nil, f.baseErrFn(backends.OpTypeIdentity)
}

func (f Function) ReduceWindow(x backends.Op, reductionType backends.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int, paddings [][2]int) (backends.Op, error) {
	return nil, f.baseErrFn(backends.OpTypeReduceWindow)
}

func (f Function) RNGBitGenerator(state backends.Op, shape shapes.Shape) (newState, values backends.Op, err error) {
	return nil, nil, f.baseErrFn(backends.OpTypeRNGBitGenerator)
}

func (f Function) BatchNormForInference(operand, scale, offset, mean, variance backends.Op, epsilon float32, axis int) (
	backends.Op, error) {
	return nil, f.baseErrFn(backends.OpTypeBatchNormForInference)
}

func (f Function) BatchNormForTraining(operand, scale, offset backends.Op, epsilon float32, axis int) (
	normalized, batchMean, batchVariance backends.Op, err error) {
	return nil, nil, nil, f.baseErrFn(backends.OpTypeBatchNormForTraining)
}

func (f Function) BatchNormGradient(operand, scale, mean, variance, gradOutput backends.Op, epsilon float32, axis int) (
	gradOperand, gradScale, gradOffset backends.Op, err error) {
	return nil, nil, nil, f.baseErrFn(backends.OpTypeBatchNormGradient)
}

func (f Function) AllReduce(inputs []backends.Op, reduceOp backends.ReduceOpType, replicaGroups [][]int) (
	[]backends.Op, error) {
	return nil, f.baseErrFn(backends.OpTypeAllReduce)
}
