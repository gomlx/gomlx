// Package notimplemented implements a backends.Builder interface that throws a "Not implemented"
// exception to all operations.
//
// This can help bootstrap any backend implementation.
package notimplemented

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// NotImplementedError is returned by every method.
var NotImplementedError = fmt.Errorf("not implemented")

// Builder implements backends.Builder and returns the NotImplementedError wrapped with the stack-trace,
// the operation name, and the custom message Builder.ErrMessage for every operation.
type Builder struct {
	// ErrFn is called to generate the error returned, if not nil. Otherwise NotImplementedError is returned directly.
	//
	// For non-ops methods (like Builder.Name and Builder.Compile) you will have to override them.
	ErrFn func(op backends.OpType) error
}

var _ backends.Builder = Builder{}

//go:generate go run ../../internal/cmd/notimplemented_generator

// baseErrFn returns the error corresponding to the op.
// It falls back to Builder.ErrFn if it is defined.
func (b Builder) baseErrFn(op backends.OpType) error {
	if b.ErrFn == nil {
		return NotImplementedError
	}
	return b.ErrFn(op)
}

func (b Builder) Name() string {
	return "Dummy \"not implemented\" backend, please override this method"
}

func (b Builder) Compile(outputs ...backends.Op) (backends.Executable, error) {
	return nil, errors.Wrapf(NotImplementedError, "in Compile()")
}

func (b Builder) OpShape(op backends.Op) (shapes.Shape, error) {
	return shapes.Invalid(), errors.Wrapf(NotImplementedError, "in OpShape()")
}

func (b Builder) Parameter(name string, shape shapes.Shape) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeParameter)
}

func (b Builder) Constant(flat any, dims ...int) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeConstant)
}

func (b Builder) Identity(x backends.Op) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeIdentity)
}

func (b Builder) ReduceWindow(x backends.Op, reductionType backends.ReduceOpType, windowDimensions, strides, baseDilations, windowDilations []int, paddings [][2]int) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeReduceWindow)
}

func (b Builder) RngBitGenerator(state backends.Op, shape shapes.Shape) (newState, values backends.Op, err error) {
	return nil, nil, b.baseErrFn(backends.OpTypeRngBitGenerator)
}

func (b Builder) BatchNormForInference(operand, scale, offset, mean, variance backends.Op, epsilon float32, axis int) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeBatchNormForInference)
}

func (b Builder) BatchNormForTraining(operand, scale, offset backends.Op, epsilon float32, axis int) (normalized, batchMean, batchVariance backends.Op, err error) {
	return nil, nil, nil, b.baseErrFn(backends.OpTypeBatchNormForTraining)
}

func (b Builder) BatchNormGradient(operand, scale, mean, variance, gradOutput backends.Op, epsilon float32, axis int) (gradOperand, gradScale, gradOffset backends.Op, err error) {
	return nil, nil, nil, b.baseErrFn(backends.OpTypeBatchNormGradient)
}
