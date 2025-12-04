// Package notimplemented implements a backends.Builder interface that throws a "Not implemented"
// exception to all operations.
//
// This can help bootstrap any backend implementation.
package notimplemented

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// NotImplementedError is returned by every method.
var NotImplementedError = fmt.Errorf("not implemented")

// Backend is a dummy backend that can be imported to create mock backends.
type Backend struct{}

var _ backends.Backend = &Backend{}

// Name returns the short name of the backend.
func (b *Backend) Name() string {
	return "notimplemented"
}

// String returns the same as Name.
func (b *Backend) String() string {
	return b.Name()
}

// Description is a longer description of the Backend.
func (b *Backend) Description() string {
	return "Not Implemented Backend (mock backend for testing)"
}

// NumDevices returns 1 as the number of devices available.
func (b *Backend) NumDevices() int {
	return 1
}

// DeviceDescription returns a description of the device.
func (b *Backend) DeviceDescription(deviceNum backends.DeviceNum) string {
	return fmt.Sprintf("Not Implemented Device %d", deviceNum)
}

// Capabilities returns empty capabilities.
func (b *Backend) Capabilities() backends.Capabilities {
	return backends.Capabilities{
		Operations: make(map[backends.OpType]bool),
		DTypes:     make(map[dtypes.DType]bool),
	}
}

// Builder creates a new builder.
func (b *Backend) Builder(name string) backends.Builder {
	return Builder{}
}

// BufferFinalize returns NotImplementedError.
func (b *Backend) BufferFinalize(buffer backends.Buffer) error {
	return errors.Wrapf(NotImplementedError, "in BufferFinalize()")
}

// BufferShape returns NotImplementedError.
func (b *Backend) BufferShape(buffer backends.Buffer) (shapes.Shape, error) {
	return shapes.Invalid(), errors.Wrapf(NotImplementedError, "in BufferShape()")
}

// BufferDeviceNum returns NotImplementedError.
func (b *Backend) BufferDeviceNum(buffer backends.Buffer) (backends.DeviceNum, error) {
	return 0, errors.Wrapf(NotImplementedError, "in BufferDeviceNum()")
}

// BufferToFlatData returns NotImplementedError.
func (b *Backend) BufferToFlatData(buffer backends.Buffer, flat any) error {
	return errors.Wrapf(NotImplementedError, "in BufferToFlatData()")
}

// BufferFromFlatData returns NotImplementedError.
func (b *Backend) BufferFromFlatData(
	deviceNum backends.DeviceNum,
	flat any,
	shape shapes.Shape,
) (backends.Buffer, error) {
	return nil, errors.Wrapf(NotImplementedError, "in BufferFromFlatData()")
}

// HasSharedBuffers returns false.
func (b *Backend) HasSharedBuffers() bool {
	return false
}

// NewSharedBuffer panics as shared buffers are not supported.
func (b *Backend) NewSharedBuffer(
	deviceNum backends.DeviceNum,
	shape shapes.Shape,
) (buffer backends.Buffer, flat any, err error) {
	return nil, nil, errors.Wrapf(NotImplementedError, "in NewSharedBuffer()")
}

// BufferData returns NotImplementedError.
func (b *Backend) BufferData(buffer backends.Buffer) (flat any, err error) {
	return nil, errors.Wrapf(NotImplementedError, "in BufferData()")
}

// BufferCopyToDevice returns NotImplementedError.
func (b *Backend) BufferCopyToDevice(
	source backends.Buffer,
	deviceNum backends.DeviceNum,
) (bufferOnDevice backends.Buffer, err error) {
	return nil, errors.Wrapf(NotImplementedError, "in BufferCopyToDevice()")
}

// Finalize does nothing for this dummy backend.
func (b *Backend) Finalize() {
	// No-op for dummy backend
}

// IsFinalized always returns false for this dummy backend.
func (b *Backend) IsFinalized() bool {
	return false
}

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

func (b Builder) DistributedSPMD(numDevices int) error {
	return errors.Wrapf(NotImplementedError, "in DistributedSPMD()")
}

func (b Builder) DistributedAutoSharding(meshes ...backends.Mesh) error {
	return errors.Wrapf(NotImplementedError, "in DistributedAutoSharding()")
}

// DeviceAssignment returns nil if it's an assignment to device #0.
// Otherwise, it returns a non-implemented error.
func (b Builder) DeviceAssignment(devices ...backends.DeviceNum) error {
	if len(devices) != 1 && devices[0] != backends.DeviceNum(0) {
		return errors.Wrapf(NotImplementedError, "in DeviceAssignment()")
	}
	return nil
}

func (b Builder) Compile(_ []backends.Op, _ []*backends.ShardingSpec) (backends.Executable, error) {
	return nil, errors.Wrapf(NotImplementedError, "in Compile()")
}

func (b Builder) OpShape(op backends.Op) (shapes.Shape, error) {
	return shapes.Invalid(), errors.Wrapf(NotImplementedError, "in OpShape()")
}

func (b Builder) Parameter(name string, shape shapes.Shape, spec *backends.ShardingSpec) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeParameter)
}

func (b Builder) Constant(flat any, dims ...int) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeConstant)
}

func (b Builder) Identity(x backends.Op) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeIdentity)
}

func (b Builder) ReduceWindow(x backends.Op, reductionType backends.ReduceOpType,
	windowDimensions, strides, baseDilations, windowDilations []int, paddings [][2]int) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeReduceWindow)
}

func (b Builder) RNGBitGenerator(state backends.Op, shape shapes.Shape) (newState, values backends.Op, err error) {
	return nil, nil, b.baseErrFn(backends.OpTypeRNGBitGenerator)
}

func (b Builder) BatchNormForInference(operand, scale, offset, mean, variance backends.Op, epsilon float32, axis int) (
	backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeBatchNormForInference)
}

func (b Builder) BatchNormForTraining(operand, scale, offset backends.Op, epsilon float32, axis int) (
	normalized, batchMean, batchVariance backends.Op, err error) {
	return nil, nil, nil, b.baseErrFn(backends.OpTypeBatchNormForTraining)
}

func (b Builder) BatchNormGradient(operand, scale, mean, variance, gradOutput backends.Op, epsilon float32, axis int) (
	gradOperand, gradScale, gradOffset backends.Op, err error) {
	return nil, nil, nil, b.baseErrFn(backends.OpTypeBatchNormGradient)
}

func (b Builder) AllReduce(inputs []backends.Op, reduceOp backends.ReduceOpType, replicaGroups [][]int) (
	[]backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeAllReduce)
}
