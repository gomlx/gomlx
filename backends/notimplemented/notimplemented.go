// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package notimplemented implements a backends.Builder interface that throws a "Not implemented"
// exception to all operations.
//
// This can help bootstrap any backend implementation.
package notimplemented

import (
	"fmt"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// NotImplementedError is returned by every method.
//
// It doesn't contain a stack, attach a stack to with with errors.Wrapf(ErrNotImplemented, "...") when using it.
var NotImplementedError = backends.ErrNotImplemented

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

	// mainFn is the main function, lazily created.
	mainFn *Function
}

var _ backends.Builder = Builder{}

//go:generate go run ../../internal/cmd/notimplemented_generator

func (b Builder) Name() string {
	return "Dummy \"not implemented\" backend, please override this method"
}

func (b Builder) Main() backends.Function {
	if b.mainFn == nil {
		b.mainFn = &Function{ErrFn: b.ErrFn}
	}
	return b.mainFn
}

func (b Builder) NewFunction(name string) (backends.Function, error) {
	return &Function{ErrFn: b.ErrFn}, nil
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

func (b Builder) Compile() (backends.Executable, error) {
	return nil, errors.Wrapf(NotImplementedError, "in Compile()")
}

func (b Builder) OpShape(op backends.Value) (shapes.Shape, error) {
	return shapes.Invalid(), errors.Wrapf(NotImplementedError, "in OpShape()")
}
