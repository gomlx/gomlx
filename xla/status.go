/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package xla

// This file implements a wrapper over C++'s Status and StatusOr objects.

// #include <string.h>
// #include <gomlx/status.h>
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"
)

// Status is a wrapper for `xla::Status`.
type Status struct {
	cStatusPtr *C.XlaStatus
}

// Finalize implements Finalizer.
func (s *Status) Finalize() {
	if s == nil || s.cStatusPtr == nil {
		return
	}
	defer runtime.KeepAlive(s)
	C.free(unsafe.Pointer(s.cStatusPtr))
	s.cStatusPtr = nil
}

// IsNil returns whether either the Client object is nil or the contained C pointer.
func (s *Status) IsNil() bool {
	return s == nil || s.cStatusPtr == nil
}

// UnsafeCPtr returns the underlying C pointer converted to unsafe.Pointer.
func (s *Status) UnsafeCPtr() unsafe.Pointer {
	return unsafe.Pointer(s.cStatusPtr)
}

// NewStatus creates a Status object that owns the underlying C.XlaStatus.
func NewStatus(unsafeStatus unsafe.Pointer) *Status {
	s := &Status{cStatusPtr: (*C.XlaStatus)(unsafeStatus)}
	RegisterFinalizer(s)
	return s
}

func (s *Status) Ok() bool {
	if s.IsNil() {
		return true
	}
	return bool(C.XlaStatusOk(s.UnsafeCPtr()))
}

func (s *Status) ErrorMessage() string {
	if s.Ok() {
		return ""
	}
	return StrFree(C.XlaStatusErrorMessage(s.UnsafeCPtr()))
}

func (s *Status) Error() error {
	if s.Ok() {
		return nil
	}
	return fmt.Errorf("C++ error %s, (%d): %s", s.Code(), s.Code(), s.ErrorMessage())
}

func (s *Status) Code() ErrorCode {
	if s.Ok() {
		return OK
	}
	return ErrorCode(C.XlaStatusCode(s.UnsafeCPtr()))
}

// UnsafePointerOrError converts a StatusOr structure to either an unsafe.Pointer with the data
// or the Status converted to an error message and then freed.
func UnsafePointerOrError(s C.StatusOr) (unsafe.Pointer, error) {
	if s.status != nil {
		status := NewStatus(s.status)
		err := status.Error()
		status.Finalize() // No longer needed, free it immediately.
		return nil, err
	}
	return s.value, nil
}

// PointerOrError converts a StatusOr structure to either a pointer to T with the data
// or the Status converted to an error message and then freed.
func PointerOrError[T any](s C.StatusOr) (t *T, err error) {
	var ptr unsafe.Pointer
	ptr, err = UnsafePointerOrError(s)
	if err != nil {
		return
	}
	t = (*T)(ptr)
	return
}

// ErrorFromStatus converts a *C.XlaStatus returned to an error or nil if there were no
// errors or if status == nil. It also frees the returned *C.XlaStatus.
func ErrorFromStatus(status *C.XlaStatus) (err error) {
	if status == nil {
		return // no error
	}
	s := NewStatus(unsafe.Pointer(status))
	if !s.Ok() {
		err = s.Error()
	}
	s.Finalize() // No longer needed, free it immediately.
	return
}
