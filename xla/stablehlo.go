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

// #include "gomlx/aot_compile.h"
// #include "gomlx/client.h"
// #include "gomlx/computation.h"
import "C"

import (
	"github.com/pkg/errors"
	"os"
	"runtime"
)

// StableHLO is a wrapper for the C++ `StableHLOHolder`.
type StableHLO struct {
	cPtr *C.StableHLOHolder
}

// NewStableHLO creates the wrapper.
func NewStableHLO(cPtr *C.StableHLOHolder) *StableHLO {
	shlo := &StableHLO{cPtr: cPtr}
	RegisterFinalizer(shlo)
	return shlo
}

// Finalize implements Finalizer.
func (shlo *StableHLO) Finalize() {
	defer runtime.KeepAlive(shlo)
	if shlo.IsNil() {
		return
	}
	C.DeleteStableHLOHolder(shlo.cPtr)
	shlo.cPtr = nil
}

// IsNil checks whether the computation is nil or it's C underlying object.
func (shlo *StableHLO) IsNil() bool {
	return shlo == nil || shlo.cPtr == nil
}

// ToStableHLO returning a holder of the C++ object representing the StableHLO.
func (comp *Computation) ToStableHLO() (*StableHLO, error) {
	if comp.IsNil() || comp.firstError != nil {
		return nil, errors.New("Computation graph is nil!?")
	}
	statusOr := C.ConvertComputationToStableHLO(comp.cCompPtr)
	ptr, err := UnsafePointerOrError(statusOr)
	if err != nil {
		return nil, errors.Wrapf(err, "failed conversion in Computation.ToStableHLO")
	}
	return NewStableHLO((*C.StableHLOHolder)(ptr)), nil
}

// String generates a human-readable version of the StableHLO.
func (shlo *StableHLO) String() string {
	if shlo.IsNil() {
		return "<nil>"
	}
	return StrFree(C.StableHLOToString(shlo.cPtr))
}

// StableHLOCurrentVersion returns the current version for the StableHLO library.
func StableHLOCurrentVersion() string {
	return StrFree(C.StableHLOCurrentVersion())
}

// SerializeWithVersion to bytecode that can presumably be used by PjRT and IREE, as well as
// embedded in one of the TensorFlow SavedModel formats.(??)
//
// It serializes to the given file descriptor, presumable a file opened for writing.
//
// For version, the usual is to use the value returned by StableHLOCurrentVersion.
func (shlo *StableHLO) SerializeWithVersion(fileDescriptor uintptr, version string) error {
	if C.SerializeStableHLO(shlo.cPtr, C.CString(version), C.int(fileDescriptor)) {
		return nil
	}
	return errors.Errorf("Failed to serialize StableHLO at version %s (current StableHLO version is %s)",
		version, StableHLOCurrentVersion())
}

// Serialize to bytecode that can presumably be used by PjRT and IREE, as well as
// embedded in one of the TensorFlow SavedModel formats.(??)
//
// It serializes to the given file path.
func (shlo *StableHLO) Serialize(filePath string) error {
	f, err := os.Create(filePath)
	if err != nil {
		return errors.Wrapf(err, "Cannot create file %q to save StableHLO", filePath)
	}
	return shlo.SerializeWithVersion(f.Fd(), StableHLOCurrentVersion())
}

func NewStableHLOFromSerialized(serialized []byte) (*StableHLO, error) {
	if len(serialized) == 0 {
		return nil, errors.Errorf("Trying to unserialized from empty buffer")
	}
	statusOr := C.UnserializeStableHLO(SliceToVectorData(serialized))
	ptr, err := UnsafePointerOrError(statusOr)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to unserialize (serialized has %d bytes) to StableHLO", len(serialized))
	}
	return NewStableHLO((*C.StableHLOHolder)(ptr)), nil
}
