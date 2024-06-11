//go:build !google3

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

// #include "gomlx/aot_exec.h"
import "C"
import (
	"fmt"
	"github.com/pkg/errors"
	"unsafe"
)

// AOTExecutable executes Ahead-Of-Time (AOT) compiled graphs.
type AOTExecutable struct {
	cPtr   *C.AOTExecutable
	client *Client
}

// NewAOTExecutable given the client and the aotResult returned by an earlier Computation.AOTCompile call. It
// may return an error.
func NewAOTExecutable(client *Client, aotResult []byte) (*AOTExecutable, error) {
	exec := &AOTExecutable{client: client}
	statusOr := C.NewAOTExecutable(client.cClientPtr, SliceToVectorData(aotResult))
	cPtr, err := UnsafePointerOrError(statusOr)
	if err != nil {
		return nil, errors.Wrapf(err, "failed constructing a AOTExecutable")
	}
	exec.cPtr = (*C.AOTExecutable)(cPtr)
	return exec, nil
}

// IsNil returns whether contents are invalid or have been freed already.
func (exec *AOTExecutable) IsNil() bool {
	return exec == nil || exec.cPtr == nil
}

func (exec *AOTExecutable) Run(params []*OnDeviceBuffer) (*OnDeviceBuffer, error) {
	if exec.IsNil() {
		return nil, fmt.Errorf("trying to run Ahead-Of-Time (AOT) computation that was not successfully loaded")
	}
	shapedBuffers := newShapedBuffersCArray(params)
	statusOr := C.ExecuteAOT(exec.client.cClientPtr, exec.cPtr, C.int(len(params)),
		(*unsafe.Pointer)(unsafe.Pointer(shapedBuffers)))
	if len(params) > 0 {
		C.free(unsafe.Pointer(shapedBuffers))
	}
	cBuffer, err := PointerOrError[C.OnDeviceBuffer](statusOr)
	if err != nil {
		return nil, err
	}
	return newOnDeviceBuffer(exec.client, cBuffer), nil
}
