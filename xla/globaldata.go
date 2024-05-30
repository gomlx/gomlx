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

// #include "gomlx/client.h"
// #include "gomlx/computation.h"
import "C"
import (
	"fmt"
	"runtime"
	"unsafe"

	"github.com/gomlx/gomlx/types/shapes"
)

// GlobalData represents a value stored in the remote accelerator (or wherever the computation
// is executed). Use ToLiteral to bring it to the local CPU.
//
// Note: GoMLX started using a different execution pathway that uses ShapedBuffer instead, and
// GlobalData is no longer being used right now. But I'm not sure if it will be used later
// when adding support for multiple accelerator devices.
type GlobalData struct {
	cGlobalDataPtr *C.XlaGlobalData
	client         *Client
}

func newGlobalData(client *Client, cGlobalDataPtr *C.XlaGlobalData) *GlobalData {
	gd := &GlobalData{
		cGlobalDataPtr: cGlobalDataPtr,
		client:         client,
	}
	RegisterFinalizer(gd)
	return gd
}

// Finalize implements Finalizer.
func (gd *GlobalData) Finalize() {
	if gd == nil || gd.cGlobalDataPtr == nil {
		return
	}
	defer runtime.KeepAlive(gd)
	C.DeleteGlobalData(unsafe.Pointer(gd.cGlobalDataPtr))
	gd.cGlobalDataPtr = nil
}

// IsNil returns whether either the Client object is nil or the contained C pointer.
func (gd *GlobalData) IsNil() bool {
	return gd == nil || gd.cGlobalDataPtr == nil
}

// Shape retrieves the shape of data stored globally.
func (gd *GlobalData) Shape() (shape shapes.Shape, err error) {
	if gd.IsNil() {
		return
	}
	statusOr := C.GlobalDataShape(unsafe.Pointer(gd.cGlobalDataPtr), gd.client.cClientPtr)
	shapePtr, err := UnsafePointerOrError(statusOr)
	if err != nil {
		return shape, err
	}
	cShape := (*C.Shape)(shapePtr)
	shape = ShapeFromCShape(cShape)
	C.DeleteShape(cShape)
	return
}

func (gd *GlobalData) Client() *Client {
	return gd.client
}

// SplitTuple splits the tuple GlobalData into its various components. This invalidates
// the current global data.
func (gd *GlobalData) SplitTuple() (gds []*GlobalData, err error) {
	shape, err := gd.Shape()
	if err != nil {
		return nil, err
	}
	if !shape.IsTuple() {
		return nil, fmt.Errorf("can't SplitTupleError if OnDeviceBuffer is not Tuple, shape=%s", shape)
	}
	numElements := shape.TupleSize()
	client := gd.client
	statusOr := C.GlobalDataDeconstructTuple(unsafe.Pointer(gd.cGlobalDataPtr), gd.client.cClientPtr)
	rawPtr, err := UnsafePointerOrError(statusOr)
	if err != nil {
		return nil, err
	}
	xlaGDs := unsafe.Slice((*unsafe.Pointer)(rawPtr), numElements) // It doesn't allow an array of C pointers.
	gds = make([]*GlobalData, 0, numElements)
	for _, xlaGD := range xlaGDs {
		gd := newGlobalData(client, (*C.XlaGlobalData)(xlaGD))
		gds = append(gds, gd)
	}
	C.free(rawPtr)
	return
}

// ToLiteral transfers from stored in accelerator server, to CPU. Returns a Literal
// with the transferred data.
func (gd *GlobalData) ToLiteral() (*Literal, error) {
	statusOr := C.TransferFromServer(unsafe.Pointer(gd.cGlobalDataPtr), gd.client.cClientPtr)
	literalPtr, err := UnsafePointerOrError(statusOr)
	if err != nil {
		return nil, err
	}
	cLiteral := (*C.Literal)(literalPtr)
	return newLiteral(cLiteral), nil
}
