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
	"fmt"
	"github.com/pkg/errors"
	"github.com/gomlx/gomlx/types/shapes"
	"log"
	"unsafe"
)

// Computation is a wrapper for the C++ `XlaComputation`.
type Computation struct {
	cCompPtr *C.Computation
	client   *Client
	ops      []*C.XlaOp

	firstError error
	compiled   bool
}

// NewComputation creates a new XlaComputation object and returns its wrapper.
func NewComputation(client *Client, name string) *Computation {
	if client == nil || client.cClientPtr == nil {
		log.Printf("NewComputation: client == nil !?")
		return nil
	}
	cCompPtr := C.NewComputation(C.CString(name))
	comp := &Computation{
		cCompPtr: cCompPtr,
		client:   client,
	}
	RegisterFinalizer(comp)
	return comp
}

// Finalize implements Finalizer.
func (comp *Computation) Finalize() {
	if comp == nil || comp.cCompPtr == nil {
		return
	}
	C.DeleteComputation(unsafe.Pointer(comp.cCompPtr))
	comp.cCompPtr = nil
	for _, op := range comp.ops {
		C.DeleteXlaOp(unsafe.Pointer(op))
	}
	comp.ops = nil
}

// IsNil checks whether the computation is nil or it's C underlying object.
func (comp *Computation) IsNil() bool {
	return comp == nil || comp.cCompPtr == nil
}

// IsCompiled returns whether this computation has already been compiled.
func (comp *Computation) IsCompiled() bool { return comp.compiled }

// constLargeArraySize is just an ad-hoc to cgo, to get a pointer to array of pointers. cgo complaints
// if we try to cast to a pointer of pointer, as in `(*unsafePointer)(unsafe.Pointer(&pointerArray[0])`.
const constLargeArraySize = 100000

// AddOp adds an Op described by node to the compilation graph. Notice after the graph
// is compiled it becomes frozen and no more ops can be added. Ops are created in order,
// and
func (comp *Computation) AddOp(node *SerializedNode) (opsNum int, shape shapes.Shape, err error) {
	//fmt.Printf("Computation(%x).AddOp: %s\n", unsafe.Pointer(comp.cCompPtr), node.Type)
	if comp.firstError != nil {
		err = comp.firstError
		return
	}
	if comp.compiled {
		err = fmt.Errorf("cannot AddOp after computation has been compiled")
		return
	}

	cNode := comp.SerializedToC(node)
	//fmt.Printf("\tcNode=%+v\n", cNode)
	status := C.ComputationAddOp(comp.cCompPtr, cNode)
	err = ErrorFromStatus((*C.XlaStatus)(status))
	if err != nil {
		return
	}
	opsNum = len(comp.ops)
	comp.ops = append(comp.ops, (*C.XlaOp)(cNode.new_op))
	//fmt.Printf("\t\tnew_op=%v\n", cNode.new_op)
	shape = ShapeFromCShape(cNode.new_shape)
	DeleteCSerializedNode(cNode)
	//fmt.Printf("\t\tnew_shape=%s\n", shape)
	return
}

var OpsCount = 0

type xlaOpPtr struct {
	cXlaOpPtr *C.XlaOp
}

func (p *xlaOpPtr) Finalize() {
	if p == nil && p.cXlaOpPtr == nil {
		return
	}
	OpsCount -= 1
	C.free(unsafe.Pointer(p))
	p.cXlaOpPtr = nil
}

func newXlaOp(op *C.XlaOp) *xlaOpPtr {
	OpsCount += 1
	p := &xlaOpPtr{op}
	RegisterFinalizer(p)
	return p
}

func newShapesCArray(paramShapes []shapes.Shape) (cShapes **C.Shape, cShapesSlice []*C.Shape) {
	numParams := len(paramShapes)
	if numParams > 0 {
		cShapes = MallocArray[*C.Shape](numParams)
		cShapesSlice = unsafe.Slice(cShapes, numParams)
		for ii, shape := range paramShapes {
			cShapesSlice[ii] = CShapeFromShape(shape)
		}
	}
	return
}

func freeShapesCArray(cShapes **C.Shape, cShapesSlice []*C.Shape) {
	if len(cShapesSlice) == 0 {
		return
	}
	// Free memory allocated in C.
	for _, cShape := range cShapesSlice {
		C.DeleteShape(cShape)
	}
	C.free(unsafe.Pointer(cShapes))
}

// Compile compiles the Computation, so it's ready for execution. After this
// no more ops can be added. The output is the index to the output node.
func (comp *Computation) Compile(paramShapes []shapes.Shape, output int) error {
	if comp.firstError != nil {
		return comp.firstError
	}
	if output < 0 || output >= len(comp.ops) {
		return fmt.Errorf("output ops (%d) out of range (only %d were defined)", output, len(comp.ops))
	}
	if comp.compiled {
		return fmt.Errorf("the Computation is already compiled")
	}
	if !comp.client.Ok() {
		return fmt.Errorf("the Client associated to Computation is not ok (finalized?)")
	}
	comp.compiled = true

	numParams := len(paramShapes)
	cShapes, cShapesSlice := newShapesCArray(paramShapes)
	status := C.ClientCompileComputation(
		comp.client.cClientPtr, comp.cCompPtr, C.int(numParams), cShapes, unsafe.Pointer(comp.ops[output]))
	freeShapesCArray(cShapes, cShapesSlice)
	comp.firstError = ErrorFromStatus((*C.XlaStatus)(status))
	return comp.firstError
}

// newShapedBuffersCArray returns a C array pointing to the params. Notice
// the array -- but not it's underlying XlaShapedBuffer -- should be manually freed.
func newShapedBuffersCArray(params []*OnDeviceBuffer) **C.XlaShapedBuffer {
	var shapedBuffers **C.XlaShapedBuffer
	if len(params) > 0 {
		shapedBuffers = MallocArray[*C.XlaShapedBuffer](len(params))
		shapedBuffersSlice := unsafe.Slice(shapedBuffers, len(params))
		for ii, buffer := range params {
			cBuffer := buffer.cOnDeviceBufferPtr
			if cBuffer.sb_buffer != nil {
				shapedBuffersSlice[ii] = (*C.XlaShapedBuffer)(cBuffer.sb_buffer)
			} else {
				shapedBuffersSlice[ii] = (*C.XlaShapedBuffer)(cBuffer.ssb_buffer)
			}
		}
	}
	return shapedBuffers
}

// Run runs a computation. The parameter values for the graph are given in params.
func (comp *Computation) Run(params []*OnDeviceBuffer) (*OnDeviceBuffer, error) {
	if !comp.IsCompiled() || comp.IsNil() {
		return nil, fmt.Errorf("trying to run computation that was not successfully compiled")
	}
	shapedBuffers := newShapedBuffersCArray(params)
	statusOr := C.ClientExecuteComputation(
		comp.client.cClientPtr, comp.cCompPtr, C.int(len(params)),
		(*unsafe.Pointer)(unsafe.Pointer(shapedBuffers)))
	if len(params) > 0 {
		C.free(unsafe.Pointer(shapedBuffers))
	}
	cBuffer, err := PointerOrError[C.OnDeviceBuffer](statusOr)
	if err != nil {
		return nil, err
	}
	return newOnDeviceBuffer(comp.client, cBuffer), nil
}

// AOTCompile returns the Ahead-Of-Time compiled version of the graph, that can be used for
// execution later.
//
// The graph needs to be compiled. And it is AOT-compiled to the same platform it was already
// compiled -- TODO: cross-compile.
//
// It returns a binary serialized format that can be executed later, without linking the whole GoMLX machinery.
// See tutorial on instructions and an example of how to do this.
func (comp *Computation) AOTCompile(paramShapes []shapes.Shape) ([]byte, error) {
	if comp.IsNil() || comp.firstError != nil {
		return nil, errors.Errorf("Computation graph is nil!?")
	}
	if comp.firstError != nil {
		return nil, comp.firstError
	}
	if !comp.client.Ok() {
		return nil, errors.Errorf("the Client associated to Computation is not ok (finalized?)")
	}
	numParams := len(paramShapes)
	cShapes, cShapesSlice := newShapesCArray(paramShapes)
	statusOr := C.ClientAOTCompileComputation(comp.client.cClientPtr, comp.cCompPtr, C.int(numParams), cShapes)
	freeShapesCArray(cShapes, cShapesSlice)
	vec, err := PointerOrError[C.VectorData](statusOr)
	if err != nil {
		return nil, errors.Wrapf(err, "failed conversion in Computation.ReadableStableHLO")
	}
	return VectorDataToSlice[byte](vec), nil
}
