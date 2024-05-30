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

// This file includes the all code that is dependent on cgo ("C" import). This needs to be a separate file
// for stringer to work.

// #include "gomlx/client.h"
// #include "gomlx/computation.h"
// #include "gomlx/status.h"
import "C"
import (
	"log"
	"unsafe"

	"github.com/gomlx/gomlx/types/shapes"
)

func init() {
	cVer := int(C.xla_wrapper_version())
	if cVer != XlaWrapperVersion {
		log.Fatalf("GoMLX XLA libraries is out of sync: Go library is version %d, but linked C++ library is "+
			"version %d. In most cases it's because of an out-dated C++ library: if that is the case simply download "+
			"the new release from github (or compile the C++ from scratch, can take some time, see instructions). But "+
			"it could be the Go version is out-of-sync as well, in which case just `go get` the latest/compatible "+
			"version.", XlaWrapperVersion, cVer)
	}
}

// SerializedNode represents a graph.Node with its arguments. This can then be passed to the XLA C++ library.
// Only used by ops implementers.
type SerializedNode struct {
	Type       NodeType
	NodeInputs []int32  // Index to other nodes that are used as inputs.
	Literal    *Literal // If a Literal (constant) is involved in the operation.

	Int   int          // Used for any static integer inputs.
	Shape shapes.Shape // If a shape is used as a static input.
	Str   string       // Used for any static string argument.
	Ints  []int        // List of integer numbers.
	Float float32      // For a float parameter.
}

func (comp *Computation) SerializedToC(node *SerializedNode) *C.SerializedNode {
	// Allocate and set C.SerializedNodes struct using Go memory. It can be discarded when this function exit.
	numInputs := len(node.NodeInputs)
	cNode := &C.SerializedNode{
		node_type:  C.int32_t(node.Type),
		num_inputs: C.int32_t(numInputs),
		integer:    C.int64_t(node.Int),
		float_v:    C.float(node.Float),
	}
	if numInputs > 0 {
		// Create the `inputs` array.
		cNode.inputs = MallocArrayAndSet[C.XlaOpPtr](numInputs, func(ii int) C.XlaOpPtr {
			return (C.XlaOpPtr)(unsafe.Pointer(comp.ops[node.NodeInputs[ii]]))
		})
	}
	if node.Shape.Ok() {
		cNode.shape = CShapeFromShape(node.Shape)
	}
	if node.Str != "" {
		cNode.string = C.CString(node.Str)
	}
	if !node.Literal.IsNil() {
		cNode.literal = node.Literal.cLiteralPtr
	}
	if len(node.Ints) > 0 {
		cNode.integer_array_size = C.int32_t(len(node.Ints))
		cNode.integer_array = MallocArrayAndSet[C.int64_t](len(node.Ints), func(ii int) C.int64_t { return C.int64_t(node.Ints[ii]) })
	}
	return cNode
}

// DeleteCSerializedNode frees the C allocated memory within cNode. Note that cNode itself is assumed to be
// allocated in Go space, hence it is (and should be) automatically garbage collected.
func DeleteCSerializedNode(cNode *C.SerializedNode) {
	if cNode.inputs != nil {
		C.free(unsafe.Pointer(cNode.inputs))
		cNode.inputs = nil
		cNode.num_inputs = 0
	}
	if cNode.shape != nil {
		C.DeleteShape(cNode.shape)
		cNode.shape = nil
	}
	if cNode.new_shape != nil {
		C.DeleteShape(cNode.new_shape)
		cNode.new_shape = nil
	}
	if cNode.string != nil {
		C.free(unsafe.Pointer(cNode.string))
		cNode.string = nil
	}
	if cNode.integer_array != nil {
		C.free(unsafe.Pointer(cNode.integer_array))
		cNode.integer_array = nil
		cNode.integer_array_size = 0
	}
}
