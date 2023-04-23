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

// #include <gomlx/on_device_buffer.h>
import "C"
import (
	"fmt"
	"github.com/gomlx/gomlx/types/shapes"
	"sync/atomic"
	"unsafe"
)

// OnDeviceBuffer represents a value stored in an execution device (CPU or accelerator).
//
// Notice there can be multiple accelerator devices, DeviceOrdinal will inform which device
// this is stored in.
type OnDeviceBuffer struct {
	cOnDeviceBufferPtr *C.OnDeviceBuffer

	client        *Client
	shape         shapes.Shape
	deviceOrdinal int
}

// Number of OnDeviceBuffer objects allocated and freed: used for profiling and debugging.
var (
	OnDeviceBufferCountDeallocated = int64(0)
	OnDeviceBufferCountAllocated   = int64(0)
)

// OnDeviceBufferCount returns the number of OnDeviceBuffer still allocated.
func OnDeviceBufferCount() int64 {
	return OnDeviceBufferCountAllocated - OnDeviceBufferCountDeallocated
}

func newOnDeviceBuffer(client *Client, cBuffer *C.OnDeviceBuffer) *OnDeviceBuffer {
	atomic.AddInt64(&OnDeviceBufferCountAllocated, 1)
	cShape := C.OnDeviceBufferShape(cBuffer)
	shape := ShapeFromCShape(cShape)
	C.DeleteShape(cShape)
	b := &OnDeviceBuffer{
		cOnDeviceBufferPtr: cBuffer,
		client:             client,
		shape:              shape,
		deviceOrdinal:      int(C.OnDeviceBufferDeviceOrdinal(cBuffer)),
	}
	RegisterFinalizer(b)
	return b
}

// Finalize implements Finalizer.
func (b *OnDeviceBuffer) Finalize() {
	if b == nil || b.cOnDeviceBufferPtr == nil {
		return
	}
	C.DeleteOnDeviceBuffer(b.cOnDeviceBufferPtr)
	atomic.AddInt64(&OnDeviceBufferCountDeallocated, 1)
	b.cOnDeviceBufferPtr = nil
}

// IsNil returns whether OnDeviceBuffer holds no data.
func (b *OnDeviceBuffer) IsNil() bool {
	return b == nil || b.cOnDeviceBufferPtr == nil
}

func (b *OnDeviceBuffer) Shape() shapes.Shape {
	return b.shape
}

func (b *OnDeviceBuffer) Client() *Client {
	return b.client
}

func (b *OnDeviceBuffer) String() string {
	return StrFree(C.OnDeviceBufferToString(b.cOnDeviceBufferPtr))
}

// DeviceOrdinal returns the ordinal number of the device -- an id in case there are several
// replicas of the device (? XLA documentation is not clear about this, just guessing).
func (b *OnDeviceBuffer) DeviceOrdinal() int {
	return b.deviceOrdinal
}

// SubTree retrieves an element from a nested tuple (tree) OnDeviceBuffer.
func (b *OnDeviceBuffer) SubTree(path []int) (*OnDeviceBuffer, error) {
	if !b.shape.IsTuple() {
		return nil, fmt.Errorf("can't index ShappedBuffer, it is not a tuple")
	}
	if len(path) == 0 {
		// Trivial case, return itself.
		return b, nil
	}

	subShape := b.shape
	subPath := path
	for len(subPath) > 0 {
		if !subShape.IsTuple() {
			return nil, fmt.Errorf("can't index ShappedBuffer at position %d of path %v", len(path)-len(subPath), path)
		}
		if subPath[0] >= len(subShape.TupleShapes) {
			return nil, fmt.Errorf("index %d to path in ShappedBuffer value %d is out of bounds (tuple at that position has only %d elements)", len(path)-len(subPath), subPath[0], len(subShape.TupleShapes))
		}
		subShape = subShape.TupleShapes[subPath[0]]
		subPath = subPath[1:]
	}

	cPath := MallocArrayAndSet[C.int64_t](len(path), func(ii int) C.int64_t { return C.int64_t(path[ii]) })
	statusOr := C.OnDeviceBufferSubTree(b.cOnDeviceBufferPtr, C.int(len(path)), cPath)
	C.free(unsafe.Pointer(cPath))
	cSubBuffer, err := PointerOrError[C.OnDeviceBuffer](statusOr)
	if err != nil {
		return nil, err
	}
	return newOnDeviceBuffer(b.client, cSubBuffer), nil
}
