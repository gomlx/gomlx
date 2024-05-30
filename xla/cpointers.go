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

// This file includes many `cgo` related tools, in particular C pointer manipulation.

// #include <string.h>
// #include "gomlx/status.h"
import "C"
import (
	"fmt"
	"reflect"
	"runtime"
	"unsafe"
)

// SizeOf returns the size of the given type in bytes. Notice some structures may be padded, and this will
// include that space.
func SizeOf[T any]() C.size_t {
	var ptr *T
	return C.size_t(reflect.TypeOf(ptr).Elem().Size())
}

// Malloc allocates a T in the C heap and initializes it to zero.
// It must be manually freed with C.free() by the user.
func Malloc[T any]() (ptr *T) {
	size := SizeOf[T]()
	cPtr := (*T)(C.malloc(size))
	C.memset(unsafe.Pointer(cPtr), 0, size)
	return cPtr
}

// MallocArray allocates space to hold n copies of T in the C heap and initializes it to zero.
// It must be manually freed with C.free() by the user.
func MallocArray[T any](n int) (ptr *T) {
	size := SizeOf[T]() * C.size_t(n)
	cPtr := (*T)(C.malloc(size))
	C.memset(unsafe.Pointer(cPtr), 0, size)
	return cPtr
}

// MallocArrayAndSet allocates space to hold n copies of T in the C heap, and set each element `i` with the result of
// `setFn(i)`.
// It must be manually freed with C.free() by the user.
func MallocArrayAndSet[T any](n int, setFn func(i int) T) (ptr *T) {
	ptr = MallocArray[T](n)
	slice := unsafe.Slice(ptr, n)
	for ii := 0; ii < n; ii++ {
		slice[ii] = setFn(ii)
	}
	return ptr
}

func CDataToSlice[T any](data unsafe.Pointer, count int) (result []T) {
	sliceHeader := (*reflect.SliceHeader)((unsafe.Pointer(&result)))
	sliceHeader.Cap = count
	sliceHeader.Len = count
	sliceHeader.Data = uintptr(data)
	return
}

// StrFree converts the allocated C string (char *) to a Go `string` and
// frees the C string immediately.
func StrFree(cstr *C.char) (str string) {
	if cstr == nil {
		return ""
	}
	str = C.GoString(cstr)
	C.free(unsafe.Pointer(cstr))
	return
}

// VectorDataToSlice and then frees the given VectorData.
func VectorDataToSlice[T any](vec *C.VectorData) (data []T) {
	if vec.count == 0 {
		return
	}
	data = make([]T, vec.count)
	original := unsafe.Slice((*T)(unsafe.Pointer(vec.data)), int(vec.count))
	copy(data, original)
	C.free(vec.data)
	C.free(unsafe.Pointer(vec))
	return
}

func SliceToVectorData[T any](slice []T) (vec *C.VectorData) {
	vec = Malloc[C.VectorData]()
	vec.count = C.int(len(slice))
	if vec.count == 0 {
		return
	}
	data := MallocArray[T](len(slice))
	copy(unsafe.Slice(data, len(slice)), slice)
	vec.data = unsafe.Pointer(data)
	return
}

// MemoryStats returns memory profiling/introspection using MallocExtension::GetStats().
func MemoryStats() string {
	return StrFree(C.memory_stats())
}

// NoGlobalLeaks indicates the head-cheker (part of tcmalloc) to output its heap profile
// immediately -- because we are in Go, it is not called at exit, so we need to manually
// call this. To be used with google-pprof.
func NoGlobalLeaks() bool {
	return bool(C.heap_checker_no_global_leaks())
}

// StrVectorFree converts a C.VectorPointers that presumably contains `char *` to []string.
// It frees everything: the individual `char *` pointers, the array that contains it (`vp.data`) and
// finally `vp` itself.
func StrVectorFree(vp *C.VectorPointers) (strs []string) {
	n := int(vp.count)
	strs = make([]string, 0, n)
	data := CDataToSlice[*C.void](unsafe.Pointer(vp.data), n) // [*C.void]
	for ii := 0; ii < n; ii++ {
		str := StrFree((*C.char)(unsafe.Pointer(data[ii])))
		strs = append(strs, str)
	}
	if vp.count > 0 {
		C.free(unsafe.Pointer(vp.data))
	}
	C.free(unsafe.Pointer(vp))
	return
}

// MemoryUsage returns the memory used by the application, as reported by
// MallocExtension::GetNumericProperty("generic.current_allocated_bytes",...).
// If it returns 0, something went wrong.
func MemoryUsage() uint64 {
	return uint64(C.memory_usage())
}

// NumberToString converts a number to string in C++. Trivial function used for testing only.
func NumberToString(n int) string {
	return StrFree(C.number_to_string(C.int(n)))
}

// MemoryUsedByFn returns the extra memory usage in C/C++ heap after calling
// the given function. Used for testing.
// If msg is not "", logs information before and after.
func MemoryUsedByFn(fn func(), msg string) int64 {
	before := MemoryUsage()
	if msg != "" {
		fmt.Printf("%s:\n\tMemory usage before test: %s\n", msg, HumanBytes(before))
	}

	// Call provided closure.
	fn()

	after := MemoryUsage()
	diff := int64(after) - int64(before)
	if msg != "" {
		fmt.Printf("\tMemory usage after test: %s\n", HumanBytes(after))
		fmt.Printf("\tMemory difference: %s\n", HumanBytes(diff))
	}
	return diff
}

func HumanBytes[T interface{ int64 | uint64 }](bytes T) string {
	const unit = 1024
	if bytes < unit {
		return fmt.Sprintf("%d B", bytes)
	}
	div, exp := T(unit), 0
	for n := bytes / unit; n >= unit; n /= unit {
		div *= unit
		exp++
	}
	return fmt.Sprintf("%.1f %ciB",
		float64(bytes)/float64(div), "KMGTPE"[exp])
}

// GarbageCollectXLAObjects interactively calls runtime.GC() until no more xla.Literal
// or xla.OnDeviceBuffer objects are being collected.
func GarbageCollectXLAObjects(verbose bool) {
	for currentLiteral, currentShaped := int64(-1), int64(-1); LiteralsCount() != currentLiteral || OnDeviceBufferCount() != currentShaped; {
		if verbose {
			fmt.Printf("\tGC: L=%d D=%d\n", LiteralsCount(), OnDeviceBufferCount())
		}
		currentLiteral = LiteralsCount()
		currentShaped = OnDeviceBufferCount()
		for ii := 0; ii < 10; ii++ {
			runtime.GC()
			runtime.GC()
			runtime.GC()
		}
	}

}

// Finalizer is any object that implements Finalize, that can be called
// when an object is deallocated using runtime.SetFinalizer.
//
// Finalize should be idem-potent: if called multiple times subsequent calls
// shouldn't affect it.
type Finalizer interface {
	// Finalize frees the underlying resources, presumably outside
	// Go runtime control, like C pointers.
	//
	// Finalize should be idem-potent: if called multiple times subsequent calls
	Finalize()
}

// RegisterFinalizer is a trivial helper function that calls the WrapperWithDestructor.Finalize.
func RegisterFinalizer[T Finalizer](o T) {
	runtime.SetFinalizer(o, func(o T) {
		o.Finalize()
	})
}

// AutoCPointer wrapps a C pointer, and frees it when AutoCPointer is garbage collected,
// or when Finalize is called.
type AutoCPointer[T any] struct {
	P *T
}

// IsNil returns whether the AutoCPointer is nil or what it's pointing.
func (w *AutoCPointer[T]) IsNil() bool {
	return w == nil || w.P == nil
}

// UnsafePtr returns the pointer to T cast as `unsafe.Pointer`.
func (w *AutoCPointer[T]) UnsafePtr() unsafe.Pointer {
	return unsafe.Pointer(w.P)
}

// Finalize implements Finalizer.
func (w *AutoCPointer[T]) Finalize() {
	defer runtime.KeepAlive(w)
	if w == nil || w.P == nil {
		return
	}
	C.free(unsafe.Pointer(w.P))
	w.P = nil
}

// NewAutoCPointer holds a C pointers, and makes sure it is freed (`C.free`) when
// it is garbage collected or when Finalize is called.
//
// Note since Go 1.20 forward declared C types (not completely known) are no
// longer supported as generic parameters -- even though we only use the pointers to them. See
// https://groups.google.com/g/golang-nuts/c/h75BwBsz4YA/m/FLBIjgFBBQAJ
// for details.
func NewAutoCPointer[T any](p *T) (w *AutoCPointer[T]) {
	w = &AutoCPointer[T]{
		P: p,
	}
	RegisterFinalizer(w)
	return
}

// UnsafeCPointer holds a C pointers, and makes sure it is freed (`C.free`) when
// it is garbage collected or when Finalize is called.
//
// Because since Go 1.20 forward declared C types (not completely known) are no
// longer supported as generic parameters, we need to keep those pointers as unsafe pointers,
// and have to be manually cast.
type UnsafeCPointer struct {
	P unsafe.Pointer
}

// UnsafePtr returns the pointer to T cast as `unsafe.Pointer`.
func (w *UnsafeCPointer) UnsafePtr() unsafe.Pointer {
	return w.P
}

// IsNil returns whether the AutoCPointer is nil or what it's pointing.
func (w *UnsafeCPointer) IsNil() bool {
	return w == nil || w.P == nil
}

// Finalize implements Finalizer.
func (w *UnsafeCPointer) Finalize() {
	defer runtime.KeepAlive(w)
	if w == nil || w.P == nil {
		return
	}
	C.free(w.P)
	w.P = nil
}

// NewUnsafeCPointer holds a forward declared C pointer as `unsafe.Pointer`,
// and makes sure it is freed (`C.free`) when it is garbage collected or
// when WrapperWithDestructor.Finalize is called.
//
// Note since Go 1.20 forward declared C types (not completely known) are no
// longer supported as type parameters -- even though we only use the pointers to them. See
// https://groups.google.com/g/golang-nuts/c/h75BwBsz4YA/m/FLBIjgFBBQAJ
// for details.
func NewUnsafeCPointer(p unsafe.Pointer) *UnsafeCPointer {
	w := &UnsafeCPointer{
		P: p,
	}
	RegisterFinalizer(w)
	return w
}

// WrapperWithDestructor wraps a pointer to an arbitrary type and adds a finalize method.
type WrapperWithDestructor[T any] struct {
	Data       T
	destructor func(p T)
}

// Empty returns true if either w is nil, or it's contents has
// already been finalized.
func (w *WrapperWithDestructor[T]) Empty() bool {
	return w == nil || w.destructor == nil
}

// Finalize frees the pointer held. Notice this version is not concurrency safe -- but then
// Finalize should be called only once anyway.
//
// Once Finalize is called, the object cannot be re-used, it will be forever marked as empty.
func (w *WrapperWithDestructor[T]) Finalize() {
	if w.Empty() {
		return
	}
	defer runtime.KeepAlive(w)
	w.destructor(w.Data)
	w.destructor = nil
}

// wrapperFinalizer is a trivial helper function that calls the WrapperWithDestructor.Finalize.
func wrapperFinalizer[T any](w *WrapperWithDestructor[T]) {
	w.Finalize()
}

// NewWrapperWithDestructor creates a WrapperWithDestructor to type T, using the given destructor to finalize the object.
//
// The `destructor` will only be called once, even if `Finalize()` is called manually.
// The wrapper sets the destructor to nil after the first time `Finalize()` is called.
//
// There are no synchronization mechanisms, manually calling `Finalize()` concurrently is
// undefined.
func NewWrapperWithDestructor[T any](data T, destructor func(data T)) (w *WrapperWithDestructor[T]) {
	w = &WrapperWithDestructor[T]{
		Data:       data,
		destructor: destructor,
	}
	runtime.SetFinalizer(w, wrapperFinalizer[T])
	return
}
