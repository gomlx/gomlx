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

// Package tensor provides a `Tensor` interface with 2 different implementations: `Local` and `Device`.
//
// Tensors are multidimensional arrays (from scalar with 0 dimensions, to arbitrarily large dimensions), defined
// by their shape (a data type and its axes dimensions) and their actual content. As a special case, a Tensor can
// also be a tuple of multiple tensors.
//
// This implementation uses `gomlx.types.Shape` to represent the shape, and (for now) only explicitly supports dense
// representations of the data. There are two types of `Tensor` implementation: `Local` and `Device`. Both are wrappers
// to XLA underlying data representations, but can be used from Go. Mostly they are means to interact with the graph
// computation.
//
// `Device` means a tensor that is stored in wherever the computation is run (GPU, TPU or on the host
// itself). `Local` is a tensor in the current local process (still, they are stored in C++ to interact
// with XLA). When one wants to print or save a tensor, it needs to be converted to `Local`. When one
// feed a tensor to a computational graph execution, it is converted to `Device`.
//
// Device tensors are considered immutable, with one exception: decomposing tuples destroy them (a
// implementation choice of the underlying C++ XLA implementation).
//
// Transferring tensors to/from local/device areas has a cost, and should be avoided. For example,
// while training weights of an ML model, one generally does not need to transfer those weights to local -- just at
// the end of training to save the model weights. Because the tensors are immutable, the transferring is
// cached, so if `Tensor.Local()` or `Tensor.Device()` is called multiple times, the price is paid only once.
//
// In addition, tensors (both `Local` and `Device`) can be tuples of other tensors -- a recursive definition that
// allow for nested values. This seems to be an XLA mechanism to return several values in one -- documentation there
// is not very clear.
//
// Tensors are not yet concurrency safe. You have to handle race conditions to prevent simultaneous changes.
// TODO: add the necessary locking mechanisms.
//
// **Advanced**: On storage and mutability:
//
// This is somewhat inefficient for large tensors: there are potentially 3 copies (or more) of values floating around:
// the "Device" data, stored in the accelerator; the "Local" data stored on the program C++ heap; the Go values.
// This is still a work in progress on ways to avoid extra copies floating around. In the short term there are the
// following "unstable" (the API may change) options:
//
//   - Call `Finalize()` or `FinalizeAll()` on tensors that one is no longer needed. This immediately frees the
//     resources, and doesn't wait for the garbage collector. Notice that the Local/Device tensors create links to each
//     other for cache purpose, so if let for the garbage collector they will only be freed once both are no longer used.
//   - For `Local` tensors, the functions `Local.ValueOf()`, `Local.Data() returns pointers to the underlying C++ data.
//     One can mutate them directly (the values, not the slice lengths), for instance if loading value from disk, one
//     can avoid an extra copy of the data in Go by loading directly to the C++ data. If you mutate them, then also call
//     ClearCache(), since a corresponding Device will become out of date -- before you feed the modified Local tensor
//     to another graph run.
package tensor

import (
	"fmt"
	"github.com/pkg/errors"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/xla"
	"reflect"
)

// Tensor represents a multidimensional arrays (from scalar with 0 dimensions, to arbitrarily large dimensions), defined
// by their shape (a data type and its axes dimensions) and their actual content. As a special case, a Tensor can
// also be a tuple of multiple tensors.
//
// Tensor can be implemented by a tensor.Local or tensor.Device, which reflects whether the data is stored in the local
// CPU on or the device actually running the computation: an accelerator like a GPU or the CPU as well.
//
// Local and Device tensors can be converted to each other -- there is a transferring cost to that. There is a cache
// system to prevent duplicate transfers, but it assumes immutability -- call ClearCache after mutating a Local tensor.
//
// More details in the `tensor` package documentation.
type Tensor interface {
	// Local version of the tensor. If the underlying tensor is Local already, it's a no-op. Otherwise, the tensor
	// contents are transferred locally. It uses a cache system, so if tensor was already local no transfer happens.
	Local() *Local

	// Device version of the tensor. If the underlying tensor is on the given Device already, it's a no-op. Otherwise,
	// the tensor contents are transferred to the device. It uses a cache system, so if tensor was already local no
	// transfer happens.
	Device(client HasClient, deviceNum int) *Device

	// Shape of the tensor.
	Shape() shapes.Shape

	// DType of the tensor's shape.
	DType() shapes.DType

	// Rank of the tensor's shape.
	Rank() int

	// String returns a printable version of the tensor. This may lead to a transfer from a Device tensor
	// with the Local().
	String() string

	// Value returns a multidimensional slice (except if shape is a scalar) containing the values.
	// If the underlying tensor is on device (e.g: GPU), it's transferred locally with Local().
	Value() any

	// Error returns the message that caused an error state.
	Error() error

	// Ok returns whether the tensor is in an invalid state.
	Ok() bool

	// FinalizeAll immediately frees the dat of all versions of the Tensor -- Local or on Device, and make the
	// tensor invalid.
	FinalizeAll()
}

var (
	// Compile time type assertions.
	localAssert  *Local
	deviceAssert *Device
	_            Tensor = localAssert
	_            Tensor = deviceAssert
)

// HasClient accepts anything that can return a xla.Client. That includes xla.Client itself and
// graph.Manager.
type HasClient interface {
	Client() *xla.Client
}

// Local represents a multidimensional array of one of the supported types (see shapes.Number).
// It can be from a scalar to an arbitrary rank (number of dimensions). See Shape() to
// get information about its dimensions and the underlying DType. Finally, it can also hold a tuple
// of tensors (recursive definition).
//
// It implements the generic Tensor interface, and provides some specialized functionality that assumes
// the data is local -- all derived from the Data() method, which returns a pointer to the underlying data directly.
//
// A Local tensor needs to be made Device before being fed (as input) to a computation graph. Conversely,
// the output of computation graphs are Device that need to be converted to Local to introspect/manipulate
// the values in Go.
//
// Local is not thread safe, if using it concurrently, you will have to protect the access.
type Local struct {
	*cache

	// literal contains a locally (in C++ heap) stored tensor.
	literal *xla.Literal
	shape   shapes.Shape

	// error that may have occurred during construction/conversion/operation.
	error error
}

// Shape of Local, includes DType.
func (local *Local) Shape() shapes.Shape { return local.shape }

// DType returns the DType of the tensor's shape.
func (local *Local) DType() shapes.DType {
	return local.shape.DType
}

// Rank returns the rank fo the tensor's shape.
func (local *Local) Rank() int { return local.shape.Rank() }

// IsTuple returns whether Local is a tuple.
func (local *Local) IsTuple() bool { return !local.Empty() && local.shape.IsTuple() }

// Empty returns whether Local is holding no data. It's similar
// to a "nil" state for Local.
func (local *Local) Empty() bool {
	return local == nil || local.literal.IsNil()
}

// Ok returns whether local is both not empty and is not in error.
func (local *Local) Ok() bool {
	return !local.Empty() && local.error == nil
}

// Error returns the message that caused an error state.
func (local *Local) Error() error {
	if local == nil {
		return errors.New("local tensor is nil")
	}
	return local.error
}

// Data returns a slice with the consecutive data of the corresponding DType type.
// Consider the generic function Data[L]() if you know the type upfront.
//
// It returns nil is Local tensor is in an invalid state, or if it is a tuple.
func (local *Local) Data() any {
	if !local.Ok() || local.IsTuple() {
		return nil
	}
	return local.literal.Data()
}

// Bytes returns the same memory as Data, but the raw slice of bytes, with the proper size in bytes.
func (local *Local) Bytes() []byte {
	if !local.Ok() || local.IsTuple() {
		return nil
	}
	return local.literal.Bytes()
}

// Scalar returns the scalar value contained in the Local. It will return a zero value
// if the shape is not scalar.
func (local *Local) Scalar() any {
	if !local.Ok() || local.IsTuple() {
		return nil
	}
	if !local.shape.IsScalar() {
		// Build a zero value.
		tType := shapes.TypeForDType(local.shape.DType)
		vPtr := reflect.New(tType)
		v := reflect.Indirect(vPtr)
		return v.Interface()
	}

	// Get the first value of the data.
	data := local.Data()
	dataV := reflect.ValueOf(data)
	return dataV.Index(0).Interface()
}

// Value returns a multidimensional slice (except if shape is a scalar) containing the values, cast to type any.
// Same as AnyValueOf(t).
func (local *Local) Value() any {
	return AnyValueOf(local)
}

// MaxStringSize is the largest Local tensor that is actually returned by String() is requested.
var MaxStringSize = 500

// String converts to string, if not too large.
func (local *Local) String() string {
	return local.StringN(MaxStringSize)
}

// StringN converts to string, displaying at most n elements.
// TODO: nice pretty-print version, even for large tensors.
func (local *Local) StringN(n int) string {
	if local.error != nil {
		return fmt.Sprintf("tensor.Local.error=%v", local.error)
	}
	if local.IsTuple() {
		return fmt.Sprintf("Tuple(%d elements)", local.shape.TupleSize())
	}
	if local.shape.Size() <= n {
		return fmt.Sprintf("%s: %v", local.shape, local.Value())
	}
	dataV := reflect.ValueOf(local.Data())
	dataV = dataV.Slice(0, n)
	return fmt.Sprintf("%s: (... too large, %d values ..., first %d values: %v)",
		local.shape, local.shape.Size(), n, dataV.Interface())
}

// GoStr converts to string, using a Go-syntax representation that can be copied&pasted back to code.
func (local *Local) GoStr() string {
	if local.error != nil {
		return fmt.Sprintf("tensor.Local.error=%v", local.error)
	}
	if local.IsTuple() {
		return fmt.Sprintf("Tuple(%d elements)", local.shape.TupleSize())
	}
	value := local.Value()
	if local.shape.IsScalar() {
		return fmt.Sprintf("%s: %v", local.shape, value)
	}
	return fmt.Sprintf("%s: %s", local.shape, slices.SliceToGoStr(value))
}

// SplitTuple splits the Tuple tensor into its components. This unfortunately destroys the current Local,
// emptying it.
func (local *Local) SplitTuple() (tensors []*Local, err error) {
	if !local.Ok() || !local.IsTuple() {
		return nil, fmt.Errorf("tensor not a tuple")
	}
	if local.literal.IsNil() {
		return nil, fmt.Errorf("tensor local data (Literal) is empty, maybe it was already decomposed")
	}
	literals := local.literal.SplitTuple()
	tensors = make([]*Local, 0, len(literals))
	for _, literal := range literals {
		t := &Local{
			cache:   &cache{},
			shape:   literal.Shape(),
			literal: literal,
		}
		t.cache.local = t
		tensors = append(tensors, t)
	}

	// Underlying literal is destroyed anyway.
	local.literal.Finalize()
	local.error = fmt.Errorf("tensor.Local destroyed when decomposing the tuple")
	local.shape = shapes.Shape{}
	local.ClearCache()

	// De-register itself from the cache of globals
	return tensors, nil
}

// MakeLocalTuple compose local tensors into a Tuple. The individual tensors are destroyed in the process,
// as the tuple takes ownership of its parts.
func MakeLocalTuple(tensors ...*Local) *Local {
	numElements := len(tensors)
	literals := make([]*xla.Literal, 0, numElements)
	elementsShapes := make([]shapes.Shape, 0, numElements)
	for _, t := range tensors {
		if t.Empty() {
			return nil
		}
		literals = append(literals, t.literal)
		elementsShapes = append(elementsShapes, t.Shape())
	}
	tuple := &Local{
		cache:   &cache{},
		shape:   shapes.MakeTuple(elementsShapes),
		literal: xla.NewLiteralTuple(literals),
	}
	tuple.cache.local = tuple
	for _, t := range tensors {
		t.Finalize()
	}
	return tuple
}

// MakeLocalTupleAny composes values into local tensor. Values can be any value that can be converted
// to a *Local tensor, or a *Local tensor. Similar to MakeLocalTuple, but more permissible.
func MakeLocalTupleAny(values ...any) *Local {
	tensors := make([]*Local, 0, len(values))
	for ii, value := range values {
		local := FromAnyValue(value)
		if !local.Ok() {
			return &Local{
				error: fmt.Errorf("failed converting %d-th element of tuple: %w", ii, local.error),
			}
		}
		tensors = append(tensors, local)
	}
	return MakeLocalTuple(tensors...)
}

// ClearCache disconnect the local tensor to any corresponding shapedBuffer data. See discussion on storage and mutability
// on the package documentation.
func (local *Local) ClearCache() {
	local.cache.ClearLocal()

	// Create a new cache with itself only.
	cache := &cache{}
	cache.AddLocal(local)
}

// Literal returns the internal storage of the Local value. Internal only, used only by new Op implementations.
func (local *Local) Literal() *xla.Literal {
	if local.Empty() {
		return nil
	}
	return local.literal
}

// Finalize releases the memory associated with the local tensor. It becomes Empty() = true.
// It mutates the tensor, but it's handy in case one is dealing with large data. See discussion on storage and
// mutability on the package documentation.
func (local *Local) Finalize() {
	local.ClearCache()
	local.literal.Finalize()
	local.shape = shapes.Shape{}
	local.error = nil
}

// FinalizeAll releases the memory associated with all copies of the tensor (local and on device),
// and mark them as empty.
func (local *Local) FinalizeAll() {
	local.cache.FinalizeAll()
}

// layoutStrides return the strides for each dimension.
func (local *Local) layoutStrides() (strides []int) {
	rank := local.shape.Rank()
	if rank == 0 {
		return
	}
	strides = make([]int, rank)
	currentStride := 1
	for dim := rank - 1; dim >= 0; dim-- {
		strides[dim] = currentStride
		currentStride *= local.shape.Dimensions[dim]
	}
	return
}

// FromValue returns a Local tensor constructed from the given multi-dimension slice (or scalar).
func FromValue[S shapes.MultiDimensionSlice](value S) *Local {
	return FromAnyValue(value)
}

// LocalWithError creates a local tensor with the given error.
func LocalWithError(err error) *Local {
	localT := &Local{error: err}
	cache := &cache{}
	cache.AddLocal(localT)
	return localT
}

// FromShape creates a Local tensor with the given shape, with the data uninitialized.
func FromShape(shape shapes.Shape) (local *Local) {
	local = &Local{shape: shape}
	local.literal = xla.NewLiteralFromShape(local.shape)
	if local.Empty() {
		return LocalWithError(errors.New("failed to create XLA literal"))
	}
	cache := &cache{}
	cache.AddLocal(local)
	return
}

// FromValueAndDimensions creates a local tensor with the given dimensions, filled with the
// given value replicated everywhere. The DType is inferred from the value.
func FromValueAndDimensions[T shapes.Supported](value T, dimensions ...int) (local *Local) {
	shape := shapes.Make(shapes.DTypeForType(reflect.TypeOf(value)), dimensions...)
	local = FromShape(shape)
	dataV := reflect.ValueOf(local.Data())
	valueV := reflect.ValueOf(value)
	for ii := 0; ii < shape.Size(); ii++ {
		dataV.Index(ii).Set(valueV)
	}
	return
}

// FromDataAndDimensions creates a local tensor with the given dimensions, filled with the
// given flat values. The DType is inferred from the values.
func FromDataAndDimensions[T shapes.Supported](data []T, dimensions ...int) (local *Local) {
	var tmp T
	shape := shapes.Make(shapes.DTypeForType(reflect.TypeOf(tmp)), dimensions...)
	if len(data) != shape.Size() {
		return LocalWithError(errors.Errorf("FromDataAndDimensions(): data size is %d, but dimensions size is %d", len(data), shape.Size()))
	}
	local = FromShape(shape)
	dataV := reflect.ValueOf(local.Data())
	reflect.Copy(dataV, reflect.ValueOf(data))
	return
}

// FromAnyValue is a non-generic version of FromValue. The input is expected to be either a scalar or a slice of
// slices with homogeneous dimensions. If the input happens to already be a Local, it is returned.
func FromAnyValue(value any) (local *Local) {
	var ok bool
	if local, ok = value.(*Local); ok {
		// Input is already a Local.
		return
	}
	local = &Local{}
	shape, err := shapeForValue(value)
	if err != nil {
		return LocalWithError(errors.Wrapf(err, "cannot create shape from %T", value))
	}
	local = FromShape(shape)
	dataV := reflect.ValueOf(local.Data())
	if local.shape.Rank() == 0 {
		// S is a scalar type.
		elem := dataV.Index(0)
		elem.Set(reflect.ValueOf(value))
		return
	}

	// Copy over multi-dimensional slice recursively.
	copySlicesRecursively(dataV, reflect.ValueOf(value), local.layoutStrides())
	return
}

// copySlicesRecursively copy values on a multi-dimension slice to a flat data slice
// assuming the strides for each dimension.
func copySlicesRecursively(data reflect.Value, mdSlice reflect.Value, strides []int) {
	if len(strides) == 1 {
		// Last level of slice, just copy over the slice.
		reflect.Copy(data, mdSlice)
		return
	}

	numElements := mdSlice.Len()
	subStrides := strides[1:]
	for ii := 0; ii < numElements; ii++ {
		start := ii * strides[0]
		end := (ii + 1) * strides[0]
		subData := data.Slice(start, end)
		copySlicesRecursively(subData, mdSlice.Index(ii), subStrides)
	}
	return
}

// Data returns the flattened data for the given Local. Returns nil
// if the DType of the tensor is not compatible with the requested number type.
func Data[T shapes.Number](t *Local) []T {
	if t.IsTuple() {
		return nil
	}
	if t.shape.DType != shapes.DTypeGeneric[T]() {
		return nil
	}
	data, err := xla.DataFromLiteral[T](t.literal)
	if err != nil {
		return nil
	}
	return data
}

// ValueOf constructs a multi-dimension-slice from the Local. Returns nil (or the zero value
// for the type T) if the Local has an error or is not compatible (shape or DType).
//
// ValueOf will do conversions of type, if possible. Example: let's say t is a [5]int local tensor. One can call
// `ValueOf[[]float64](t)` and get a Go `[]float64` back of size 5.
//
// The slices themselves shouldn't be modified -- the underlying storage is not
// owned by Go, and tensor objects are supposed to be immutable. See discussion on data storage
// and exceptions to mutability on the package description if you really need to mutate its values.
func ValueOf[T shapes.MultiDimensionSlice](local *Local) (result T) {
	if local.Empty() || local.IsTuple() {
		return
	}
	if local.error != nil {
		return
	}
	depth, dtype, baseType := depthDTypeAndBaseType(reflect.TypeOf(result))
	_ = baseType
	if dtype == shapes.InvalidDType || depth != local.shape.Rank() {
		return
	}
	if dtype == local.DType() {
		// No conversion needed.
		return AnyValueOf(local).(T)
	}

	// Check that type is convertible.
	if !shapes.TypeForDType(local.DType()).ConvertibleTo(baseType) {
		return
	}

	// Convert the data as one large slice and then split into slices.
	numElements := local.Shape().Size()
	dataV := reflect.ValueOf(local.Data())
	newDataV := reflect.MakeSlice(reflect.SliceOf(baseType), local.Shape().Size(), numElements)
	for ii := 0; ii < numElements; ii++ {
		element := dataV.Index(ii)
		newElement := newDataV.Index(ii)
		newElement.Set(element.Convert(baseType))
	}

	// Use same recursion as AnyValueOf to return the appropriate slice.
	return convertDataToSlices(newDataV, local.Shape().Dimensions...).Interface().(T)
}

// AnyValueOf constructs a multi-dimension-slice from the Local tensor, and returns it
// as type `any` (`interface{}`). Works the same way as `ValueOf` without the
// generics interface.
//
// If local holds an error, returns the error.
//
// The slices themselves shouldn't be modified -- the underlying storage is not
// owned by Go, and tensor objects are supposed to be immutable. See discussion on data storage
// and exceptions to mutability on the package description if you really need to mutate its values.
func AnyValueOf(local *Local) (result any) {
	if local == nil {
		return errors.WithStack(errors.Errorf("tensor.Local is nil, no value associated"))
	}
	if local.error != nil {
		return local.error
	}
	if local.IsTuple() {
		return errors.WithStack(errors.Errorf("tensor.Local is a tuple (shape %s), unable to generate a value", local.Shape()))
	}
	rank := local.shape.Rank()
	if rank == 0 {
		return local.Scalar()
	}

	dataV := reflect.ValueOf(local.Data())
	return convertDataToSlices(dataV, local.Shape().Dimensions...).Interface()
	//resultT := types.TypeForDType(local.shape.DType)
	//for range local.shape.Dimensions {
	//	resultT = reflect.SliceOf(resultT)
	//}
	//sliceV := createSlicesRecursively(resultT, dataV, local.shape.Dimensions, local.layoutStrides())
	//return sliceV.Interface()
}

// convertDataToSlices take data as a flat slice and convert to a multi-dimensional slices with the given dimensions.
func convertDataToSlices(dataV reflect.Value, dimensions ...int) reflect.Value {
	if len(dimensions) <= 1 {
		return dataV
	}
	resultT := dataV.Type().Elem()
	for range dimensions {
		resultT = reflect.SliceOf(resultT)
	}
	strides := make([]int, len(dimensions))
	currentStride := 1
	for dim := len(dimensions) - 1; dim >= 0; dim-- {
		strides[dim] = currentStride
		currentStride *= dimensions[dim]
	}
	return createSlicesRecursively(resultT, dataV, dimensions, strides)
}

// createSlicesRecursively recursively creates slices copy values on a multi-dimension slice to a flat data slice
// assuming the strides for each dimension.
func createSlicesRecursively(resultT reflect.Type, data reflect.Value, dimensions []int, strides []int) reflect.Value {
	if len(strides) == 1 {
		// Last level of slice, just copy over the slice (not the data, just the slice).
		return data
	}

	numElements := dimensions[0]
	slice := reflect.MakeSlice(resultT, numElements, numElements)

	subStrides := strides[1:]
	subDimensions := dimensions[1:]
	subResultT := resultT.Elem()
	for ii := 0; ii < numElements; ii++ {
		start := ii * strides[0]
		end := (ii + 1) * strides[0]
		subData := data.Slice(start, end)
		subSlice := createSlicesRecursively(subResultT, subData, subDimensions, subStrides)
		slice.Index(ii).Set(subSlice)
	}
	return slice
}

// ToScalar returns the scalar stored in the Tensor. If Tensor is a Device tensor, transfer it
// locally -- presumably a small value. Returns 0 if the DType
// given is incompatible, or if the Local is not a scalar.
//
// Notice the Tensor DType doesn't need to be exactly the corresponding to the type parameter T.
// If they are different it will be automatically converted. E.g:
// `ToScalar[float64](<some types.Int64 tensor>)` will also work and return a `types.Int64` value
// converted to float64.
func ToScalar[T shapes.Number](t Tensor) (result T) {
	if t.Error() != nil || t.Shape().IsTuple() {
		return
	}
	vAny := t.Local().Scalar()
	resultT := reflect.TypeOf(result)
	if reflect.TypeOf(vAny) == resultT {
		return vAny.(T)
	}
	rValue := reflect.ValueOf(vAny)
	if !rValue.CanConvert(resultT) {
		return
	}
	return rValue.Convert(resultT).Interface().(T)
}

func shapeForValue(v any) (shape shapes.Shape, err error) {
	err = shapeForValueRecursive(&shape, reflect.ValueOf(v), reflect.TypeOf(v))
	return
}

func shapeForValueRecursive(shape *shapes.Shape, v reflect.Value, t reflect.Type) error {
	if t.Kind() == reflect.Slice {
		// Recurse into inner slices.
		t = t.Elem()
		shape.Dimensions = append(shape.Dimensions, v.Len())
		shapePrefix := shape.Copy()

		// The first element is the reference
		v0 := v.Index(0)
		err := shapeForValueRecursive(shape, v0, t)
		if err != nil {
			return err
		}

		// Test that other elements have the same shape as the first one.
		for ii := 1; ii < v.Len(); ii++ {
			shapeTest := shapePrefix.Copy()
			err = shapeForValueRecursive(&shapeTest, v.Index(ii), t)
			if err != nil {
				return err
			}
			if !shape.Eq(shapeTest) {
				return fmt.Errorf("sub-slices have irregular shapes, found shapes %q, and %q", shape, shapeTest)
			}
		}
	} else if t.Kind() == reflect.Pointer {
		return fmt.Errorf("cannot convert Pointer (%s) to a concrete value for tensors", t)
	} else {
		shape.DType = shapes.DTypeForType(t)
		if shape.DType == shapes.InvalidDType {
			return fmt.Errorf("cannot convert type %s to a value concrete tensor type (maybe type not supported yet?)", t)
		}
	}
	return nil
}

func depthDTypeAndBaseType(t reflect.Type) (int, shapes.DType, reflect.Type) {
	if t.Kind() == reflect.Slice {
		depth, dtype, baseType := depthDTypeAndBaseType(t.Elem())
		return depth + 1, dtype, baseType
	}
	return 0, shapes.DTypeForType(t), t

}

// Zeros returns a zero initialized Local of the given shape (including DType).
func Zeros(shape shapes.Shape) *Local {
	local := &Local{
		shape:   shape,
		literal: xla.NewLiteralWithZeros(shape),
	}
	cache := &cache{}
	cache.AddLocal(local)
	return local
}

// Device represents a tensor (or tuple) stored on device. The object doesn't offer much functionality,
// except re-use it as input to a graph execution, or converting to a `Local` copy.
//
// To create it, either create a Local tensor first and then convert, or use the output of
// the execution of a computation graph -- they return Device tensors.
//
// It implements the Tensor interface.
type Device struct {
	*cache

	shape shapes.Shape

	shapedBuffer *xla.OnDeviceBuffer
	clientId     xla.ClientId
	deviceNum    int

	// error reports back any error during the creation or transfer of this Tensor.
	error error
}

// Empty returns whether Local is holding no data or is in an error state. It's similar
// to a "nil" state for Local.
func (device *Device) Empty() bool {
	return device == nil || device.shapedBuffer.IsNil()
}

// Ok returns whether the shapedBuffer is not empty and has no error.
func (device *Device) Ok() bool {
	return !device.Empty() && device.error == nil
}

// Error returns the message that caused an error state.
func (device *Device) Error() error {
	if device == nil {
		return errors.New("device tensor is nil")
	}
	if device.error != nil {
		return device.error
	}
	if device.Empty() {
		return errors.New("device tensor is holds no data!?")
	}
	return device.error
}

// String converts to string, by converting (transferring) the tensor to local and then using Local.String().
func (device *Device) String() string {
	if device.error != nil {
		return fmt.Sprintf("tensor.Device.error=%v", device.error)
	}
	if device.IsTuple() {
		return fmt.Sprintf("Tuple(%d elements)", device.shape.TupleSize())
	}
	if device.shape.Size() < MaxStringSize {
		return device.Local().String()
	}
	return fmt.Sprintf("%s: (... too large, %d values ...)", device.shape, device.shape.Size())
}

// Shape returns the shape of the Device.
func (device *Device) Shape() shapes.Shape {
	if !device.Ok() {
		return shapes.Shape{}
	}
	return device.shape
}

// DType returns the DType of the tensor's shape.
func (device *Device) DType() shapes.DType {
	return device.shape.DType
}

// Rank returns the rank fo the tensor's shape.
func (device *Device) Rank() int {
	return device.shape.Rank()
}

// Value returns a multidimensional slice (except if shape is a scalar) containing the values.
// If there isn't yet a cached local copy of tensor, it first copies the tensor from the device
// to a local tensor.
func (device *Device) Value() any {
	return device.Local().Value()
}

// FromShapedBuffer creates a Device tensor from XLA's OnDeviceBuffer structure. Internal implementation,
// most users don't need to use this.
func FromShapedBuffer(buffer *xla.OnDeviceBuffer) (deviceT *Device) {
	if buffer == nil {
		deviceT = &Device{error: fmt.Errorf("invalid (nil) underlying OnDeviceBuffer")}
	} else {
		deviceT = &Device{
			shapedBuffer: buffer,
			clientId:     buffer.Client().Id,
			shape:        buffer.Shape(),
			deviceNum:    buffer.DeviceOrdinal(),
		}
	}
	cache := &cache{}
	cache.AddDevice(deviceT)
	return
}

// DeviceWithError creates a device tensor with the given error.
func DeviceWithError(err error) *Device {
	deviceT := &Device{error: err}
	cache := &cache{}
	cache.AddDevice(deviceT)
	return deviceT
}

// ShapedBuffer returns the underlying XLA structure. Internal usage only.
func (device *Device) ShapedBuffer() *xla.OnDeviceBuffer {
	if device.Empty() {
		return nil
	}
	return device.shapedBuffer
}

// IsTuple returns whether Local is a tuple.
func (device *Device) IsTuple() bool { return !device.Empty() && device.shape.IsTuple() }

// Finalize releases the memory associated with the shapedBuffer. It becomes empty.
// It mutates the tensor, but it's handy in case one is dealing with large data. See discussion on storage and
// mutability on the package documentation.
func (device *Device) Finalize() {
	device.ClearCache()
	device.shapedBuffer.Finalize()
	device.shape = shapes.Shape{}
	device.error = nil
}

// FinalizeAll releases the memory associated with all copies of the tensor (local and on device),
// and mark them as empty.
func (device *Device) FinalizeAll() {
	device.cache.FinalizeAll()
}

// ClearCache disconnects the device tensor to any corresponding local data. See discussion on storage and mutability
// on the package documentation.
func (device *Device) ClearCache() {
	device.cache.ClearDevice(device)

	// Create a new cache with itself only.
	cache := &cache{}
	cache.AddDevice(device)
}

// SplitTupleError splits a device tensor into its elements. In case of error, return the error.
//
// This makes the current device tensor invalid.
func (device *Device) SplitTupleError() ([]*Device, error) {
	if !device.Ok() {
		return nil, fmt.Errorf("cannot split device tensor that has error: %w", device.error)
	}
	if !device.IsTuple() {
		return nil, fmt.Errorf("cannot split device tensor that is not a tuple")
	}
	return device.splitTupleImpl(true)
}

// SplitTuple splits a device tensor into its elements. In case of error returns nil, or
// individual tuple element errors are reported in the tensors themselves.
//
// This makes the current device tensor invalid.
func (device *Device) SplitTuple() []*Device {
	if !device.Ok() {
		return nil
	}
	if !device.IsTuple() {
		return nil
	}
	parts, _ := device.splitTupleImpl(false)
	return parts
}

func (device *Device) splitTupleImpl(returnError bool) ([]*Device, error) {
	deviceTensors := make([]*Device, 0, device.shape.TupleSize())
	for ii := 0; ii < device.shape.TupleSize(); ii++ {
		subDeviceT, err := device.shapedBuffer.SubTree([]int{ii})
		if err == nil {
			deviceTensors = append(deviceTensors, FromShapedBuffer(subDeviceT))
		} else {
			if returnError {
				return nil, fmt.Errorf("failed generating split %d: %w", ii, err)
			}
			deviceTensors = append(deviceTensors,
				DeviceWithError(errors.Wrapf(err, "failed extracting split %d", ii)))
		}
	}
	device.FinalizeAll()
	return deviceTensors, nil
}

// cache stores links to materializations of the tensor (local or on device).
type cache struct {
	local *Local

	// Maps ClientId->Map of deviceNum->Device tensor.
	onDevices map[xla.ClientId]map[int]*Device
}

// FinalizeAll will finalize (free) all associated data.
func (c *cache) FinalizeAll() {
	if c == nil {
		return
	}
	local := c.local
	var devices []*Device
	for _, tensorsPerDevice := range c.onDevices {
		for _, d := range tensorsPerDevice {
			devices = append(devices, d)
		}
	}

	if local != nil {
		local.Finalize()
	}
	for _, d := range devices {
		d.Finalize()
	}
}

// AddDevice to the internal cache, and returns itself for convenience.
func (c *cache) AddDevice(device *Device) *Device {
	device.cache = c
	if !device.Ok() {
		return device
	}
	if c.onDevices == nil {
		c.onDevices = make(map[xla.ClientId]map[int]*Device)
	}
	clientMap, ok := c.onDevices[device.clientId]
	if !ok {
		clientMap = make(map[int]*Device)
		c.onDevices[device.clientId] = clientMap
	}
	clientMap[device.deviceNum] = device
	return device
}

// ClearDevice from cache, and leaves the device tensor passed without a cache.
func (c *cache) ClearDevice(device *Device) {
	if device == nil {
		return
	}
	if device.cache == c {
		device.cache = nil
	}
	if c.onDevices == nil {
		return
	}
	clientMap, ok := c.onDevices[device.clientId]
	if !ok {
		return
	}
	if clientMap == nil {
		return
	}
	delete(clientMap, device.deviceNum)
	device.cache = nil
}

// AddLocal to cache and returns the local tensor for convenience.
func (c *cache) AddLocal(local *Local) *Local {
	c.local = local
	local.cache = c
	return local
}

// ClearLocal cache, and leaves the local cached tensor without a cache.
func (c *cache) ClearLocal() {
	if c.local != nil {
		c.local.cache = nil
	}
	c.local = nil
}

// Local will transfer data from the Device storage to a Local tensor.
func (c *cache) Local() *Local {
	if c == nil {
		return nil
	}
	if c.local != nil {
		return c.local
	}
	if c.onDevices == nil {
		return nil
	}
	// Pick first onDevice tensor and convert to local.
	for _, perDeviceOrdinal := range c.onDevices {
		for _, device := range perDeviceOrdinal {
			if device == nil {
				continue
			}
			if device.error != nil {
				c.local = &Local{
					error: fmt.Errorf("tensor.Local created from bad Device: %w", device.error),
				}
				return c.local
			}
			literal, err := xla.FromOnDeviceBuffer(device.shapedBuffer)
			if err != nil {
				c.local = &Local{
					error: fmt.Errorf("tensor.Local failed to transfer from Device: %w", err),
				}
				return c.local
			}
			return c.AddLocal(&Local{
				shape:   literal.Shape(),
				literal: literal,
			})
		}
	}
	return nil
}

// Device either uses a cached value on device already or it transfers local data to the
// shapedBuffer store of values (OnDeviceBuffer) and returns a tensor.Device reference -- value
// is cached for future calls.
// This is used for instance to transfer parameters when executing a graph.
func (c *cache) Device(hasClient HasClient, deviceNum int) *Device {
	if c == nil {
		return nil
	}

	// Search in cache.
	client := hasClient.Client()
	if c.onDevices != nil {
		if deviceNumMap, found := c.onDevices[client.Id]; found {
			if device, found := deviceNumMap[deviceNum]; found {
				return device
			}
		}
	}

	local := c.local
	if local == nil || local.Empty() {
		if local != nil && local.Error() != nil {
			return DeviceWithError(local.Error())
		}
		return DeviceWithError(errors.Errorf("cannot convert empty tensor.Local to tensor.Device"))
	}
	cid := client.Id
	if local.error != nil {
		return c.AddDevice(&Device{
			clientId:  cid,
			deviceNum: deviceNum,
			error:     fmt.Errorf("tensor.Device transferred from bad Local: %w", local.error),
		})
	}
	device := &Device{deviceNum: deviceNum}
	device.shapedBuffer, device.error = local.literal.ToOnDeviceBuffer(client, deviceNum)
	c.AddDevice(device)
	if device.error != nil {
		return device
	}
	device.shape = device.shapedBuffer.Shape()
	return device
}
