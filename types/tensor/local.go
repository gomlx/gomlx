package tensor

import (
	"encoding/gob"
	"fmt"
	"github.com/gomlx/gomlx/types/keepalive"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/xla"
	"github.com/pkg/errors"
	"io"
	"reflect"
	"runtime"
)

// Local tensor represents a multidimensional array of one of the supported types (see shapes.Number)
// that is stored "locally" (in CPU memory), albeit under XLA (that is C++) management. It is
// meant to be used means of feeding and retrieving data for GoMLX computation graphs.
//
// It's shape can be from a scalar to an arbitrary rank (number of dimensions). See Shape() to
// get information about its dimensions and the underlying DType.
//
// As a special case it can also hold a tuple of Local tensors (recursive definition).
//
// It implements the generic Tensor interface, and provides some specialized functionality that assumes
// the data is local (in CPU). The Tensor interface includes conversion back-and-forth to a Device tensor.
// The converted tensors are cached, so there is no extra cost in doing it multiple times.
//
// When dealing with large tensors, one will want to carefully manage its life cycle. It provides a method
// called Finalize() to immediately release its data memory (managed in XLA C++) -- it is also called automatically
// during garbage collection. And to release all associated versions of the tensor, including the copies on Device,
// there is FinalizeAll().
//
// There are various ways to construct a Local tensor:
//
//   - `FromValue[S shapes.MultiDimensionSlice](value S)`: Generic conversion, works with the scalar supported `DType`s
//     as well as with any arbitrary multidimensional slice of them. Slices of rank > 1 must be regular, that is
//     all the sub-slices must have the same shape. E.g: `FromValue([][]float{{1,2}, {3, 5}, {7, 11}})`
//   - `FromAnyValue(value any)`: same as `FromValue` but non-generic, it takes an anonymous type `any`. The exception
//     is if `value` is already a tensor, it is itself returned.
//   - `FromShape(shape shapes.Shape)`: creates a Local tensor with the given shape, and uninitialized values. See
//     below how to mutate its values with a `LocalRef`.
//   - `FromScalarAndDimensions[T shapes.Supported](value T, dimensions ...int)`: creates a Local tensor with the
//     given dimensions, filled with the scalar value given. `T` must be one of the supported types.
//   - `FromFlatDataAndDimensions[T shapes.Supported](data []T, dimensions ...int)`: creates a Local tensor with the
//     given dimensions, and set the flattened values with the given data. `T` must be one of the supported types.
//
// There are various ways to access the contents of a Local tensor:
//
//   - `Local.Value() any`: it creates a copy of the tensor contents to a Go type. A scalar if the underlying shape
//     is a scalar or a (multidimensional) slice. If returning a multidimensional slice, it first creates a flat
//     slice (see Local.Flat below) with the flat data, and then creates the sub-slices point to the middle of it.
//   - `Local.Flat() any`: similar to Value() it creates a copy of the flattened (only one dimension) contents to
//     a Go slice -- even if it is a scalar, it will return a slice with one value.
//   - `Local.CopyFlat(dst any) error`: similar to Local.Flat, but instead of creating a new slice, it copies to an already
//     provided slice. Useful when looping over results. It returns an error if `dst` type is not compatible
//     or not if there is not the correct space in `dst`, that is, if `len(dst) != Local.Shape().Size()`.
//
// The methods above are all for reading. To mutate the contents of the Local tensor after it was created, or to
// access it directly without any copies -- relevant if using large tensors -- one has to "acquire" the data. That
// makes sure the GC is not going to collect the data (since it's in C++, Go doesn't know about it). Call
// `Local.AcquireData()` to get a `LocalRef`. Then do something like `defer LocalRef.Release()` to make sure
// the data is released after its used (otherwise it will never be freed). And with the LocalRef one can use:
//
//   - `LocalRef.Flat() any`: like `Local.Flat` returns a slice with the flattened data, but it points to the underlying c++
//     flattened data. Mutating this slice contents changes the tensor content directly. The returned slice is only
//     valid while LocalRef is not released (LocalRef.Release), and the tensor is not finalized (Local.Finalize).
//   - `FlatFromRef[T shapes.Number](ref *LocalRef) []T`: a generic function version of LocalRef.Flat (it does a couple
//     fewer casting in the process).
//   - `LocalRef.Bytes() []byte` returns a slice of bytes pointing directly to the storage bytes of tensor. Convenient
//     for saving or restoring tensor contents. See also GobSerialize and GobDeserialize.
//
// Notice there is not a library of tensor functions (math, or otherwise) to operate on them. To do math on tensors,
// instead use the computation graph engine (package `graph`).
//
// Local can be in an empty state (for instance after calling Finalize), and in an erroneous state -- after some invalid
// operation. These can be checked and tested with the methods Ok, Empty and Error.
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

// DType returns the DType of the tensor's shape. Shortcut to `local.Shape().DType`
func (local *Local) DType() shapes.DType {
	return local.shape.DType
}

// Rank returns the rank fo the tensor's shape. Shortcut to `local.Shape().Rank()`
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

// ClearCache disconnect the local tensor to the corresponding cache data, which holds the pointers to the
// on Device versions of the tensor. This should be called if the Local tensor contents are mutated but its cache
// is pointing to a lagging version of a Device tensor. See discussion in Tensor interface.
func (local *Local) ClearCache() {
	local.cache.ClearLocal()

	// Create a new cache with itself only.
	cache := &cache{}
	cache.AddLocal(local)
}

// Literal returns the internal storage of the Local value. Internal usage only, used only by new Op implementations.
// If you need access to the underlying data, use AcquireData instead.
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

// AcquireData returns a LocalRef that can be used to access the underlying
// C++ data directly.
//
// Once acquired it has to be manually released -- it may leak tensors if
// not released. The recommended way it to pair it with a deferred release,
// as in the example below:
//
//	dataRef := local.AcquireData()
//	defer dataRef.Release()
//	// do something with dataRef...
//
// It does not work for tuples: those need to be split first.
//
// It returns nil if Local tensor is in an invalid state, or if it is a tuple.
func (local *Local) AcquireData() *LocalRef {
	if !local.Ok() || local.IsTuple() {
		return nil
	}
	return &LocalRef{
		local:     local,
		keepAlive: keepalive.Acquire(local),
	}
}

// LocalRef is a live-reference to a Local tensors data (managed by C++).
// It keeps it alive (from the GC), and should be released with Release at
// the end of its use -- see example in Local.AcquireData.
//
// It provides direct access to the underlying data.
type LocalRef struct {
	local     *Local
	keepAlive keepalive.KeepAlive
}

// Release acquired data reference to a Local tensor. If LocalRef is
// nil (in case a call to Local.AcquireData failed), it is a no-op.
func (ref *LocalRef) Release() {
	if ref == nil || ref.local == nil {
		return
	}
	ref.local = nil
	ref.keepAlive.Release()
}

// Ok returns whether ref, and the Local tensor it points to are valid.
func (ref *LocalRef) Ok() bool {
	if ref == nil || ref.local == nil {
		return false
	}
	return ref.local.Ok()
}

// Flat returns the flattened data as a slice of the corresponding DType type.
//
// See Local.LayoutStrides to calculate the offset of individual positions.
//
// It is only valid while LocalRef hasn't been released.
//
// It returns nil is Local tensor is in an invalid state, or if it is a tuple.
func (ref *LocalRef) Flat() any {
	if !ref.Ok() {
		return nil
	}
	// Notice LocalRef only take Local tensors that are not tuples.
	return ref.local.literal.Data()
}

// FlatFromRef returns the flattened data acquired by `ref` as a slice of the corresponding DType type.
// It is the generics version of LocalRef.Flat()
//
// See Local.LayoutStrides to calculate the offset of individual positions.
//
// If the DType of the tensor is not compatible with the requested number type, or if tensor is a tuple, it returns nil.
func FlatFromRef[T shapes.Number](ref *LocalRef) []T {
	if ref.local.IsTuple() {
		return nil
	}
	if ref.local.shape.DType != shapes.DTypeGeneric[T]() {
		return nil
	}
	data, err := xla.DataFromLiteral[T](ref.local.literal)
	if err != nil {
		return nil
	}
	return data
}

// Bytes returns the same memory as Flat, but the raw slice of bytes, with the proper size in bytes.
//
// It is only valid while LocalRef hasn't been released.
func (ref *LocalRef) Bytes() []byte {
	if !ref.Ok() {
		return nil
	}
	return ref.local.literal.Bytes()
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

// CopyFlat contents of the tensor to dst. The parameter `dst` must be a slice of the corresponding
// type that matches the tensor DType, see shapes.TypeForDType.
func (local *Local) CopyFlat(dst any) error {
	if local.Empty() {
		return errors.Errorf("cannot copy contents of empty tensor")
	}
	if !local.Ok() {
		return local.Error()
	}
	if local.IsTuple() {
		return errors.Errorf("cannot copy contents of tuple tensor")
	}
	dstV := reflect.ValueOf(dst)
	dstT := dstV.Type()
	if dstT.Kind() != reflect.Slice {
		return errors.Errorf("dst (%s) is not a slice, cannot copy tensor contents", dstT)
	}
	if dstT.Elem() != shapes.TypeForDType(local.DType()) {
		return errors.Errorf("dst element type (%s) is incompatible with DType (%s), expected %s", dstT.Elem(), local.DType(), shapes.TypeForDType(local.DType()))
	}
	if dstV.Len() != local.Shape().Size() {
		return errors.Errorf("length of slice doesn't match: len(dst)=%d != Local.Shape().Size()=%d", dstV.Len(), local.Shape().Size())
	}
	reflect.Copy(dstV, reflect.ValueOf(local.literal.Data()))
	runtime.KeepAlive(local) // Make sure that local doesn't get garbage collected in the middle of the copy.
	return nil
}

// Flat returns a copy of the flattened contents of the Local tensor as a slice of the
// type matching the tensor's DType (see shapes.TypeForDType).
//
// See Local.LayoutStrides to calculate the offset of individual positions.
//
// It the tensor is a scalar, it still returns a slice with one element.
//
// If tensor is empty, in error of if it is a tuple, returns nil.
func (local *Local) Flat() any {
	if !local.Ok() || local.IsTuple() {
		return nil
	}
	srcV := reflect.ValueOf(local.literal.Data())
	size := local.Shape().Size()
	dstV := reflect.MakeSlice(reflect.SliceOf(shapes.TypeForDType(local.DType())), size, size)
	reflect.Copy(dstV, srcV)
	runtime.KeepAlive(local) // Make sure that local doesn't get garbage collected in the middle of the copy.
	return dstV.Interface()
}

// LayoutStrides return the strides for each axis. This can be handy when manipulating the flat data.
func (local *Local) LayoutStrides() (strides []int) {
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

// Value returns a multidimensional slice (except if shape is a scalar) containing the values. The returned `any`
// value can be cast to the appropriate Go type.
//
// It returns a copy of the underlying data. See AcquireData to access a version the one can mutate.
//
// If tensor is empty, in error of if it is a tuple, returns nil.
func (local *Local) Value() any {
	if !local.Ok() || local.IsTuple() {
		return nil
	}
	if local.shape.IsScalar() {
		// Avoid creating yet another slice:
		srcV := reflect.ValueOf(local.literal.Data())
		defer runtime.KeepAlive(local) // Make sure that local doesn't get garbage collected in the middle of the copy.
		return srcV.Index(0).Interface()
	}

	// Create a flat slice with all data.
	flatCopy := local.Flat()
	if local.shape.Rank() == 1 {
		return flatCopy
	}

	// If multi-dimensional slice, returns slice pointing to the flatCopy.
	flatV := reflect.ValueOf(flatCopy)
	return convertDataToSlices(flatV, local.Shape().Dimensions...).Interface()
}

// GobSerialize Local tensor in binary format.
func (local *Local) GobSerialize(encoder *gob.Encoder) (err error) {
	if !local.Ok() {
		return errors.Errorf("tensor.Local is in an invalid state")
	}
	err = local.shape.GobSerialize(encoder)
	if err != nil {
		return
	}
	dataRef := local.AcquireData()
	defer dataRef.Release()
	data := dataRef.Bytes()
	err = encoder.Encode(data)
	if err != nil {
		err = errors.Wrapf(err, "failed to write tensor.Local data")
	}
	return
}

// GobDeserialize a Tensor from the reader. Returns new tensor.Local or an error.
func GobDeserialize(decoder *gob.Decoder) (local *Local, err error) {
	shape, err := shapes.GobDeserialize(decoder)
	if err != nil {
		return
	}
	local = FromShape(shape)
	dataRef := local.AcquireData()
	defer dataRef.Release()
	data := dataRef.Bytes()
	err = decoder.Decode(&data)
	if err != nil {
		err = errors.Wrapf(err, "failed to read tensor.Local data")
		return
	}
	return
}

// Save the Local tensor to the given writer w.
func (local *Local) Save(w io.Writer) (err error) {
	enc := gob.NewEncoder(w)
	return local.GobSerialize(enc)
}

// Load a Local tensor from the given reader r.
//
// Notice it uses `encoding/gob` package, which uses buffering when reading, so it will read-ahead on r. In other words,
// it assumes it's the only thing in this reader: this works well if the tensor is the only thing in a file,
// but not if it's a file that holds other things. Use GobSerialize/GobDeserialize directly if you need to store several
// things in a file.
func Load(r io.Reader) (local *Local, err error) {
	dec := gob.NewDecoder(r)
	return GobDeserialize(dec)
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
		return fmt.Sprintf("tensor.Local in error: %v", local.error)
	}
	if local.IsTuple() {
		return fmt.Sprintf("Tuple(%d elements)", local.shape.TupleSize())
	}
	if local.shape.Size() <= n {
		return fmt.Sprintf("%s: %v", local.shape, local.Value())
	}

	dataRef := local.AcquireData()
	defer dataRef.Release()
	dataV := reflect.ValueOf(dataRef.Flat())
	dataV = dataV.Slice(0, n)
	return fmt.Sprintf("%s: (... too large, %d values ..., first %d values: %v)",
		local.shape, local.shape.Size(), n, dataV.Interface())
}

// GoStr converts to string, using a Go-syntax representation that can be copied&pasted back to code.
func (local *Local) GoStr() string {
	if local.error != nil || local.IsTuple() {
		return local.String()
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
		local := FromAnyValue(value).Local()
		if !local.Ok() {
			return &Local{
				error: fmt.Errorf("failed converting %d-th element of tuple: %w", ii, local.error),
			}
		}
		tensors = append(tensors, local)
	}
	return MakeLocalTuple(tensors...)
}

// MakeLocalWithError creates a local tensor with the given error.
func MakeLocalWithError(err error) *Local {
	localT := &Local{error: err}
	cache := &cache{}
	cache.AddLocal(localT)
	return localT
}

// FromShape creates a Local tensor with the given shape, with the data uninitialized. See Local.AcquireData
// to mutate the data after the Local tensor is created.
func FromShape(shape shapes.Shape) (local *Local) {
	local = &Local{shape: shape}
	local.literal = xla.NewLiteralFromShape(local.shape)
	if local.Empty() {
		return MakeLocalWithError(errors.New("failed to create XLA literal"))
	}
	cache := &cache{}
	cache.AddLocal(local)
	return
}

// FromScalarAndDimensions creates a local tensor with the given dimensions, filled with the
// given scalar value replicated everywhere. The DType is inferred from the value.
func FromScalarAndDimensions[T shapes.Supported](value T, dimensions ...int) (local *Local) {
	shape := shapes.Make(shapes.DTypeForType(reflect.TypeOf(value)), dimensions...)
	local = FromShape(shape)
	dataRef := local.AcquireData()
	defer dataRef.Release()
	dataV := reflect.ValueOf(dataRef.Flat())
	valueV := reflect.ValueOf(value)
	for ii := 0; ii < shape.Size(); ii++ {
		dataV.Index(ii).Set(valueV)
	}
	return
}

// FromFlatDataAndDimensions creates a local tensor with the given dimensions, filled with the
// flattened values given in `data`. The DType is inferred from the values.
func FromFlatDataAndDimensions[T shapes.Supported](data []T, dimensions ...int) (local *Local) {
	var tmp T
	shape := shapes.Make(shapes.DTypeForType(reflect.TypeOf(tmp)), dimensions...)
	if len(data) != shape.Size() {
		return MakeLocalWithError(errors.Errorf("FromFlatDataAndDimensions(): data size is %d, but dimensions size is %d", len(data), shape.Size()))
	}
	local = FromShape(shape)
	dataRef := local.AcquireData()
	defer dataRef.Release()
	dataV := reflect.ValueOf(dataRef.Flat())
	reflect.Copy(dataV, reflect.ValueOf(data))
	return
}

// FromValue returns a Local tensor constructed from the given multi-dimension slice (or scalar). If rank of the `value`
// is larger than 1, the shape of all sub-slices must be the same.
//
// It returns a tensor.Local with error if shape is not regular.
func FromValue[S shapes.MultiDimensionSlice](value S) *Local {
	return FromAnyValue(value).Local()
}

// FromAnyValue is a non-generic version of FromValue that returns a tensor.Tensor (not specified if local or on device).
// The input is expected to be either a scalar or a slice of slices with homogeneous dimensions.
// If the input is a tensor already (Local or Device), it is simply returned.
// If value is anything but a Device tensor, it will return a Local tensor.
//
// It returns a tensor. with an error (see `Tensor.Error()`) if `value` type is unsupported or the shape is not
// regular.
func FromAnyValue(value any) Tensor {
	if t, ok := value.(*Local); ok {
		// Input is already a Local.
		return t
	}
	local := &Local{}
	shape, err := shapeForValue(value)
	if err != nil {
		return MakeLocalWithError(errors.Wrapf(err, "cannot create shape from %T", value))
	}
	local = FromShape(shape)
	dataRef := local.AcquireData()
	defer dataRef.Release()
	dataV := reflect.ValueOf(local.Data())
	if local.shape.Rank() == 0 {
		// S is a scalar type.
		elem := dataV.Index(0)
		elem.Set(reflect.ValueOf(value))
		return local
	}

	// Copy over multi-dimensional slice recursively.
	copySlicesRecursively(dataV, reflect.ValueOf(value), local.LayoutStrides())
	return local
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

// convertDataToSlices take data as a flat slice and creates a multidimensional slices with the given dimensions that
// points to the given data.
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
