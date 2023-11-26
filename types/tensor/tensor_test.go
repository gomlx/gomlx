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

package tensor

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/xla"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"reflect"
	"strconv"
	"testing"
)

func cmpShapes(t *testing.T, shape, wantShape shapes.Shape, err error) {
	if err != nil {
		t.Fatalf("Failed to get shape (wanted %q) from value: %v", wantShape, err)
	}
	if !wantShape.Eq(shape) {
		t.Fatalf("Invalid shape %q, wanted %q", shape, wantShape)
	}
}

func TestFromValue(t *testing.T) {
	wantShape := shapes.Shape{DType: shapes.Float32, Dimensions: []int{3, 2}}
	shape, err := shapeForValue([][]float32{{0, 0}, {1, 1}, {2, 2}})
	cmpShapes(t, shape, wantShape, err)

	wantShape = shapes.Shape{DType: shapes.Float64, Dimensions: []int{1, 1, 1}}
	shape, err = shapeForValue([][][]float64{{{1}}})
	cmpShapes(t, shape, wantShape, err)

	if strconv.IntSize == 64 {
		wantShape = shapes.Shape{DType: shapes.Int64, Dimensions: nil}
		shape, err = shapeForValue(5)
		cmpShapes(t, shape, wantShape, err)
	} else if strconv.IntSize == 32 {
		wantShape = shapes.Shape{DType: shapes.Int32, Dimensions: nil}
		shape, err = shapeForValue(5)
		cmpShapes(t, shape, wantShape, err)
	} else {
		// For any other int size, it should panic.
		wantShape = shapes.Shape{DType: shapes.Int32, Dimensions: nil}
		require.Panics(t, func() {
			shape, err = shapeForValue(5)
		})
	}

	wantShape = shapes.Shape{DType: shapes.Int64, Dimensions: nil}
	shape, err = shapeForValue(5)
	cmpShapes(t, shape, wantShape, err)

	wantShape = shapes.Shape{DType: shapes.Bool, Dimensions: []int{3, 2}}
	shape, err = shapeForValue([][]bool{{true, false}, {false, false}, {false, true}})
	cmpShapes(t, shape, wantShape, err)

	wantShape = shapes.Shape{DType: shapes.Complex64, Dimensions: []int{2}}
	shape, err = shapeForValue([]complex64{1.0i, 1.0})
	cmpShapes(t, shape, wantShape, err)

	wantShape = shapes.Shape{DType: shapes.Complex128, Dimensions: []int{2}}
	shape, err = shapeForValue([]complex128{1.0i, 1.0})
	cmpShapes(t, shape, wantShape, err)

	// Test for invalid `DType`.
	shape, err = shapeForValue([][]uint16{{3}})
	if shape.DType != shapes.InvalidDType {
		t.Fatalf("Wanted InvalidDType for uint16, instead got %q", shape.DType)
	}
	if err == nil {
		t.Fatalf("Should have returned error for unsupported DType")
	}

	// Test for irregularly shaped slices.
	shape, err = shapeForValue([][][]int32{{{1}}, {{1, 2}}})
	if err == nil {
		t.Fatalf("Should have returned error for irregularly shaped slices")
	}
	fmt.Printf("\tExpected error: %v\n", err)

	// Test correct setting of scalar value, dtype=Int64.
	{
		want := int64(5)
		var local *Local
		require.NotPanics(t, func() { local = FromValue(want) })
		assert.Equal(t, want, local.Value())
	}

	// Test correct setting of scalar value for Go type `int` (maybe dtype=Int64 or Int32).
	if strconv.IntSize == 64 {
		want := int64(5)
		var local *Local
		require.NotPanics(t, func() { local = FromValue(5) })
		assert.Equal(t, want, local.Value())
	} else if strconv.IntSize == 32 {
		want := int32(5)
		var local *Local
		require.NotPanics(t, func() { local = FromValue(5) })
		assert.Equal(t, want, local.Value())
	} else {
		// For any other int size, it should panic.
		require.Panics(t, func() {
			_ = FromValue(5)
		})
	}

	// Test correct setting of 1D slice, dtype=float64
	{
		want := []float64{2, 5}
		var local *Local
		require.NotPanics(t, func() { local = FromValue(want) })
		assert.Equal(t, want, local.Value())
	}

	// Test correct setting of 1D slice, dtype=float64
	{
		want := []float32{1, 2, 3, 10, 11, 12}
		local := FromValue([][]float32{{1, 2, 3}, {10, 11, 12}})
		if local.error != nil {
			t.Fatalf("Failed to build scalar local: %v", err)
		}
		got, _ := local.FlatCopy().([]float32)
		if !slices.DeepSliceCmp(want, got, slices.Equal[float32]) {
			t.Fatalf("Local read out got %v, wanted %v", got, want)
		}
	}

	// Test correct 2D slice, dtype=Bool
	{
		want := []bool{true, false, false, false, false, true}
		local := FromValue([][]bool{{true, false}, {false, false}, {false, true}})
		if local.error != nil {
			t.Fatalf("Failed to build scalar local: %v", err)
		}
		got, _ := local.FlatCopy().([]bool)
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("Local read out got %v, wanted %v", got, want)
		}
	}

	// Test correct 2D slice, Go type=int, dtype=Int32 or Int64
	{
		local := FromValue([][]int{{1, 3}, {5, 7}})
		if strconv.IntSize == 64 {
			want := []int64{1, 3, 5, 7}
			got, _ := local.FlatCopy().([]int64)
			assert.Equal(t, want, got)
		} else if strconv.IntSize == 32 {
			want := []int32{1, 3, 5, 7}
			got, _ := local.FlatCopy().([]int32)
			assert.Equal(t, want, got)
		} // Other int sizes will panic on `FromValue`.
	}
}

func TestLocal_CopyData(t *testing.T) {
	want := []float32{1, 2, 3, 10, 11, 12}
	var local *Local
	require.NotPanics(t, func() { local = FromValue([][]float32{{1, 2, 3}, {10, 11, 12}}) })
	dst := make([]float32, len(want))
	require.NotPanics(t, func() { local.CopyData(dst) })
	require.Equal(t, want, dst)

	// Check failures:
	require.Panics(t, func() { local.CopyData(dst[:1]) }) // Wrong size.
	dst64 := make([]float64, len(want))
	require.Panics(t, func() { local.CopyData(dst64) }) // Wrong type.
}

// We test using FromAnyValue and AnyValueOf, due to Go generics limitations. See discussion in:
//
//	https://stackoverflow.com/questions/73591149/generics-type-inference-when-cascaded-calls-of-generic-functions
//	https://groups.google.com/g/golang-nuts/c/abILUXiD8-k
func testValueOf[T shapes.Number | complex64 | complex128](t *testing.T) {
	want := [][]T{{1, 2, 3}, {10, 11, 12}}
	var valueT Tensor
	require.NotPanics(t, func() { valueT = FromAnyValue(want) })
	got, ok := valueT.Value().([][]T)
	require.Truef(t, ok, "Failed to convert converted tensor to 2-dimensional slice -- value=%v", valueT.Value())
	assert.Equal(t, want, got)
}

func TestValueOf(t *testing.T) {
	// No conversion of different types, just from tensor.Local to a Go slice.
	testValueOf[float32](t)
	testValueOf[float64](t)
	testValueOf[int32](t)
	testValueOf[int64](t)
	testValueOf[uint8](t)
	testValueOf[uint32](t)
	testValueOf[uint64](t)
	testValueOf[complex64](t)
	testValueOf[complex128](t)
}

func TestTuples(t *testing.T) {
	elem0 := 5.0
	elem1 := []int64{1, 3}

	client, err := xla.NewClient("Host", 1, -1)
	if err != nil {
		t.Fatalf("Failed to create XLA client: %v", err)
	}
	localT := MakeLocalTuple(FromValue(elem0), FromValue(elem1))
	fmt.Printf("localT=%s\n", localT)
	var splits []*Local
	require.NotPanics(t, func() { splits = localT.SplitTuple() })
	for ii, split := range splits {
		fmt.Printf("\tSplit %d: %s\n", ii, split)
	}
	assert.Equal(t, elem0, splits[0].Value())
	assert.Equal(t, elem1, splits[1].Value())

	// Since SplitTupleError destroys local, we re-create it.
	localT = MakeLocalTuple(FromValue(elem0), FromValue(elem1))
	deviceT := localT.Device(client, client.DefaultDeviceOrdinal)
	if !deviceT.Shape().Eq(localT.shape) {
		t.Fatalf("Local tuple shape is %q, shapedBuffer is %q", localT.Shape(), deviceT.Shape())
	}
	fmt.Printf("deviceT.ShapedBuffer().ToString()=%s\n", deviceT.ShapedBuffer().String())

	// Split on-device.
	var deviceSplits []*Device
	require.NotPanics(t, func() { deviceSplits = deviceT.SplitTuple() })
	assert.Equalf(t, elem0, deviceSplits[0].Local().Value(), "Tuple[0] on device")
	assert.Equalf(t, elem1, deviceSplits[1].Local().Value(), "Tuple[1] on device")

	// Convert back to localT: recreate remote tuple, then clear the cache (otherwise it will simply use
	// the previous localT value), and re-convert to local.
	localT = MakeLocalTuple(FromValue(elem0), FromValue(elem1))
	deviceT = localT.Device(client, client.DefaultDeviceOrdinal)
	deviceT.ClearCache()
	localT = deviceT.Local() // New Local tensor.
	fmt.Printf("localT=%s\n", localT)

	require.NotPanics(t, func() { splits = localT.SplitTuple() })
	for ii, split := range splits {
		fmt.Printf("\tSplit %d: %s\n", ii, split)
	}
	assert.Equal(t, elem0, splits[0].Value())
	assert.Equal(t, elem1, splits[1].Value())
}

func TestSerialize(t *testing.T) {
	{
		values := [][]float64{{2}, {3}, {5}, {7}, {11}}
		localT := FromValue(values)
		buf := &bytes.Buffer{}
		enc := gob.NewEncoder(buf)
		require.NoError(t, localT.GobSerialize(enc))
		var err error
		dec := gob.NewDecoder(buf)
		localT, err = GobDeserialize(dec)
		require.NoError(t, err)
		require.Equal(t, values, localT.Value().([][]float64))
	}

	{
		values := [][]complex128{{2}, {3}, {5}, {7}, {11}}
		localT := FromValue(values)
		buf := &bytes.Buffer{}
		enc := gob.NewEncoder(buf)
		require.NoError(t, localT.GobSerialize(enc))
		var err error
		dec := gob.NewDecoder(buf)
		localT, err = GobDeserialize(dec)
		require.NoError(t, err)
		require.Equal(t, values, localT.Value().([][]complex128))
	}
}
