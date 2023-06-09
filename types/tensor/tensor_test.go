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

	wantShape = shapes.Shape{DType: shapes.Int64, Dimensions: nil}
	shape, err = shapeForValue(5)
	cmpShapes(t, shape, wantShape, err)

	wantShape = shapes.Shape{DType: shapes.Bool, Dimensions: []int{3, 2}}
	shape, err = shapeForValue([][]bool{{true, false}, {false, false}, {false, true}})
	cmpShapes(t, shape, wantShape, err)

	// Test for invalid dtypes.
	shape, err = shapeForValue([][]uint16{{3}})
	if shape.DType != shapes.InvalidDType {
		t.Fatalf("Wanted InvalidDType for uint16, instead got %q", shape.DType)
	}
	if err == nil {
		t.Fatalf("Should have returned error for unsupported DType")
	}

	// Test for irregularly shaped slices.
	shape, err = shapeForValue([][][]int{{{1}}, {{1, 2}}})
	if err == nil {
		t.Fatalf("Should have returned error for irregularly shaped slices")
	}
	fmt.Printf("\tExpected error: %v\n", err)

	// Test correct setting of scalar value, dtype=int.
	{
		want := 5
		local := FromValue[int](want)
		if local.error != nil {
			t.Fatalf("Failed to build scalar local tensor: %v", err)
		}
		if local.Empty() {
			t.Fatalf("Failed to build scalar local tensor: got IsNil Local tensor")
		}
		got := local.Value().(int)
		if got != want {
			t.Fatalf("Local read out got %d, wanted %d", got, want)
		}
	}

	// Test correct setting of 1D slice, dtype=float64
	{
		want := []float64{2, 5}
		local := FromValue(want)
		if local.error != nil {
			t.Fatalf("Failed to build scalar local: %v", err)
		}
		got, _ := local.Flat().([]float64)
		if !slices.DeepSliceCmp(want, got, slices.Equal[float64]) {
			t.Fatalf("Local read out got %v, wanted %v", got, want)
		}
	}

	// Test correct setting of 1D slice, dtype=float64
	{
		want := []float32{1, 2, 3, 10, 11, 12}
		local := FromValue([][]float32{{1, 2, 3}, {10, 11, 12}})
		if local.error != nil {
			t.Fatalf("Failed to build scalar local: %v", err)
		}
		got, _ := local.Flat().([]float32)
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
		got, _ := local.Flat().([]bool)
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("Local read out got %v, wanted %v", got, want)
		}
	}
}

func TestLocal_CopyFlat(t *testing.T) {
	want := []float32{1, 2, 3, 10, 11, 12}
	local := FromValue([][]float32{{1, 2, 3}, {10, 11, 12}})
	require.NoError(t, local.Error())
	dst := make([]float32, len(want))
	require.NoError(t, local.CopyFlat(dst))
	require.Equal(t, want, dst)

	// Check failures:
	require.Error(t, local.CopyFlat(dst[:1])) // Wrong size.
	dst64 := make([]float64, len(want))
	require.Error(t, local.CopyFlat(dst64)) // Wrong type.
}

// We test using FromAnyValue and AnyValueOf, due to Go generics limitations. See discussion in:
//
//	https://stackoverflow.com/questions/73591149/generics-type-inference-when-cascaded-calls-of-generic-functions
//	https://groups.google.com/g/golang-nuts/c/abILUXiD8-k
func testValueOf[T shapes.Number](t *testing.T) {
	want := [][]T{{1, 2, 3}, {10, 11, 12}}
	valueT := FromAnyValue(want)
	require.NoError(t, valueT.Error())
	got, ok := valueT.Value().([][]T)
	require.Truef(t, ok, "Failed to convert converted tensor to 2-dimensional slice -- value=%v", valueT.Value())
	assert.Equal(t, want, got)
}

func TestValueOf(t *testing.T) {
	// No conversion of different types, just from tensor.Local to a Go slice.
	testValueOf[uint8](t)
	testValueOf[int32](t)
	testValueOf[int](t)
	testValueOf[float32](t)
	testValueOf[float64](t)
}

func TestTuples(t *testing.T) {
	elem0 := 5.0
	elem1 := []int{1, 3}

	client, err := xla.NewClient("Host", 1, -1)
	if err != nil {
		t.Fatalf("Failed to create XLA client: %v", err)
	}
	localT := MakeLocalTuple(FromValue(elem0), FromValue(elem1))
	fmt.Printf("localT=%s\n", localT)
	splits, err := localT.SplitTuple()
	if err != nil {
		t.Fatalf("Failed to split local tuple: %v", err)
	}
	for ii, split := range splits {
		fmt.Printf("\tSplit %d: %s\n", ii, split)
	}
	if splits[0].Value().(float64) != elem0 {
		t.Errorf("Tuple[0] transferred back: got %f, wanted %f", splits[0].Value().(float64), elem0)
	}
	if !reflect.DeepEqual(splits[1].Value(), elem1) {
		t.Errorf("Tuple[1] transferred back: got %v, wanted %v", splits[1].Value(), elem1)
	}

	// Since SplitTupleError destroys local, we re-create it.
	localT = MakeLocalTuple(FromValue(elem0), FromValue(elem1))
	deviceT := localT.Device(client, client.DefaultDeviceOrdinal)
	if !deviceT.Shape().Eq(localT.shape) {
		t.Fatalf("Local tuple shape is %q, shapedBuffer is %q", localT.Shape(), deviceT.Shape())
	}
	fmt.Printf("deviceT.ShapedBuffer().ToString()=%s\n", deviceT.ShapedBuffer().String())

	// Split on device.
	deviceSplits, err := deviceT.SplitTupleError()
	if err != nil {
		t.Fatalf("Failed to split tuple on device: %v", err)
	}
	assert.Equalf(t, elem0, deviceSplits[0].Local().Value(), "Tuple[0] on device")
	assert.Equalf(t, elem1, deviceSplits[1].Local().Value(), "Tuple[1] on device")

	// Convert back to localT: recreate remote tuple, then clear the cache (otherwise it will simply use
	// the previous localT value), and re-convert to local.
	localT = MakeLocalTuple(FromValue(elem0), FromValue(elem1))
	deviceT = localT.Device(client, client.DefaultDeviceOrdinal)
	deviceT.ClearCache()
	localT = deviceT.Local()
	fmt.Printf("localT=%s\n", localT)
	splits, err = localT.SplitTuple()
	if err != nil {
		t.Fatalf("Failed to split local tuple: %v", err)
	}
	for ii, split := range splits {
		fmt.Printf("\tSplit %d: %s\n", ii, split)
	}
	if splits[0].Value().(float64) != elem0 {
		t.Errorf("Tuple[0] transferred back: got %f, wanted %f", splits[0].Value().(float64), elem0)
	}
	if !reflect.DeepEqual(splits[1].Value(), elem1) {
		t.Errorf("Tuple[1] transferred back: got %v, wanted %v", splits[1].Value(), elem1)
	}
}

func TestSerialize(t *testing.T) {
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
