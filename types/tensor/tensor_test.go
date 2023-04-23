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
	"fmt"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/xla"
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
	shape, err = shapeForValue([][]int32{{3}})
	if shape.DType != shapes.InvalidDType {
		t.Fatalf("Wanted InvalidDType for int32, instead got %q", shape.DType)
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
		got := ToScalar[int](local)
		if got != want {
			t.Fatalf("Local read out got %d, wanted %d", got, want)
		}
	}

	// Test correct setting of 1D slice, dtype=float32
	{
		want := []float64{2, 5}
		local := FromValue(want)
		if local.error != nil {
			t.Fatalf("Failed to build scalar local: %v", err)
		}
		got, _ := local.Data().([]float64)
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
		got, _ := local.Data().([]float32)
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
		got, _ := local.Data().([]bool)
		if !reflect.DeepEqual(want, got) {
			t.Fatalf("Local read out got %v, wanted %v", got, want)
		}
	}
}

// We test using FromAnyValue and AnyValueOf, due to Go generics limitations. See discussion in:
//
//	https://stackoverflow.com/questions/73591149/generics-type-inference-when-cascaded-calls-of-generic-functions
//	https://groups.google.com/g/golang-nuts/c/abILUXiD8-k
func testValueOf[T shapes.Number](t *testing.T) {
	want := [][]T{{1, 2, 3}, {10, 11, 12}}
	local := FromAnyValue(want)
	if local.error != nil {
		t.Fatalf("Failed to build scalar local tensor: %v", local.error)
	}
	got, _ := AnyValueOf(local).([][]T)
	if !reflect.DeepEqual(want, got) {
		t.Fatalf("Local read out got %v, wanted %v", got, want)
	}
}

func TestValueOf(t *testing.T) {
	// No conversion of different types, just from tensor.Local to a Go slice.
	testValueOf[int](t)
	testValueOf[float32](t)
	testValueOf[float64](t)

	// Test conversion
	t0 := FromValue([][]float32{{1, 2}, {3, 4}})

	if got, want := ValueOf[[][]float64](t0), [][]float64{{1, 2}, {3, 4}}; !reflect.DeepEqual(got, want) {
		t.Errorf("Failed to convert (float32) %v to (float64) %v got (%s) %v", t0, want, reflect.TypeOf(got), got)
	}
	if got := ValueOf[[][][]int](t0); got != nil {
		// Wrong rank, should have returned nil.
		t.Errorf("Converting (float32) %v to [][][]64 returned (%s) %v!?", t0, reflect.TypeOf(got), got)
	}
	if got, want := ValueOf[[][]int](t0), [][]int{{1, 2}, {3, 4}}; !reflect.DeepEqual(got, want) {
		t.Errorf("Failed to convert (float32) %v to (int) %v got (%s) %v", t0, want, reflect.TypeOf(got), got)
	}
}

func TestToScalar(t *testing.T) {
	// Test conversion among different types.
	t0 := FromValue(3)
	if got := ToScalar[int](t0); got != 3 {
		t.Errorf("Failed to convert %s to int(3), got %d", t0, got)
	}
	if got := ToScalar[float64](t0); got != 3.0 {
		t.Errorf("Failed to convert %s to float64(3), got %f", t0, got)
	}

	t1 := FromValue(float32(5))
	if got := ToScalar[int](t1); got != 5 {
		t.Errorf("Failed to convert %s to int(5), got %d", t1, got)
	}
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
	if ToScalar[float64](splits[0]) != elem0 {
		t.Errorf("Tuple[0] transferred back: got %f, wanted %f", ToScalar[float64](splits[0]), elem0)
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
	fmt.Printf("deviceT.ToString()=%s\n", deviceT.ShapedBuffer().String())

	// Split on device.
	deviceSplits, err := deviceT.SplitTupleError()
	if err != nil {
		t.Fatalf("Failed to split tuple on device: %v", err)
	}
	if ToScalar[float64](deviceSplits[0].Local()) != elem0 {
		t.Errorf("Tuple[0] on device: got %f, wanted %f", ToScalar[float64](deviceSplits[0].Local()), elem0)
	}
	if !reflect.DeepEqual(deviceSplits[1].Local().Value(), elem1) {
		t.Errorf("Tuple[1] on device: got %v, wanted %v", deviceSplits[1].Local().Value(), elem1)
	}

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
	if ToScalar[float64](splits[0]) != elem0 {
		t.Errorf("Tuple[0] transferred back: got %f, wanted %f", ToScalar[float64](splits[0]), elem0)
	}
	if !reflect.DeepEqual(splits[1].Value(), elem1) {
		t.Errorf("Tuple[1] transferred back: got %v, wanted %v", splits[1].Value(), elem1)
	}
}
