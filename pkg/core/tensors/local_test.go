// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package tensors

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"strconv"
	"testing"
	"unsafe"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/dtypes/bfloat16"
	"github.com/gomlx/compute/dtypes/float16"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func cmpShapes(t *testing.T, shape, wantShape shapes.Shape, err error) {
	if err != nil {
		t.Fatalf("Failed to get shape (wanted %q) from value: %v", wantShape, err)
	}
	if !wantShape.Equal(shape) {
		t.Fatalf("Invalid shape %q, wanted %q", shape, wantShape)
	}
}

func TestFromValue(t *testing.T) {
	t.Run("shapeForValue", func(t *testing.T) {
		wantShape := shapes.Shape{DType: dtypes.Float32, Dimensions: []int{3, 2}}
		shape, err := shapeForValue([][]float32{{0, 0}, {1, 1}, {2, 2}})
		cmpShapes(t, shape, wantShape, err)

		wantShape = shapes.Shape{DType: dtypes.Float64, Dimensions: []int{1, 1, 1}}
		shape, err = shapeForValue([][][]float64{{{1}}})
		cmpShapes(t, shape, wantShape, err)

		if strconv.IntSize == 64 {
			wantShape = shapes.Shape{DType: dtypes.Int64, Dimensions: nil}
			shape, err = shapeForValue(5)
			cmpShapes(t, shape, wantShape, err)
		} else if strconv.IntSize == 32 {
			wantShape = shapes.Shape{DType: dtypes.Int32, Dimensions: nil}
			shape, err = shapeForValue(5)
			cmpShapes(t, shape, wantShape, err)
		} else {
			// For any other int size, it should panic.
			require.Panics(t, func() {
				shape, err = shapeForValue(5)
			})
		}

		wantShape = shapes.Shape{DType: dtypes.Int64, Dimensions: nil}
		shape, err = shapeForValue(5)
		cmpShapes(t, shape, wantShape, err)

		wantShape = shapes.Shape{DType: dtypes.Bool, Dimensions: []int{3, 2}}
		shape, err = shapeForValue([][]bool{{true, false}, {false, false}, {false, true}})
		cmpShapes(t, shape, wantShape, err)

		wantShape = shapes.Shape{DType: dtypes.Complex64, Dimensions: []int{2}}
		shape, err = shapeForValue([]complex64{1.0i, 1.0})
		cmpShapes(t, shape, wantShape, err)

		wantShape = shapes.Shape{DType: dtypes.Complex128, Dimensions: []int{2}}
		shape, err = shapeForValue([]complex128{1.0i, 1.0})
		cmpShapes(t, shape, wantShape, err)

		wantShape = shapes.Shape{DType: dtypes.Uint16, Dimensions: []int{1, 1}}
		shape, err = shapeForValue([][]uint16{{3}})
		cmpShapes(t, shape, wantShape, err)

		// Test for invalid `DType`.
		shape, err = shapeForValue([][]string{{"blah"}})
		if shape.DType != dtypes.InvalidDType {
			t.Fatalf("Wanted InvalidDType for string, instead got %q", shape.DType)
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
	})

	// Test the correct setting of scalar value, dtype=Int64.
	t.Run("int64", func(t *testing.T) {
		want := int64(5)
		var tensor *Tensor
		require.NotPanics(t, func() { tensor = FromValue(want) })
		assert.Equal(t, want, tensor.Value())
	})

	// Test the correct setting of scalar value for Go type `int` (maybe dtype=Int64 or Int32).
	t.Run("int", func(t *testing.T) {
		if strconv.IntSize == 64 {
			want := int64(5)
			var tensor *Tensor
			require.NotPanics(t, func() { tensor = FromValue(5) })
			assert.Equal(t, want, tensor.Value())
		} else if strconv.IntSize == 32 {
			want := int32(5)
			var tensor *Tensor
			require.NotPanics(t, func() { tensor = FromValue(5) })
			assert.Equal(t, want, tensor.Value())
		} else {
			// For any other int size, it should panic.
			require.Panics(t, func() {
				_ = FromValue(5)
			})
		}
	})

	// Test the correct setting of 1D slice, dtype=float64
	t.Run("float64", func(t *testing.T) {
		want := []float64{2, 5}
		var tensor *Tensor
		require.NotPanics(t, func() { tensor = FromValue(want) })
		assert.Equal(t, want, tensor.Value())
	})

	// Test the correct setting of 1D slice, dtype=float64
	t.Run("float32", func(t *testing.T) {
		want := []float32{1, 2, 3, 10, 11, 12}
		var tensor *Tensor
		require.NotPanics(t, func() { tensor = FromValue([][]float32{{1, 2, 3}, {10, 11, 12}}) })
		tensor.MustConstFlatData(func(flat any) {
			got, _ := flat.([]float32)
			require.Equal(t, want, got)
		})
		tensor.MustMutableFlatData(func(flat any) {
			got, _ := flat.([]float32)
			require.Equal(t, want, got)
		})
	})

	// Test 2D slice, dtype=Bool
	t.Run("bool", func(t *testing.T) {
		want := []bool{true, false, false, false, false, true}
		var tensor *Tensor
		require.NotPanics(t, func() {
			tensor = FromFlatDataAndDimensions(want, 3, 2)
		})
		require.NoError(t, tensor.Shape().Check(dtypes.Bool, 3, 2))
		tensor.MustConstFlatData(func(flat any) {
			got, _ := flat.([]bool)
			require.Equal(t, want, got)
		})
		tensor.MustMutableFlatData(func(flat any) {
			got, _ := flat.([]bool)
			require.Equal(t, want, got)
		})
	})

	// Test 2D slice, Go type=int, dtype=Int32 or Int64
	t.Run("2D-int", func(t *testing.T) {
		var tensor *Tensor
		require.NotPanics(t, func() {
			tensor = FromValue([][]int{{1, 3}, {5, 7}})
		})
		if strconv.IntSize == 64 {
			want := []int64{1, 3, 5, 7}
			tensor.MustConstFlatData(func(flat any) {
				got, _ := flat.([]int64)
				require.Equal(t, want, got)
			})
		} else if strconv.IntSize == 32 {
			want := []int32{1, 3, 5, 7}
			tensor.MustConstFlatData(func(flat any) {
				got, _ := flat.([]int32)
				require.Equal(t, want, got)
			})
		} // Other int sizes will panic on `FromValue`.
	})

	t.Run("BFloat16", func(t *testing.T) {
		var tensor *Tensor
		require.NotPanics(t, func() {
			// Test with infinite numbers, it should work as well.
			tensor = FromAnyValue(bfloat16.FromFloat32(float32(math.Inf(-1))))
		})
		require.Equal(t, tensor.Shape().DType, dtypes.BFloat16)
		require.True(t, tensor.IsScalar())
	})
}

// We test using FromAnyValue and AnyValueOf, due to Go generics limitations. See discussion in:
//
//	https://stackoverflow.com/questions/73591149/generics-type-inference-when-cascaded-calls-of-generic-functions
//	https://groups.google.com/g/golang-nuts/c/abILUXiD8-k
func testValueOf[T dtypes.Number | complex64 | complex128](t *testing.T) {
	want := [][]T{{1, 2, 3}, {10, 11, 12}}
	var tensor *Tensor
	require.NotPanics(t, func() { tensor = FromAnyValue(want) })
	got, ok := tensor.Value().([][]T)
	require.Truef(
		t,
		ok,
		"Failed to convert converted tensor to 2-dimensional slice -- want=%v, value=%v",
		want,
		tensor.Value(),
	)

	// assert.Equal is not deep, so we have to assert the sub-slices.
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

func TestSerialization(t *testing.T) {
	{
		values := [][]float64{{2}, {3}, {5}, {7}, {11}}
		var tensor *Tensor
		require.NotPanics(t, func() { tensor = FromValue(values) })
		buf := &bytes.Buffer{}
		enc := gob.NewEncoder(buf)
		require.NoError(t, tensor.GobSerialize(enc))
		fmt.Printf("\t%#v serialized to %d bytes\n", values, buf.Len())
		var err error
		dec := gob.NewDecoder(buf)
		tensor, err = GobDeserialize(dec)
		require.NoError(t, err)
		require.Equal(t, values, tensor.Value().([][]float64))
	}

	{
		values := [][]complex128{{2}, {3}, {5}, {7}, {11}}
		var tensor *Tensor
		require.NotPanics(t, func() { tensor = FromValue(values) })
		buf := &bytes.Buffer{}
		enc := gob.NewEncoder(buf)

		// Serialized repeats times:
		repeats := 10
		for range repeats {
			require.NoError(t, tensor.GobSerialize(enc))
		}
		fmt.Printf("\t%#v serialized %d times to %d bytes\n", values, repeats, buf.Len())

		// Deserialize repeats times:
		dec := gob.NewDecoder(buf)
		for range repeats {
			var err error
			tensor, err = GobDeserialize(dec)
			require.NoError(t, err)
			require.Equal(t, values, tensor.Value().([][]complex128))
			tensor.MustFinalizeAll()
		}
	}
}

func testSaveLoadStumpImpl(t *testing.T, tensor *Tensor) (loadedTensor *Tensor) {
	dtype := tensor.DType()
	fmt.Printf("\ttesting Save&Load for dtype %s\n", dtype)

	// Create a temporary file and get its name.
	tempFile, err := os.CreateTemp("", fmt.Sprintf(
		"gomlx_tensor_test_%s_*.txt", dtype))
	if err != nil {
		t.Fatal("Error creating temp file:", err)
	}
	fileName := tempFile.Name()
	_ = tempFile.Close()
	defer func() { _ = os.Remove(tempFile.Name()) }()

	// Save tensor.
	require.NoErrorf(t, tensor.Save(fileName), "Saving tensor of dtype %s", dtype)

	// Re-load tensor.
	loadedTensor, err = Load(fileName)
	require.NoErrorf(t, err, "Loading tensor of dtype %s", dtype)
	return
}

func testSaveLoadGenericsImpl[T dtypes.Number](t *testing.T) {
	values := []T{0, 1, 2, 3, 4, 11}
	dtype := dtypes.FromGenericsType[T]()
	var tensor *Tensor
	require.NotPanics(t, func() { tensor = FromFlatDataAndDimensions(values, 3, 2) })

	// Save and re-load the tensor:
	loadedTensor := testSaveLoadStumpImpl(t, tensor)

	// Check loadedTensor contents.
	require.NoErrorf(
		t,
		loadedTensor.Shape().Check(dtype, 3, 2),
		"Loaded tensor for dtype %s got shape %s",
		dtype,
		loadedTensor.Shape(),
	)
	loadedTensor.MustConstFlatData(func(flatAny any) {
		flat := flatAny.([]T)
		require.Equal(t, values, flat)
	})
}

func testSaveLoadBool(t *testing.T) {
	values := []bool{false, true, true, false, true, false}
	var tensor *Tensor
	require.NotPanics(t, func() { tensor = FromFlatDataAndDimensions(values, 3, 2) })
	dtype := tensor.DType()

	// Save and re-load the tensor:
	loadedTensor := testSaveLoadStumpImpl(t, tensor)

	// Check loadedTensor contents.
	require.NoErrorf(
		t,
		loadedTensor.Shape().Check(dtype, 3, 2),
		"Loaded tensor for dtype %s got shape %s",
		dtype,
		loadedTensor.Shape(),
	)
	loadedTensor.MustConstFlatData(func(flatAny any) {
		flat := flatAny.([]bool)
		require.Equal(t, values, flat)
	})
}

func testSaveLoadFloat16(t *testing.T) {
	values := make([]float16.Float16, 6)
	for ii, v := range []float32{0, 1, 2, 3, 4, 11} {
		values[ii] = float16.FromFloat32(v)
	}
	var tensor *Tensor
	require.NotPanics(t, func() { tensor = FromFlatDataAndDimensions(values, 3, 2) })
	dtype := tensor.DType()

	// Save and re-load the tensor:
	loadedTensor := testSaveLoadStumpImpl(t, tensor)

	// Check loadedTensor contents.
	require.NoErrorf(
		t,
		loadedTensor.Shape().Check(dtype, 3, 2),
		"Loaded tensor for dtype %s got shape %s",
		dtype,
		loadedTensor.Shape(),
	)
	loadedTensor.MustConstFlatData(func(flatAny any) {
		flat := flatAny.([]float16.Float16)
		require.Equal(t, values, flat)
	})
}

func testSaveLoadBFloat16(t *testing.T) {
	values := make([]bfloat16.BFloat16, 6)
	for ii, v := range []float32{0, 1, 2, 3, 4, 11} {
		values[ii] = bfloat16.FromFloat32(v)
	}
	var tensor *Tensor
	require.NotPanics(t, func() { tensor = FromFlatDataAndDimensions(values, 3, 2) })
	dtype := tensor.DType()

	// Save and re-load the tensor:
	loadedTensor := testSaveLoadStumpImpl(t, tensor)

	// Check loadedTensor contents.
	require.NoErrorf(
		t,
		loadedTensor.Shape().Check(dtype, 3, 2),
		"Loaded tensor for dtype %s got shape %s",
		dtype,
		loadedTensor.Shape(),
	)
	loadedTensor.MustConstFlatData(func(flatAny any) {
		flat := flatAny.([]bfloat16.BFloat16)
		require.Equal(t, values, flat)
	})
}

func TestSaveLoad(t *testing.T) {
	testSaveLoadGenericsImpl[int8](t)
	testSaveLoadGenericsImpl[int16](t)
	testSaveLoadGenericsImpl[int32](t)
	testSaveLoadGenericsImpl[int64](t)

	testSaveLoadGenericsImpl[uint8](t)
	testSaveLoadGenericsImpl[uint16](t)
	testSaveLoadGenericsImpl[uint32](t)
	testSaveLoadGenericsImpl[uint64](t)

	testSaveLoadGenericsImpl[float32](t)
	testSaveLoadGenericsImpl[float64](t)

	testSaveLoadGenericsImpl[complex64](t)
	testSaveLoadGenericsImpl[complex128](t)

	testSaveLoadBool(t)
	testSaveLoadFloat16(t)
	testSaveLoadBFloat16(t)
}

func TestClone(t *testing.T) {
	tensor := FromValue([][]int32{{0, 1}, {3, 5}, {7, 11}})
	clone, err := tensor.LocalClone()
	require.NoError(t, err)

	// Change the original tensor and check that the cloned version is unchanged
	tensor.MustMutableFlatData(func(flatAny any) {
		flat := flatAny.([]int32)
		flat[0] = 100
	})
	require.NoError(t, clone.Shape().Check(dtypes.Int32, 3, 2))
	require.Equal(t, []int32{0, 1, 3, 5, 7, 11}, MustCopyFlatData[int32](clone))
}

func TestBytes(t *testing.T) {
	tensor := FromValue([][]int32{{0, 1}, {3, 5}, {7, 11}})
	require.NoError(t, tensor.ConstBytes(func(data []byte) {
		require.Equal(t, 6*4 /* sizeof(int32) */, len(data))
		flat := unsafe.Slice((*int32)(unsafe.Pointer(&data[0])), 6)
		require.Equal(t, []int32{0, 1, 3, 5, 7, 11}, flat)
	}))
	require.NoError(t, tensor.MutableBytes(func(data []byte) {
		require.Equal(t, 6*4 /* sizeof(int32) */, len(data))
		flat := unsafe.Slice((*int32)(unsafe.Pointer(&data[0])), 6)
		flat[0] = 13
		flat[5] = 17
	}))
	require.Equal(t, [][]int32{{13, 1}, {3, 5}, {7, 17}}, tensor.Value())
}

func TestAssign(t *testing.T) {
	tensor := FromShape(shapes.Make(dtypes.Float64, 2, 3))

	// Wrong dtype:
	require.Error(t, AssignFlatData(tensor, []float32{0, 1, 2, 3, 4, 5}))

	// Wrong flat size:
	require.Error(t, AssignFlatData(tensor, []float64{0, 1, 2, 3, 4, 5, 6}))

	// Check assignment happened:
	values := []float64{0, 1, 2, 3, 4, 5}
	require.NoError(t, AssignFlatData(tensor, values))
	require.Equal(t, values, MustCopyFlatData[float64](tensor))
}

func TestPackedSubByteValueSafe(t *testing.T) {
	t.Run("Int4", func(t *testing.T) {
		// Shape [2, 4]: 8 values packed into 4 bytes, 2 nibbles per byte (little-endian).
		tensor := FromShape(shapes.Make(dtypes.Int4, 2, 4))
		require.NoError(t, tensor.MutableBytes(func(data []byte) {
			data[0] = 0x21 // values: 1, 2
			data[1] = 0x43 // values: 3, 4
			data[2] = 0x65 // values: 5, 6
			data[3] = 0xF7 // values: 7, -1 (0xF sign-extended)
		}))
		val, err := tensor.ValueSafe()
		require.NoError(t, err)
		got, ok := val.([][]int8)
		require.True(t, ok, "expected [][]int8, got %T", val)
		assert.Equal(t, [][]int8{{1, 2, 3, 4}, {5, 6, 7, -1}}, got)

		s := tensor.Summary(4)
		assert.Contains(t, s, dtypes.Int4.String())
	})

	t.Run("Uint4", func(t *testing.T) {
		// Shape [2, 2]: 4 values in 2 bytes.
		tensor := FromShape(shapes.Make(dtypes.Uint4, 2, 2))
		require.NoError(t, tensor.MutableBytes(func(data []byte) {
			data[0] = 0xBA // values: 0xA, 0xB
			data[1] = 0xDC // values: 0xC, 0xD
		}))
		val, err := tensor.ValueSafe()
		require.NoError(t, err)
		got, ok := val.([][]int8)
		require.True(t, ok, "expected [][]uint8, got %T", val)
		assert.Equal(t, [][]int8{{0xA, 0xB}, {0xC, 0xD}}, got)
	})

	t.Run("Int2", func(t *testing.T) {
		// Shape [2, 4]: 8 values packed into 2 bytes, 4 crumbs per byte.
		// Byte layout: bits [1:0]=val0, [3:2]=val1, [5:4]=val2, [7:6]=val3.
		tensor := FromShape(shapes.Make(dtypes.Int2, 2, 4))
		require.NoError(t, tensor.MutableBytes(func(data []byte) {
			// Row 0: values 0, 1, -2, -1 → crumbs 00, 01, 10, 11 → 0b11_10_01_00 = 0xE4
			data[0] = 0xE4
			// Row 1: values 1, 0, -1, -2 → crumbs 01, 00, 11, 10 → 0b10_11_00_01 = 0xB1
			data[1] = 0xB1
		}))
		val, err := tensor.ValueSafe()
		require.NoError(t, err)
		got, ok := val.([][]int8)
		require.True(t, ok, "expected [][]int8, got %T", val)
		assert.Equal(t, [][]int8{{0, 1, -2, -1}, {1, 0, -1, -2}}, got)
	})

	t.Run("Uint2", func(t *testing.T) {
		// Shape [2, 4]: 8 values packed into 2 bytes.
		tensor := FromShape(shapes.Make(dtypes.Uint2, 2, 4))
		require.NoError(t, tensor.MutableBytes(func(data []byte) {
			// Row 0: values 0, 1, 2, 3 → crumbs 00, 01, 10, 11 → 0xE4
			data[0] = 0xE4
			// Row 1: values 3, 2, 1, 0 → crumbs 11, 10, 01, 00 → 0x1B
			data[1] = 0x1B
		}))
		val, err := tensor.ValueSafe()
		require.NoError(t, err)
		got, ok := val.([][]int8)
		require.True(t, ok, "expected [][]uint8, got %T", val)
		assert.Equal(t, [][]int8{{0, 1, 2, 3}, {3, 2, 1, 0}}, got)
	})

	t.Run("ScalarInt4", func(t *testing.T) {
		tensor := FromShape(shapes.Make(dtypes.Int4))
		require.NoError(t, tensor.MutableBytes(func(data []byte) {
			data[0] = 0xF5 // low nibble = 5, high nibble unused for scalar
		}))
		val, err := tensor.ValueSafe()
		require.NoError(t, err)
		assert.Equal(t, int8(5), val)
	})

	t.Run("Summary()", func(t *testing.T) {
		raw := xslices.Iota(byte(0), 256)
		tensor, err := FromRaw(nil, 0, shapes.Make(dtypes.Uint4, 16, 16, 2), raw)
		require.NoError(t, err)
		summary := tensor.Summary(1)
		want := `[16][16][2]U4{
 {{0, 0},
  {1, 0},
  {2, 0},
  ...,
  {13, 0},
  {14, 0},
  {15, 0}},
 ...,
 {{0, 15},
  {1, 15},
  {2, 15},
  ...,
  {13, 15},
  {14, 15},
  {15, 15}}}`
		assert.Equal(t, want, summary)
	})
}

func TestFromShape(t *testing.T) {
	testCases := []struct {
		dtype   dtypes.DType
		size    int
		wantLen int
	}{
		{dtypes.Uint2, 10, 3},
		{dtypes.Int2, 10, 3},
		{dtypes.Uint4, 10, 5},
		{dtypes.Int4, 10, 5},
		{dtypes.Uint2, 12, 3},
		{dtypes.Uint4, 11, 6},
		{dtypes.Uint2, 0, 0},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s_size_%d", tc.dtype, tc.size), func(t *testing.T) {
			shape := shapes.Make(tc.dtype, tc.size)
			tensor := FromShape(shape)
			err := tensor.ConstFlatData(func(flat any) {
				slice, ok := flat.([]byte)
				require.Truef(t, ok, "expected []byte flat data for packed dtype %s", tc.dtype)
				require.Equalf(t, tc.wantLen, len(slice), "invalid flat data length for %s", tc.dtype)
			})
			require.NoError(t, err)
		})
	}
}

func TestEmptySizeAccess(t *testing.T) {
	testCases := []struct {
		dtype dtypes.DType
	}{
		{dtypes.Float32},
		{dtypes.Int8},
	}

	for _, tc := range testCases {
		t.Run(fmt.Sprintf("%s", tc.dtype), func(t *testing.T) {
			shape := shapes.Make(tc.dtype, 10, 0, 3)
			tensor := FromShape(shape)

			// t.MutableFlatData
			err := tensor.MutableFlatData(func(flat any) {
				switch tc.dtype {
				case dtypes.Float32:
					slice := flat.([]float32)
					require.Len(t, slice, 0)
				case dtypes.Int8:
					slice := flat.([]int8)
					require.Len(t, slice, 0)
				}
			})
			require.NoError(t, err)

			// t.ConstFlatData
			err = tensor.ConstFlatData(func(flat any) {
				switch tc.dtype {
				case dtypes.Float32:
					slice := flat.([]float32)
					require.Len(t, slice, 0)
				case dtypes.Int8:
					slice := flat.([]int8)
					require.Len(t, slice, 0)
				}
			})
			require.NoError(t, err)

			// t.MutableBytes
			err = tensor.MutableBytes(func(data []byte) {
				require.Len(t, data, 0)
			})
			require.NoError(t, err)

			// tensors.MutableFlatData[T]
			if tc.dtype == dtypes.Float32 {
				err = MutableFlatData(tensor, func(flat []float32) {
					require.Len(t, flat, 0)
				})
				require.NoError(t, err)
			} else if tc.dtype == dtypes.Int8 {
				err = MutableFlatData(tensor, func(flat []int8) {
					require.Len(t, flat, 0)
				})
				require.NoError(t, err)
			}

			// tensors.ConstFlatData[T]
			if tc.dtype == dtypes.Float32 {
				err = ConstFlatData(tensor, func(flat []float32) {
					require.Len(t, flat, 0)
				})
				require.NoError(t, err)
			} else if tc.dtype == dtypes.Int8 {
				err = ConstFlatData(tensor, func(flat []int8) {
					require.Len(t, flat, 0)
				})
				require.NoError(t, err)
			}
		})
	}
}
