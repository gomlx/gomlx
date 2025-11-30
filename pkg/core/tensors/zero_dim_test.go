package tensors

import (
	"reflect"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func TestZeroDimFromScalarAndDimensions(t *testing.T) {
	testCases := []struct {
		name       string
		makeTensor func() *Tensor
		dimensions []int
	}{
		{
			name: "rank_1_zero_dim",
			makeTensor: func() *Tensor {
				tensor := FromScalarAndDimensions(float32(0), 0)
				tensor.MustConstFlatData(func(flat any) {
					got, _ := flat.([]float32)
					require.Equal(t, 0, len(got))
				})
				return tensor
			},
			dimensions: []int{0},
		},
		{
			name: "rank_2_zero_first",
			makeTensor: func() *Tensor {
				tensor := FromScalarAndDimensions(int32(0), 0, 5)
				tensor.MustConstFlatData(func(flat any) {
					got, _ := flat.([]int32)
					require.Equal(t, 0, len(got))
				})
				return tensor
			},
			dimensions: []int{0, 5},
		},
		{
			name: "rank_2_zero_second",
			makeTensor: func() *Tensor {
				tensor := FromScalarAndDimensions(int64(0), 3, 0)
				tensor.MustConstFlatData(func(flat any) {
					got, _ := flat.([]int64)
					require.Equal(t, 0, len(got))
				})
				return tensor
			},
			dimensions: []int{3, 0},
		},
		{
			name: "rank_3_zero_middle",
			makeTensor: func() *Tensor {
				tensor := FromScalarAndDimensions(float64(0), 2, 0, 4)
				tensor.MustConstFlatData(func(flat any) {
					got, _ := flat.([]float64)
					require.Equal(t, 0, len(got))
				})
				return tensor
			},
			dimensions: []int{2, 0, 4},
		},
		{
			name: "rank_4_multiple_zeros",
			makeTensor: func() *Tensor {
				tensor := FromScalarAndDimensions(float64(0), 1, 0, 0, 3)
				tensor.MustConstFlatData(func(flat any) {
					got, _ := flat.([]float64)
					require.Equal(t, 0, len(got))
				})
				return tensor
			},
			dimensions: []int{1, 0, 0, 3},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			tensor := tc.makeTensor()
			require.Equal(t, tc.dimensions, tensor.shape.Dimensions)
		})
	}
}

func TestZeroDimFromShape(t *testing.T) {
	testCases := []struct {
		name       string
		dimensions []int
		dtype      dtypes.DType
	}{
		{"rank_1_zero_dim", []int{0}, dtypes.Int32},
		{"rank_2_zero_first", []int{0, 5}, dtypes.Float64},
		{"rank_2_zero_second", []int{3, 0}, dtypes.Int64},
		{"rank_3_zero_middle", []int{2, 0, 4}, dtypes.Int8},
		{"rank_4_multiple_zeros", []int{1, 0, 0, 3}, dtypes.Float32},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			shape := shapes.Make(tc.dtype, tc.dimensions...)
			tensor := FromShape(shape)
			tensor.MustConstFlatData(func(flat any) {
				got := reflect.ValueOf(flat)
				require.Equal(t, 0, got.Len())
			})
			require.Equal(t, tc.dimensions, tensor.shape.Dimensions)
		})
	}
}

func TestZeroDimFromFlat(t *testing.T) {
	testCases := []struct {
		name       string
		makeTensor func() *Tensor
		dimensions []int
	}{
		{
			name: "rank_1_zero_dim",
			makeTensor: func() *Tensor {
				tensor := FromScalarAndDimensions(float32(0), 0)
				tensor.MustConstFlatData(func(flat any) {
					got, _ := flat.([]float32)
					require.Equal(t, len(got), 0)
				})
				return tensor
			},
			dimensions: []int{0},
		},
		{
			name: "rank_2_zero_first",
			makeTensor: func() *Tensor {
				tensor := FromFlatDataAndDimensions([]int32{}, 0, 5)
				tensor.MustConstFlatData(func(flat any) {
					got, _ := flat.([]int32)
					require.Equal(t, len(got), 0)
				})
				return tensor
			},
			dimensions: []int{0, 5},
		},
		{
			name: "rank_2_zero_second",
			makeTensor: func() *Tensor {
				tensor := FromFlatDataAndDimensions([]int64{}, 3, 0)
				tensor.MustConstFlatData(func(flat any) {
					got, _ := flat.([]int64)
					require.Equal(t, len(got), 0)
				})
				return tensor
			},
			dimensions: []int{3, 0},
		},
		{
			name: "rank_3_zero_middle",
			makeTensor: func() *Tensor {
				tensor := FromFlatDataAndDimensions([]float64{}, 2, 0, 4)
				tensor.MustConstFlatData(func(flat any) {
					got, _ := flat.([]float64)
					require.Equal(t, len(got), 0)
				})
				return tensor
			},
			dimensions: []int{2, 0, 4},
		},
		{
			name: "rank_4_multiple_zeros",
			makeTensor: func() *Tensor {
				tensor := FromFlatDataAndDimensions([]float64{}, 1, 0, 0, 3)
				tensor.MustConstFlatData(func(flat any) {
					got, _ := flat.([]float64)
					require.Equal(t, len(got), 0)
				})
				return tensor
			},
			dimensions: []int{1, 0, 0, 3},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			testTensor := tc.makeTensor()
			require.Equal(t, testTensor.shape.Dimensions, tc.dimensions)
		})
	}
}
