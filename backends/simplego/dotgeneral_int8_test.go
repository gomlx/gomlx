//go:build !noasm && arm64

package simplego

import (
	"testing"
	"unsafe"
)

func TestDotProductInt8Assembly(t *testing.T) {
	tests := []struct {
		name string
		a    []int8
		b    []int8
		want int32
	}{
		{
			name: "simple_4_elements",
			a:    []int8{1, 2, 3, 4},
			b:    []int8{1, 1, 1, 1},
			want: 10, // 1+2+3+4
		},
		{
			name: "16_elements_smmla_path",
			a:    []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			b:    []int8{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			want: 136, // sum(1..16)
		},
		{
			name: "17_elements_smmla_plus_scalar",
			a:    []int8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17},
			b:    []int8{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			want: 153, // sum(1..17)
		},
		{
			name: "negative_values",
			a:    []int8{-1, -2, 3, 4},
			b:    []int8{2, 2, 2, 2},
			want: 8, // -2 + -4 + 6 + 8
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			aPtr := unsafe.Pointer(&tt.a[0])
			bPtr := unsafe.Pointer(&tt.b[0])
			got := dotProductInt8_neon_asm(aPtr, bPtr, int64(len(tt.a)))
			if got != tt.want {
				t.Errorf("dotProductInt8_neon_asm() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestDotProductUint8Assembly(t *testing.T) {
	tests := []struct {
		name string
		a    []uint8
		b    []uint8
		want int32
	}{
		{
			name: "simple_4_elements",
			a:    []uint8{1, 2, 3, 4},
			b:    []uint8{1, 1, 1, 1},
			want: 10,
		},
		{
			name: "16_elements_ummla_path",
			a:    []uint8{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},
			b:    []uint8{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
			want: 136,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			aPtr := unsafe.Pointer(&tt.a[0])
			bPtr := unsafe.Pointer(&tt.b[0])
			got := dotProductUint8_neon_asm(aPtr, bPtr, int64(len(tt.a)))
			if got != tt.want {
				t.Errorf("dotProductUint8_neon_asm() = %v, want %v", got, tt.want)
			}
		})
	}
}
