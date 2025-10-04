package xslices

import (
	"flag"
	"fmt"
	"strconv"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMap(t *testing.T) {
	count := 17
	in := make([]int, count)
	for ii := 0; ii < count; ii++ {
		in[ii] = ii
	}
	out := Map(in, func(v int) int32 { return int32(v + 1) })
	for ii := 0; ii < count; ii++ {
		assert.Equalf(t, int32(ii+1), out[ii], "element %d doesn't match", ii)
	}
}

func TestMapParallel(t *testing.T) {
	count := 17
	in := make([]int, count)
	for ii := 0; ii < count; ii++ {
		in[ii] = ii
	}
	out := MapParallel(in, func(v int) int32 { return int32(v + 1) })
	for ii := 0; ii < count; ii++ {
		assert.Equalf(t, int32(ii+1), out[ii], "element %d doesn't match", ii)
	}
}

func TestAtAndLast(t *testing.T) {
	slice := []int{0, 1, 2, 3, 4, 5}
	assert.Equal(t, 5, At(slice, -1))
	assert.Equal(t, 4, At(slice, -2))
	assert.Equal(t, 5, Last(slice))
}

func TestPop(t *testing.T) {
	slice := []int{0, 1, 2, 3, 4, 5}
	var got int
	got, slice = Pop(slice)
	assert.Equal(t, 5, got)
	assert.Len(t, slice, 5)

	got, slice = Pop(slice)
	assert.Equal(t, 4, got)
	assert.Len(t, slice, 4)
}

type StringerFloat float64

func (f StringerFloat) String() string {
	return fmt.Sprintf("%.02f", float64(f))
}

func TestSliceFlag(t *testing.T) {
	f1Ptr := Flag("f1", []int{2, 3}, "f1 flag test", strconv.Atoi)
	assert.Equal(t, []int{2, 3}, *f1Ptr)
	require.NoError(t, flag.Set("f1", "3,4,5"))
	assert.Equal(t, []int{3, 4, 5}, *f1Ptr)
	f1Flag := flag.Lookup("f1")
	require.NotNil(t, f1Flag)
	assert.Equal(t, "2,3", f1Flag.DefValue)

	f2Ptr := Flag("f2", []StringerFloat{2.0, 3.0}, "f2 flag test",
		func(v string) (StringerFloat, error) {
			f, err := strconv.ParseFloat(v, 64)
			return StringerFloat(f), err
		})
	assert.Equal(t, []StringerFloat{2, 3}, *f2Ptr)
	require.NoError(t, flag.Set("f2", "3,4,5"))
	assert.Equal(t, []StringerFloat{3, 4, 5}, *f2Ptr)
	f2Flag := flag.Lookup("f2")
	require.NotNil(t, f2Flag)
	assert.Equal(t, "2.00,3.00", f2Flag.DefValue)
}

func TestOperations(t *testing.T) {
	s := []float32{1.0, -3.0, 2.0}
	assert.Equal(t, float32(2), Max(s))
	assert.Equal(t, float32(-3), Min(s))

	SetAt(s, 0, 10)
	assert.Equal(t, float32(10), s[0])
	SetLast(s, 100)
	assert.Equal(t, float32(100), s[2])

	FillSlice(s, -7)
	assert.Equal(t, []float32{-7, -7, -7}, s)
	FillAnySlice(s, float32(11))
	assert.Equal(t, []float32{11, 11, 11}, s)

	md := MultidimensionalSliceWithValue(int64(13), 2, 1, 1)
	assert.Equal(t, [][][]int64{{{13}}, {{13}}}, md)

	s2d := Slice2DWithValue(int8(67), 2, 1)
	assert.Equal(t, [][]int8{{67}, {67}}, s2d)

	s3d := Slice3DWithValue(uint8(23), 1, 1, 3)
	assert.Equal(t, [][][]uint8{{{23, 23, 23}}}, s3d)
}
