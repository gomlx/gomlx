package slices

import (
	"github.com/stretchr/testify/assert"
	"testing"
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
