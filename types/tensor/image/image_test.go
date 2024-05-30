package image

import (
	"testing"

	"github.com/gomlx/gomlx/types/shapes"
	"google3/third_party/golang/testify/assert/assert"
)

func TestGetUpSampledSizes(t *testing.T) {
	s := shapes.Make(shapes.F32, 2, 3, 4, 5)
	assert.Equal(t, []int{2, 6, 8, 5}, GetUpSampledSizes(s, ChannelsLast, 2))
	assert.Equal(t, []int{2, 3, 12, 15}, GetUpSampledSizes(s, ChannelsFirst, 3))
}
