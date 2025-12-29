package notimplemented

import (
	"github.com/gomlx/gomlx/backends"
)

// DynamicReshapeWithBounds reshapes operand to the shape specified by outputShape tensor,
// using explicit dimension bounds for XLA compilation.
func (b Builder) DynamicReshapeWithBounds(operand backends.Op, outputShape backends.Op, bounds []int) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeDynamicReshape)
}

// DynamicBroadcastInDimWithBounds broadcasts operand to a shape specified by outputDimensions tensor,
// using explicit dimension bounds for XLA compilation.
func (b Builder) DynamicBroadcastInDimWithBounds(operand backends.Op, outputDimensions backends.Op, broadcastDimensions []int, bounds []int) (backends.Op, error) {
	return nil, b.baseErrFn(backends.OpTypeDynamicBroadcastInDim)
}
