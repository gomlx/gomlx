package simplego

import "github.com/gomlx/gopjrt/dtypes/bfloat16"

// dgNormalizeShape reshapes the source to a rank-3 shape [batchSize, crossSize, contractingSize].
//
// It returns a buffer with the transposed/reshaped source.
//
// In the chance that the source needs no transposing, output is returned nil.
func dgNormalizeShape[T interface {
	PODNumericConstraints | bfloat16.BFloat16
}](backend *Backend, source *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int) (output *Buffer) {
	return nil
}

func execDotGeneralSmall(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) error {
	//dtype := lhs.shape.DType
	return nil
}
