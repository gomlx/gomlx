// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// Exported helpers for subpackages (e.g. highway) to implement fused op executors.
// These extract the parameters from opaque node data and allocate output buffers,
// following the same pattern as UnaryOperandAndOutput.

// FusedOpOutput allocates an output buffer for a fused op based on the node's output shape.
func FusedOpOutput(backend *Backend, node *Node) *Buffer {
	return backend.getBufferForShape(node.shape)
}

// FusedOpOutputShape returns the output shape for a fused op node.
func FusedOpOutputShape(node *Node) shapes.Shape {
	return node.shape
}

// SoftmaxParams extracts the axis from a Softmax node.
func SoftmaxParams(node *Node) (axis int) {
	return node.data.(*nodeFusedSoftmax).axis
}

// LayerNormParams extracts axes and epsilon from a LayerNorm node.
func LayerNormParams(node *Node) (axes []int, epsilon float64) {
	data := node.data.(*nodeFusedLayerNorm)
	return data.axes, data.epsilon
}

// DenseActivationParams extracts the activation type from a DenseActivation node.
func DenseActivationParams(node *Node) backends.ActivationType {
	return node.data.(*nodeFusedDenseActivation).activation
}

// LayerNormFloat32Fallback is the scalar implementation of LayerNorm for float32.
// Used by the highway subpackage for non-trailing axis combinations where SIMD
// acceleration is not applicable.
func LayerNormFloat32Fallback(input, output, gamma, beta *Buffer, axes []int, epsilon float64) {
	layerNormFloat32(input, output, gamma, beta, axes, epsilon)
}

// LayerNormFloat64Fallback is the scalar implementation of LayerNorm for float64.
func LayerNormFloat64Fallback(input, output, gamma, beta *Buffer, axes []int, epsilon float64) {
	layerNormFloat64(input, output, gamma, beta, axes, epsilon)
}
