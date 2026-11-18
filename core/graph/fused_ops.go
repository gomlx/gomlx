// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"fmt"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/gomlx/internal/graphimpl"
)

// FusionEnv is the name of the environment variable that controls whether fused ops are used.
// The default value is true.
const FusionEnv = graphimpl.FusionEnv

// Quantization describes how a graph-level quantized tensor maps between quantized codes and real values.
// This is the graph-layer counterpart of compute.Quantization, holding *Node references for Scale
// and ZeroPoint instead of compute.Value so GoMLX tracks them as graph dependencies.
type Quantization struct {
	Scheme         compute.QuantizationScheme
	Scale          *Node
	ZeroPoint      *Node
	BlockAxis      int
	BlockSize      int
	GGMLType       compute.GGMLQuantType
	QuantizedDType dtypes.DType
	PackedDType    dtypes.DType
}

func (q *Quantization) String() string {
	if q == nil {
		return "nil"
	}
	strNode := func(n *Node) string {
		if n == nil {
			return "nil"
		}
		return fmt.Sprintf("[#%d]", n.Id())
	}
	return fmt.Sprintf("{Scheme:%s Scale:%s ZeroPoint:%s BlockAxis:%d BlockSize:%d GGMLType:%s QuantizedDType:%s PackedDType:%s}",
		q.Scheme, strNode(q.Scale), strNode(q.ZeroPoint), q.BlockAxis, q.BlockSize, q.GGMLType, q.QuantizedDType, q.PackedDType)
}

// QuantizedDenseConfig holds configuration and quantization metadata for nn.QuantizedDense.
type QuantizedDenseConfig struct {
	WeightQuantization *Quantization
	XQuantization      *Quantization
	OutputQuantization *Quantization
	QuantizedMatmul    bool
	Activation         compute.ActivationConfig
	WeightLayout       compute.DenseLayout
}
