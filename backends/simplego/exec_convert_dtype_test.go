// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/stretchr/testify/assert"

	"github.com/gomlx/gomlx/pkg/core/graph"
)

func TestExecSpecialOps_ConvertDType(t *testing.T) {
	// Test int32 to float32
	y0 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Float32)
	}, int32(42))
	// fmt.Printf("\ty0=%s\n", y0.GoStr())
	assert.Equal(t, float32(42.0), y0.Value())

	// Test float32 to bfloat16
	y1 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.BFloat16)
	}, float32(3.14))
	// fmt.Printf("\ty1=%s\n", y1.GoStr())
	assert.Equal(t, bf16(3.14), y1.Value())

	// Test bfloat16 to int32
	y2 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Int32)
	}, bf16(7.8))
	// fmt.Printf("\ty2=%s\n", y2.GoStr())
	assert.Equal(t, int32(7), y2.Value())

	// Test bool to int32
	y3 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Int32)
	}, true)
	// fmt.Printf("\ty3=%s\n", y3.GoStr())
	assert.Equal(t, int32(1), y3.Value())

	// Test float32 to bool
	y4 := graph.MustExecOnce(backend, func(x *graph.Node) *graph.Node {
		return graph.ConvertDType(x, dtypes.Bool)
	}, float32(1.0))
	// fmt.Printf("\ty4=%s\n", y4.GoStr())
	assert.Equal(t, true, y4.Value())
}
