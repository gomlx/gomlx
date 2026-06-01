// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDynamicShapes_BasicExec(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)

	// x is [batch, 3], returns x + x
	exec, err := NewExec(backend, func(x *Node) *Node {
		return Add(x, x)
	})
	require.NoError(t, err)
	exec.WithDynamicAxes([]string{"batch", ""})

	// Run with batch=2
	input2 := [][]float32{{1, 2, 3}, {4, 5, 6}}
	out2, err := exec.Call(input2)
	require.NoError(t, err)
	assert.Equal(t, [][]float32{{2, 4, 6}, {8, 10, 12}}, out2[0].Value())

	// Run with batch=3
	input3 := [][]float32{{1, 1, 1}, {2, 2, 2}, {3, 3, 3}}
	out3, err := exec.Call(input3)
	require.NoError(t, err)
	assert.Equal(t, [][]float32{{2, 2, 2}, {4, 4, 4}, {6, 6, 6}}, out3[0].Value())

	// Verify that they used the same cached graph!
	exec.cacheMu.Lock()
	cacheLen := len(exec.cache)
	exec.cacheMu.Unlock()
	assert.Equal(t, 1, cacheLen, "Graph should only be compiled once for dynamic batch dimension")
}

func TestDynamicShapes_Broadcast(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)

	// x has shape [batch, 1], broadcast to target shape [batch, 3]
	exec, err := NewExec(backend, func(x *Node) *Node {
		// Target shape has dynamic dimension "batch"
		targetShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 3}, []string{"batch", ""})
		return BroadcastToShape(x, targetShape)
	})
	require.NoError(t, err)
	exec.WithDynamicAxes([]string{"batch", ""})

	input := [][]float32{{1}, {2}}
	out, err := exec.Call(input)
	require.NoError(t, err)
	assert.Equal(t, [][]float32{{1, 1, 1}, {2, 2, 2}}, out[0].Value())
}

func TestDynamicShapes_Reshape(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)

	// x has shape [batch, 4], reshaped to [batch, 2, 2]
	exec, err := NewExec(backend, func(x *Node) *Node {
		targetShape := shapes.MakeDynamic(dtypes.Float32, []int{shapes.DynamicDim, 2, 2}, []string{"batch", "", ""})
		return ReshapeWithShape(x, targetShape)
	})
	require.NoError(t, err)
	exec.WithDynamicAxes([]string{"batch", ""})

	input := [][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}}
	out, err := exec.Call(input)
	require.NoError(t, err)
	assert.Equal(t, [][][]float32{{{1, 2}, {3, 4}}, {{5, 6}, {7, 8}}}, out[0].Value())
}

func TestDynamicShapes_RankMismatchPanic(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)

	exec, err := NewExec(backend, func(x *Node) *Node {
		return Add(x, x)
	})
	require.NoError(t, err)
	
	// x is 2D but dynamicAxes expects 3D
	exec.WithDynamicAxes([]string{"batch", "time", ""})

	input := [][]float32{{1, 2}, {3, 4}}
	_, err = exec.Call(input)
	assert.Error(t, err)
	assert.Contains(t, err.Error(), "rank mismatch")
}
