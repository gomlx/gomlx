// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/stretchr/testify/require"
)

func TestExecCompileUncached(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	builds := 0
	offset := float32(1)
	exec, err := NewExec(backend, func(x *Node) *Node {
		builds++
		return Add(x, Scalar(x.Graph(), x.DType(), offset))
	})
	require.NoError(t, err)
	defer exec.Finalize()
	shape := shapes.Make(dtypes.Float32, 1)
	cached, err := exec.Compile(shape)
	require.NoError(t, err)
	offset = 10
	fresh, err := exec.CompileUncached(shape)
	require.NoError(t, err)
	defer fresh.Finalize()
	require.NotSame(t, cached, fresh)
	another, err := exec.CompileUncached(shape)
	require.NoError(t, err)
	defer another.Finalize()
	require.NotSame(t, fresh, another)
	again, err := exec.Compile(shape)
	require.NoError(t, err)
	require.Same(t, cached, again)
	require.Equal(t, 3, builds)
	exec.Finalize()
	values := fresh.Run([]float32{1})
	require.Equal(t, []float32{11}, values[0].Value())
	require.NoError(t, values[0].FinalizeAll())
	_, err = exec.CompileUncached(shape)
	require.ErrorIs(t, err, ErrExecFinalized)
}
