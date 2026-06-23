// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestCustomCall_GoBackendNotImplemented checks that emitting a custom_call on the
// SimpleGo backend (no custom-call support) surfaces a wrapped compute.ErrNotImplemented.
// Callers rely on this to recover and fall back to a decomposed implementation.
func TestCustomCall_GoBackendNotImplemented(t *testing.T) {
	graphtest.BuildTestBackend() // ensure backends are registered / XLA auto-installed
	backend, err := compute.NewWithConfig("go")
	require.NoError(t, err)

	g := graph.NewGraph(backend, "custom_call_go")
	x := graph.Parameter(g, "x", shapes.Make(dtypes.Float32, 2, 3))
	spec := compute.CustomCallSpec{
		Target:       "__test$unsupported",
		APIVersion:   2,
		OutputShapes: []shapes.Shape{shapes.Make(dtypes.Float32, 2, 3)},
	}
	err = exceptions.TryCatch[error](func() { graph.CustomCall(spec, nil, x) })
	require.Error(t, err)
	assert.True(t, compute.IsNotImplemented(err),
		"expected the failure to wrap ErrNotImplemented, got: %v", err)
}

// TestCustomCall_XLAConstruction checks that on the XLA backend graph.CustomCall builds a
// multi-output node with the requested result shapes (cuDNN flash attention returns its
// output plus a scratch buffer). Executing cuDNN targets needs a GPU and is covered by
// go-xla's pjrt tests.
func TestCustomCall_XLAConstruction(t *testing.T) {
	graphtest.BuildTestBackend() // auto-installs the XLA PJRT plugin
	backend, err := compute.NewWithConfig("xla:cpu")
	if err != nil {
		t.Skipf("xla:cpu backend unavailable: %v", err)
	}

	g := graph.NewGraph(backend, "custom_call_xla")
	qkv := shapes.Make(dtypes.BFloat16, 2, 2048, 12, 64) // [B,S,H,D]
	q := graph.Parameter(g, "q", qkv)
	k := graph.Parameter(g, "k", qkv)
	v := graph.Parameter(g, "v", qkv)

	out := shapes.Make(dtypes.BFloat16, 2, 12, 2048, 64) // [B,H,S,D]
	scratch := shapes.Make(dtypes.Uint8, 0)
	// A generic (non-cuDNN) target: the XLA backend gates __cudnn$ targets to the cuda plugin,
	// and this construction check runs on cpu. It exercises the multi-output node + result shapes,
	// the same machinery the cuDNN flash forward uses.
	spec := compute.CustomCallSpec{
		Target:       "__test$multi_output",
		APIVersion:   2,
		OutputShapes: []shapes.Shape{out, scratch},
	}
	nodes := graph.CustomCall(spec, nil, q, k, v)
	require.Len(t, nodes, 2)
	assert.True(t, nodes[0].Shape().Equal(out), "output shape: got %s, want %s", nodes[0].Shape(), out)
	assert.True(t, nodes[1].Shape().Equal(scratch), "scratch shape: got %s, want %s", nodes[1].Shape(), scratch)
}
