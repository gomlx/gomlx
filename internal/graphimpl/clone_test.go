// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/internal/graphimpl"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
)

func TestCloneWithInputs(t *testing.T) {
	backend := testutil.BuildTestBackend()

	t.Run("basic", func(t *testing.T) {
		g := graphimpl.NewGraphImpl(backend, "test_clone")

		x := graphimpl.Const(g, float32(10.0))
		y := graphimpl.Const(g, float32(20.0))
		z := graphimpl.Add(x, y)

		a := graphimpl.Const(g, float32(2.0))
		b := graphimpl.Const(g, float32(3.0))
		zCloned := z.CloneWithInputs(a, b)

		err := g.Compile(z, zCloned)
		assert.NoError(t, err)

		results := g.Run()
		gotZ := results[0].Value().(float32)
		gotZCloned := results[1].Value().(float32)

		assert.Equal(t, float32(30.0), gotZ)
		assert.Equal(t, float32(5.0), gotZCloned)
	})

	t.Run("if", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		if !backend.Capabilities().Functions {
			t.Skipf("Backend %q does not support functions", backend.Name())
		}
		g := graphimpl.NewGraphImpl(backend, "test_clone_if")

		pred := graphimpl.Const(g, true)

		trueBranch := graphimpl.NewClosure(g, func(g *graphimpl.GraphImpl) []*graphimpl.NodeImpl {
			return []*graphimpl.NodeImpl{graphimpl.Const(g, float32(100.0))}
		})
		falseBranch := graphimpl.NewClosure(g, func(g *graphimpl.GraphImpl) []*graphimpl.NodeImpl {
			return []*graphimpl.NodeImpl{graphimpl.Const(g, float32(200.0))}
		})

		ifNodeSlice := graphimpl.If(pred, trueBranch, falseBranch)
		ifNode := ifNodeSlice[0]

		predFalse := graphimpl.Const(g, false)
		ifNodeCloned := ifNode.CloneWithInputs(predFalse)

		err := g.Compile(ifNode, ifNodeCloned)
		assert.NoError(t, err)

		results := g.Run()
		gotIf := results[0].Value().(float32)
		gotIfCloned := results[1].Value().(float32)

		assert.Equal(t, float32(100.0), gotIf)
		assert.Equal(t, float32(200.0), gotIfCloned)
	})

	t.Run("while", func(t *testing.T) {
		backend := testutil.BuildTestBackend()
		if !backend.Capabilities().Functions {
			t.Skipf("Backend %q does not support functions", backend.Name())
		}
		g := graphimpl.NewGraphImpl(backend, "test_clone_while")

		condValOriginal := graphimpl.Const(g, int32(5))
		condValNew := graphimpl.Const(g, int32(3))

		cond := graphimpl.NewClosure(g, func(g *graphimpl.GraphImpl) []*graphimpl.NodeImpl {
			counter := graphimpl.Parameter(g, "counter", shapes.Make(dtypes.Int32))
			condLimit := graphimpl.Parameter(g, "cond_limit", shapes.Make(dtypes.Int32))
			return []*graphimpl.NodeImpl{graphimpl.LessThan(counter, condLimit)}
		})
		body := graphimpl.NewClosure(g, func(g *graphimpl.GraphImpl) []*graphimpl.NodeImpl {
			counter := graphimpl.Parameter(g, "counter", shapes.Make(dtypes.Int32))
			condLimit := graphimpl.Parameter(g, "cond_limit", shapes.Make(dtypes.Int32))
			return []*graphimpl.NodeImpl{
				graphimpl.Add(counter, graphimpl.Const(g, int32(1))),
				condLimit,
			}
		})

		initialCounter := graphimpl.Const(g, int32(0))
		whileNodeSlice := graphimpl.While(cond, body, initialCounter, condValOriginal)
		whileNode := whileNodeSlice[0]

		multiOutputNode := whileNode.Inputs()[0]
		multiOutputCloned := multiOutputNode.CloneWithInputs(initialCounter, condValNew)
		whileNodeCloned := whileNode.CloneWithInputs(multiOutputCloned)

		err := g.Compile(whileNode, whileNodeCloned)
		assert.NoError(t, err)

		results := g.Run()
		gotOriginal := results[0].Value().(int32)
		gotCloned := results[1].Value().(int32)

		assert.Equal(t, int32(5), gotOriginal)
		assert.Equal(t, int32(3), gotCloned)
	})
}
