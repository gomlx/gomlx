// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
)

func TestCloneWithInputs(t *testing.T) {
	backend := testutil.BuildTestBackend()

	t.Run("basic", func(t *testing.T) {
		g := NewGraph(backend, "test_clone")

		x := Const(g, float32(10.0))
		y := Const(g, float32(20.0))
		z := Add(x, y)

		a := Const(g, float32(2.0))
		b := Const(g, float32(3.0))
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
		g := NewGraph(backend, "test_clone_if")

		x := Const(g, float32(10.0))
		y := Const(g, float32(20.0))
		_ = x
		_ = y
		pred := Const(g, true)

		trueBranch := NewClosure(g, func(g *Graph) []*Node {
			return []*Node{Const(g, float32(100.0))}
		})
		falseBranch := NewClosure(g, func(g *Graph) []*Node {
			return []*Node{Const(g, float32(200.0))}
		})

		ifNodeSlice := If(pred, trueBranch, falseBranch)
		ifNode := ifNodeSlice[0]

		// Clone with a false predicate:
		predFalse := Const(g, false)

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
		g := NewGraph(backend, "test_clone_while")

		// We count from 0 to 5 (original), and we clone to count from 0 to 3.
		condValOriginal := Const(g, int32(5))
		condValNew := Const(g, int32(3))

		cond := NewClosure(g, func(g *Graph) []*Node {
			counter := Parameter(g, "counter", shapes.Make(dtypes.Int32))
			condLimit := Parameter(g, "cond_limit", shapes.Make(dtypes.Int32))
			return []*Node{LessThan(counter, condLimit)}
		})
		body := NewClosure(g, func(g *Graph) []*Node {
			counter := Parameter(g, "counter", shapes.Make(dtypes.Int32))
			condLimit := Parameter(g, "cond_limit", shapes.Make(dtypes.Int32))
			return []*Node{
				Add(counter, Const(g, int32(1))),
				condLimit,
			}
		})

		initialCounter := Const(g, int32(0))
		whileNodeSlice := While(cond, body, initialCounter, condValOriginal)
		whileNode := whileNodeSlice[0]

		multiOutputNode := whileNode.Inputs()[0]

		// Clone with condValNew instead of condValOriginal
		// multiOutputNode has inputs: [initialCounter, condValOriginal]
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
