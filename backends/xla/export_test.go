package xla

// This file exports internal helpers for testing only.
// These functions are only available in test builds.

import "github.com/gomlx/go-xla/pkg/stablehlo"

// TestClosure exposes fn.Closure() for testing Sort/While operations.
func (b *Builder) TestClosure() *stablehlo.Function {
	return b.fn.Closure()
}

// TestNodeValue exposes the internal stablehlo.Value from a Node.
func (n *Node) TestValue() *stablehlo.Value {
	return n.value
}
