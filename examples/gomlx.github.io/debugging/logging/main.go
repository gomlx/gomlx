// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default" // Import the default compute backends (XLA / SimpleGo).
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
)

//md_start:logging

func SquareAndAddOne(x *Node) *Node {
	xSquared := Mul(x, x)
	// Mark the intermediate square calculation to be logged:
	xSquared.SetLoggedf("#full x^2 intermediate value")

	one := Scalar(x.Graph(), x.DType(), 1.0)
	return Add(xSquared, one)
}

//md_end:logging

//md:logging(-1)

func main() {
	backend := compute.MustNew()
	fmt.Println("md:logging")

	//md_start:logging

	exec := MustNewExec(backend, SquareAndAddOne)

	// Option 1: Run with the default logger (prints to stdout)
	exec.MustCall([]float32{2.0, 3.0})

	// Option 2: Run with a custom logger
	exec.SetNodeLogger(func(g *Graph, messages []string, values []*tensors.Tensor, nodes []NodeId) {
		for i, msg := range messages {
			fmt.Printf("Custom Logger: [Node #%d] %s = %v\n", nodes[i], msg, values[i].Value())
		}
	})
	exec.MustCall([]float32{5.0, 7.0})

	//md_end:logging
}
