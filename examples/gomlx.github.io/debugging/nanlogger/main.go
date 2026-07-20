// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/nanlogger"
)

//md_start:nanlogger

func MyLayer(l *nanlogger.NanLogger, x *Node) *Node {
	l.PushScope("dense_layer_1")
	defer l.PopScope()

	denominator := Scalar(x.Graph(), x.DType(), 0.0)
	y := Div(x, denominator)
	l.TraceFirstNaN(y, "output_division")
	return y
}

//md_end:nanlogger

//md:nanlogger(-1)

func main() {
	backend := compute.MustNew()

	// Mark output for nanlogger example.
	fmt.Println("md:nanlogger")

	//md_start:nanlogger

	l := nanlogger.New().WithHandler(nanlogger.ReportAndPanicHandler)
	exec := MustNewExec(backend, func(x *Node) *Node {
		return MyLayer(l, x)
	})
	l.AttachToExec(exec)
	l.WithHandler(nanlogger.ReportScopeHandler)
	_, err := exec.Call([]float32{1.0, 2.0})

	//md_end:nanlogger

	if err != nil {
		fmt.Printf("Error: %+v\n", err)
	}
}
