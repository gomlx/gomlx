// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// backends_generator generates parts of the backends.Builder interface based on the github.com/gomlx/gopjrt/xlabuilder implementation.
//
// Although GoMLX can support more than one backend, the XlaBuilder is the reference implementation for now.
//
// If the environment variable GOPJRT_SRC is set, it parses the ops from there.
// Otherwise it clones the gopjrt repository to a temporary sub-directory.
package main

import (
	"flag"

	"github.com/gomlx/gomlx/internal/backendparser"
	"github.com/gomlx/gomlx/internal/must"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()
	klog.V(1).Info("notimplemented_generator:")
	methods := must.M1(backendparser.ParseBuilder())
	GenerateStandardOpsInterface(methods)
}
