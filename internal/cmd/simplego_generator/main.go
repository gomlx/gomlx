// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// simplego_generator auto-generates parts of the SimpleGo backend:
//
// - exec_binary.go: binary ops execution, e.g.: Add, Mul, Div, Sub, Pow, etc.
package main

import (
	"flag"

	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()
	klog.V(1).Info("\tinternal/cmd/simplego_generator:")
	GenerateExecBinary()
}
