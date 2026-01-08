// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package notebooks allows one to check if running within a notebook.
// It supports GoNB [1] and bash_kernel [2].
//
// [1] GoNB: https://github.com/janpfeifer/gonb
// [2] bash_kernel: https://github.com/takluyver/bash_kernel
package notebooks

import "os"

// IsNotebook returns whether running inside a Jupyter notebook.
func IsNotebook() bool {
	return IsBashKernel() || IsGoNB()
}

const bashKernelEnv = "NOTEBOOK_BASH_KERNEL_CAPABILITIES"

// IsBashKernel returns whether running in a Jupyter notebook with a bash_kernel.
func IsBashKernel() bool {
	_, found := os.LookupEnv(bashKernelEnv)
	return found
}

const goNBKernelEnv = "GONB_PIPE"

// IsGoNB returns whether running in a Jupyter notebook with a GoNB kernel.
func IsGoNB() bool {
	_, found := os.LookupEnv(goNBKernelEnv)
	return found
}
