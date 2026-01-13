// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build noasm || !arm64

package simplego

// execDotGeneralSmallMatMulFloat32NEON stub for non-ARM64 platforms.
// This should never be called since hasNEON will be false on these platforms.
func execDotGeneralSmallMatMulFloat32NEON(_ *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) {
	panic("NEON SmallMatMul not available on this platform")
}
