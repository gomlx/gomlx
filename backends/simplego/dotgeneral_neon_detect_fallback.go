// Copyright 2025 The GoMLX Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build noasm || !arm64

package simplego

// hasNEON indicates whether NEON SIMD optimizations are available.
// NEON is only available on ARM64, so it's always false on other platforms.
const hasNEON = false
