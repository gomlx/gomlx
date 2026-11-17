//go:build !linux && !darwin && !windows && !freebsd && !openbsd && !netbsd && !dragonfly && !solaris

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"math/rand/v2"
	"time"
)

// initializeRandomBytes provides a fallback for systems without a known strong entropy
// source in Go's crypto/rand package.
// WARNING: This method is NOT cryptographically secure as it uses the current time.
func initializeRandomBytes(dst []byte) error {
	now := uint64(time.Now().UnixNano())
	rng := rand.NewPCG(now, now^0x9e3779b97f4a7c15)
	for i := 0; i < len(dst); {
		v := rng.Uint64()
		for j := 0; j < 8 && i < len(dst); j++ {
			dst[i] = byte(v)
			v >>= 8
			i++
		}
	}
	return nil
}
