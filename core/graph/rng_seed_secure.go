//go:build linux || darwin || windows || freebsd || openbsd || netbsd || dragonfly || solaris

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"crypto/rand"

	"github.com/pkg/errors"
)

// initializeRandomBytes fills dst with cryptographically secure random bytes from the OS.
func initializeRandomBytes(dst []byte) error {
	if len(dst) == 0 {
		return nil
	}
	if _, err := rand.Read(dst); err != nil {
		return errors.Wrapf(err, "could not read random bytes from OS")
	}
	return nil
}
