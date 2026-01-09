//go:build linux || darwin || windows || freebsd || openbsd || netbsd || dragonfly || solaris

// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0


package graph

import (
	"crypto/rand"
	"encoding/binary"
	"github.com/pkg/errors"
)

// initializeRNGState uses the OS's cryptographically secure random number
// generator to initialize the state. This is the preferred method.
// In Go, crypto/rand abstracts the underlying source (e.g., /dev/urandom on Linux,
// CryptGenRandom on Windows).
func initializeRNGState(state *[3]uint64) error {
	// The total size of our state is 3 * 8 = 24 bytes.
	const stateSize = 24

	// Create a byte slice with the exact size of our state.
	randomBytes := make([]byte, stateSize)

	// Read reads random data from the OS into the byte slice.
	// It will return an error if the OS's entropy source fails.
	// rand.Read is guaranteed to fill the entire slice.
	_, err := rand.Read(randomBytes)
	if err != nil {
		return errors.Wrapf(err, "could not read random bytes from OS")
	}

	// Convert the 24 random bytes into three uint64 values.
	// We use BigEndian, but LittleEndian would be just as valid.
	// This is just a way to consistently interpret the bytes.
	state[0] = binary.BigEndian.Uint64(randomBytes[0:8])
	state[1] = binary.BigEndian.Uint64(randomBytes[8:16])
	state[2] = binary.BigEndian.Uint64(randomBytes[16:24])

	return nil
}
