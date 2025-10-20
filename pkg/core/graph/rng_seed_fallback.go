//go:build !linux && !darwin && !windows && !freebsd && !openbsd && !netbsd && !dragonfly && !solaris

package graph

import (
	"math/rand"
	"time"
)

// initializeRNGState provides a fallback for systems without a known
// strong entropy source in Go's crypto/rand package.
// WARNING: This method is NOT cryptographically secure as it uses the
// current time, which is predictable.
func initializeRNGState(state *[3]uint64) error {

	// Create a new random source seeded with the current nanosecond time.
	// This is a pseudo-random generator, not a true random source.
	source := rand.NewSource(time.Now().UnixNano())
	r := rand.New(source)

	state[0] = r.Uint64()
	state[1] = r.Uint64()
	state[2] = r.Uint64()

	return nil
}
