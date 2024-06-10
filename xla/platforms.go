package xla

import "unsafe"

// Platform is a XLA platform providing a XLA implementation.
type Platform interface {
	// Name of the platform.
	Name() string

	// Client returns a new client for the platform.
	Client(numReplicas, numThreads int) (unsafe.Pointer, error)
}

var platforms = make(map[string]Platform)

// RegisterPlatform register a XLA platform.
func RegisterPlatform(platform Platform) {
	platforms[platform.Name()] = platform
}
