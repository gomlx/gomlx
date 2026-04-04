//go:build !darwin || !cgo

// Package metal provides a stub for the Metal backend on non-darwin platforms.
package metal

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/pkg/errors"
)

const BackendName = "metal"

func init() {
	backends.Register(BackendName, New)
}

type Backend struct {
	notimplemented.Backend
}

func New(config string) (backends.Backend, error) {
	return nil, errors.New("metal backend not available on this platform")
}
