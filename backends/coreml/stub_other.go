//go:build !darwin

// Package coreml provides a CoreML backend for GoMLX on macOS.
// This file is a stub for non-darwin platforms.
package coreml

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// BackendName to be used in GOMLX_BACKEND to specify this backend.
const BackendName = "coreml"

// New returns an error on non-darwin platforms, as CoreML is only available on macOS.
func New(config string) (backends.Backend, error) {
	return nil, errors.New("CoreML backend is only available on macOS (darwin)")
}
