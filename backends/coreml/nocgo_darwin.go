//go:build darwin && !cgo

// Package coreml provides a CoreML backend for GoMLX on macOS.
// This file is a stub for when CGO is disabled. The CoreML backend requires CGO
// to interface with Apple's CoreML framework through Objective-C.
package coreml

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// BackendName to be used in GOMLX_BACKEND to specify this backend.
const BackendName = "coreml"

// New returns an error when CGO is disabled, as CoreML requires CGO.
func New(config string) (backends.Backend, error) {
	return nil, errors.New("CoreML backend requires CGO to be enabled; rebuild with CGO_ENABLED=1")
}
