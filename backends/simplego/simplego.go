// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package simplego is a stub for the newly created github.com/gomlx/compute/gobackend.
// It simply aliases the public types and functions.
package simplego

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/gobackend"
)

// BackendName to be used in GOMLX_BACKEND to specify this backend.
const BackendName = gobackend.BackendName

// GetBackend returns a singleton backend for SimpleGo, created with the default configuration.
// The backend is only created at the first call of the function.
//
// The singleton is never destroyed.
//
// Deprecated: use gobackend.GetBackend instead.
func GetBackend() compute.Backend {
	return gobackend.GetBackend()
}

// New constructs a new SimpleGo Backend.
//
// Deprecated: use gobackend.New instead.
func New(config string) (compute.Backend, error) {
	return gobackend.New(config)
}
