// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package xla provides the XLA backend for gomlx.
// It is a wrapper around github.com/gomlx/go-xla/compute/xla.
//
// Deprecated: This package is deprecated. Use github.com/gomlx/go-xla/compute/xla instead.
package xla

import (
	"github.com/gomlx/compute"
	xlabackend "github.com/gomlx/go-xla/compute/xla"
	"github.com/gomlx/go-xla/pjrt"
)

// BackendName to be used in GOMLX_BACKEND to specify this backend.
//
// Deprecated: use xlabackend.Backend instead.
const BackendName = xlabackend.BackendName

// NoAutoInstallEnv is an environment variable that can be used to disable the
// auto-installation of the XLA backend. See EnableAutoInstall to disable it.
//
// Deprecated: use xlabackend.NoAutoInstallEnv instead.
const NoAutoInstallEnv = xlabackend.NoAutoInstallEnv

// New constructs a new SimpleGo Backend.
//
// Deprecated: use gobackend.New instead.
func New(config string) (compute.Backend, error) {
	return xlabackend.New(config)
}

// GetAvailablePlugins returns the list of available plugins for the XLA backend.
// This is the same as xlabackend.GetAvailablePlugins.
//
// Deprecated: use xlabackend.GetAvailablePlugins instead.
func GetAvailablePlugins() []string {
	return xlabackend.GetAvailablePlugins()
}

// NewWithOptions creates a StableHLO backend with the given client options.
// It allows more control, not available with the default New constructor.
//
// Deprecated: use xlabackend.NewWithOptions instead.
func NewWithOptions(config string, options pjrt.NamedValuesMap) (compute.Backend, error) {
	return xlabackend.NewWithOptions(config, options)
}

// AutoInstall the standard plugin version tested for the current go-xla version.
// If GPU or TPU are detected, it will also install the corresponding plugins.
//
// This simply calls github.com/gomlx/go-xla/installer.AutoInstall().
// If you want more control over the installation path, cache usage, or verbosity,
// you can use the AutoInstall function from go-xla's installer package directly.
//
// Deprecated: use xlabackend.AutoInstall instead.
func AutoInstall() error {
	return xlabackend.AutoInstall()
}

// EnableAutoInstall sets whether AutoInstall should be triggered automatically for GetAvailablePlugins or New.
//
// If enabled, the default, the AutoInstall function will be called automatically when GetAvailablePlugins or New is called.
//
// Deprecated: use xlabackend.EnableAutoInstall instead.
func EnableAutoInstall(enable bool) {
	xlabackend.EnableAutoInstall(enable)
}
