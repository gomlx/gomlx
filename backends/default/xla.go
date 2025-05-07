//go:build linux && amd64 && !noxla

// For now XLA is only supported for linux/amd64.
// TODO: change when more platforms are supported (linux/arm64, darwin/amd64, darwin/arm64, etc.)

package _default

import _ "github.com/gomlx/gomlx/backends/xla"
