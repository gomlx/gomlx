//go:build ((linux && amd64) || darwin) && !noxla

package _default

import _ "github.com/gomlx/gomlx/backends/xla"
