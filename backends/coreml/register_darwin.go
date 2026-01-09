//go:build darwin && cgo

package coreml

import (
	"github.com/gomlx/gomlx/backends"
)

func init() {
	backends.Register(BackendName, New)
}
