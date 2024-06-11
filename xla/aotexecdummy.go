//go:build google3

package xla

// Dummy package for AOT interface when using "google3" build tag: this allows the dependency to the corresponding
// C++ code to be dropped.

import (
	"github.com/pkg/errors"
)

// AOTExecutable executes Ahead-Of-Time (AOT) compiled graphs.
type AOTExecutable struct {
}

// NewAOTExecutable given the client and the aotResult returned by an earlier Computation.AOTCompile call. It
// may return an error.
func NewAOTExecutable(client *Client, aotResult []byte) (*AOTExecutable, error) {
	return nil, errors.Errorf("AOT disabled on google3")
}

// IsNil returns whether contents are invalid or have been freed already.
func (exec *AOTExecutable) IsNil() bool {
	return true
}

// Run is disabled on google3.
func (exec *AOTExecutable) Run(params []*OnDeviceBuffer) (*OnDeviceBuffer, error) {
	return nil, errors.Errorf("AOT disabled on google3")
}
