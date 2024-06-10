// Package cpu registers the Host platform to XLA.
package cpu

import (
	"unsafe"

	"github.com/gomlx/gomlx/xla"
)

// #include "gomlx/client.h"
import "C"

// Platform is the name of the platform imported by this package.
const Platform = "Host"

func init() {
	xla.RegisterPlatform(&cpu{})
}

type cpu struct{}

func (cpu) Name() string {
	return Platform
}

func (cpu) Client(numReplicas, numThreads int) (unsafe.Pointer, error) {
	statusOr := C.NewClient(C.CString(Platform), C.int(numReplicas), C.int(numThreads))
	if statusOr.status != nil {
		return nil, xla.NewStatus(statusOr.status).Error()
	}
	return statusOr.value, nil
}
