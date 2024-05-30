// Package cuda registers the Host platform to XLA.
package cuda

import (
	"unsafe"

	"github.com/gomlx/gomlx/xla"
)

// #include "gomlx/client.h"
import "C"

// Platform is the name of the platform imported by this package.
const Platform = "CUDA"

func init() {
	xla.RegisterPlatform(&cuda{})
}

type cuda struct{}

func (cuda) Name() string {
	return Platform
}

func (cuda) Client(numReplicas, numThreads int) (unsafe.Pointer, error) {
	statusOr := C.NewClient(C.CString(Platform), C.int(numReplicas), C.int(numThreads))
	if statusOr.status != nil {
		return nil, xla.NewStatus(statusOr.status).Error()
	}
	return statusOr.value, nil
}
