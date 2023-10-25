/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package xla

// This file includes wrappers around xla::Service and xla::ClientLibrary.

// #include <gomlx/client.h>
// #include <gomlx/aot_exec.h>
import "C"
import (
	"fmt"
	"gopkg.in/errgo.v2/fmt/errors"
	"os"
	"runtime"
	"unsafe"
)

// ClientId is a unique identifier to clients. Starts with 0 and increases.
type ClientId int

const InvalidClientId = ClientId(-1)

var nextClientId = ClientId(0)

// Client is a wrapper for the C++ `Client`, itself a wrapper to xla::Client, described as
// "XLA service's client object -- wraps the service with convenience and lifetime-oriented methods."
//
// It's a required parameter to most high level XLA functionality: compilation and execution of graphs,
// transfer of data, etc.
type Client struct {
	cClientPtr                        *C.Client
	Id                                ClientId
	DeviceCount, DefaultDeviceOrdinal int
}

// Finalize implements Finalizer.
func (c *Client) Finalize() {
	defer runtime.KeepAlive(c)
	if c == nil || c.cClientPtr == nil {
		return
	}
	C.DeleteClient(c.cClientPtr)
	c.cClientPtr = nil
}

// IsNil returns whether either the Client object is nil or the contained C pointer.
func (c *Client) IsNil() bool {
	return c == nil || c.cClientPtr == nil
}

// GetPlatforms lists the available platforms
func GetPlatforms() ([]string, error) {
	StatusOr := C.GetPlatforms()
	value, err := UnsafePointerOrError(StatusOr)
	if err != nil {
		return nil, err
	}
	vp := (*C.VectorPointers)(value)
	return StrVectorFree(vp), nil
}

const DefaultPlatformEnv = "GOMLX_PLATFORM"

var (
	PlatformPreferences = []string{"TPU", "CUDA"}
	AvoidPlatform       = "Host"
)

// GetDefaultPlatform returns the default platform from the list of available platforms. It avoids choosing "Host",
// that is, it tries to pick the first ML accelerator available.
// The choice can be overridden by the value of the environment variable GOMLX_PLATFORM.
func GetDefaultPlatform() (string, error) {
	defaultPlatform, defaultGiven := os.LookupEnv(DefaultPlatformEnv)
	platforms, err := GetPlatforms()
	if err != nil {
		return "", err
	}
	if len(platforms) == 0 {
		return "", fmt.Errorf("no XLA platform found")
	}
	if len(platforms) == 1 {
		return platforms[0], nil
	}
	pSet := make(map[string]bool, len(platforms))
	for _, p := range platforms {
		pSet[p] = true
	}

	// If default is given by the environment variable:
	if defaultGiven {
		if _, found := pSet[defaultPlatform]; found {
			return defaultPlatform, nil
		}
	}

	// Pick first platform from list of preferences.
	for _, p := range PlatformPreferences {
		if _, found := pSet[p]; found {
			return p, nil
		}
	}

	// If none found, anything different from AvoidPlatform.
	for _, p := range platforms {
		if p != AvoidPlatform {
			return p, nil
		}
	}

	// Otherwise (repeated AvoidPlatform values), just pick the first.
	return platforms[0], nil
}

// NewClient constructs the Client.
// If platform is empty, use what is returned by GetDefaultPlatform. If numThreads is -1, use the number of available
// cores.
// Not thread-safe.
func NewClient(platform string, numReplicas, numThreads int) (*Client, error) {
	var err error
	if platform == "" {
		platform, err = GetDefaultPlatform()
		if err != nil {
			return nil, err
		}
	}
	if platform == "CUDA" {
		if !LibDeviceFound {
			return nil, errors.New(LibDeviceNotFoundErrorMessage)
		} else if !DirHasLibdevice(LibDeviceDir) {
			return nil, errors.Newf(
				"GPU/CUDA directory provided (%q) doesn't have a `nvvm/libdevice/libdevice.10.bc`.\n%s",
				LibDeviceDir, LibDeviceNotFoundErrorMessage)
		}
	}
	statusOr := C.NewClient(C.CString(platform), C.int(numReplicas), C.int(numThreads))
	cPtr, err := UnsafePointerOrError(statusOr)
	if err != nil {
		return nil, err
	}
	clientId := nextClientId
	nextClientId += 1
	var deviceCount, defaultDeviceOrdinal int
	cClientPtr := (*C.Client)(cPtr)
	C.ClientDevices(cClientPtr, (*C.int64_t)(unsafe.Pointer(&deviceCount)), (*C.int64_t)(unsafe.Pointer(&defaultDeviceOrdinal)))
	return &Client{
		cClientPtr:           cClientPtr,
		Id:                   clientId,
		DeviceCount:          deviceCount,
		DefaultDeviceOrdinal: defaultDeviceOrdinal,
	}, nil
}

// Ok returns whether the client was created successful and is still valid.
func (c *Client) Ok() bool {
	return !c.IsNil()
}

// Client implements the tensor.HasClient interface, by returning itself.
func (c *Client) Client() *Client {
	return c
}
