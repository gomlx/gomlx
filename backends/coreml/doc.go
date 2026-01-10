// Package coreml implements a GoMLX backend using Apple's CoreML framework.
//
// This backend enables GoMLX computations to run on Apple Silicon's Neural Engine (ANE),
// GPU, and CPU. It provides hardware acceleration on macOS devices through the CoreML
// framework.
//
// The backend is only available on darwin (macOS) platforms. On other platforms,
// importing this package is a no-op.
//
// # Usage
//
// To use the CoreML backend, simply import it:
//
//	import _ "github.com/gomlx/gomlx/backends/coreml"
//
// Then set the GOMLX_BACKEND environment variable:
//
//	export GOMLX_BACKEND=coreml
//
// Or create the backend directly:
//
//	backend, err := coreml.New("")
//
// # Configuration
//
// The backend supports the following configuration options (passed to New()):
//
//   - "cpu_only": Use only CPU for computation
//   - "gpu": Use CPU and GPU
//   - "ane": Use CPU and Apple Neural Engine
//   - (default): Use all available compute units
//
// # Limitations
//
// The CoreML backend currently supports a subset of GoMLX operations.
// See the ops.go file for the list of implemented operations.
package coreml
