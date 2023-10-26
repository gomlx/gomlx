package xla

import (
	"fmt"
	"golang.org/x/exp/slices"
	"os"
	"path"
	"strings"
)

// This file implements the workaround to help XLA find the (infamous) `libdevice.10.bc` file
// needed by the NVidia CUDA drivers.
// Unfortunately, it is a required file and there is not a good default way of finding it.
//
// XLA uses the environment variable XLA_FLAGS, with the flag --xla_gpu_cuda_data_dir set to the
// CUDA directory to be searched. If it's not set, it searches the current directory.
//
// The strategies for GoMLX around this limitation, in case it is compiled with GPU/CUDA support, are:
//
// 1. If --xla_gpu_cuda_data_dir is not set:
//   - If `CUDA_DIR` env variable is set, set that in XLA_FLAGS instead.
//   - Try to find in standard (for now only Ubuntu/debian) locations for CUDA driver files, and if they find
//     `libdevice.10.bc` there, set it in XLA_FLAGS accordingly.
//     It starts with `./cuda_sdk_lib`, the default used by XLA (see file
//     https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc).
//     2. Independent of (1) changing or not `XLA_FLAGS`, parse errors from XLA in search for `libdevice not found`
//     and provide a much more detailed error message to the end user.
const (
	CudaDirKey            = "CUDA_DIR"
	XlaFlagsKey           = "XLA_FLAGS"
	XlaFlagGpuCudaDataDir = "--xla_gpu_cuda_data_dir"
)

// LibDeviceFound indicates whether a location for `libdevice` CUDA file was found or pre-given by the user
// (in XLA_FLAGS) at init time.
//
// If one attempts to use CUDA with LibDeviceFound == false, the library will return an error.
var LibDeviceFound bool

const LibDeviceNotFoundErrorMessage = `The GPU/CUDA drivers require knowing where is the CUDA data directory, to find
a file called "libdevice.10.bc" (*).

GoMLX could not find the CUDA directory given (if any was given) or in the usual places:
"./cuda_sdk_lib", "/usr/lib/cuda", "/usr/lib/nvidia-cuda-toolkit", "/usr/local/cuda-*", 
therefore we cannot continue using CUDA.

If you are ok using CPU, just configure "GOMLX_PLATFORM=Host".

If you know where the CUDA directory is (it should contain a subdirectory named "nvvm/libdevice"), you
can configure it in the environment variable "$CUDA_DIR", and GoMLX will pick it up.

Alternatively, configure it in a flag for XLA, doing something like
$ export XLA_FLAGS=--xla_gpu_cuda_data_dir=<your_cuda_dir>

If you think there is a standard CUDA directory GoMLX should also be automatically trying to use for some
common OS/architecture, please contact the team by creating an issue in github.com/gomlx/gomlx, and we
will include it in the default search.

(*) The file is needed because it contains implementations of a wide range of mathematical functions.
`

var LibDeviceDir string

// ParseXlaFlags returns the flags passed in the environment variable `XLA_FLAGS`.
func ParseXlaFlags() []string {
	xlaFlags := os.Getenv(XlaFlagsKey)
	return slices.DeleteFunc(
		strings.Split(xlaFlags, " "),
		func(s string) bool { return s == "" })
}

// PresetXlaFlagsCudaDir will update `XLA_FLAGS` env variable if `xla_gpu_cuda_data_dir` is not set, in an attempt
// to make the file `libdevice.10.bc` available (it is required by NVidia CUDA):
//
//   - If `CUDA_DIR` env variable is set, set that in XLA_FLAGS instead.
//   - Try to find in standard (for now only Ubuntu/debian) locations for CUDA driver files, and if they find
//     `libdevice.10.bc` there, set it in XLA_FLAGS accordingly.
//     It starts with `./cuda_sdk_lib`, the default used by XLA (see file
//     https://github.com/openxla/xla/blob/main/xla/debug_options_flags.cc).
func PresetXlaFlagsCudaDir() {
	xlaFlags := ParseXlaFlags()

	// Find index where XlaFlagGpuCudaDataDir is set.
	found := -1
	for ii, f := range xlaFlags {
		if f != XlaFlagGpuCudaDataDir && !strings.HasPrefix(f, XlaFlagGpuCudaDataDir+"=") {
			continue // Not the flag we are looking for.
		}

		// Discard multiple occurrences of flag.
		if found >= 0 {
			xlaFlags[found] = ""
		}
		found = ii

		// Flag set in separate arguments: merge them.
		if f == XlaFlagGpuCudaDataDir {
			if ii < len(xlaFlags)-1 {
				// Merge flag "--foo bar" into "--foo=bar"
				LibDeviceDir = xlaFlags[ii+1]
				xlaFlags[ii+1] = ""
				xlaFlags[ii] = fmt.Sprintf("%s=%s", XlaFlagGpuCudaDataDir, LibDeviceDir)
			}
			continue
		}

		// Take value from after the "=".
		nextPos := len(XlaFlagGpuCudaDataDir) + 1
		if len(f) == nextPos {
			LibDeviceDir = ""
		} else {
			LibDeviceDir = f[nextPos:]
		}
	}

	// If flag not set, let's try to find `libdevice` for the user:
	if found == -1 {
		cudaDir := findLibDevice()
		if cudaDir != "" {
			LibDeviceFound = true
			LibDeviceDir = cudaDir
			xlaFlags = append(xlaFlags, fmt.Sprintf("%s=%s", XlaFlagGpuCudaDataDir, cudaDir))
		}
	} else {
		// If provided by the user, assume it was correctly set.
		LibDeviceFound = true
	}

	// Set result.
	xlaFlags = slices.DeleteFunc(xlaFlags, func(s string) bool { return s == "" })
	if len(xlaFlags) > 0 {
		_ = os.Setenv(XlaFlagsKey, strings.Join(xlaFlags, " "))
	} else {
		_ = os.Unsetenv(XlaFlagsKey)
	}
}

// findLibDevice returns a string with the directory where to find "/nvvm/libdevice" subdirectory, presumably with the
// file.
// It will return `$CUDA_DIR` if it is defined.
func findLibDevice() string {
	// First use CUDA_DIR is available.
	cudaDir := os.Getenv(CudaDirKey)
	if cudaDir != "" {
		return cudaDir
	}

	// Search for standard locations, starting with XLA's default `./cuda_sdk_lib`.
	for _, candidate := range []string{"./cuda_sdk_lib", "/usr/lib/cuda", "/usr/lib/nvidia-cuda-toolkit"} {
		if DirHasLibdevice(candidate) {
			return candidate
		}
	}

	// Search for `/usr/local/cuda-*` directories.
	localPath := "/usr/local"
	entries, err := os.ReadDir(localPath)
	if err == nil {
		var candidate string
		for _, e := range entries {
			dir := e.Name()
			fullPath := path.Join(localPath, dir)
			if e.IsDir() && strings.HasPrefix(dir, "cuda-") && DirHasLibdevice(fullPath) {
				// Take just the last (in alphabetical order) candidate.
				if candidate == "" || strings.Compare(candidate, dir) == -1 {
					candidate = fullPath
				}
			}
		}
		if candidate != "" {
			return candidate
		}
	}

	return ""
}

// DirHasLibdevice checks whether the directory has "libdevice..." needed by GPU CUDA.
//
// This is somewhat based on function `GetLibdeviceDir` defined in
// https://github.com/openxla/xla/blob/main/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc
func DirHasLibdevice(dir string) bool {
	tgtDir := path.Join(dir, "nvvm", "libdevice")
	fi, err := os.Stat(tgtDir)
	if err != nil {
		return false
	}
	return fi.IsDir()
}

func init() {
	PresetXlaFlagsCudaDir()
}
