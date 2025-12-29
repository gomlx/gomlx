// Package xla implements a GoMLX backend using Google's XLA (see github.com/gomlx/go-xla).
//
// The backend is registered with the aliases "xla", "stablehlo", "shlo" or "hlo" (all aliases to the same backend).
//
// By default, the XLA/PJRT backend loads the requested plugins after the program starts and specifies the desired
// plugin name (default to "cpu") using `dlopen`.
//
// If the plugins are not available, the backend will download them automatically ("auto-install):
//
// - From github.com/gomlx/pjrt-cpu-binaries for CPU PJRT plugins.
// - From pypi.org, using the Jax pacakges for the CUDA and TPU PJRT plugins.
//
// Auto-install has no effect if default plugins are already installed. But to control it you can:
//
//   - Call xla.AutoInstall() directly if you want to call it immediately.
//   - Configure it with xla.EnableAutoInstall() if you want to enable/disable it globally (default is enabled).
//   - Set GOMLX_NO_AUTO_INSTALL, which sets the global auto-install flag to false -- but it can be overridden by
//     calling xla.EnableAutoInstall().
//
// Experimentally, one can get this backend to work with pre-linked PJRT plugins, but it will require the user to
// add the `.so` files in a library in LD_LIBRARY_PATH, or precompile a `.a` static library.
//
//   - Pre-link the CPU PJRT plugin statically: this will generate a bigger binary (+ ~200Mb, so slower to build),
//     but allows one to build a static binary that can be deployed without extra dependencies (except the standard C and C++ libraries,
//     usually available in most machines).
//     To enable, build using the tag `pjrt_cpu_static` (e.g.: `go build --tags pjrt_cpu_static ...`),
//     or import `github.com/gomlx/gomlx/backends/stablehlo/cpu/static`. Both methods have the same effect.
//   - Pre-link the CPU PJRT plugin dynamically: build with the build tag `pjrt_cpu_dynamic` (e.g.: `go test --tags pjrt_cpu_dynamic ...`),
//     or import `github.com/gomlx/gomlx/backends/stablehlo/cpu/dynamic`. Not much difference from linking the PJRT plugin
//     after the program starts, as default.
//
// # Shared Buffers Support:
//
// XLA/PJRT for CPU allows the "device buffer" (where device=CPU) to be addressed directly, which
// saves the copy from "host/local tensor" to the "on-device tensor" when executing a computation.
// This is enabled by default if the plugin is called "cpu". To force advertising support for this
// for other PJRTs provide the "shared_buffers" option, e.g.: GOMLX_BACKEND="xla:my_pjrt,shared_buffers".
// Or to force disabling the support, provide the "noshared_buffers" option.
package xla

import (
	"os"
	"path/filepath"
	"slices"
	"strings"
	"unsafe"

	"github.com/gomlx/go-xla/pkg/installer"
	"github.com/gomlx/go-xla/pkg/pjrt"
	xladtypes "github.com/gomlx/go-xla/pkg/types/dtypes"
	xlabfloat16 "github.com/gomlx/go-xla/pkg/types/dtypes/bfloat16"
	xlashapes "github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

//go:generate go run ../../internal/cmd/stablehlo_generator

// BackendName is the name of the backend.
//
// The stablehlo backend also accepts the "xla", "hlo" and "pjrt" aliases.
const BackendName = "xla"

// Disable XLA logging by default by setting TF_CPP_MIN_LOG_LEVEL to 2 (errors level), if it is not already set.
// This won't work if the PJRT is linked statically or dynamically before the go program start (without `dlopen` that is).
func init() {
	const TensorflowCPPMinLogLevelEnv = "TF_CPP_MIN_LOG_LEVEL"
	tfLogLevel := os.Getenv(TensorflowCPPMinLogLevelEnv)
	if tfLogLevel == "" {
		err := os.Setenv(TensorflowCPPMinLogLevelEnv, "2")
		if err != nil {
			klog.Errorf("Failed to set $%s to 2: %v", TensorflowCPPMinLogLevelEnv, err)
		}
	}
}

// New returns a new Backend using the config as a configuration.
// The config string should be the name of the PJRT plugin to use.
//
// This function triggers AutoInstall if it is enabled (the default). See EnableAutoInstall to disable it.
func New(config string) (backends.Backend, error) {
	return NewWithOptions(config, nil)
}

// NewWithOptions creates a StableHLO backend with the given client options.
// It allows more control, not available with the default New constructor.
//
// This function triggers AutoInstall if it is enabled (the default). See EnableAutoInstall to disable it.
func NewWithOptions(config string, options pjrt.NamedValuesMap) (*Backend, error) {
	pluginName := config
	var pluginOptions []string
	parts := strings.Split(config, ",")
	if len(parts) > 1 {
		// Plugin options (exclude empty).
		pluginOptions = slices.DeleteFunc(parts[1:], func(s string) bool { return s == "" })
		pluginName = parts[0]
	}

	if !filepath.IsAbs(pluginName) {
		if autoInstall {
			err := AutoInstall()
			if err != nil {
				return nil, errors.WithMessagef(err, "backend %q failed to auto-install default plugins", BackendName)
			}
		}

		// Verify the pluginName is available.
		plugins := GetAvailablePlugins()
		if len(plugins) == 0 {
			return nil, errors.Errorf("no plugins found for backend %q -- either use the absolute "+
				"path to the pluginName as the configuration or set PJRT_PLUGIN_LIBRARY_PATH to the path where to search for "+
				"PJRT plugins", BackendName)
		}
		if pluginName == "" {
			pluginName = plugins[0]
		} else if slices.Index(plugins, pluginName) == -1 {
			return nil, errors.Errorf("Plugin %q for backend %q not found: available plugins found %q", pluginName, BackendName, plugins)
		}
	}

	plugin, err := pjrt.GetPlugin(pluginName)
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q:", BackendName)
	}
	var client *pjrt.Client
	client, err = plugin.NewClient(options)
	if err != nil {
		return nil, errors.WithMessagef(err, "while creating plugin %s for backend %q", pluginName, BackendName)
	}
	klog.V(1).Infof("created new plugin %q for backend %q", pluginName, BackendName)
	backend := &Backend{
		plugin:       plugin,
		client:       client,
		pluginName:   pluginName,
		capabilities: Capabilities.Clone(),
		numDevices:   len(client.AddressableDevices()),
	}

	// Support "shared buffers":
	backend.hasSharedBuffers = pluginName == "cpu"
	if idx := slices.Index(pluginOptions, "shared_buffers"); idx != -1 {
		backend.hasSharedBuffers = true
		pluginOptions = slices.Delete(pluginOptions, idx, idx+1)
	} else if idx := slices.Index(pluginOptions, "noshared_buffers"); idx != -1 {
		backend.hasSharedBuffers = false
		pluginOptions = slices.Delete(pluginOptions, idx, idx+1)
	}

	// Support for tf32 DotGeneral.
	if idx := slices.Index(pluginOptions, "tf32"); idx != -1 {
		backend.DotGeneralConfig.UseTF32 = true
		pluginOptions = slices.Delete(pluginOptions, idx, idx+1)
	}

	// Any leftover plugin options are unknown.
	if len(pluginOptions) != 0 {
		klog.Errorf("backend %q: unknown plugin options %q", BackendName, pluginOptions)
	}
	return backend, nil
}

// Registers New() as the default constructor for "xla" backend.
func init() {
	backends.Register(BackendName, New)

	// Other aliases for this backend.
	backends.Register("stablehlo", New)
	backends.Register("hlo", New)
	backends.Register("shlo", New)
}

var (
	// DefaultPlugins is the list of plugins to use in preference order, if not otherwise specified.
	DefaultPlugins = []string{"cuda", "cpu"}

	// availablePluginsList are the keys to the available plugins sorted by DefaultPlugins.
	availablePluginsList []string
)

var autoInstall bool = true // Whether it should always auto-install at every call to New()

func init() {
	_, found := os.LookupEnv(NoAutoInstallEnv)
	if found {
		autoInstall = false
	}
}

const NoAutoInstallEnv = "GOMLX_NO_AUTO_INSTALL"

// AutoInstall the standard plugin version tested for the current go-xla version.
// If GPU or TPU are detected, it will also install the corresponding plugins.
//
// This simply calls github.com/gomlx/go-xla/pkg/installer.AutoInstall().
// If you want more control over the installation path, cache usage, or verbosity,
// you can use the AutoInstall function from go-xla's installer package directly.
func AutoInstall() error {
	return installer.AutoInstall("", true, installer.Normal)
}

// EnableAutoInstall sets whether AutoInstall should be triggered automatically for GetAvailablePlugins or New.
//
// If enabled, the default, the AutoInstall function will be called automatically when GetAvailablePlugins or New is called.
func EnableAutoInstall(enable bool) {
	autoInstall = enable
}

// GetAvailablePlugins lists the available platforms -- it caches and reuses the result in future calls.
//
// This function triggers AutoInstall if it is enabled (the default). See EnableAutoInstall to disable it.
//
// Plugins are searched in the PJRT_PLUGIN_LIBRARY_PATH directory -- or directories if it is a ":" separated list.
// If it is not set, it will search the system "/usr/local/lib/go-xla", the users $HOME/.local/lib/go-xla (or
// "$HOME/Library/Application Support/go-xla" in MacOS) and the standard libraries directories of the
// system (in linux in LD_LIBRARY_PATH and /etc/ld.so.conf file) in that order.
//
// If there are plugins with the same name but different versions in different directories, it respects the order
// of the directories given by PJRT_PLUGIN_LIBRARY_PATH or by the system.
//
// See details in pjrt.AvailablePlugins.
func GetAvailablePlugins() []string {
	if autoInstall {
		err := AutoInstall()
		if err != nil {
			klog.Errorf("Error auto-installing plugins: %+v", err)
		}
	}

	if len(availablePluginsList) > 0 {
		// Use cache results.
		return availablePluginsList
	}

	availablePluginsMap := pjrt.AvailablePlugins()
	pluginNames := sets.MakeWith(xslices.Keys(availablePluginsMap)...)
	klog.V(1).Infof("Available plugins: %v\n", pluginNames)
	availablePluginsList = make([]string, 0, len(pluginNames))

	// Add DefaultPlugins first.
	for _, pluginName := range DefaultPlugins {
		if pluginNames.Has(pluginName) {
			availablePluginsList = append(availablePluginsList, pluginName)
			delete(pluginNames, pluginName)
		}
	}

	// Add the other plugins in some random order.
	for pluginName := range pluginNames {
		availablePluginsList = append(availablePluginsList, pluginName)
	}
	return availablePluginsList
}

// DTypeToXLA converts a GoMLX dtypes.DType to a go-xla xladtypes.DType.
// Currently, they are identical types, but this function centralizes the conversion
// in case they diverge in the future.
func DTypeToXLA(dtype dtypes.DType) xladtypes.DType {
	return xladtypes.DType(dtype)
}

// DTypeFromXLA converts a go-xla xladtypes.DType to a GoMLX dtypes.DType.
// Currently, they are identical types, but this function centralizes the conversion
// in case they diverge in the future.
func DTypeFromXLA(xlaDType xladtypes.DType) dtypes.DType {
	return dtypes.DType(xlaDType)
}

// BFloat16SliceToXLA converts a GoMLX []bfloat16.BFloat16 slice to a go-xla []xlabfloat16.BFloat16 slice.
// Both types are defined as `type BFloat16 uint16`, so this is a zero-copy conversion using unsafe.
func BFloat16SliceToXLA(slice []bfloat16.BFloat16) []xlabfloat16.BFloat16 {
	return unsafe.Slice((*xlabfloat16.BFloat16)(unsafe.Pointer(unsafe.SliceData(slice))), len(slice))
}

// BFloat16SliceFromXLA converts a go-xla []xlabfloat16.BFloat16 slice to a GoMLX []bfloat16.BFloat16 slice.
// Both types are defined as `type BFloat16 uint16`, so this is a zero-copy conversion using unsafe.
func BFloat16SliceFromXLA(slice []xlabfloat16.BFloat16) []bfloat16.BFloat16 {
	return unsafe.Slice((*bfloat16.BFloat16)(unsafe.Pointer(unsafe.SliceData(slice))), len(slice))
}

// ShapeToXLA converts a GoMLX shape to a go-xla shape.
// Dynamic dimensions (negative values) are replaced with 1 since XLA doesn't
// support unbounded dimensions directly. The dynamic dimensions are tracked
// at the GoMLX layer for pattern matching and gradients.
func ShapeToXLA(shape shapes.Shape) xlashapes.Shape {
	if !shape.Ok() || shape.IsTuple() {
		return xlashapes.Invalid()
	}
	// Clone dimensions and replace dynamic (negative) values with 1
	dims := slices.Clone(shape.Dimensions)
	for i, d := range dims {
		if d < 0 {
			dims[i] = 1 // Use 1 as placeholder for dynamic dimensions
		}
	}
	return xlashapes.Make(DTypeToXLA(shape.DType), dims...)
}

// ShapeFromXLA converts a go-xla shape to a GoMLX shape.
func ShapeFromXLA(shape xlashapes.Shape) shapes.Shape {
	if !shape.Ok() || shape.IsTuple() {
		return shapes.Invalid()
	}
	dims := slices.Clone(shape.Dimensions)
	// Check if any dimension is dynamic (negative)
	if shape.IsDynamic() {
		return shapes.MakeDynamic(DTypeFromXLA(shape.DType), dims...)
	}
	return shapes.Make(DTypeFromXLA(shape.DType), dims...)
}
