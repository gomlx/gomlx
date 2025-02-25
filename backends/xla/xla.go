// Package xla implements the XLA/PJRT (https://openxla.org/) based backend for GoMLX.
//
// Simply import it with:
//
//	import _ "github.com/gomlx/gomlx/backends/xla"
//
// To make it available in your program.
// It will register itself as an available backend during initialization.
//
// By default, XLA/PJRT backend loads requested plugins after the program starts and specifies the desired
// plugin name (default to "cpu") using `dlopen`. Now there are cases that one may simply want to pre-link
// a plugin with the program. There are two options here (at most one can be selected):
//
//   - Pre-link the CPU PJRT plugin statically: this will generate a bigger binary (+ ~200Mb, so slower to build),
//     but allows one to build a static binary that can be deployed without extra dependencies (except the standard C and C++ libraries,
//     usually available in most machines).
//     To enable, build using the tag `pjrt_cpu_static` (e.g.: `go build --tags pjrt_cpu_static ...`),
//     or import `github.com/gomlx/gomlx/backends/xla/cpu/static`. Both methods have the same effect.
//   - Pre-link the CPU PJRT plugin dynamically: build with the build tag `pjrt_cpu_dynamic` (e.g.: `go test --tags pjrt_cpu_dynamic ...`),
//     or import `github.com/gomlx/gomlx/backends/xla/cpu/dynamic`. Not much difference from linking the PJRT plugin
//     after the program starts, as default.
//
// Darwin (MacOS): currently dynamic linking XLA/PJRT is not working, so it links the CPU PJRT plugin by default,
// no need to manually link `github.com/gomlx/gomlx/backends/xla/cpu/static`.
//
// # Shared Buffers Support:
//
// XLA/PJRT for CPU allows the "device buffer" (where device=CPU) to be addressed directly, which
// saves the copy from "host/local tensor" to the "on-device tensor" when executing a computation.
// This is enabled by default if the plugin is called "cpu". To force advertising support for this
// for other PJRTs provide the "shared_buffers" option, e.g.: GOMLX_BACKEND="xla:my_pjrt,shared_buffers".
// Or to force disabling the support, provide the "noshared_buffers" option.
package xla

//go:generate go run ../../cmd/xla_generator

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"path"
	"slices"
	"strings"
)

const BackendName = "xla"

// New returns a new Backend using the config as a configuration.
// The config string should be the name of the PJRT plugin to use.
func New(pluginName string) backends.Backend {
	return NewWithOptions(pluginName, nil)
}

// NewWithOptions creates a XlaBackend with the given client options.
// It allows more control, not available with the default New constructor.
func NewWithOptions(pluginName string, options pjrt.NamedValuesMap) *Backend {
	var pluginOptions []string
	parts := strings.Split(pluginName, ",")
	if len(parts) > 1 {
		// Plugin options (exclude empty).
		pluginOptions = slices.DeleteFunc(parts[1:], func(s string) bool { return s == "" })
		pluginName = parts[0]
	}

	if !path.IsAbs(pluginName) {
		// Verify the pluginName is available.
		plugins := GetAvailablePlugins()
		if len(plugins) == 0 {
			exceptions.Panicf("no plugins found for backend %q -- either use the absolute "+
				"path to the pluginName as the configuration or set PJRT_PLUGIN_LIBRARY_PATH to the path where to search for "+
				"PJRT plugins", BackendName)
		}
		if pluginName == "" {
			pluginName = plugins[0]
		} else if slices.Index(plugins, pluginName) == -1 {
			exceptions.Panicf("Plugin %q for backend %q not found: available plugins found %q", pluginName, BackendName, plugins)
		}
	}

	plugin, err := pjrt.GetPlugin(pluginName)
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q:", BackendName))
	}
	var client *pjrt.Client
	client, err = plugin.NewClient(options)
	if err != nil {
		panic(errors.WithMessagef(err, "while creating plugin %s for backend %q", pluginName, BackendName))
	}
	klog.V(1).Infof("created new plugin %q for backend %q", pluginName, BackendName)
	backend := &Backend{
		plugin:     plugin,
		client:     client,
		pluginName: pluginName,
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
	if len(pluginOptions) != 0 {
		klog.Errorf("backend %q: unknown plugin options %q", BackendName, pluginOptions)
	}
	return backend
}

// Registers New() as the default constructor for "xla" backend.
func init() {
	backends.Register(BackendName, New)
}

var (
	// DefaultPlugins is the list of plugins to use in preference order, if not otherwise specified.
	DefaultPlugins = []string{"cuda", "cpu"}

	// availablePluginsList are the keys to availablePluginsMap sorted by DefaultPlugins.
	availablePluginsList []string
)

// GetAvailablePlugins lists the available platforms -- it caches and reuses the result in future calls.
//
// Plugins are searched in the PJRT_PLUGIN_LIBRARY_PATH directory -- or directories, if it is a ":" separated list.
// If it is not set it will search in "/usr/local/lib/gomlx/pjrt" and the standard libraries directories of the
// system (in linux in LD_LIBRARY_PATH and /etc/ld.so.conf file) in that order.
//
// If there are plugins with the same name but different versions in different directories, it respects the order of the directories given by
// PJRT_PLUGIN_LIBRARY_PATH or by the system.
//
// See details in pjrt.AvailablePlugins.
func GetAvailablePlugins() []string {
	if len(availablePluginsList) > 0 {
		// Use cache results.
		return availablePluginsList
	}

	availablePluginsMap := pjrt.AvailablePlugins()
	pluginNames := types.SetWith(xslices.Keys(availablePluginsMap)...)
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
