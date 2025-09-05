// Package stablehlo implements a GoMLX backend using StableHLO (see github.com/gomlx/stablehlo) as a language to
// talk to PJRT, the C++ engine for XLA (see github.com/gomlx/gopjrt/pjrt).
//
// The backend is registered as "stablehlo", "shlo" or "hlo" (all aliases to the same backend).
package stablehlo

import (
	"path"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/pjrt"
	stablehloshapes "github.com/gomlx/stablehlo/types/shapes"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

//go:generate go run ../../internal/cmd/stablehlo_generator

// BackendName is the name of the backend.
//
// The stablehlo backend also accepts the "hlo" and "pjrt" aliases.
const BackendName = "stablehlo"

// New returns a new Backend using the config as a configuration.
// The config string should be the name of the PJRT plugin to use.
func New(config string) (backends.Backend, error) {
	return NewWithOptions(config, nil)
}

// NewWithOptions creates a StableHLO backend with the given client options.
// It allows more control, not available with the default New constructor.
func NewWithOptions(config string, options pjrt.NamedValuesMap) (*Backend, error) {
	pluginName := config
	var pluginOptions []string
	parts := strings.Split(config, ",")
	if len(parts) > 1 {
		// Plugin options (exclude empty).
		pluginOptions = slices.DeleteFunc(parts[1:], func(s string) bool { return s == "" })
		pluginName = parts[0]
	}

	if !path.IsAbs(pluginName) {
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
	backends.Register("hlo", New)
	backends.Register("shlo", New)
}

var (
	// DefaultPlugins is the list of plugins to use in preference order, if not otherwise specified.
	DefaultPlugins = []string{"cuda", "cpu"}

	// availablePluginsList are the keys to the available plugins sorted by DefaultPlugins.
	availablePluginsList []string
)

// GetAvailablePlugins lists the available platforms -- it caches and reuses the result in future calls.
//
// Plugins are searched in the PJRT_PLUGIN_LIBRARY_PATH directory -- or directories if it is a ":" separated list.
// If it is not set, it will search in "/usr/local/lib/gomlx/pjrt" and the standard libraries directories of the
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

// ShapeToStableHLO converts a GomlX shape to a StableHLO shape.
func ShapeToStableHLO(shape shapes.Shape) stablehloshapes.Shape {
	if !shape.Ok() || shape.IsTuple() {
		return stablehloshapes.Invalid()
	}
	return stablehloshapes.Make(shape.DType, slices.Clone(shape.Dimensions)...)
}

// ShapeFromStableHLO converts a StableHLO shape to a GomlX shape.
func ShapeFromStableHLO(shape stablehloshapes.Shape) shapes.Shape {
	if !shape.Ok() || shape.IsTuple() {
		return shapes.Invalid()
	}
	return shapes.Make(shape.DType, slices.Clone(shape.Dimensions)...)
}
