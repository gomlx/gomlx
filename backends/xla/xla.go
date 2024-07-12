// Package xla implements the XLA/PJRT (https://openxla.org/) based backend for GoMLX.
//
// Simply import it with import _ "github.com/gomlx/gomlx/backends/xla" to make it available in your program.
// It will register itself as an available backend during initialization.
package xla

import (
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/pkg/errors"
	"slices"
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

	plugin, err := pjrt.GetPlugin(pluginName)
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q:", BackendName))
	}
	client, err := plugin.NewClient(options)
	if err != nil {
		panic(errors.WithMessagef(err, "backend %q:", BackendName))
	}
	return &Backend{
		plugin:     plugin,
		client:     client,
		pluginName: pluginName,
	}
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
