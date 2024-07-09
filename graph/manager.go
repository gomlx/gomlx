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

package graph

import (
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/pkg/errors"
	"os"
	"sync"
)

var (
	// DefaultPlugins is the list of plugins to use in preference order, if not otherwise specified.
	// It is "trumped" by the value of the environment variable GOMLX_PJRT_PLUGIN (see DefaultPluginEnv) if
	// it is set.
	DefaultPlugins = []string{"cuda", "cpu"}

	// availablePluginsList are the keys to availablePluginsMap sorted by DefaultPlugins.
	availablePluginsList []string
)

// Manager sets up an execution "server" (?? whatever runs stuff in XLA ?), including managing
// the memory in the accelerator.
//
// The Manager is used to create computation graphs (Graph), and then JIT-compile them to a
// "computation" (in XLA), that can be executed.
type Manager struct {
	plugin     *pjrt.Plugin
	client     *pjrt.Client
	pluginName string
}

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
	pluginNames := types.SetWith(xslices.Keys(availablePluginsMap))
	availablePluginsList = make([]string, 0, len(pluginNames))

	// Add GOMLX_PJRT_PLUGIN first.
	selectedPlugin := os.Getenv(DefaultPluginEnv)
	if selectedPlugin != "" && pluginNames.Has(selectedPlugin) {
		availablePluginsList = append(availablePluginsList, selectedPlugin)
		delete(pluginNames, selectedPlugin)
	}

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

// DefaultPluginEnv is the environment variable name for the default plugin name.
// It can be a name (e.g.: "cpu", "cuda") or a full-path to the plugin.
// See pjrt.GetPlugin.
var DefaultPluginEnv = "GOMLX_PJRT_PLUGIN"

// ManagerBuilder allow setting of options to build a Manager object.
type ManagerBuilder struct {
	pluginName string
}

// NewManager creates a new `Manager` object using the default plugin and configuration.
// For more fine-grained control, see BuildManager.
func NewManager() *Manager {
	return BuildManager().Done()
}

// BuildManager allows the creations a Manager object, used to create computation graphs and execute them.
// Optional parameters are PluginDescription, NumReplicas and NumThreads (see ManagerBuilder methods).
// At the end call IsNil().
func BuildManager() *ManagerBuilder {
	return &ManagerBuilder{}
}

// Plugin specifies the plugin to use: it can be a name or an absolute path.
//
// If left empty, it will try to load the plugin selected in GOMLX_PJRT_PLUGIN environment variable.
// And if that is also empty (or not set), it will pick one in DefaultPlugins order.
// If those are not available, it will pick the first returned by GetAvailablePlugins.
//
// See details on Plugin selection in pjrt.GetPlugin.
func (b *ManagerBuilder) Plugin(p string) *ManagerBuilder {
	b.pluginName = p
	return b
}

// WithDefaultPlugin takes the given pluginName if one is not overwritten with GOMLX_PLATFORM.
// This is useful for testing, where one wants to force something except if explicitly overwritten.
func (b *ManagerBuilder) WithDefaultPlugin(p string) *ManagerBuilder {
	b.pluginName = p
	plugin, found := os.LookupEnv(DefaultPluginEnv)
	if found {
		b.pluginName = plugin
	}
	return b
}

// Done constructs the Manager.
//
// Errors are reported/thrown back with `panic`.
func (b *ManagerBuilder) Done() (m *Manager) {
	pluginName := b.pluginName
	if b.pluginName == "" {
		var found bool
		pluginName, found = os.LookupEnv(DefaultPluginEnv)
		if !found {
			allPlugins := GetAvailablePlugins()
			if len(allPlugins) == 0 {
				exceptions.Panicf("can't find any PJRT plugin to use to compile+execute computation graphs")
			}
			pluginName = allPlugins[0]
		}
	}
	if pluginName == "" {
		exceptions.Panicf("can't find any PJRT plugin to use to compile+execute computation graphs")
	}
	plugin, err := pjrt.GetPlugin(pluginName)
	if err != nil {
		panic(errors.WithMessagef(err, "failed to load PJRT plugin %q", pluginName))
	}
	client, err := plugin.NewClient()
	m = &Manager{
		client: client,
		plugin: plugin,
	}
	return
}

// GraphId has to globally unique.
var (
	muGraphCount sync.Mutex
	graphCount   int
)

// NewGraph constructs an empty Graph. If `name` is set to "" a unique name is picked.
// It uses the manager's default device number.
func (m *Manager) NewGraph(name string) *Graph {
	return m.NewGraphWithDeviceNum(name, m.DefaultDeviceNum())
}

// NewGraphWithDeviceNum constructs an empty Graph, and sets to use the given device number.
// If name is set to "" a unique name is picked.
func (m *Manager) NewGraphWithDeviceNum(name string, deviceNum int) *Graph {
	muGraphCount.Lock()
	defer muGraphCount.Unlock()

	if name == "" {
		name = fmt.Sprintf("graph_#%d", graphCount)
	}
	g := newGraph(m, name, GraphId(graphCount), deviceNum)
	graphCount += 1
	return g
}

// PluginDescription returns a printable description of the PJRT plugin used by manager.
func (m *Manager) PluginDescription() string {
	if m == nil || m.plugin == nil {
		return "nil"
	}
	return m.plugin.String()
}

// PluginName returns the name of the plugin loaded for this manager.
func (m *Manager) PluginName() string {
	return m.pluginName
}

// DefaultDeviceNum returns the default device number for the device associated with this Manager.
func (m *Manager) DefaultDeviceNum() int {
	return 0
}

// DeviceCount returns the device count associated with this Manager.
func (m *Manager) DeviceCount() int {
	return len(m.client.AddressableDevices())
}
