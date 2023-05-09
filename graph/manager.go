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
	"github.com/gomlx/gomlx/xla"
	"os"
)

// GetPlatforms lists the available platforms. Returns `([]string, error)`.
var GetPlatforms = xla.GetPlatforms

// GetDefaultPlatform returns the default list of platforms. Returns `(string, error)`.
var GetDefaultPlatform = xla.GetDefaultPlatform

// ManagerBuilder allow setting of options to build a Manager object.
type ManagerBuilder struct {
	platform                string
	numReplicas, numThreads int
}

// BuildManager allows the creations a Manager object, used to create computation graphs and execute them.
// Optional parameters are Platform, NumReplicas and NumThreads (see ManagerBuilder methods).
// At the end call IsNil().
func BuildManager() *ManagerBuilder {
	return &ManagerBuilder{numReplicas: 1, numThreads: -1}
}

// Platform can be left empty (it will pick one per GetDefaultPlatform) or can
// be selected from one returned by GetPlatforms.
func (b *ManagerBuilder) Platform(p string) *ManagerBuilder {
	b.platform = p
	return b
}

// WithDefaultPlatform takes the given platform if one is not overwritten with GOMLX_PLATFORM.
// This is useful for testing, where one wants to force something except if explicitly overwritten.
func (b *ManagerBuilder) WithDefaultPlatform(p string) *ManagerBuilder {
	b.platform = p
	platform, found := os.LookupEnv(xla.DefaultPlatformEnv)
	if found {
		b.platform = platform
	}
	return b
}

// NumReplicas sets number of replicas to use when building Manager. Defaults to 1.
func (b *ManagerBuilder) NumReplicas(n int) *ManagerBuilder {
	b.numReplicas = n
	return b
}

// NumThreads sets number of threads to use when building Manager. Defaults to -1, which indicates to use what
// is available.
func (b *ManagerBuilder) NumThreads(n int) *ManagerBuilder {
	b.numThreads = n
	return b
}

// Done constructs the Manager.
func (b *ManagerBuilder) Done() (m *Manager, err error) {
	platform := b.platform
	if b.platform == "" {
		platform, err = GetDefaultPlatform()
		if err != nil {
			return
		}
	}
	client, err := xla.NewClient(platform, b.numReplicas, b.numThreads)
	if err != nil {
		return nil, err
	}
	m = &Manager{
		client:     client,
		platform:   platform,
		graphCount: 0,
	}
	return
}

// MustDone constructs the Manager. It panics if there was an error.
func (b *ManagerBuilder) MustDone() *Manager {
	manager, err := b.Done()
	if err != nil {
		panic(fmt.Sprintf("Failed to build gomlx.computation.Manager: %+v", err))
	}
	return manager
}

// Manager sets up an execution "server" (?? whatever runs stuff in XLA ?), including managing
// the memory in the accelerator.
//
// The Manager is used to create computation graphs (Graph), and then JIT-compile them to a
// "computation" (in XLA), that can be executed.
type Manager struct {
	client   *xla.Client
	platform string

	// graphCount is the number of graphs created.
	graphCount int
}

// NewGraph constructs an empty Graph. If name is set to "", a unique name is picked. Uses
// DeviceNumber == 0.
func (m *Manager) NewGraph(name string) *Graph {
	return m.NewGraphWithDeviceNum(name, m.DefaultDeviceNum())
}

// NewGraphWithDeviceNum constructs an empty Graph, and sets to use the given device number.
// If name is set to "", a unique name is picked.
func (m *Manager) NewGraphWithDeviceNum(name string, deviceNum int) *Graph {
	if name == "" {
		name = fmt.Sprintf("graph_#%d", m.graphCount)
	}
	g := newGraph(m, name, GraphId(m.graphCount), deviceNum)
	m.graphCount += 1
	return g
}

// Platform returns the platform used by manager -- which may be different from the one requested,
// depending on availability.
func (m *Manager) Platform() string {
	return m.platform
}

// DefaultDeviceNum returns the default device number for the device associated with this Manager.
func (m *Manager) DefaultDeviceNum() int {
	return m.client.DefaultDeviceOrdinal
}

// DeviceCount returns the device count associated with this Manager.
func (m *Manager) DeviceCount() int {
	return m.client.DeviceCount
}

func (m *Manager) ClientId() xla.ClientId {
	return m.client.Id
}

func (m *Manager) Client() *xla.Client {
	return m.client
}
