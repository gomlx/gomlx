// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/pkg/errors"
)

// SetAutoSharding sets the distributed strategy to AutoSharding and sets the device meshes.
func (g *GraphImpl) SetAutoSharding(meshes ...*distributed.DeviceMesh) error {
	if !g.IsValid() {
		return errors.New("cannot set AutoSharding on an invalid graph")
	}
	if g.IsCompiled() || g.IsBuilding() {
		return errors.New("cannot set AutoSharding on a graph that is being built or already compiled")
	}
	if g.distStrategy != distributed.None {
		return errors.New("cannot set AutoSharding on a graph that already has a distributed strategy set")
	}
	g.distStrategy = distributed.AutoSharding
	g.deviceMeshes = slices.Clone(meshes)
	g.numDevices = 0
	for _, mesh := range meshes {
		g.numDevices = max(g.numDevices, mesh.NumDevices())
	}
	if g.numDevices > g.backend.NumDevices() {
		return errors.Errorf("number of devices in the mesh (%d) exceeds the number of devices in the backend (%d)",
			g.numDevices, g.backend.NumDevices())
	}
	return nil
}

// SetDeviceAssignment specifies which concrete devices to use for the graph.
func (g *GraphImpl) SetDeviceAssignment(devices []compute.DeviceNum) error {
	if !g.IsValid() {
		return errors.New("cannot set device assignment on an invalid graph")
	}
	if g.IsCompiled() || g.IsBuilding() {
		return errors.New("cannot set device assignment on a graph that is being built or already compiled")
	}
	if g.distStrategy == distributed.None {
		return errors.New("cannot set device assignment on a graph that doesn't have a distribution strategy, " +
			"please use WithAutoSharding or WithSPMD first.")
	}
	if len(devices) != g.numDevices {
		return errors.Errorf("number of devices (%d) doesn't match the number of devices in the largest mesh (%d)",
			len(devices), g.numDevices)
	}
	numDevicesAvailable := compute.DeviceNum(g.backend.NumDevices())
	seen := sets.Make[compute.DeviceNum]()
	for _, device := range devices {
		if device >= numDevicesAvailable {
			return errors.Errorf("device number (%d) exceeds the number of devices in the backend (%d)",
				device, numDevicesAvailable)
		}
		if seen.Has(device) {
			return errors.Errorf("device number (%d) is used more than once in the assignment", device)
		}
		seen.Insert(device)
	}
	g.deviceAssignment = devices
	return nil
}

// SetSPMD sets the distributed strategy to SPMD.
func (g *GraphImpl) SetSPMD(mesh *distributed.DeviceMesh) error {
	if g.IsCompiled() || g.IsBuilding() {
		return errors.New("cannot set SPMD on a graph that is being built or already compiled")
	}
	if g.distStrategy != distributed.None {
		return errors.New("cannot set SPMD on a graph that already has a distributed strategy set")
	}
	g.distStrategy = distributed.SPMD
	g.deviceMeshes = []*distributed.DeviceMesh{mesh}
	g.numDevices = mesh.NumDevices()
	if g.numDevices > g.backend.NumDevices() {
		return errors.Errorf("number of devices in the mesh (%d) exceeds the number of devices in the backend (%d)",
			g.numDevices, g.backend.NumDevices())
	}
	return nil
}

func (g *GraphImpl) setupBuilderDistribution() error {
	switch g.distStrategy {
	case distributed.None:
	case distributed.SPMD:
		if err := g.builder.DistributedSPMD(g.NumDevices()); err != nil {
			return errors.WithMessagef(err, "graph failed to create distributed SPMD builder with backend %s", g.backend.Name())
		}
	case distributed.AutoSharding:
		bMeshes := xslices.Map(
			g.deviceMeshes,
			func(m *distributed.DeviceMesh) compute.Mesh { return m.ToBackendsMesh() },
		)
		if err := g.builder.DistributedAutoSharding(bMeshes...); err != nil {
			panic(errors.WithMessagef(err, "Graph failed to create distributed builder with backend %s", g.backend.Name()))
		}
	}
	if g.deviceAssignment == nil && g.numDevices > 1 {
		g.deviceAssignment = xslices.Iota(compute.DeviceNum(0), g.numDevices)
	}
	if g.deviceAssignment != nil {
		if err := g.builder.DeviceAssignment(g.deviceAssignment...); err != nil {
			return errors.WithMessagef(err, "Graph failed to set device assignment with backend %s", g.backend.Name())
		}
	}
	return nil
}

// NumDevices returns the number of devices configured for this graph.
func (g *GraphImpl) NumDevices() int {
	return g.numDevices
}

// DeviceMeshes returns the DeviceMeshes configured for this graph.
func (g *GraphImpl) DeviceMeshes() []*distributed.DeviceMesh {
	return g.deviceMeshes
}

// DeviceAssignment returns the DeviceAssignment configured for this graph.
func (g *GraphImpl) DeviceAssignment() []compute.DeviceNum {
	return g.deviceAssignment
}

// DistributedStrategy returns the distributed strategy configured for this graph.
func (g *GraphImpl) DistributedStrategy() distributed.Strategy {
	return g.distStrategy
}
