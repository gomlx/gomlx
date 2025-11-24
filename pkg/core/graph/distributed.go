package graph

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

// SetAutoSharding sets the distributed strategy to AutoSharding and sets the device
// meshes that will be used for this Graph computation.
//
// This is the supported and recommended strategy for distributed training.
// It uses XLA Shardy [1] framework to automatically shard the computation across the
// devices (both data and model), given sharded input and output shapes of the computations.
// (and optional in between hints).
//
// [1] https://github.com/openxla/shardy
func (g *Graph) SetAutoSharding(meshes ...*distributed.DeviceMesh) error {
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
//
// These must be valid numbers for the backend and must match the number of devices of the
// largest mesh given to WithAutoSharding or WithSPMD.
//
// The default assignment is simply using the devices in the order they were added to the backend
// (sequential DeviceNum values, starting from 0).
func (g *Graph) SetDeviceAssignment(devices []backends.DeviceNum) error {
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
	numDevicesAvailable := backends.DeviceNum(g.backend.NumDevices())
	seen := sets.Make[backends.DeviceNum]()
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
//
// This is a Single Program Multiple Data (SPMD) strategy, where synchronization is done
// by the user with the "collective" operations (see Graph.Distributed()).
//
// EXPERIMENTAL: For normal use WithAutoSharding instead. If you want to use this consider
// reaching out to the GoMLX team to discuss what you will need.
func (g *Graph) SetSPMD(mesh *distributed.DeviceMesh) error {
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

// setupBuilderDistribution should be called as soon as the backends.Builder object is created,
// and it will set it up with the Graph configuration for distribution.
func (g *Graph) setupBuilderDistribution() error {
	switch g.distStrategy {
	case distributed.None:
		// Nothing to do.
	case distributed.SPMD:
		err := g.builder.DistributedSPMD(g.NumDevices())
		if err != nil {
			return errors.WithMessagef(err,
				"graph failed to create distributed SPMD builder with backend %s",
				g.backend.Name())
		}
	case distributed.AutoSharding:
		bMeshes := xslices.Map(
			g.deviceMeshes,
			func(m *distributed.DeviceMesh) backends.Mesh { return m.ToBackendsMesh() },
		)
		err := g.builder.DistributedAutoSharding(bMeshes...)
		if err != nil {
			panic(errors.WithMessagef(err,
				"Graph failed to create distributed builder with backend %s",
				g.backend.Name()))
		}
	}

	// Create a default device assignment if numDevices > 1 -- computations for one device only may be portable.
	if g.deviceAssignment == nil && g.numDevices > 1 {
		g.deviceAssignment = xslices.Iota(backends.DeviceNum(0), g.numDevices)
	}
	if g.deviceAssignment != nil {
		err := g.builder.DeviceAssignment(g.deviceAssignment...)
		if err != nil {
			return errors.WithMessagef(err,
				"Graph failed to set device assignment with backend %s", g.backend.Name())
		}
	}
	return nil
}

// DistributedStrategy returns the distributed strategy set for the graph.
// This is changed by configuring the Graph with SetAutoSharding or SetSPMD.
func (g *Graph) DistributedStrategy() distributed.Strategy {
	return g.distStrategy
}

// DeviceMeshes returns the graph's DeviceMeshes.
// These are owned by the graph and should not be modified.
func (g *Graph) DeviceMeshes() []*distributed.DeviceMesh {
	return g.deviceMeshes
}

// NumDevices participating in this computation graph.
//
// This is 1 for non-distributed graphs. Otherwise, it's the number of devices in the mesh (see Graph.WithDeviceMesh).
func (g *Graph) NumDevices() int {
	return g.numDevices
}

// DistributedOps provides a namespace for all distributed and collective
// operations on a graph. It is accessed via graph.Graph.Distributed().
//
// It also acts as a builder, allowing optional parameters (like mesh axes)
// to be set via chaining.
type DistributedOps struct {
	g    *Graph
	axes []string
}

// Distributed returns a helper object that provides access to all distributed and collective operations.
func (g *Graph) Distributed() *DistributedOps {
	d := &DistributedOps{
		g: g,
	}
	switch g.distStrategy {
	case distributed.SPMD:
		// For SPMD the default is to operate over all the axes of the mesh.
		if g.deviceMeshes == nil {
			exceptions.Panicf("graph.Distributed() with SPMD requires a device mesh to be set")
		}
		mesh := g.deviceMeshes[0]
		if mesh == nil {
			exceptions.Panicf("graph.Distributed() with SPMD requires a non-empty device mesh")
		}
		d.Along(mesh.AxesNames()...)
	case distributed.AutoSharding:
		exceptions.Panicf("if using AutoSharding you should not use graph.Distributed() operations: the " +
			"sharding of the operations happens automatically, without any explicit distributed calls.")
	case distributed.None:
		// No axes defined.
	}
	return d
}

// Along specifies which DeviceMeshes axes the *next* collective
// operation should apply to.
//
// For example:
//
//	g.Distributed().Along("data").AllReduceOne(x, backends.ReduceOpSum)
//
// This will perform an AllReduceOne along the "data" axis of the mesh.
func (d *DistributedOps) Along(meshAxes ...string) *DistributedOps {
	d.axes = meshAxes
	return d
}

// AllReduce performs a reduce operation across the devices specified in the chained options.
//
// It takes a collection of operands and reduces each of them across the devices as outputs.
// So the output shapes are the same as the operand shapes.
//
// If no device axes were specified via DistributedOps.Along(), it performs the operation
// across *all* devices in the mesh.
func (d *DistributedOps) AllReduce(operands []*Node, reductionType backends.ReduceOpType) []*Node {
	if len(operands) == 0 {
		return nil // Or panic
	}
	mesh := d.g.deviceMeshes[0]
	if mesh == nil {
		// Single-device graph: this is a no-op.
		return operands
	}
	replicaGroups, err := mesh.ComputeReplicaGroups(d.axes)
	if err != nil {
		panic(errors.WithMessagef(err, "failed compute replicaGroups for AllReduceOne"))
	}
	return backendAllReduce(operands, reductionType, replicaGroups)
}

// AllReduceOne performs a reduce operation across the devices specified
// in the chained options.
// So the output shapes are the same as the operand shapes.
//
// If no device axes were specified via DistributedOps.Along(), it performs the operation
// across *all* devices in the mesh.
func (d *DistributedOps) AllReduceOne(input *Node, reductionType backends.ReduceOpType) *Node {
	return d.AllReduce([]*Node{input}, reductionType)[0]
}
