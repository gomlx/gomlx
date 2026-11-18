// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/gomlx/internal/graphimpl"
	"github.com/gomlx/gomlx/support/exceptions"
)

// SetAutoSharding sets the distributed strategy to AutoSharding and sets the device meshes.
func (g *Graph) SetAutoSharding(meshes ...*distributed.DeviceMesh) error {
	return g.impl.SetAutoSharding(meshes...)
}

// SetDeviceAssignment specifies which concrete devices to use for the graph.
func (g *Graph) SetDeviceAssignment(devices []compute.DeviceNum) error {
	return g.impl.SetDeviceAssignment(devices)
}

// SetSPMD sets the distributed strategy to SPMD.
func (g *Graph) SetSPMD(mesh *distributed.DeviceMesh) error {
	return g.impl.SetSPMD(mesh)
}

// DistributedStrategy returns the distributed strategy set for the graph.
func (g *Graph) DistributedStrategy() distributed.Strategy {
	return g.impl.DistributedStrategy()
}

// DeviceMeshes returns the graph's DeviceMeshes.
func (g *Graph) DeviceMeshes() []*distributed.DeviceMesh {
	return g.impl.DeviceMeshes()
}

// NumDevices participating in this computation graph.
func (g *Graph) NumDevices() int {
	return g.impl.NumDevices()
}

// DistributedOps provides a namespace for distributed and collective operations on a graph.
type DistributedOps struct {
	g    *Graph
	axes []string
}

// Distributed returns a helper object that provides access to all distributed and collective operations.
func (g *Graph) Distributed() *DistributedOps {
	d := &DistributedOps{
		g: g,
	}
	switch g.DistributedStrategy() {
	case distributed.SPMD:
		meshes := g.DeviceMeshes()
		if len(meshes) == 0 || meshes[0] == nil {
			exceptions.Panicf("graph.Distributed() with SPMD requires a non-empty device mesh")
		}
		d.Along(meshes[0].AxesNames()...)
	case distributed.AutoSharding:
		exceptions.Panicf("if using AutoSharding you should not use graph.Distributed() operations: the " +
			"sharding of the operations happens automatically, without any explicit distributed calls.")
	case distributed.None:
	}
	return d
}

// Along specifies which mesh axes the distributed operation will operate on.
func (d *DistributedOps) Along(meshAxes ...string) *DistributedOps {
	d.axes = meshAxes
	return d
}

// AllReduce performs an all-reduce across the devices specified by Along().
func (d *DistributedOps) AllReduce(operands []*Node, reductionType compute.ReduceOpType) []*Node {
	if len(operands) == 0 {
		exceptions.Panicf("Distributed().AllReduce() requires at least one operand")
	}
	if d.g.DistributedStrategy() != distributed.SPMD {
		exceptions.Panicf("Distributed().AllReduce() requires SPMD distribution strategy, got %s", d.g.DistributedStrategy())
	}
	meshes := d.g.DeviceMeshes()
	if len(meshes) == 0 || meshes[0] == nil {
		exceptions.Panicf("Distributed().AllReduce() requires a device mesh")
	}
	mesh := meshes[0]
	replicaGroups, err := mesh.ComputeReplicaGroups(d.axes)
	if err != nil {
		panic(err)
	}
	return wrapNodes(graphimpl.BackendAllReduce(unwrapNodes(operands), reductionType, replicaGroups))
}

// AllReduceOne is a convenience wrapper for AllReduce with a single operand.
func (d *DistributedOps) AllReduceOne(input *Node, reductionType compute.ReduceOpType) *Node {
	return d.AllReduce([]*Node{input}, reductionType)[0]
}
