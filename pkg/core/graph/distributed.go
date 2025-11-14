package graph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/pkg/errors"
)

// WithDistributedStrategy sets the distributed strategy for the graph.
// This must be set before compiling the graph or adding any distributed operations.
//
// The default is distributed.NoStrategy.
// If setting to something else, you must also call WithDeviceMesh.
func (g *Graph) WithDistributedStrategy(s distributed.Strategy) *Graph {
	g.distStrategy = s
	return g
}

// DistributedStrategy returns the distributed strategy set for the graph.
func (g *Graph) DistributedStrategy() distributed.Strategy {
	return g.distStrategy
}

// WithDeviceMesh sets the device mesh for the graph.
//
// This is required is the WithDistributedStrategy is set to something different than
// distributed.NoStrategy.
// This must be set before compiling the graph or adding any distributed operations.
func (g *Graph) WithDeviceMesh(mesh *distributed.DeviceMesh) *Graph {
	g.deviceMesh = mesh
	return g
}

// DeviceMesh returns the graph's DeviceMesh.
func (g *Graph) DeviceMesh() *distributed.DeviceMesh {
	return g.deviceMesh
}

// NumDevices participating in this computation graph.
//
// This is 1 for non-distributed graphs. Otherwise, it's the number of devices in the mesh (see Graph.WithDeviceMesh).
func (g *Graph) NumDevices() int {
	if g.deviceMesh == nil {
		return 1
	}
	return g.deviceMesh.NumDevices()

}

// nextChannelID returns the next channel ID to use for synchronization.
// This should be unique, and we use them incrementally.
func (g *Graph) nextChannelID() int {
	next := g.currentChannelID
	g.currentChannelID++
	return next
}

// DistributedOps provides a namespace for all distributed and collective
// operations on a graph. It is accessed via graph.Graph.Distributed().
//
// It also acts as a builder, allowing optional parameters (like mesh axes)
// to be set via chaining.
type DistributedOps struct {
	g    *Graph
	axes []string // Stores the mesh axes for the next op.

	// channelID is used a unique id for channel synchronization.
	// If nil, it will use the Graph's incremental channelID.
	channelID *int
}

// Distributed returns a helper object that provides access to all distributed and collective operations.
func (g *Graph) Distributed() DistributedOps {
	d := DistributedOps{g: g}
	switch g.distStrategy {
	case distributed.SPMD:
		if g.deviceMesh == nil {
			exceptions.Panicf("graph.Distributed() with SPMD requires a device mesh to be set")
		}
		d.axes = g.deviceMesh.AxisNames()
	case distributed.GSPMD:
		exceptions.Panicf("graph.Distributed() with GSPMD is not supported yet")
	case distributed.None:
		// No axes defined.
	}
	return d
}

// Along specifies which DeviceMesh axes the *next* collective
// operation should apply to.
//
// For example:
//
//	g.Distributed().Along("data").AllReduce(x)
//
// This will perform an AllReduce along the "data" axis of the mesh.
func (d DistributedOps) Along(meshAxes ...string) DistributedOps {
	dOut := d
	dOut.axes = meshAxes
	return dOut
}

// OnChannel specifies the channel ID to use for synchronization.
// All communicating devices must use the same channel ID, so it must be agreed upon in some fashion.
//
// For SPMD programs, you usually don't need to use this, since the default is to use an incremental
// channelID. And since it's the same Graph executed in every device, it's usually fine.
func (d DistributedOps) OnChannel(channelID int) DistributedOps {
	dOut := d
	dOut.channelID = &channelID
	return dOut
}

// AllReduce performs an AllReduce operation across the devices specified
// in the chained options.
//
// If no axes were specified via DistributedOps.Along(), it performs the operation
// across *all* devices in the mesh.
// If no channelID was specified via DistributedOps.OnChannel(), it uses the Graph's incremental channelID.
func (d DistributedOps) AllReduce(input *Node, reductionType backends.ReduceOpType) *Node {
	return d.AllReduceMany([]*Node{input}, reductionType)[0]
}

// AllReduceMany performs an AllReduce operation across the devices specified
// in the chained options.
//
// It takes a collection of operands and returns them reduced operands across the devices as outputs.
// So the output shapes are the same as the operand shapes.
//
// If no axes were specified via .Along(), it performs the operation
// across *all* devices in the mesh.
func (d DistributedOps) AllReduceMany(operands []*Node, reductionType backends.ReduceOpType) []*Node {
	if len(operands) == 0 {
		return nil // Or panic
	}
	g := operands[0].Graph()

	// 1. Get the DeviceMesh from the graph.
	// (This implies you need a new method on graph.Graph)
	mesh := d.g.deviceMesh
	if mesh == nil {
		// Single-device graph: this is a no-op.
		return operands
	}
	replicaGroups, err := mesh.ComputeReplicaGroups(d.axes)
	if err != nil {
		panic(errors.WithMessagef(err, "failed compute replicaGroups for AllReduce"))
	}
	channelID := g.nextChannelID()
	return backendAllReduce(operands, reductionType, replicaGroups, channelID)
}
