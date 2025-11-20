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

	// channelIDGenerator is used to generate unique ids for channel synchronization.
	// The default is the Graph's incremental channelID, which works fine for SPMD graphs.
	channelIDGenerator func() int
}

// Distributed returns a helper object that provides access to all distributed and collective operations.
func (g *Graph) Distributed() *DistributedOps {
	d := &DistributedOps{
		g:                  g,
		channelIDGenerator: g.nextChannelID,
	}
	switch g.distStrategy {
	case distributed.SPMD:
		if g.deviceMesh == nil {
			exceptions.Panicf("graph.Distributed() with SPMD requires a device mesh to be set")
		}
		d.axes = g.deviceMesh.AxisNames()
	case distributed.AutoSharding:
		exceptions.Panicf("if using AutoSharding you should not use graph.Distributed() operations: the " +
			"sharding of the operations happens automatically, without any explicit distributed calls.")
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
//	g.Distributed().Along("data").AllReduceOne(x, backends.ReduceOpSum)
//
// This will perform an AllReduceOne along the "data" axis of the mesh.
func (d *DistributedOps) Along(meshAxes ...string) *DistributedOps {
	d.axes = meshAxes
	return d
}

// WithChannelIDGenerator specifies a generator of channel IDs to use for synchronization.
// All communicating devices must use the same channel ID, so it must be agreed upon in some fashion.
//
// We use a channel ID generator because we don't know upfront how many IDs will be needed: different
// backends may need different numbers of IDs. Plus, extra IDs may be needed for the gradient computation
// (in case it is done).
//
// For SPMD programs, you usually don't need to use this, since the default is to use an incremental
// channelID generator provided by the Graph itself.
// Since it's the same Graph executed in every device, it's usually fine.
func (d *DistributedOps) WithChannelIDGenerator(channelIDGenerator func() int) *DistributedOps {
	d.channelIDGenerator = channelIDGenerator
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
	mesh := d.g.deviceMesh
	if mesh == nil {
		// Single-device graph: this is a no-op.
		return operands
	}
	replicaGroups, err := mesh.ComputeReplicaGroups(d.axes)
	if err != nil {
		panic(errors.WithMessagef(err, "failed compute replicaGroups for AllReduceOne"))
	}
	return backendAllReduce(operands, reductionType, replicaGroups, d.channelIDGenerator)
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
