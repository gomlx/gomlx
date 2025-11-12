package graph

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/distributed"
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

// DistributedOps provides a namespace for all distributed and collective
// operations on a graph. It is accessed via graph.Graph.Distributed().
//
// It also acts as a builder, allowing optional parameters (like mesh axes)
// to be set via chaining.
type DistributedOps struct {
	g    *Graph
	axes []string // Stores the mesh axes for the next op.
}

// Distributed returns a helper object that provides access to all distributed and collective operations.
func (g *Graph) Distributed() DistributedOps {
	d := &DistributedOps{g: g}
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
	return DistributedOps{
		g:    d.g,
		axes: meshAxes,
	}
}

// AllReduce performs an AllReduce operation across the devices specified
// in the chained options.
//
// If no axes were specified via .Along(), it performs the operation
// across *all* devices in the mesh.
func (d DistributedOps) AllReduce(op backends.ReduceOpType, input *Node) *Node {
	return d.AllReduceMany(op, []*Node{input})[0]
}

// AllReduceMany performs an AllReduce operation across the devices specified
// in the chained options.
//
// It takes a slice of inputs and returns a slice of the reduced inputs across the devices as outputs.
//
// If no axes were specified via .Along(), it performs the operation
// across *all* devices in the mesh.
func (d DistributedOps) AllReduceMany(op backends.ReduceOpType, inputs []*Node) []*Node {
	if len(inputs) == 0 {
		return nil // Or panic
	}

	// 1. Get the DeviceMesh from the graph.
	// (This implies you need a new method on graph.Graph)
	mesh := d.g.deviceMesh
	if mesh == nil {
		// Single-device graph: this is a no-op.
		return
	}
	var replicaGroups [][]int
	if mesh == nil {
		replicaGroups = [][]int{{0}}
	} else {
		replicaGroups, err := mesh.ComputeReplicaGroups(d.axes)
		if err != nil {
			return Panicf(d.g, "AllReduce: %v", err)
		}
	}

	// 2. THIS IS YOUR LOGIC:
	//    Translate the logical 'd.axes' names into the physical 'replica_groups'.

	// 3. Pass the replica_groups as the argument to the backend op.
	//    The backend op argument is now a struct/list.
	opArgs := backends.AllReduceArgs{
		ReduceOp:      op,
		ReplicaGroups: replicaGroups,
	}
	return d.g.newNode(inputs, nil, backends.AllReduce, opArgs)
}
