// Package distributed defines the following objects related to cross-device execution:
//
// - DeviceMesh: expresses the topology of a set of devices, in terms of axis and their sizes.
// - ShardSpec: defines how a logical tensor is sharded across a DeviceMesh.
// - Tensor: a logical tensor distributed across multiple devices organized as a DeviceMesh.
package distributed

import (
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"

	"github.com/pkg/errors"
)

// Tensor is a logical tensor distributed across multiple devices (organized as a DeviceMesh).
// It is a container for its shards (concrete tensors.Tensor), one per device.
//
// The logical distributed Tensor can be either replicated or sharded at each axis.
// When an axis is sharded, it is shared over a DeviceMesh axis, and the distributed tensor is
// equally split across that configured mesh axis.
// When an axis is replicated, the tensor is replicated across all mesh axes.
//
// The idea is that when executing a distributed computation, each device will receive the corresponding
// tensor shard, replicated and/or sharded, according to the specification.
type Tensor struct {
	// mesh is the DeviceMesh this tensor is distributed on.
	mesh *DeviceMesh

	// spec defines how this tensor is sharded across the mesh.
	spec ShardSpec

	// shards holds the physical tensor data for each device.
	// The map key is the device's global ordinal index (0 to NumDevices-1).
	shards []*tensors.Tensor

	// logicalShape is the shape of the distributed tensor seen as one.
	// shardShape is the shape of the individual shards.
	logicalShape, shardShape shapes.Shape
}

// New creates a new distributed Tensor.
// It assumes the provided shards are already on their respective devices.
func New(mesh *DeviceMesh, spec ShardSpec, shards []*tensors.Tensor) (*Tensor, error) {
	if err := spec.Validate(mesh); err != nil {
		return nil, errors.Wrap(err, "invalid ShardSpec")
	}
	if len(shards) != mesh.NumDevices() {
		return nil, errors.Errorf("number of shards (%d) does not match number of devices in mesh (%d)", len(shards), mesh.NumDevices())
	}
	// TODO: Validate that shard shapes are consistent with the ShardSpec, and that all shards have the same shape.
	return &Tensor{
		mesh:       mesh,
		spec:       spec,
		shards:     shards,
		shardShape: shards[0].Shape(),
	}, nil
}

// calculateLogicalShape based on the shardShape and the ShardSpec.
func (dt *Tensor) calculateLogicalShape() {
	// TODO: calculate and sets logicalShape.
	dt.logicalShape = dt.shardShape
}

// Mesh redt.logicaturns the DeviceMesh for this tensor.
func (dt *Tensor) Mesh() *DeviceMesh {
	return dt.mesh
}

// ShardSpec returns the sharding specification for this tensor.
func (dt *Tensor) ShardSpec() ShardSpec {
	return dt.spec
}

// Shards returns the physical tensor shards.
// They are owned by the distributed.Tensor object, and the slice shouldn't be modified -- the contents of the
// individual shards can be modified directly, if they are stored locally.
func (dt *Tensor) Shards() []*tensors.Tensor {
	return dt.shards
}

// Shape returns the logical, unsharded shape of the tensor.
func (dt *Tensor) Shape() shapes.Shape {
	return dt.logicalShape
}

// ShardShape returns the shape of the individual shards.
func (dt *Tensor) ShardShape() shapes.Shape {
	return dt.shardShape
}
