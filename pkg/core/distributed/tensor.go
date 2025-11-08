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

// Tensor is a logical tensor distributed across
// multiple devices organized as a DeviceMesh.
//
// It holds the physical tensor shards (one per device) and the
// sharding specification as a ShardSpec.
type Tensor struct {
	// mesh is the DeviceMesh this tensor is distributed on.
	mesh *DeviceMesh

	// spec defines how this tensor is sharded across the mesh.
	spec ShardSpec

	// shards holds the physical tensor data for each device.
	// The map key is the device's global ordinal index (0 to NumDevices-1).
	shards map[int]*tensors.Tensor
}

// New creates a new Tensor.
// It assumes the provided shards are already on their respective devices.
func New(mesh *DeviceMesh, spec ShardSpec, shards map[int]*tensors.Tensor) (*Tensor, error) {
	if err := spec.Validate(mesh); err != nil {
		return nil, errors.Wrap(err, "invalid ShardSpec")
	}
	if len(shards) != mesh.NumDevices() {
		return nil, errors.Errorf("number of shards (%d) does not match number of devices in mesh (%d)", len(shards), mesh.NumDevices())
	}
	// TODO: Validate that shard shapes are consistent with the ShardSpec.
	return &Tensor{
		mesh:   mesh,
		spec:   spec,
		shards: shards,
	}, nil
}

// Mesh returns the DeviceMesh for this tensor.
func (dt *Tensor) Mesh() *DeviceMesh {
	return dt.mesh
}

// ShardSpec returns the sharding specification for this tensor.
func (dt *Tensor) ShardSpec() ShardSpec {
	return dt.spec
}

// Shards returns the map of physical tensor shards.
func (dt *Tensor) Shards() map[int]*tensors.Tensor {
	return dt.shards
}

// Shape returns the logical, unsharded shape of the tensor.
// (This requires calculating it from the shard shapes and ShardSpec)
func (dt *Tensor) Shape() shapes.Shape {
	// TODO: Implement shape calculation.
	// For now, just return the shape of the first shard as a placeholder.
	if shard, ok := dt.shards[0]; ok {
		return shard.Shape() // This is the per-shard shape not the logical shape.
	}
	return shapes.Shape{} // Should not happen.
}
