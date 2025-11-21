package distributed

import (
	"github.com/pkg/errors"
)

// ShardingSpec (also known as PartitionSpec in JAX) defines how a logical tensor is to be sharded (partitioned) across
// a DeviceMesh. This is used by Shardy, and is based on its documentation in [1].
//
// The definition is per axis of the logical tensor -- and not per axis of the Mesh, a common confusion.
// If not all axes of the Tensor are defined, the tail axes are considered simply to be replicated across the whole
// mesh.
//
// Each tensor axis can be replicated or sharded across one or more mesh axes.
//
// Example:
//
//	mesh := NewDeviceMesh(backend, []int{2, 2}, []string{"data", "model"})
//
//	// Input's "batch" axis is sharded across the "data" axis of the mesh.
//	inputSharding := distributed.NewShardingSpec(mesh, mesh.AddShardedAxis("data")
//
//	// First axis is replicated, second is shared across "model" devices
//	variableSharding := NewShardingSpec(mesh).AddReplicated().AddShardedAxis("model")
//
//	// Second axis is sharded across both "data" and "model" devices.
//	 largeWeights := NewShardingSpec(mesh).AddReplicated().AddShardedAxis("data", "model")
//
// There are two advanced features supported but not tested (pls if you need let us know how it goes, or if you find
// any issues):
//
//  1. The tensor can also be sharded across mesh "sub-axes" -- seed detailed documentation in [1]
//  2. If using ShardingSpec for hints, instead of mesh axes one can give an "open" (in StableHLO marked as "?")
//     axis, with the semantics that XLA Shardy can choose any mesh axis (or axes) to shard the tensor. See [1].
//
// [1] https://github.com/openxla/shardy/blob/main/docs/sharding_representation.md
type ShardingSpec struct {
	Mesh *DeviceMesh
	Axes []AxisSpec
}

// AxisSpec specifies how a tensor axis is to be sharded (or replicated).
// See details in ShardingSpec.
//
// It's a list of mesh axes names, in order. An empty list means the axis is replicated.
type AxisSpec []string

// ReplicatedAxis is a special AxisSpec that means the tensor axis is replicated.
var ReplicatedAxis = AxisSpec(nil)

// NewShardSpec creates a new ShardingSpec for a tensor, defined over the given mesh axes.
//
// It takes an axisSpec for each axis of the tensor (omitted axes are assumed to be replicated).
func NewShardSpec(mesh *DeviceMesh, axisSpec ...AxisSpec) (*ShardingSpec, error) {
	s := &ShardingSpec{mesh, axisSpec}
	meshAxesUsed := make(map[string]bool)
	// Validate mesh axes names.
	for axisIdx, tensorAxisSpec := range s.Axes {
		for _, axisName := range tensorAxisSpec {
			if _, ok := mesh.nameToAxis[axisName]; !ok {
				return nil, errors.Errorf("ShardingSpec axis #%d refers to unknown mesh axis %q", axisIdx, axisName)
			}
			if meshAxesUsed[axisName] {
				return nil, errors.Errorf("mesh axis %q used more than once in ShardingSpec", axisName)
			}
			meshAxesUsed[axisName] = true
		}
	}
	return s, nil
}

// Rank returns the rank of the tensor this ShardingSpec describes.
func (s *ShardingSpec) Rank() int {
	return len(s.Axes)
}

// IsReplicated returns true if the tensor is fully replicated
// (i.e., not sharded along any axis).
func (s *ShardingSpec) IsReplicated() bool {
	if len(s.Axes) == 0 {
		return true
	}
	for _, meshAxes := range s.Axes {
		if len(meshAxes) > 0 {
			return false
		}
	}
	return true
}

// NumDevicesShardingAxis returns the number of devices that will be used to shard the tensor along the given
// tensor axis. If the axis is replicated, it returns 1.
//
// Notice this is about the tensor axis, not the mesh axis. A tensor axis can be sharded across multiple mesh axes.
func (s *ShardingSpec) NumDevicesShardingAxis(axis int) int {
	if axis >= len(s.Axes) {
		return 1 // Replicated.
	}
	meshAxes := s.Axes[axis]
	if len(meshAxes) == 0 {
		return 1 // Replicated.
	}
	size := 1
	for _, meshAxis := range meshAxes {
		size *= s.Mesh.axesSizes[s.Mesh.nameToAxis[meshAxis]]
	}
	return size
}
