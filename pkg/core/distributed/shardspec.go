package distributed

import (
	"github.com/pkg/errors"
)

// ShardSpec (also known as PartitionSpec in JAX) defines how a logical
// tensor (or more concretely a DistributedTensor) is sharded (partitioned) across a DeviceMesh.
//
// It must have the same rank as the tensor it describes.
// Each element of the ShardSpec is a string, which can be:
//  1. An axis name from the DeviceMesh: The corresponding tensor axis
//     is sharded (split) across this mesh axis.
//  2. An empty string (""): The corresponding tensor axis is replicated
//     across all mesh axes.
type ShardSpec []string

// NewShardSpec creates a new ShardSpec.
func NewShardSpec(axes ...string) ShardSpec {
	return axes
}

// Rank returns the rank of the tensor this ShardSpec describes.
func (s ShardSpec) Rank() int {
	return len(s)
}

// IsReplicated returns true if the tensor is fully replicated
// (i.e., not sharded along any axis).
func (s ShardSpec) IsReplicated() bool {
	for _, axisName := range s {
		if axisName != "" {
			return false
		}
	}
	return true
}

// Validate checks that the ShardSpec is valid for the given mesh.
func (s ShardSpec) Validate(mesh *DeviceMesh) error {
	meshAxesUsed := make(map[string]bool)
	for i, axisName := range s {
		if axisName == "" {
			continue // Replicated axis is always valid.
		}
		if _, ok := mesh.nameToAxis[axisName]; !ok {
			return errors.Errorf("ShardSpec axis %d refers to unknown mesh axis %q", i, axisName)
		}
		if meshAxesUsed[axisName] {
			return errors.Errorf("mesh axis %q used more than once in ShardSpec", axisName)
		}
		meshAxesUsed[axisName] = true
	}
	return nil
}
