// Package distributed defines the following objects related to cross-device execution:
//
// - DeviceMesh: expresses the topology of a set of devices, in terms of axis and their sizes.
// - ShardingSpec: defines how a logical tensor is sharded across a DeviceMesh.
// - Tensor: a logical tensor distributed across multiple devices organized as a DeviceMesh.
package distributed

import (
	"fmt"
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
)

// Tensor is a logical tensor meant to be distributed across multiple devices (organized as a DeviceMesh).
// It is a container for its shards (concrete tensors.Tensor), one per device.
// The shards may or may not be stored on a backend (or local on the host), it's up to the caller to manage
// the shards storage.
//
// The logical distributed Tensor can be either replicated or sharded at each axis.
// When an axis is sharded, it is shared over a DeviceMesh axis, and the distributed tensor is
// equally split across that configured mesh axis.
// When an axis is replicated, the tensor is replicated across all mesh axes.
//
// The idea is that when executing a distributed computation, each device will receive the corresponding
// tensor shard, replicated and/or sharded, according to the specification.
type Tensor struct {
	// spec defines how this tensor is sharded across the mesh.
	spec *ShardingSpec

	// mesh comes from the spec.
	mesh *DeviceMesh

	// shards holds the physical tensor data for each device.
	// The map key is the device's global ordinal index (0 to NumDevices-1).
	shards []*tensors.Tensor

	// logicalShape is the shape of the distributed tensor seen as one.
	// shardShape is the shape of the individual shards.
	logicalShape, shardShape shapes.Shape
}

// NewTensor creates a new distributed Tensor.
// It assumes the provided shards are already on their respective devices.
func NewTensor(spec *ShardingSpec, shards []*tensors.Tensor) (*Tensor, error) {
	if len(shards) == 0 {
		return nil, errors.New("distributed.NewTensor requires shards for initialization, none was provided")
	}
	mesh := spec.Mesh
	if len(shards) != mesh.NumDevices() {
		return nil, errors.Errorf(
			"number of shards (%d) does not match number of devices in mesh (%d)",
			len(shards),
			mesh.NumDevices(),
		)
	}

	// Create the distributed tensor.
	dt := &Tensor{
		mesh:   mesh,
		spec:   spec,
		shards: slices.Clone(shards),
	}
	if err := dt.validateShards(); err != nil {
		return nil, err
	}
	dt.shardShape = shards[0].Shape()
	err := dt.calculateLogicalShape()
	if err != nil {
		return nil, err
	}
	return dt, nil
}

// NumShards returns the number of shards in the distributed tensor.
func (dt *Tensor) NumShards() int {
	return len(dt.shards)
}

// calculateLogicalShape based on the shardShape and the ShardingSpec.
func (dt *Tensor) calculateLogicalShape() error {
	logicalShape := dt.shardShape.Clone()
	for tensorAxis, axisSpec := range dt.spec.Axes {
		if len(axisSpec) == 0 {
			// Replicated axis, it's always valid.
			continue
		}
		meshSize := 1
		for _, meshAxisName := range axisSpec {
			s, err := dt.mesh.AxisSize(meshAxisName)
			if err != nil {
				return errors.WithMessagef(err,
					"inconsistency in distributed.Tensor, sharding spec references mesh tensorAxis %q not in mesh %s",
					meshAxisName, dt.mesh)
			}
			meshSize *= s
		}
		logicalShape.Dimensions[tensorAxis] *= meshSize
	}
	dt.logicalShape = logicalShape
	return nil
}

// Mesh redt.logicaturns the DeviceMesh for this tensor.
func (dt *Tensor) Mesh() *DeviceMesh {
	return dt.mesh
}

// Shards returns the physical tensor shards.
// They are owned by the distributed.Tensor object, and the slice shouldn't be modified -- the contents of the
// individual shards can be modified directly, if they are stored locally.
//
// It returns Tensor.NumDevices() shards.
func (dt *Tensor) Shards() []*tensors.Tensor {
	return dt.shards
}

// Shape returns the logical, unsharded shape of the tensor.
func (dt *Tensor) Shape() shapes.Shape {
	return dt.logicalShape
}

// DType returns the logical, unsharded shape of the tensor.
func (dt *Tensor) DType() dtypes.DType {
	return dt.logicalShape.DType
}

// ShardingSpec returns the sharding specification for this tensor.
func (dt *Tensor) ShardingSpec() *ShardingSpec {
	return dt.spec
}

// validateShards that all shards have the same shape and are consistent with the `ShardingSpec`.
func (dt *Tensor) validateShards() error {
	shards := dt.shards
	if len(shards) == 0 {
		return errors.New("cannot create a distributed tensor with no shards")
	}
	if err := shards[0].CheckValid(); err != nil {
		return errors.WithMessagef(err, "invalid shard 0 for distributed.Tensor")
	}
	shardShape := shards[0].Shape()
	for i, shard := range shards {
		if err := shard.CheckValid(); err != nil {
			return errors.WithMessagef(err, "invalid shard %d for distributed.Tensor", i)
		}
		if !shard.Shape().Equal(shardShape) {
			return errors.Errorf("shard %d has shape %s, but shard 0 has shape %s",
				i, shard.Shape(), shardShape)
		}
	}

	// Make sure all on-device shards are on the same backend, if any is on-device.
	var refBackend backends.Backend
	for i, shard := range shards {
		if shard.IsOnAnyDevice() {
			backend, err := shard.Backend()
			if err != nil {
				return errors.WithMessagef(err, "failed to get backend of shard %d", i)
			}
			if refBackend == nil {
				refBackend = backend
			} else if refBackend != backend {
				return errors.Errorf("shard #%d is on backend %s, but shard 0 is on backend %s",
					i, backend.Name(), refBackend.Name())
			}
		}
	}

	// Check that the spec is valid for the Mesh.
	mesh := dt.mesh
	spec := dt.spec
	if spec.Rank() > shardShape.Rank() {
		return errors.Errorf("shard shape %s is too small (rank) for sharding spec %s", shardShape, spec)
	}
	for _, axisSpec := range spec.Axes {
		if len(axisSpec) == 0 {
			// Replicated axis, it's always valid.
			continue
		}
		for _, meshAxisName := range axisSpec {
			_, err := mesh.AxisSize(meshAxisName)
			if err != nil {
				return errors.WithMessagef(err,
					"inconsistency in distributed.Tensor, sharding spec references mesh axis %q not in mesh %s",
					meshAxisName, mesh)
			}
		}
	}
	return nil
}

// ShardShape returns the shape of the individual shards.
func (dt *Tensor) ShardShape() shapes.Shape {
	return dt.shardShape
}

// Ok returns whether the distributed tensor is in a valid state -- and hasn't been finalized.
func (dt *Tensor) Ok() bool {
	return dt != nil && len(dt.shards) > 0 && dt.spec != nil && dt.mesh != nil
}

// Finalize releases the memory associated with the distributed tensor.
// It calls FinalizeAll on each of the shards.
// It is safe to call Finalize on an already finalized tensor.
func (dt *Tensor) Finalize() error {
	if dt.shards == nil {
		return nil
	}
	var firstErr error
	for _, shard := range dt.shards {
		if err := shard.FinalizeAll(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	dt.shards = nil
	dt.spec = nil
	dt.mesh = nil
	return firstErr
}

// FinalizeAll is an alias to Finalize.
func (dt *Tensor) FinalizeAll() error {
	return dt.Finalize()
}

// Clone creates a copy of the distributed Tensor.
// It clones each shard on the device it currently resides.
func (dt *Tensor) Clone() (*Tensor, error) {
	newShards := make([]*tensors.Tensor, len(dt.shards))
	for i, shard := range dt.shards {
		var err error
		newShards[i], err = shard.Clone()
		if err != nil {
			return nil, errors.WithMessagef(err, "distributed.Tensor.Clone: failed to clone shard %d", i)
		}
	}
	// NewTensor will validate the new shards and calculate shapes.
	return NewTensor(dt.spec, newShards)
}

// Merge merges the tensors into one concrete logical tensor.
func (dt *Tensor) Merge() (*tensors.Tensor, error) {
	// Create a new tensor with the logical shape.
	rank := dt.logicalShape.Rank()
	t := tensors.FromShape(dt.logicalShape)
	toStrides := dt.logicalShape.Strides()

	elementSize := dt.logicalShape.DType.Size()
	if elementSize == 0 {
		return nil, errors.Errorf("merge of tensors with sub-byte sizes not implemented (for DType %s)",
			dt.logicalShape.DType)
	}

	// Pre-calculate mesh info for efficiency
	mesh := dt.mesh
	spec := dt.spec
	meshRank := mesh.Rank()
	meshSizes := mesh.AxesSizes()
	meshAxesNames := mesh.AxesNames()

	// Build a mapping from mesh axis name to its index for efficient lookup
	meshNameToIdx := make(map[string]int, meshRank)
	for i, name := range meshAxesNames {
		meshNameToIdx[name] = i
	}

	// Track which shard positions have already been processed to skip replicated shards.
	processedShardPositions := make(map[string]bool)

	var innerErr error
	err := t.MutableBytes(func(tBytes []byte) {
		// Iterate over all devices, including replicated ones.
		for deviceIdx := range len(dt.shards) {
			// Calculate mesh coordinates for this device.
			// Device indices are laid out in row-major order across mesh dimensions.
			meshCoords := make([]int, meshRank)
			remaining := deviceIdx
			for i := meshRank - 1; i >= 0; i-- {
				meshCoords[i] = remaining % meshSizes[i]
				remaining /= meshSizes[i]
			}

			// Calculate the shard position for each tensor axis based on which mesh axes shard it.
			shardPos := make([]int, rank)
			for tensorAxis := 0; tensorAxis < rank; tensorAxis++ {
				if tensorAxis >= spec.Rank() || len(spec.Axes[tensorAxis]) == 0 {
					// Replicated tensor axis: shard position is 0.
					shardPos[tensorAxis] = 0
				} else {
					// Sharded tensor axis: combine mesh coordinates for all mesh axes that shard this axis.
					axisSpec := spec.Axes[tensorAxis]
					pos := 0
					multiplier := 1
					for i := len(axisSpec) - 1; i >= 0; i-- {
						meshAxisName := axisSpec[i]
						meshAxisIdx := meshNameToIdx[meshAxisName]
						pos += meshCoords[meshAxisIdx] * multiplier
						multiplier *= meshSizes[meshAxisIdx]
					}
					shardPos[tensorAxis] = pos
				}
			}

			// Create a unique key for this shard position to detect duplicates.
			// Convert shard position to a string representation for use as a map key.
			shardPosKey := fmt.Sprintf("%v", shardPos)

			// Skip if we've already processed this shard position (replicated shard).
			if processedShardPositions[shardPosKey] {
				continue
			}
			processedShardPositions[shardPosKey] = true

			shard := dt.shards[deviceIdx]
			innerErr = shard.ConstBytes(func(shardBytes []byte) {
				// Calculate the slice of the logical tensor that corresponds to this shard.
				sliceStarts := make([]int, rank)

				// Calculate the base offset in the destination (logical) tensor buffer.
				dstBaseOffset := 0
				for axis := range rank {
					sliceStarts[axis] = shardPos[axis] * dt.shardShape.Dimensions[axis]

					// Add stride offset: coordinate * stride * bytes_per_element
					dstBaseOffset += sliceStarts[axis] * toStrides[axis]
				}
				dstBaseOffset *= elementSize

				// Optimization: Determine the largest contiguous block of bytes.
				// We iterate from the innermost dimension outwards. If the shard dimension
				// matches the logical dimension, the data is contiguous along that axis.
				contiguousRank := rank
				blockSize := elementSize
				for i := rank - 1; i >= 0; i-- {
					if dt.shardShape.Dimensions[i] != dt.logicalShape.Dimensions[i] {
						// Found a split dimension. The dimensions below this (i+1 to rank)
						// are the contiguous block.
						contiguousRank = i + 1
						break
					}
					// This dimension is not split, so it contributes to the contiguous block.
					contiguousRank = i
					blockSize *= dt.shardShape.Dimensions[i]
				}

				// Recursive copy function.
				// srcOffset: tracks the linear position in the Shard (source).
				//            Since shards are dense, this just increments by blockSize.
				// dstOffset: tracks the position in the Logical Tensor (destination).
				//            The jumps are based on strides.
				var srcOffset int
				var copier func(axis int, dstOffset int)

				copier = func(axis int, dstOffset int) {
					// Base Case: We reached the contiguous block. Perform a direct memory copy.
					if axis == contiguousRank {
						copy(tBytes[dstOffset:dstOffset+blockSize], shardBytes[srcOffset:srcOffset+blockSize])
						srcOffset += blockSize
						return
					}

					// Recursive Step: Iterate along the split dimensions.
					dimSize := dt.shardShape.Dimensions[axis]
					step := toStrides[axis] * elementSize
					for i := 0; i < dimSize; i++ {
						copier(axis+1, dstOffset+i*step)
					}
				}

				// Start the copy process
				copier(0, dstBaseOffset)
			})
			if innerErr != nil {
				return
			}
		}
	})
	if err != nil {
		return nil, err
	}
	if innerErr != nil {
		return nil, innerErr
	}
	return t, nil
}

// ShardTensor splits a tensor into individual shards.
// This all happen on the host, and the shards of the returned distributed.Tensor are local tensors.
func ShardTensor(spec *ShardingSpec, t *tensors.Tensor) (*Tensor, error) {
	logicalShape := t.Shape()
	shardShape := logicalShape.Clone()
	mesh := spec.Mesh

	// Calculate shard shape by dividing tensor dimensions by the product of mesh axes that shard them.
	for tensorAxis, axisLen := range logicalShape.Dimensions {
		if tensorAxis >= spec.Rank() {
			break
		}
		axisSpec := spec.Axes[tensorAxis]
		if len(axisSpec) == 0 {
			continue
		}
		meshSize := 1
		for _, meshAxisName := range axisSpec {
			meshAxisSize, err := mesh.AxisSize(meshAxisName)
			if err != nil {
				return nil, errors.WithMessagef(
					err,
					"inconsistency in distributed.ShardTensor, sharding spec references mesh tensorAxis %q not in mesh %s",
					meshAxisName,
					mesh,
				)
			}
			meshSize *= meshAxisSize
		}
		if axisLen%meshSize != 0 {
			return nil, errors.Errorf(
				"tensor shape %s is not divisible at axis %d by mesh axes %q (total size %d) for sharding",
				logicalShape, tensorAxis, axisSpec, meshSize)
		}
		shardShape.Dimensions[tensorAxis] /= meshSize
	}

	numDevices := mesh.NumDevices()
	shards := make([]*tensors.Tensor, numDevices)
	for i := range shards {
		shards[i] = tensors.FromShape(shardShape)
	}

	elementSize := logicalShape.DType.Size()
	if elementSize == 0 {
		return nil, errors.Errorf("sharding of tensors with sub-byte sizes not implemented (for DType %s)",
			logicalShape.DType)
	}

	// Pre-calculate source strides and mesh info for efficiency
	srcStrides := logicalShape.Strides()
	rank := logicalShape.Rank()
	meshRank := mesh.Rank()
	meshSizes := mesh.AxesSizes()
	meshAxesNames := mesh.AxesNames()

	// Build a mapping from mesh axis name to its index for efficient lookup
	meshNameToIdx := make(map[string]int, meshRank)
	for i, name := range meshAxesNames {
		meshNameToIdx[name] = i
	}

	var innerErr error
	err := t.ConstBytes(func(tBytes []byte) {
		// Iterate over all devices, including replicated ones.
		for deviceIdx := range numDevices {
			// Calculate mesh coordinates for this device.
			// Device indices are laid out in row-major order across mesh dimensions.
			meshCoords := make([]int, meshRank)
			remaining := deviceIdx
			for i := meshRank - 1; i >= 0; i-- {
				meshCoords[i] = remaining % meshSizes[i]
				remaining /= meshSizes[i]
			}

			// Calculate the shard position for each tensor axis based on which mesh axes shard it.
			shardPos := make([]int, rank)
			for tensorAxis := 0; tensorAxis < rank; tensorAxis++ {
				if tensorAxis >= spec.Rank() || len(spec.Axes[tensorAxis]) == 0 {
					// Replicated tensor axis: shard position is 0.
					shardPos[tensorAxis] = 0
				} else {
					// Sharded tensor axis: combine mesh coordinates for all mesh axes that shard this axis.
					axisSpec := spec.Axes[tensorAxis]
					pos := 0
					multiplier := 1
					for i := len(axisSpec) - 1; i >= 0; i-- {
						meshAxisName := axisSpec[i]
						meshAxisIdx := meshNameToIdx[meshAxisName]
						pos += meshCoords[meshAxisIdx] * multiplier
						multiplier *= meshSizes[meshAxisIdx]
					}
					shardPos[tensorAxis] = pos
				}
			}

			shard := shards[deviceIdx]
			innerErr = shard.MutableBytes(func(shardBytes []byte) {
				// 1. Calculate where this shard begins in the logical tensor (Source Base Offset)
				srcBaseOffset := 0
				for axis := range rank {
					start := shardPos[axis] * shardShape.Dimensions[axis]
					srcBaseOffset += start * srcStrides[axis]
				}
				srcBaseOffset *= elementSize

				// 2. Determine Contiguous Block
				// We scan from the innermost dimension outwards. If the shard dimension matches the logical
				// dimension, that data is contiguous in memory.
				contiguousRank := rank
				blockSize := elementSize
				for i := rank - 1; i >= 0; i-- {
					if shardShape.Dimensions[i] != logicalShape.Dimensions[i] {
						contiguousRank = i + 1
						break
					}
					contiguousRank = i
					blockSize *= shardShape.Dimensions[i]
				}

				// 3. Recursive Copy
				// We fill the shard linearly (dstOffset increments by blockSize),
				// but we jump around the source tensor (tBytes) based on strides.
				dstOffset := 0

				var copier func(axis int, srcOffset int)
				copier = func(axis int, srcOffset int) {
					// Base Case: Contiguous block found.
					if axis == contiguousRank {
						copy(shardBytes[dstOffset:dstOffset+blockSize], tBytes[srcOffset:srcOffset+blockSize])
						dstOffset += blockSize
						return
					}

					// Recursive Step: Iterate over the split dimension.
					dimSize := shardShape.Dimensions[axis]
					step := srcStrides[axis] * elementSize

					for i := range dimSize {
						copier(axis+1, srcOffset+i*step) //nolint:goimports
					}
				}

				copier(0, srcBaseOffset)
			})
			if innerErr != nil {
				return
			}
		}
	})
	if err != nil {
		return nil, err
	}
	if innerErr != nil {
		return nil, innerErr
	}
	return NewTensor(spec, shards)
}
