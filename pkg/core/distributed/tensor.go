// Package distributed defines the following objects related to cross-device execution:
//
// - DeviceMesh: expresses the topology of a set of devices, in terms of axis and their sizes.
// - ShardingSpec: defines how a logical tensor is sharded across a DeviceMesh.
// - Tensor: a logical tensor distributed across multiple devices organized as a DeviceMesh.
package distributed

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
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
	backend backends.Backend

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
func NewTensor(backend backends.Backend, spec *ShardingSpec, shards []*tensors.Tensor) (*Tensor, error) {
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
	dt := &Tensor{
		backend: backend,
		mesh:    mesh,
		spec:    spec,
		shards:  slices.Clone(shards),
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
		_, err := shard.Device()
		if err == nil {
			// Shard is on-device: we need to check that it is on the correct backend.
			// We can't yet check that it is the correct device because we don't yet have access here to
			// the assignment of devices.
			shardBackend, err := shard.Backend()
			if err != nil {
				return errors.WithMessagef(err, "shard %d is on-device, but has invalid backend", i)
			}
			if shardBackend != dt.backend {
				return errors.Errorf("shard %d is on backend %s, but distributed.Tensor configured for backend %s",
					i, shardBackend, dt.backend)
			}
		}
	}

	// Check that the shard shape is divisible by the mesh axis sizes for sharded axes.
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

//// Merge merges the tensors into one concrete logical tensor.
//// For the replicated axes, it takes the values from the first replica.
//func (dt *Tensor) Merge() (*tensors.Tensor, error) {
//	// Create a new tensor with the logical shape.
//	rank := dt.logicalShape.Rank()
//	t := tensors.FromShape(dt.logicalShape)
//	toStrides := dt.logicalShape.Strides()
//	shardStrides := dt.shardShape.Strides()
//	shapeRatio := dt.logicalShape.Clone()
//	for axis, logicalDim := range shapeRatio.Dimensions {
//		shapeRatio.Dimensions[axis] = logicalDim / dt.shardShape.Dimensions[axis]
//	}
//	if shapeRatio.Size() != len(dt.shards) {
//		return nil, errors.Errorf("number of shards (%d) does not match logical shape (%s)",
//			len(dt.shards), dt.logicalShape)
//	}
//	elementSize := dt.logicalShape.DType.Size()
//	if elementSize == 0 {
//		return nil, errors.Errorf("merge of tensors with sub-byte sizes not implemented (for DType %s)",
//			dt.logicalShape.DType)
//	}
//
//	t.MutableBytes(func(tBytes []byte) {
//		for shardIdx, shardPos := range shapeRatio.Iter() {
//			shard := dt.shards[shardIdx]
//			shard.ConstBytes(func(shardBytes []byte) {
//				// Calculate the slice of the logical tensor that corresponds to this shard.
//				sliceStarts := make([]int, rank)
//				sliceEnds := make([]int, rank)
//				for axis := range rank {
//					sliceStarts[axis] = shardPos[axis] * dt.shardShape.Dimensions[axis]
//					sliceEnds[axis] = sliceStarts[axis] + dt.shardShape.Dimensions[axis]
//				}
//
//			})
//		}
//	})
//	return t, nil
//}

// Merge merges the tensors into one concrete logical tensor.
// For the replicated axes, it takes the values from the first replica.
func (dt *Tensor) Merge() (*tensors.Tensor, error) {
	// Create a new tensor with the logical shape.
	rank := dt.logicalShape.Rank()
	t := tensors.FromShape(dt.logicalShape)
	toStrides := dt.logicalShape.Strides()

	// shapeRatio calculation (from your snippet)
	shapeRatio := dt.logicalShape.Clone()
	for axis, logicalDim := range shapeRatio.Dimensions {
		shapeRatio.Dimensions[axis] = logicalDim / dt.shardShape.Dimensions[axis]
	}
	if shapeRatio.Size() != len(dt.shards) {
		return nil, errors.Errorf("number of shards (%d) does not match logical shape (%s)",
			len(dt.shards), dt.logicalShape)
	}

	elementSize := dt.logicalShape.DType.Size()
	if elementSize == 0 {
		return nil, errors.Errorf("merge of tensors with sub-byte sizes not implemented (for DType %s)",
			dt.logicalShape.DType)
	}

	t.MutableBytes(func(tBytes []byte) {
		for shardIdx, shardPos := range shapeRatio.Iter() {
			shard := dt.shards[shardIdx]
			shard.ConstBytes(func(shardBytes []byte) {
				// Calculate the slice of the logical tensor that corresponds to this shard.
				sliceStarts := make([]int, rank)
				// We don't strictly need sliceEnds for the copy logic, but keeping for context if needed.
				sliceEnds := make([]int, rank)

				// Calculate the base offset in the destination (logical) tensor buffer.
				dstBaseOffset := 0
				for axis := range rank {
					sliceStarts[axis] = shardPos[axis] * dt.shardShape.Dimensions[axis]
					sliceEnds[axis] = sliceStarts[axis] + dt.shardShape.Dimensions[axis]

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
				//            This jumps based on strides.
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
		}
	})
	return t, nil
}

// ShardTensor splits a tensor into individual shards.
func ShardTensor(t *tensors.Tensor, mesh *DeviceMesh, spec ShardingSpec) (*Tensor, error) {
	if err := spec; err != nil {
		return nil, errors.Wrap(err, "invalid ShardingSpec")
	}
	logicalShape := t.Shape()
	shardShape := logicalShape.Clone()
	for tensorAxis, meshAxisName := range spec {
		if meshAxisName == "" {
			continue
		}
		meshAxisSize, _ := mesh.AxisSize(meshAxisName)
		if logicalShape.Dimensions[tensorAxis]%meshAxisSize != 0 {
			return nil, errors.Errorf(
				"tensor shape %s is not divisible by mesh axis %q (size %d) for sharding",
				logicalShape, meshAxisName, meshAxisSize)
		}
		shardShape.Dimensions[tensorAxis] /= meshAxisSize
	}

	shards := make([]*tensors.Tensor, mesh.NumDevices())
	for i := 0; i < mesh.NumDevices(); i++ {
		shards[i] = tensors.FromShape(shardShape)
	}

	// Create a temporary tensor with the logical shape to read from.
	tmpT := t.Clone()
	for i := 0; i < mesh.NumDevices(); i++ {
		// Calculate the slice of the logical tensor that corresponds to this shard.
		sliceStarts := make([]int, logicalShape.Rank())
		sliceEnds := make([]int, logicalShape.Rank())
		for j := 0; j < logicalShape.Rank(); j++ {
			sliceEnds[j] = shardShape.Dimensions[j]
		}
		for tensorAxis, meshAxisName := range spec {
			if meshAxisName == "" {
				continue
			}
			meshAxisSize, _ := mesh.AxisSize(meshAxisName)
			shardIndex := i % meshAxisSize
			sliceStarts[tensorAxis] = shardIndex * shardShape.Dimensions[tensorAxis]
			sliceEnds[tensorAxis] = sliceStarts[tensorAxis] + shardShape.Dimensions[tensorAxis]
		}
		// Slice the logical tensor and copy the data to the shard.
		// Slicing is not implemented in `tensors.Tensor`, so we do it manually.
		copySlice(tmpT, shards[i], sliceStarts, sliceEnds)
	}

	return NewTensor(mesh, spec, shards)
}

func copySlice(from, to *tensors.Tensor, starts, ends []int) {
	from.ConstBytes(func(fromBytes []byte) {
		to.MutableBytes(func(toBytes []byte) {
			fromShape := from.Shape()
			toShape := to.Shape()

			// Calculate element size in bytes
			elementSize := len(fromBytes) / fromShape.Size()

			// Copy the data from the source slice to the destination slice.
			fromStrides := fromShape.Strides()
			for i := 0; i < toShape.Size(); i++ {
				// Calculate destination indices
				toIndices := make([]int, toShape.Rank())
				toOffset := i
				for j := toShape.Rank() - 1; j >= 0; j-- {
					toIndices[j] = toOffset % toShape.Dimensions[j]
					toOffset /= toShape.Dimensions[j]
				}

				// Calculate source indices
				fromIndices := make([]int, fromShape.Rank())
				for j, toIndex := range toIndices {
					fromIndices[j] = toIndex + starts[j]
				}

				// Calculate source offset
				fromOffset := 0
				for j, fromIndex := range fromIndices {
					fromOffset += fromIndex * fromStrides[j]
				}

				// Copy the bytes for this element
				fromByteOffset := fromOffset * elementSize
				toByteOffset := i * elementSize
				copy(toBytes[toByteOffset:toByteOffset+elementSize],
					fromBytes[fromByteOffset:fromByteOffset+elementSize])
			}
		})
	})
}
