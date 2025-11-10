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
	if err := validateShards(mesh, spec, shards); err != nil {
		return nil, err
	}
	dt := &Tensor{
		mesh:       mesh,
		spec:       spec,
		shards:     shards,
		shardShape: shards[0].Shape(),
	}
	dt.calculateLogicalShape()
	return dt, nil
}

// calculateLogicalShape based on the shardShape and the ShardSpec.
func (dt *Tensor) calculateLogicalShape() {
	logicalShape := dt.shardShape.Clone()
	for tensorAxis, meshAxisName := range dt.spec {
		if meshAxisName == "" {
			// Replicated, dimension is the same.
			continue
		}
		// Sharded, multiply dimension by mesh axis size.
		meshAxisSize, err := dt.mesh.AxisSize(meshAxisName)
		if err != nil {
			// This should have been caught by ShardSpec.Validate, so it's a panic situation.
			panic(errors.Wrapf(err, "inconsistency in distributed.Tensor, sharding spec references mesh axis %q not in mesh %s", meshAxisName, dt.mesh))
		}
		logicalShape.Dimensions[tensorAxis] *= meshAxisSize
	}
	dt.logicalShape = logicalShape
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

// validateShards that all shards have the same shape and are consistent with the `ShardSpec`.
func validateShards(mesh *DeviceMesh, spec ShardSpec, shards []*tensors.Tensor) error {
	if len(shards) == 0 {
		return errors.New("cannot create a distributed tensor with no shards")
	}
	shardShape := shards[0].Shape()
	for i, shard := range shards {
		if !shard.Shape().Equal(shardShape) {
			return errors.Errorf("shard %d has shape %s, but shard 0 has shape %s", i, shard.Shape(), shardShape)
		}
	}
	// Check that the shard shape is divisible by the mesh axis sizes for sharded axes.
	for tensorAxis, meshAxisName := range spec {
		if meshAxisName == "" {
			continue
		}
		meshAxisSize, _ := mesh.AxisSize(meshAxisName)
		if shardShape.Dimensions[tensorAxis]*meshAxisSize%meshAxisSize != 0 {
			return errors.Errorf(
				"shard shape %s is not consistent with sharding spec %s for mesh %s: "+
					"tensor axis %d is sharded across mesh axis %q (size %d), but the shard dimension %d is not divisible by it",
				shardShape, spec, mesh, tensorAxis, meshAxisName, meshAxisSize, shardShape.Dimensions[tensorAxis])
		}
	}
	return nil
}

// ShardShape returns the shape of the individual shards.
func (dt *Tensor) ShardShape() shapes.Shape {
	return dt.shardShape
}

// Merge merges the tensors into one concrete logical tensor.
// For the replicated axes, it takes the values from the first replica.
func (dt *Tensor) Merge() *tensors.Tensor {
	// Create a new tensor with the logical shape.
	t := tensors.FromShape(dt.logicalShape)
	t.MutableBytes(func(tBytes []byte) {
		for i, shard := range dt.shards {
			shard.ConstBytes(func(shardBytes []byte) {
				// Calculate the slice of the logical tensor that corresponds to this shard.
				sliceStarts := make([]int, dt.logicalShape.Rank())
				sliceEnds := make([]int, dt.logicalShape.Rank())
				for j := 0; j < dt.logicalShape.Rank(); j++ {
					sliceEnds[j] = dt.shardShape.Dimensions[j]
				}
				for tensorAxis, meshAxisName := range dt.spec {
					if meshAxisName == "" {
						continue
					}
					meshAxisSize, _ := dt.mesh.AxisSize(meshAxisName)
					shardIndex := i % meshAxisSize
					sliceStarts[tensorAxis] = shardIndex * dt.shardShape.Dimensions[tensorAxis]
					sliceEnds[tensorAxis] = sliceStarts[tensorAxis] + dt.shardShape.Dimensions[tensorAxis]
				}

				// Copy the data from the shard to the logical tensor.
				toStrides := dt.logicalShape.Strides()
				elementSize := len(shardBytes) / shard.Shape().Size()

				for j := 0; j < shard.Shape().Size(); j++ {
					fromIndices := make([]int, dt.shardShape.Rank())
					fromOffset := j
					for k := dt.shardShape.Rank() - 1; k >= 0; k-- {
						fromIndices[k] = fromOffset % dt.shardShape.Dimensions[k]
						fromOffset /= dt.shardShape.Dimensions[k]
					}

					toIndices := make([]int, dt.logicalShape.Rank())
					for k, fromIndex := range fromIndices {
						toIndices[k] = fromIndex + sliceStarts[k]
					}

					toOffset := 0
					for k, toIndex := range toIndices {
						toOffset += toIndex * toStrides[k]
					}

					fromByteOffset := j * elementSize
					toByteOffset := toOffset * elementSize
					copy(tBytes[toByteOffset:toByteOffset+elementSize], shardBytes[fromByteOffset:fromByteOffset+elementSize])
				}
			})
		}
	})
	return t
}

// ShardTensor splits a tensor into individual shards.
func ShardTensor(t *tensors.Tensor, mesh *DeviceMesh, spec ShardSpec) (*Tensor, error) {
	if err := spec.Validate(mesh); err != nil {
		return nil, errors.Wrap(err, "invalid ShardSpec")
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

	return New(mesh, spec, shards)
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
