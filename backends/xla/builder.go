package xla

import (
	"slices"

	"github.com/gomlx/go-xla/pkg/stablehlo"
	xla_dtypes "github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/go-xla/pkg/types/shardy"
	xla_shapes "github.com/gomlx/go-xla/pkg/types/shapes"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Builder keeps track of the computation graph being defined.
type Builder struct {
	name     string
	backend  *Backend
	compiled bool

	builder *stablehlo.Builder
	fn      *stablehlo.Function

	numDevices       int // numDevices used by the builder <= backend.numDevices.
	deviceAssignment []int
	distStrategy     distributed.Strategy
	meshes           []*shardy.DeviceMesh

	parameterNames  []string
	parameterShapes []shapes.Shape
	parameterSpecs  []*backends.ShardingSpec

	// Various caches.
	cacheReductions map[reductionKey]*stablehlo.Function
	cacheArgMinMax  map[argMinMaxKey]*stablehlo.Function
	cacheSelections map[reductionKey]*stablehlo.Function
}

// reductionKey for the cache of inlined functions for reductions.
type reductionKey struct {
	dtype  dtypes.DType
	opType backends.OpType
}

// argMinMaxKey for the cache of inlined functions for ArgMinMax.
type argMinMaxKey struct {
	valuesDType, outputDType dtypes.DType
	isMin                    bool
}

var _ backends.Builder = (*Builder)(nil)

// Builder creates a new builder used to define a new computation.
func (backend *Backend) Builder(name string) backends.Builder {
	if err := backend.CheckValid(); err != nil {
		klog.Error(err)
		return nil
	}
	b := &Builder{
		backend:         backend,
		builder:         stablehlo.New(name),
		name:            name,
		cacheReductions: make(map[reductionKey]*stablehlo.Function),
		cacheArgMinMax:  make(map[argMinMaxKey]*stablehlo.Function),
		cacheSelections: make(map[reductionKey]*stablehlo.Function),
	}
	b.fn = b.builder.Main()
	return b
}

// Name returns the name of the builder.
func (b *Builder) Name() string {
	return b.name
}

// Node represents the output of an operation and implements a "backends.Op" interface.
type Node struct {
	value   *stablehlo.Value
	shape   shapes.Shape
	builder *Builder
}

// CheckValid returns an error if the backend or the builder are not ok.
//
// E.g.: they have been finalized or the builder has already been compiled.
func (b *Builder) CheckValid() error {
	if b == nil || b.builder == nil {
		return errors.Errorf("builder is nil or undefined for %q", BackendName)
	}
	return b.backend.CheckValid()
}

// verifyAndCastValues sanity checks that the values (backends.Op) are valid and created with this builder.
// It returns the underlying *Node of the values.
func (b *Builder) verifyAndCastValues(name string, values ...backends.Op) ([]*Node, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	nodes := make([]*Node, len(values))
	for i, input := range values {
		if input == nil {
			return nil, errors.Errorf("nil Op given as an input to %q", name)
		}
		node, ok := input.(*Node)
		if !ok {
			return nil, errors.Errorf(
				"nil or invalid Op (%T: %v) given as an input to %q, it must be an input created by the same "+
					"backend builder (%s:%s)", input, input, name, b.backend.Name(), b.name)
		}
		if node.builder != b {
			return nil, errors.Errorf(
				"input given to parameter #%d (%q) was created with a different builder (%s) than the builder"+
					" (%s) it is being used in -- Ops cannot cross to different builders",
				i, name, node.builder.Name(), b.Name())
		}
		nodes[i] = node
	}
	return nodes, nil
}

// OpShape returns the shape of a computation Op.
func (b *Builder) OpShape(op backends.Op) (shapes.Shape, error) {
	if err := b.CheckValid(); err != nil {
		return shapes.Invalid(), err
	}
	nodes, err := b.verifyAndCastValues("OpShape", op)
	if err != nil {
		return shapes.Invalid(), err
	}
	return nodes[0].shape, nil
}

func (b *Builder) newNode(value *stablehlo.Value) *Node {
	return &Node{
		value:   value,
		shape:   ShapeFromXLA(value.Shape()),
		builder: b,
	}
}

func (b *Builder) Parameter(name string, shape shapes.Shape, sharding *backends.ShardingSpec) (
	backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	normalizedName := stablehlo.NormalizeIdentifier(name)
	if slices.Index(b.parameterNames, normalizedName) != -1 {
		if name == normalizedName {
			return nil, errors.Errorf("parameter named %q already exists", name)
		}
		return nil, errors.Errorf("parameter named %q (normalized to %q) already exists",
			name, normalizedName)
	}
	b.parameterNames = append(b.parameterNames, normalizedName)
	b.parameterShapes = append(b.parameterShapes, shape)
	b.parameterSpecs = append(b.parameterSpecs, sharding)
	var shardySpec *shardy.ShardingSpec
	if sharding != nil {
		var err error
		shardySpec, err = b.shardingSpecToShardy(sharding)
		if err != nil {
			return nil, errors.WithMessagef(err, "while creating sharding spec for parameter %q", name)
		}
	}
	value, err := b.fn.NamedInputWithSharding(name, ShapeToXLA(shape), shardySpec)
	if err != nil {
		return nil, errors.WithMessagef(err, "while building parameter %q", name)
	}
	return b.newNode(value), nil
}

// Constant creates a constant in the graph with the given flat values and the shape defined by the dimensions.
//
// The flat value must be a slice of a basic type supported -- that can be converted to a DType.
//
// The value is copied into the graph. It's recommended that for very large tensors,
// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
func (b *Builder) Constant(flat any, dimensions ...int) (backends.Op, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	if flat == nil {
		return nil, errors.Errorf("nil value given to Constant")
	}
	if bf16Slice, ok := flat.([]bfloat16.BFloat16); ok {
		flat = any(BFloat16SliceToXLA(bf16Slice))
	}
	value, err := b.fn.ConstantFromFlatAndDimensions(flat, dimensions...)
	if err != nil {
		return nil, errors.WithMessagef(err, "while building op Constant()")
	}
	return b.newNode(value), nil
}

// DistributedSPMD creates a computation that will be executed on multiple devices in SPMD fashion
// (SPMD = single program, multiple data).
func (b *Builder) DistributedSPMD(numDevices int) error {
	if err := b.CheckValid(); err != nil {
		return err
	}
	if b.compiled {
		return errors.Errorf("DistributedSPMD cannot be called after the computation has been compiled")
	}
	b.distStrategy = distributed.SPMD
	b.builder.WithNumReplicas(numDevices)
	b.numDevices = numDevices
	return nil
}

// DistributedAutoSharding creates a computation that will be executed on multiple devices with auto-sharding.
// This currently aims at XLA Shardy [1] framework. But other backends can implement it with the same semantics,
// if appropriate.
//
// [1] https://github.com/openxla/shardy
func (b *Builder) DistributedAutoSharding(meshes ...backends.Mesh) error {
	if err := b.CheckValid(); err != nil {
		return err
	}
	if b.compiled {
		return errors.Errorf("DistributedAutoSharding cannot be called after the computation has been compiled")
	}
	b.distStrategy = distributed.AutoSharding
	b.meshes = make([]*shardy.DeviceMesh, len(meshes))
	var err error
	b.numDevices = 0
	for i, mesh := range meshes {
		b.meshes[i], err = shardy.NewDeviceMesh(mesh.Name, mesh.AxesSizes, mesh.AxesNames)
		if err != nil {
			return errors.WithMessagef(err, "while creating mesh %q", mesh.Name)
		}
		meshNumDevices := b.meshes[i].NumDevices()
		if meshNumDevices > b.backend.NumDevices() {
			return errors.Errorf("mesh %q has %d devices, but the backend only has %d devices",
				mesh.Name, meshNumDevices, b.backend.NumDevices())
		}
		b.numDevices = max(b.numDevices, meshNumDevices)
	}
	b.builder.WithShardy(b.meshes...)
	b.builder.WithNumReplicas(1)
	b.builder.WithNumPartitions(b.numDevices)
	return nil
}

func (b *Builder) meshByName(meshName string) (*shardy.DeviceMesh, error) {
	for _, mesh := range b.meshes {
		if mesh.Name() == meshName {
			return mesh, nil
		}
	}
	return nil, errors.Errorf("mesh %q not found", meshName)
}

// StableHLOFunction returns the underlying stablehlo.Function for advanced use cases.
// This allows direct access to StableHLO features like While loops.
func (b *Builder) StableHLOFunction() *stablehlo.Function {
	return b.fn
}

func (b *Builder) shardingSpecToShardy(sharding *backends.ShardingSpec) (*shardy.ShardingSpec, error) {
	if sharding == nil {
		return nil, nil
	}
	name := sharding.Mesh
	mesh, err := b.meshByName(sharding.Mesh)
	if err != nil {
		return nil, errors.WithMessagef(err, "in sharding spec with mesh %q", name)
	}
	shardySpec := shardy.NewShardingSpec(mesh)
	for _, meshAxes := range sharding.Axes {
		if len(meshAxes) == 0 {
			shardySpec.AddReplicated()
		} else {
			shardySpec.AddShardedAxis(meshAxes...)
		}
	}
	return shardySpec, nil
}

// DeviceAssignment assigns the devices to the computation.
//
// The number of devices must match the number of devices in the computation.
// Usually, that is 1. But if DistributedSPMD was used, it can be more.
func (b *Builder) DeviceAssignment(devices ...backends.DeviceNum) error {
	numReplicas := max(1, b.numDevices)
	if len(devices) != numReplicas {
		return errors.Errorf("DeviceAssignment expects %d devices, got %d", numReplicas, len(devices))
	}
	b.deviceAssignment = make([]int, 0, numReplicas)
	for _, device := range devices {
		deviceInt := int(device)
		b.deviceAssignment = append(b.deviceAssignment, deviceInt)
		if deviceInt < 0 || deviceInt >= b.backend.NumDevices() {
			return errors.Errorf("device %d is out of range for the number of devices %d in the backend %q",
				deviceInt, b.backend.NumDevices(), b.backend.Name())
		}
	}
	return nil
}

func broadcastShapeForBinaryOps(
	opType backends.OpType,
	lhsShape, rhsShape shapes.Shape,
) (output shapes.Shape, err error) {
	if lhsShape.IsScalar() {
		return rhsShape, nil
	}
	if rhsShape.IsScalar() {
		return lhsShape, nil
	}

	// Other cases, either the dimensions match or one of them is 1.
	if lhsShape.Rank() != rhsShape.Rank() {
		err = errors.Errorf(
			"if operands are not scalars, their rank must match for BinaryOp (%s), got shapes %s and %s",
			opType,
			lhsShape,
			rhsShape,
		)
		return
	}
	output = lhsShape.Clone()
	for axis := range output.Rank() {
		lhsDim := lhsShape.Dimensions[axis]
		rhsDim := rhsShape.Dimensions[axis]

		// Handle dynamic dimensions
		switch {
		case lhsDim < 0 && rhsDim < 0:
			// Both dynamic
			if lhsDim == rhsDim {
				// Same dynamic marker - preserve it
				output.Dimensions[axis] = lhsDim
			} else {
				// Different dynamic markers - use generic dynamic
				output.Dimensions[axis] = shapes.DynamicDim
			}
		case lhsDim < 0 && rhsDim == 1:
			// Left is dynamic, right broadcasts
			output.Dimensions[axis] = lhsDim
		case rhsDim < 0 && lhsDim == 1:
			// Right is dynamic, left broadcasts
			output.Dimensions[axis] = rhsDim
		case lhsDim > 0 && rhsDim > 0:
			// Both static - use existing max logic for broadcasting
			if lhsDim != 1 && rhsDim != 1 && lhsDim != rhsDim {
				err = errors.Errorf(
					"dimension of axis #%d doesn't match and cannot be broadcast for BinaryOp (%s), got shapes %s and %s",
					axis,
					opType,
					lhsShape,
					rhsShape,
				)
				return
			}
			output.Dimensions[axis] = max(lhsDim, rhsDim)
		case lhsDim < 0 && rhsDim > 1:
			// Dynamic vs concrete > 1: use concrete (dynamic must be compatible)
			output.Dimensions[axis] = rhsDim
		case rhsDim < 0 && lhsDim > 1:
			// Concrete > 1 vs dynamic: use concrete
			output.Dimensions[axis] = lhsDim
		default:
			err = errors.Errorf("incompatible dynamic dimensions at axis %d: %d vs %d for BinaryOp (%s)",
				axis, lhsDim, rhsDim, opType)
			return
		}
	}
	return

}

// broadcastForBinaryOps returns the broadcasted versions of the two ops,
// converting them to Nodes in the process.
func (b *Builder) broadcastForBinaryOps(
	opType backends.OpType,
	lhs, rhs backends.Op,
) (lhsNode, rhsNode *Node, err error) {
	opName := opType.String()
	nodes, err := b.verifyAndCastValues(opName, lhs, rhs)
	if err != nil {
		return
	}
	lhsNode, rhsNode = nodes[0], nodes[1]
	if lhsNode.shape.DType != rhsNode.shape.DType {
		return nil, nil, errors.Errorf("cannot broadcast %s and %s for %q: they have different dtypes",
			lhsNode.shape.DType, rhsNode.shape.DType, opType)
	}
	if rhsNode.shape.Equal(lhsNode.shape) {
		// No casting needed.
		return
	}

	// If any is a scalar, just broadcast it to the other one.
	if lhsNode.shape.IsScalar() {
		var value *stablehlo.Value
		// Check if the GoMLX shape has dynamic dimensions
		// (not the StableHLO shape, which may have dynamic dims even when GoMLX shape is concrete)
		if rhsNode.shape.IsDynamic() {
			// Use DynamicBroadcastInDim for shapes with dynamic dimensions
			shapeTensor, shapeErr := b.shapeToTensor(rhsNode.value)
			if shapeErr != nil {
				return nil, nil, errors.WithMessagef(shapeErr, "while creating shape tensor for broadcasting in op %q", opType)
			}
			value, err = stablehlo.DynamicBroadcastInDim(lhsNode.value, shapeTensor, nil)
		} else {
			// Use static BroadcastInDim with the concrete GoMLX shape
			targetShape := ShapeToXLA(rhsNode.shape)
			value, err = stablehlo.BroadcastInDim(lhsNode.value, targetShape, nil)
		}
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while building op %q", opType)
		}
		lhsNode = b.newNode(value)
		return
	} else if rhsNode.shape.IsScalar() {
		var value *stablehlo.Value
		// Check if the GoMLX shape has dynamic dimensions
		// (not the StableHLO shape, which may have dynamic dims even when GoMLX shape is concrete)
		if lhsNode.shape.IsDynamic() {
			// Use DynamicBroadcastInDim for shapes with dynamic dimensions
			shapeTensor, shapeErr := b.shapeToTensor(lhsNode.value)
			if shapeErr != nil {
				return nil, nil, errors.WithMessagef(shapeErr, "while creating shape tensor for broadcasting in op %s", opName)
			}
			value, err = stablehlo.DynamicBroadcastInDim(rhsNode.value, shapeTensor, nil)
		} else {
			// Use static BroadcastInDim with the concrete GoMLX shape
			targetShape := ShapeToXLA(lhsNode.shape)
			value, err = stablehlo.BroadcastInDim(rhsNode.value, targetShape, nil)
		}
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while building op %s", opName)
		}
		rhsNode = b.newNode(value)
		return
	}

	// Find the larger shape that fits both operands.
	broadcastShape, err := broadcastShapeForBinaryOps(opType, lhsNode.shape, rhsNode.shape)
	if err != nil {
		return nil, nil, err
	}
	newShapeStableHLO := ShapeToXLA(broadcastShape)
	broadcastAxes := xslices.Iota(0, broadcastShape.Rank())

	// Check if the broadcast shape has dynamic dimensions
	useDynamicBroadcast := broadcastShape.IsDynamic()

	if !broadcastShape.Equal(lhsNode.shape) {
		var value *stablehlo.Value
		if useDynamicBroadcast {
			// Use DynamicBroadcastInDim for shapes with dynamic dimensions
			// Try to get shape from the concrete operand if available
			var shapeTensor *stablehlo.Value
			if !rhsNode.shape.IsDynamic() {
				shapeTensor, err = b.shapeToTensor(rhsNode.value)
			} else if !lhsNode.shape.IsDynamic() {
				shapeTensor, err = b.shapeToTensor(lhsNode.value)
			} else {
				// Both have dynamic dims, create shape tensor from broadcast shape
				shapeTensor, err = b.createShapeTensorForBroadcast(broadcastShape, lhsNode.value, rhsNode.value)
			}
			if err != nil {
				return nil, nil, errors.WithMessagef(err, "while creating shape tensor for broadcasting lhs in op %q", opType)
			}
			value, err = stablehlo.DynamicBroadcastInDim(lhsNode.value, shapeTensor, broadcastAxes)
		} else {
			value, err = stablehlo.BroadcastInDim(lhsNode.value, newShapeStableHLO, broadcastAxes)
		}
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while broadcasting lhs for op %q", opType)
		}
		lhsNode = b.newNode(value)
	}
	if !broadcastShape.Equal(rhsNode.shape) {
		var value *stablehlo.Value
		if useDynamicBroadcast {
			// Use DynamicBroadcastInDim for shapes with dynamic dimensions
			// Try to get shape from the concrete operand if available
			var shapeTensor *stablehlo.Value
			if !lhsNode.shape.IsDynamic() {
				shapeTensor, err = b.shapeToTensor(lhsNode.value)
			} else if !rhsNode.shape.IsDynamic() {
				shapeTensor, err = b.shapeToTensor(rhsNode.value)
			} else {
				// Both have dynamic dims, create shape tensor from broadcast shape
				shapeTensor, err = b.createShapeTensorForBroadcast(broadcastShape, lhsNode.value, rhsNode.value)
			}
			if err != nil {
				return nil, nil, errors.WithMessagef(err, "while creating shape tensor for broadcasting rhs in op %q", opType)
			}
			value, err = stablehlo.DynamicBroadcastInDim(rhsNode.value, shapeTensor, broadcastAxes)
		} else {
			value, err = stablehlo.BroadcastInDim(rhsNode.value, newShapeStableHLO, broadcastAxes)
		}
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while broadcasting rhs for op %q", opType)
		}
		rhsNode = b.newNode(value)
	}
	return
}

// shapeToTensor creates a 1D tensor containing the runtime dimensions of the given value.
// This is used for dynamic broadcast operations when the target shape has dynamic dimensions.
func (b *Builder) shapeToTensor(value *stablehlo.Value) (*stablehlo.Value, error) {
	shape := value.Shape()
	rank := shape.Rank()
	fn := b.fn

	// Create a slice to hold individual dimension sizes
	dimValues := make([]*stablehlo.Value, rank)

	for i := 0; i < rank; i++ {
		var dimValue *stablehlo.Value
		var err error

		if shape.Dimensions[i] >= 0 {
			// Concrete dimension - use constant (as 1D tensor with single element)
			dimValue, err = fn.ConstantFromFlatAndDimensions([]int64{int64(shape.Dimensions[i])}, 1)
			if err != nil {
				return nil, errors.WithMessagef(err, "creating constant for dimension %d", i)
			}
		} else {
			// Dynamic dimension - use get_dimension_size
			dimSize, err := stablehlo.GetDimensionSize(value, i)
			if err != nil {
				return nil, errors.WithMessagef(err, "getting dimension size for axis %d", i)
			}
			// Convert to int64 as required by DynamicBroadcastInDim
			// Note: GetDimensionSize returns i32, we need i64
			if dimSize.Shape().DType != xla_dtypes.Int64 {
				dimSize, err = stablehlo.Convert(dimSize, xla_dtypes.Int64)
				if err != nil {
					return nil, errors.WithMessagef(err, "converting dimension size to int64")
				}
			}
			// Reshape scalar to 1D tensor
			targetShape := xla_shapes.Make(dimSize.Shape().DType, 1)
			dimValue, err = stablehlo.Reshape(dimSize, targetShape)
			if err != nil {
				return nil, errors.WithMessagef(err, "reshaping dimension size to 1D")
			}
		}

		dimValues[i] = dimValue
	}

	// Concatenate all dimensions into a single 1D tensor
	if len(dimValues) == 0 {
		// Scalar case - return empty shape
		return fn.ConstantFromFlatAndDimensions([]int64{}, 0)
	} else if len(dimValues) == 1 {
		// Single dimension - already 1D
		return dimValues[0], nil
	} else {
		// Multiple dimensions - concatenate along axis 0
		return stablehlo.Concatenate(0, dimValues...)
	}
}

// createShapeTensorForBroadcast creates a shape tensor for broadcast operations when
// the broadcast shape has dynamic dimensions. It tries to resolve dimensions from
// the concrete operands when possible.
func (b *Builder) createShapeTensorForBroadcast(broadcastShape shapes.Shape, lhsValue, rhsValue *stablehlo.Value) (*stablehlo.Value, error) {
	fn := b.fn
	rank := broadcastShape.Rank()

	// Create a slice to hold individual dimension sizes
	dimValues := make([]*stablehlo.Value, rank)

	for i := 0; i < rank; i++ {
		var dimValue *stablehlo.Value
		var err error

		if broadcastShape.Dimensions[i] >= 0 {
			// Concrete dimension in broadcast shape - use constant
			dimValue, err = fn.ConstantFromFlatAndDimensions([]int64{int64(broadcastShape.Dimensions[i])}, 1)
			if err != nil {
				return nil, errors.WithMessagef(err, "creating constant for dimension %d", i)
			}
		} else {
			// Dynamic dimension - try to get from operands
			// First try lhs
			if i < lhsValue.Shape().Rank() {
				lhsDim := lhsValue.Shape().Dimensions[i]
				if lhsDim >= 0 {
					dimValue, err = fn.ConstantFromFlatAndDimensions([]int64{int64(lhsDim)}, 1)
					if err != nil {
						return nil, errors.WithMessagef(err, "creating constant from lhs dimension %d", i)
					}
				} else {
					// Get runtime dimension from lhs
					dimSize, err := stablehlo.GetDimensionSize(lhsValue, i)
					if err != nil {
						return nil, errors.WithMessagef(err, "getting lhs dimension size for axis %d", i)
					}
					if dimSize.Shape().DType != xla_dtypes.Int64 {
						dimSize, err = stablehlo.Convert(dimSize, xla_dtypes.Int64)
						if err != nil {
							return nil, errors.WithMessagef(err, "converting dimension size to int64")
						}
					}
					targetShape := xla_shapes.Make(dimSize.Shape().DType, 1)
					dimValue, err = stablehlo.Reshape(dimSize, targetShape)
					if err != nil {
						return nil, errors.WithMessagef(err, "reshaping dimension size to 1D")
					}
				}
			} else if i < rhsValue.Shape().Rank() {
				// Try rhs if lhs doesn't have this dimension
				rhsDim := rhsValue.Shape().Dimensions[i]
				if rhsDim >= 0 {
					dimValue, err = fn.ConstantFromFlatAndDimensions([]int64{int64(rhsDim)}, 1)
					if err != nil {
						return nil, errors.WithMessagef(err, "creating constant from rhs dimension %d", i)
					}
				} else {
					// Get runtime dimension from rhs
					dimSize, err := stablehlo.GetDimensionSize(rhsValue, i)
					if err != nil {
						return nil, errors.WithMessagef(err, "getting rhs dimension size for axis %d", i)
					}
					if dimSize.Shape().DType != xla_dtypes.Int64 {
						dimSize, err = stablehlo.Convert(dimSize, xla_dtypes.Int64)
						if err != nil {
							return nil, errors.WithMessagef(err, "converting dimension size to int64")
						}
					}
					targetShape := xla_shapes.Make(dimSize.Shape().DType, 1)
					dimValue, err = stablehlo.Reshape(dimSize, targetShape)
					if err != nil {
						return nil, errors.WithMessagef(err, "reshaping dimension size to 1D")
					}
				}
			} else {
				return nil, errors.Errorf("cannot resolve dynamic dimension %d in broadcast shape", i)
			}
		}

		dimValues[i] = dimValue
	}

	// Concatenate all dimensions into a single 1D tensor
	if len(dimValues) == 0 {
		return fn.ConstantFromFlatAndDimensions([]int64{}, 0)
	} else if len(dimValues) == 1 {
		return dimValues[0], nil
	} else {
		return stablehlo.Concatenate(0, dimValues...)
	}
}
