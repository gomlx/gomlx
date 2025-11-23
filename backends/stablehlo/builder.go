package stablehlo

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types/shardy"
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
		shape:   ShapeFromStableHLO(value.Shape()),
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
	value, err := b.fn.NamedInputWithSharding(name, ShapeToStableHLO(shape), shardySpec)
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
		value, err = stablehlo.BroadcastInDim(lhsNode.value, rhsNode.value.Shape(), nil)
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while building op %q", opType)
		}
		lhsNode = b.newNode(value)
		return
	} else if rhsNode.shape.IsScalar() {
		var value *stablehlo.Value
		value, err = stablehlo.BroadcastInDim(rhsNode.value, lhsNode.value.Shape(), nil)
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
	newShapeStableHLO := ShapeToStableHLO(broadcastShape)
	broadcastAxes := xslices.Iota(0, broadcastShape.Rank())
	if !broadcastShape.Equal(lhsNode.shape) {
		value, err := stablehlo.BroadcastInDim(lhsNode.value, newShapeStableHLO, broadcastAxes)
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while broadcasting lhs for op %q", opType)
		}
		lhsNode = b.newNode(value)
	}
	if !broadcastShape.Equal(rhsNode.shape) {
		value, err := stablehlo.BroadcastInDim(rhsNode.value, newShapeStableHLO, broadcastAxes)
		if err != nil {
			return nil, nil, errors.WithMessagef(err, "while broadcasting rhs for op %q", opType)
		}
		rhsNode = b.newNode(value)
	}
	return
}
