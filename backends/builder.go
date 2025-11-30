package backends

import (
	"slices"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// Op represents the output of an operation, during the computation graph building time.
//
// It is opaque from the GoMLX perspective: it passes Op as input to the other methods.
type Op any

// Builder defines the set of ops to support building a computation.
// It is the sub-interface of Backend.
//
// Each Builder can also:
//  1. Not implement standard operations by returning an error -- this restricts what type of models it can support.
//     See Backend.Capabilities and package github.com/gomlx/gomlx/backends/notimplemented
//  2. Support specialized operations beyond those defined in this interface -- this requires
//     careful interface casting by the caller (in package github.com/gomlx/gomlx/pkg/core/graph) and
//     fallback to backends that don't support these specialized ops.
type Builder interface {
	// Compile the computation built. This immediately invalidates the Builder and returns an Executable that
	// can now be used to run the computation.
	//
	// It optionally also takes the corresponding output sharding specs: this is only needed for distributed
	// computations with AutoSharding strategy and can be set to nil otherwise.
	//
	// It is given the list of outputs.
	Compile(outputs []Op, shardings []*ShardingSpec) (Executable, error)

	// Name of the computation being built.
	Name() string

	// OpShape returns the shape of a computation Op.
	// Notice this is not an operation and doesn't change the graph being built.
	//
	// One can use the shape and create a constant out of it.
	OpShape(op Op) (shapes.Shape, error)

	// Parameter creates an input parameter for the computation.
	// During the execution of a compiled computation (returned by Builder.Compile), this value will need to be fed
	// in the same order it is created.
	//
	// The sharding defines how the parameter will be shared for distributed operations.
	// Set it to nil if not using distribution.
	Parameter(name string, shape shapes.Shape, sharding *ShardingSpec) (Op, error)

	// Constant creates a constant in the graph with the given flat values and the shape defined by
	// the dimensions in dim.
	//
	// The flat value must be a slice of a basic type supported -- that can be converted to a DType.
	//
	// The value is copied into the graph. It's recommended that for very large tensors,
	// even if constants, that they are passed as side inputNodes (or variables, see context package) instead.
	Constant(flat any, dims ...int) (Op, error)

	// DistributedSPMD creates a computation that will be executed on multiple devices in SPMD fashion
	// (SPMD = single program, multiple data).
	//
	// Use DeviceAssignment to assign the devices to the computation -- the default assignment is incremental
	// devices starting from 0.
	DistributedSPMD(numDevices int) error

	// DistributedAutoSharding creates a computation that will be executed on multiple devices with auto-sharding.
	// This currently aims at XLA Shardy [1] framework. But other backends can implement it with the same semantics,
	// if appropriate.
	//
	// [1] https://github.com/openxla/shardy
	DistributedAutoSharding(meshes ...Mesh) error

	// DeviceAssignment assigns the concrete devices to the computation.
	//
	// The number of devices must match the number of devices in the computation.
	// Usually, that is 1. But if DistributedSPMD was used, it can be more.
	DeviceAssignment(devices ...DeviceNum) error

	// StandardOps include all other standard math (or ML) operations.
	StandardOps

	// CollectiveOps include all collective (distributed cross-device) operations.
	CollectiveOps
}

// ConvolveAxesConfig defines the interpretation of the input/kernel/output tensor axes.
// There must be the same number of spatial dimensions (axes) for each of the 3 tensors.
// Input and output have batch and channel axes. Kernel has inputChannel and outputChannel axes.
//
// See Builder.ConvGeneral.
type ConvolveAxesConfig struct {
	InputBatch, InputChannels int
	InputSpatial              []int

	KernelInputChannels, KernelOutputChannels int
	KernelSpatial                             []int

	OutputBatch, OutputChannels int
	OutputSpatial               []int
}

// Clone returns a deep copy of the structure.
func (c ConvolveAxesConfig) Clone() ConvolveAxesConfig {
	var c2 ConvolveAxesConfig
	c2 = c
	c2.InputSpatial = slices.Clone(c.InputSpatial)
	c2.KernelSpatial = slices.Clone(c.KernelSpatial)
	c2.OutputSpatial = slices.Clone(c.OutputSpatial)
	return c2
}

// PadAxis defines the amount of padding preceding one axis (Start), at the end of axis (End)
// or in between the inputs (Interior).
// This is used as a parameter for the Pad operation.
type PadAxis struct {
	Start, End, Interior int
}

type FFTType int

const (
	// FFTForward - complex in, complex out.
	FFTForward FFTType = iota

	// FFTInverse - complex in, complex out.
	FFTInverse

	// FFTForwardReal - real in, fft_length / 2 + 1 complex out.
	FFTForwardReal

	// FFTInverseReal - fft_length / 2 + 1 complex in.
	FFTInverseReal
)

//go:generate go tool enumer -type FFTType -trimprefix=FFT -output=gen_ffttype_enumer.go builder.go

// ReduceOpType select among the basic types of reduction supported.
type ReduceOpType int

const (
	// ReduceOpUndefined is an undefined value.
	ReduceOpUndefined ReduceOpType = iota

	// ReduceOpSum reduces by summing all elements being reduced.
	ReduceOpSum

	// ReduceOpProduct reduces by multiplying all elements being reduced.
	ReduceOpProduct

	// ReduceOpMax reduces by taking the maximum value.
	ReduceOpMax

	// ReduceOpMin reduces by taking the minimum value.
	ReduceOpMin
)

// RNGStateShape is the default shape for the random number generator state.
// It dependents on the algorithm, but for now we are using Philox.
var RNGStateShape = shapes.Make(dtypes.Uint64, 3) //nolint:mnd // This is a constant.

//go:generate go tool enumer -type ReduceOpType -trimprefix=ReduceOp -output=gen_reduceoptype_enumer.go builder.go

// Mesh represents a mesh of devices, passed to the Builder.DistributedAutoSharding method.
//
// AxesSizes and AxesNames define the mesh topology.
type Mesh struct {
	Name      string
	AxesSizes []int
	AxesNames []string

	// LogicalDeviceAssignment is the logical assignment of devices to the mesh.
	// The numbers here correspond to the indices on the "hard" device assignment set with
	// Builder.DeviceAssignment() method.
	//
	// If left empty, the default assignment is incremental devices starting from 0.
	LogicalDeviceAssignment []int
}

// ShardingSpec holds a list of per tensor (or Node) axis of a list of Mesh axes names.
// Any tensor axis that doesn't have a corresponding ShardingSpec is considered replicated.
// And any tensor axis for which the list of Mesh axes is empty is also considered replicated.
//
// The ShardingSpec also holds the Mesh name over which it is defined.
type ShardingSpec struct {
	Mesh string
	Axes [][]string
}
