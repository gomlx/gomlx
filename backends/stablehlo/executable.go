package stablehlo

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/stablehlo"
	"github.com/gomlx/stablehlo/types/shardy"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Executable implements the backends.Executable for XLA/PJRT github.com/gomlx/gopjrt.
type Executable struct {
	backend         *Backend
	exec            *pjrt.LoadedExecutable
	name            string
	parameterNames  []string
	parameterShapes []shapes.Shape
	parameterSpecs  []*backends.ShardingSpec
	outputShapes    []shapes.Shape
	outputSpecs     []*backends.ShardingSpec

	distStrategy     distributed.Strategy
	numDevices       int
	deviceAssignment []int
	portable         bool
}

func (b *Builder) Compile(outputs []backends.Op, shardings []*backends.ShardingSpec) (backends.Executable, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	if len(outputs) == 0 {
		return nil, errors.Errorf(
			"backend %q, computation %q: you must have at least one output to a computation",
			BackendName, b.name)
	}

	outputNodes, err := b.verifyAndCastValues("Compile", outputs...)
	if err != nil {
		return nil, err
	}
	outputValues := make([]*stablehlo.Value, len(outputs))
	outputShapes := make([]shapes.Shape, len(outputs))

	for ii, outputNode := range outputNodes {
		outputValues[ii] = outputNode.value
		outputShapes[ii] = outputNode.shape
	}

	// Verify shardings:
	var shardySpecs []*shardy.ShardingSpec
	if len(shardings) > 0 {
		if b.distStrategy != distributed.AutoSharding {
			return nil, errors.Errorf(
				"backend %q, computation %q: sharding of the outputs are only supported with AutoSharding strategy",
				BackendName, b.name)
		}
		shardySpecs = make([]*shardy.ShardingSpec, len(outputs))
		for i, spec := range shardings {
			shardySpecs[i], err = b.shardingSpecToShardy(spec)
			if err != nil {
				return nil, errors.WithMessagef(err,
					"backend %q, computation %q: failed to convert sharding spec for output #%d",
					BackendName, b.name, i)
			}
		}
	}

	// Finish StableHLO "main" function:
	err = b.fn.ReturnWithShardingAndAttributes(outputValues, shardySpecs, nil)
	if err != nil {
		return nil, errors.WithMessagef(err,
			"backend %q: failed to finish StableHLO program %q", BackendName, b.name)
	}
	program, err := b.builder.Build()
	if err != nil {
		return nil, errors.WithMessagef(err,
			"backend %q: failed to build StableHLO from computation %q", BackendName, b.name)
	}
	if klog.V(2).Enabled() { //nolint:mnd // Log-level numbers are ok.
		klog.Infof("StableHLO program:\n%s\n", program)
	}

	compileConfig := b.backend.client.Compile().WithStableHLO(program)
	var portable bool
	switch b.distStrategy {
	case distributed.SPMD:
		compileConfig = compileConfig.
			WithSPMD(b.numDevices).
			WithDeviceAssignment(b.deviceAssignment)
	case distributed.AutoSharding:
		compileConfig = compileConfig.
			WithShardy(b.numDevices).
			WithDeviceAssignment(b.deviceAssignment)
	case distributed.None:
		// Device Assignment is optional, only if set.
		// Otherwise, the compilation is "portable" and can be executed on any device.
		if b.deviceAssignment != nil {
			compileConfig = compileConfig.
				WithDeviceAssignment(b.deviceAssignment)
		} else {
			portable = true
		}
	}
	exec, err := compileConfig.Done()
	if err != nil {
		return nil, errors.WithMessagef(err,
			"backend %q: failed to compile computation %q", BackendName, b.name)
	}
	return &Executable{
		backend:         b.backend,
		exec:            exec,
		name:            b.name,
		parameterNames:  b.parameterNames,
		parameterShapes: b.parameterShapes,
		parameterSpecs:  b.parameterSpecs,
		outputShapes:    outputShapes,

		distStrategy:     b.distStrategy,
		numDevices:       max(1, len(b.deviceAssignment)),
		deviceAssignment: b.deviceAssignment,
		portable:         portable,
	}, nil
}

// CheckValid returns an error if the backend or the executable are not ok -- e.g.: if they have been finalized or the builder
// has already been compiled.
func (e *Executable) CheckValid() error {
	if e == nil || e.exec == nil || e.backend == nil {
		return errors.Errorf("backend %q: Executable nil or already finalized", BackendName)
	}
	return e.backend.CheckValid()
}

// Finalize immediately frees resources associated with the executable.
func (e *Executable) Finalize() {
	if e == nil || e.exec == nil || e.backend == nil {
		return
	}
	err := e.exec.Destroy()
	if err != nil {
		klog.Warningf("Error while destroying executable %q on backend %q: %+v", e.name, BackendName, err)
	}
	e.exec = nil
	e.backend = nil
	e.parameterNames = nil
	e.parameterShapes = nil
	e.outputShapes = nil
}

// Inputs returns the parameters' names and shapes, in order created by the Builder.Parameter calls.
func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	return e.parameterNames, e.parameterShapes
}

// Outputs return the computation's output shapes, in the order given to the Builder.Compile call.
func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	return e.outputShapes
}

// Execute the executable on the default device (0). The number and shapes of the inputs must match those returned by Inputs.
func (e *Executable) Execute(
	inputs []backends.Buffer,
	donate []bool,
	defaultDevice backends.DeviceNum,
) ([]backends.Buffer, error) {
	if err := e.CheckValid(); err != nil {
		return nil, err
	}
	numParams := len(e.parameterShapes)
	numDevices := e.numDevices
	if len(inputs) != numParams*numDevices {
		return nil, errors.Errorf(
			"backend %q: wrong number of parameters to Execute %q: %d given, %d * %d expected",
			BackendName, e.name, len(inputs), numParams, numDevices)
	}
	if len(donate) > 0 && len(donate) != numParams*numDevices {
		return nil, errors.Errorf(
			"backend %q: wrong number of donate values to Execute %q: %d given, nil or %d * %d expected",
			BackendName, e.name, len(donate), numParams, numDevices)
	}
	pInputs := xslices.Map(inputs, castToPJRT)
	execBuilder := e.exec.Execute(pInputs...)
	if len(donate) == 0 {
		execBuilder = execBuilder.DonateNone()
	} else {
		execBuilder = execBuilder.SetDonate(donate)
	}
	if e.portable {
		execBuilder = execBuilder.OnDeviceByNum(int(defaultDevice))
	}
	pOutputs, err := execBuilder.Done()
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q: failed to execute computation %q", BackendName, e.name)
	}
	return xslices.Map(pOutputs, func(e *pjrt.Buffer) backends.Buffer { return e }), nil
}
