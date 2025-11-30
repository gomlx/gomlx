package xla

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/pjrt"
	"github.com/gomlx/gopjrt/xlabuilder"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// Executable implements backends.Executable for XLA/PJRT github.com/gomlx/gopjrt
type Executable struct {
	backend         *Backend
	exec            *pjrt.LoadedExecutable
	name            string
	parameterNames  []string
	parameterShapes []shapes.Shape
	outputShapes    []shapes.Shape
}

func (b *Builder) Compile(outputs []backends.Op, shardings []*backends.ShardingSpec) (backends.Executable, error) {
	if err := b.CheckValid(); err != nil {
		return nil, err
	}
	if len(outputs) == 0 {
		return nil, errors.Errorf(
			"backend %q, computation %q: you must have at least one output to a computation",
			BackendName,
			b.name,
		)
	}
	if len(shardings) != 0 {
		return nil, errors.Errorf("sharding or distributed execution are not supported by SimpleGo backend")
	}
	xOutputs := make([]*xlabuilder.Op, len(outputs))
	outputShapes := make([]shapes.Shape, len(outputs))
	for ii, output := range outputs {
		xOutputs[ii] = castToXlaOp(output)
		outputShapes[ii] = xshapeToShape(xOutputs[ii].Shape)
	}

	// If there are more than 1 outputs, use a tuple output -- PJRT un-tuples them during execution..
	tupleOutput := xOutputs[0]
	if len(xOutputs) > 1 {
		var err error
		tupleOutput, err = xlabuilder.Tuple(xOutputs...)
		if err != nil {
			return nil, errors.WithMessagef(
				err,
				"backend %q: failed to tuple the outputs to compile computation %q",
				BackendName,
				b.name,
			)
		}
	}
	comp, err := b.builder.Build(tupleOutput)
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q: failed to build HLO from computation %q", BackendName, b.name)
	}
	if klog.V(2).Enabled() {
		if comp.HasStableHLO() {
			stableHLO, err := comp.TextStableHLO()
			if err != nil {
				return nil, errors.WithMessagef(
					err,
					"backend %q: failed to print out StableHLO from computation %q",
					BackendName,
					b.name,
				)
			}
			klog.Infof("StableHLO program:\n%s\n", stableHLO)
		} else {
			klog.Infof("HLO program:\n%s\n", comp.TextHLO())
		}
	}
	exec, err := b.backend.client.Compile().WithComputation(comp).Done()
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q: failed to compile computation %q", BackendName, b.name)
	}
	return &Executable{
		backend:         b.backend,
		exec:            exec,
		name:            b.name,
		parameterNames:  b.parameterNames,
		parameterShapes: b.parameterShapes,
		outputShapes:    outputShapes,
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

// Finalize immediately frees resources associated to the executable.
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

// Outputs returns the computation's output shapes, in the order given to the Builder.Compile call.
func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	return e.outputShapes
}

// Execute the executable on the default device (0). The number and shapes of the inputs must match those returned by Inputs.
func (e *Executable) Execute(inputs []backends.Buffer, donate []bool, _ backends.DeviceNum) ([]backends.Buffer, error) {
	if err := e.CheckValid(); err != nil {
		return nil, err
	}
	if len(inputs) != len(e.parameterShapes) {
		return nil, errors.Errorf(
			"backend %q: wrong number of parameters to Execute %q: %d given, %d expected",
			BackendName,
			e.name,
			len(inputs),
			len(e.parameterShapes),
		)
	}
	if len(donate) > 0 && len(donate) != len(e.parameterShapes) {
		return nil, errors.Errorf(
			"backend %q: wrong number of donate values to Execute %q: %d given, nil or %d expected",
			BackendName,
			e.name,
			len(donate),
			len(e.parameterShapes),
		)
	}
	pInputs := xslices.Map(inputs, castToPJRT)
	var pOutputs []*pjrt.Buffer
	var err error
	if len(donate) == 0 {
		pOutputs, err = e.exec.Execute(pInputs...).DonateNone().Done()
	} else {
		pOutputs, err = e.exec.Execute(pInputs...).SetDonate(donate).Done()
	}
	if err != nil {
		return nil, errors.WithMessagef(err, "backend %q: failed to execute computation %q", BackendName, e.name)
	}
	return xslices.Map(pOutputs, func(e *pjrt.Buffer) backends.Buffer { return e }), nil
}
