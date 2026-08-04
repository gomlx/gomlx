// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

//go:build onnx

package onnx

import (
	"io"
	"os"

	"github.com/gomlx/compute"
	onnxbackend "github.com/gomlx/compute-onnx"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
)

// IsONNX checks if the given backend is an ONNX backend.
func IsONNX(backend compute.Backend) bool {
	_, ok := backend.(*onnxbackend.Backend)
	return ok
}

// Save exports the computation graph associated with the given model.Exec as an ONNX model to w.
//
// It temporary sets store.WithVariableAsConst(true) and backend.SetKeepModelProto(true) while compiling
// the graph so that variable weights are embedded directly into the ONNX graph constants.
func Save(backend compute.Backend, exec *model.Exec, w io.Writer, inputShapes []shapes.Shape, inputNames, outputNames []string) error {
	onnxBackend, ok := backend.(*onnxbackend.Backend)
	if !ok {
		return errors.Errorf("backend %q (%T) is not an \"onnx\" backend (*onnxbackend.Backend) required to generate a .onnx model", backend.Name(), backend)
	}

	prevKeepModelProto := onnxBackend.KeepModelProto()
	onnxBackend.SetKeepModelProto(true)
	defer onnxBackend.SetKeepModelProto(prevKeepModelProto)

	store := exec.Store()
	if store == nil {
		return errors.Errorf("exec model has no associated Store")
	}
	prevVariablesAsConst := store.VariablesAsConst()
	store.WithVariableAsConst(true)
	defer store.WithVariableAsConst(prevVariablesAsConst)

	g, err := exec.Compile(inputShapes...)
	if err != nil {
		return errors.Wrap(err, "failed to compile computation graph for ONNX export")
	}

	onnxExecutable := g.Executable()
	if onnxExecutable == nil {
		return errors.Errorf("compiled graph has no compute.Executable")
	}
	if onExec, ok := onnxExecutable.(*onnxbackend.Executable); ok && onExec.ModelProto() == nil {
		g, err = exec.Compile(inputShapes...)
		if err != nil {
			return errors.Wrap(err, "failed to re-compile computation graph for ONNX export with retained proto")
		}
		onnxExecutable = g.Executable()
	}
	err = onnxbackend.SaveModel(backend, onnxExecutable, w, inputNames, outputNames)
	if err != nil {
		return errors.Wrap(err, "failed to save ONNX model")
	}
	return nil
}

// SaveToFile exports the computation graph associated with the given model.Exec as an ONNX model to filePath.
func SaveToFile(backend compute.Backend, exec *model.Exec, filePath string, inputShapes []shapes.Shape, inputNames, outputNames []string) error {
	if !IsONNX(backend) {
		return errors.Errorf("backend %q (%T) is not an \"onnx\" backend (*onnxbackend.Backend) required to generate a .onnx model", backend.Name(), backend)
	}

	f, err := os.Create(filePath)
	if err != nil {
		return errors.Wrapf(err, "failed to create ONNX file %q", filePath)
	}
	defer func() {
		_ = f.Close()
	}()

	if err := Save(backend, exec, f, inputShapes, inputNames, outputNames); err != nil {
		return err
	}
	if err := f.Close(); err != nil {
		return errors.Wrapf(err, "failed to close ONNX file %q", filePath)
	}
	return nil
}

// Executable represents a loaded ONNX model executable that can be called with input tensors.
type Executable struct {
	backend    compute.Backend
	executable compute.Executable
}

// Load loads an ONNX model from r and compiles it into a runnable Executable.
func Load(backend compute.Backend, r io.Reader) (*Executable, error) {
	onnxBackend, ok := backend.(*onnxbackend.Backend)
	if !ok {
		return nil, errors.Errorf("backend %T is not an *onnxbackend.Backend", backend)
	}

	exec, err := onnxbackend.LoadModel(onnxBackend, r)
	if err != nil {
		return nil, errors.Wrap(err, "failed to load ONNX model")
	}

	return &Executable{
		backend:    backend,
		executable: exec,
	}, nil
}

// LoadFromFile loads an ONNX model from filePath and compiles it into a runnable Executable.
func LoadFromFile(backend compute.Backend, filePath string) (*Executable, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open ONNX file %q", filePath)
	}
	defer func() {
		_ = f.Close()
	}()

	exec, err := Load(backend, f)
	if err != nil {
		return nil, err
	}
	return exec, nil
}

// Call converts inputs to tensors and executes the ONNX graph, returning output tensors.
func (e *Executable) Call(inputs ...any) ([]*tensors.Tensor, error) {
	inputTensors := make([]*tensors.Tensor, len(inputs))
	for i, in := range inputs {
		switch v := in.(type) {
		case *tensors.Tensor:
			inputTensors[i] = v
		default:
			t, err := tensors.FromAnyValue(v)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to convert input %d (%T) to tensor", i, in)
			}
			inputTensors[i] = t
		}
	}

	buffers := make([]compute.Buffer, len(inputTensors))
	donate := make([]bool, len(inputTensors))
	for i, t := range inputTensors {
		buf, err := t.Buffer(e.backend, 0)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to get buffer for input %d", i)
		}
		buffers[i] = buf
	}

	outBuffers, err := e.executable.Execute(buffers, donate, 0)
	if err != nil {
		return nil, errors.Wrap(err, "failed to execute ONNX model")
	}

	outTensors := make([]*tensors.Tensor, len(outBuffers))
	for i, buf := range outBuffers {
		t, err := tensors.FromBuffer(buf)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to create tensor from output buffer %d", i)
		}
		outTensors[i] = t
	}
	return outTensors, nil
}

// Finalize frees resources associated with the loaded ONNX executable.
func (e *Executable) Finalize() {
	if e.executable != nil {
		e.executable.Finalize()
	}
}
