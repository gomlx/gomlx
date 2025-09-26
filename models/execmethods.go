package models

import (
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/pkg/errors"
)

// Exec1 executes the model with the given inputs and returns the output directly (as opposed to a slice of tensors).
//
// It is a simple wrapper around Call, but it is more convenient to use.
func (e *Exec) Exec1(inputs ...any) (*tensors.Tensor, error) {
	if e.numBuilderOutputs >= 0 && e.numBuilderOutputs != 1 {
		return nil, errors.Errorf("model Build method has %d outputs, cannot use Exec1", e.numBuilderOutputs)
	}
	outputs, err := e.Exec(inputs...)
	if err != nil {
		return nil, err
	}
	if len(outputs) != 1 {
		return nil, errors.Errorf("wrong number of outputs for model for Exec1, expected 1, got %d", len(outputs))
	}
	return outputs[0], nil
}

// Exec2 executes the model with the given inputs and returns two outputs directly (as opposed to a slice of tensors).
//
// It is a simple wrapper around Call, but it is more convenient to use.
func (e *Exec) Exec2(inputs ...any) (*tensors.Tensor, *tensors.Tensor, error) {
	if e.numBuilderOutputs >= 0 && e.numBuilderOutputs != 2 {
		return nil, nil, errors.Errorf("model Build method has %d outputs, cannot use Exec2", e.numBuilderOutputs)
	}
	outputs, err := e.Exec(inputs...)
	if err != nil {
		return nil, nil, err
	}
	if len(outputs) != 2 {
		return nil, nil, errors.Errorf("wrong number of outputs for model for Exec2, expected 2, got %d", len(outputs))
	}
	return outputs[0], outputs[1], nil
}

// Exec3 executes the model with the given inputs and returns three outputs directly (as opposed to a slice of tensors).
//
// It is a simple wrapper around Call, but it is more convenient to use.
func (e *Exec) Exec3(inputs ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	if e.numBuilderOutputs >= 0 && e.numBuilderOutputs != 3 {
		return nil, nil, nil, errors.Errorf("model Build method has %d outputs, cannot use Exec3", e.numBuilderOutputs)
	}
	outputs, err := e.Exec(inputs...)
	if err != nil {
		return nil, nil, nil, err
	}
	if len(outputs) != 3 {
		return nil, nil, nil, errors.Errorf("wrong number of outputs for model for Exec3, expected 3, got %d", len(outputs))
	}
	return outputs[0], outputs[1], outputs[2], nil
}

// Exec4 executes the model with the given inputs and returns four outputs directly (as opposed to a slice of tensors).
//
// It is a simple wrapper around Call, but it is more convenient to use.
func (e *Exec) Exec4(inputs ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor, error) {
	if e.numBuilderOutputs >= 0 && e.numBuilderOutputs != 4 {
		return nil, nil, nil, nil, errors.Errorf("model Build method has %d outputs, cannot use Exec4", e.numBuilderOutputs)
	}
	outputs, err := e.Exec(inputs...)
	if err != nil {
		return nil, nil, nil, nil, err
	}
	if len(outputs) != 4 {
		return nil, nil, nil, nil, errors.Errorf("wrong number of outputs for model for Exec4, expected 4, got %d", len(outputs))
	}
	return outputs[0], outputs[1], outputs[2], outputs[3], nil
}

// Call is a variation of Exec that panics if there is an error.
func (e *Exec) Call(inputs ...any) []*tensors.Tensor {
	return must.M1(e.Exec(inputs...))
}

// Call1 is a variation of Exec that panics if there is an error and returns the one output directly (as opposed to a slice of tensors).
func (e *Exec) Call1(inputs ...any) *tensors.Tensor {
	return must.M1(e.Exec1(inputs...))
}

// Call2 is a variation of Exec that panics if there is an error and returns two outputs directly (as opposed to a slice of tensors).
func (e *Exec) Call2(inputs ...any) (*tensors.Tensor, *tensors.Tensor) {
	return must.M2(e.Exec2(inputs...))
}

// Call3 is a variation of Exec that panics if there is an error and returns three outputs directly (as opposed to a slice of tensors).
func (e *Exec) Call3(inputs ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	return must.M3(e.Exec3(inputs...))
}

// Call4 is a variation of Exec that panics if there is an error and returns four outputs directly (as opposed to a slice of tensors).
func (e *Exec) Call4(inputs ...any) (*tensors.Tensor, *tensors.Tensor, *tensors.Tensor, *tensors.Tensor) {
	return must.M4(e.Exec4(inputs...))
}
