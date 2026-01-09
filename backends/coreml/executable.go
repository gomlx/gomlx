//go:build darwin && cgo

package coreml

import (
	"reflect"
	"sync"
	"unsafe"

	"github.com/gomlx/go-coreml/runtime"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Compile-time check that Executable implements backends.Executable.
var _ backends.Executable = (*Executable)(nil)

// Executable wraps a compiled CoreML model and implements the backends.Executable interface.
// It manages execution of CoreML models through the go-coreml runtime.
type Executable struct {
	backend *Backend
	runtime *runtime.Executable

	inputNames   []string
	outputNames  []string
	inputShapes  []shapes.Shape
	outputShapes []shapes.Shape

	mu sync.Mutex
}

// newExecutable creates a new Executable from a Builder and go-coreml runtime executable.
func newExecutable(builder *Builder, runtimeExec *runtime.Executable) *Executable {
	return &Executable{
		backend:      builder.backend,
		runtime:      runtimeExec,
		inputNames:   builder.inputNames,
		outputNames:  builder.outputNames,
		inputShapes:  builder.inputShapes,
		outputShapes: builder.outputShapes,
	}
}

// Finalize immediately frees resources associated with the executable.
// After calling Finalize, the executable should not be used.
func (e *Executable) Finalize() {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.runtime != nil {
		_ = e.runtime.Close()
		e.runtime = nil
	}
}

// Inputs returns the list of parameter names and shapes, in order created by the Builder.Parameter calls.
func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	return e.inputNames, e.inputShapes
}

// Outputs returns the output shapes of the computation, in order given to the Builder.Compile call.
func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	return e.outputShapes
}

// Execute runs the CoreML model with the given inputs.
// The number and shapes of the inputs must match those returned by Inputs.
//
// The inputs marked in `donate` will become invalid after use.
// Note: CoreML backend currently ignores the donate parameter as CoreML manages its own memory.
//
// The defaultDevice parameter is ignored as CoreML manages device selection internally.
func (e *Executable) Execute(inputs []backends.Buffer, donate []bool, defaultDevice backends.DeviceNum) ([]backends.Buffer, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.runtime == nil {
		return nil, errors.New("executable has been finalized")
	}

	// Validate number of inputs
	if len(inputs) != len(e.inputNames) {
		return nil, errors.Errorf("Execute: expected %d inputs, got %d", len(e.inputNames), len(inputs))
	}

	// Validate input shapes and convert to runtime format
	runtimeInputs := make(map[string]interface{}, len(inputs))
	for i, input := range inputs {
		if input == nil {
			return nil, errors.Errorf("Execute: input #%d is nil", i)
		}

		// Get the input buffer
		buffer, ok := input.(*Buffer)
		if !ok {
			return nil, errors.Errorf("Execute: input #%d is not a CoreML buffer", i)
		}

		// Validate shape
		if !buffer.shape.Equal(e.inputShapes[i]) {
			return nil, errors.Errorf("Execute: input %q (input #%d): expected shape %s, got %s",
				e.inputNames[i], i, e.inputShapes[i], buffer.shape)
		}

		// Convert buffer data to runtime format
		data, err := bufferToRuntimeData(buffer)
		if err != nil {
			return nil, errors.Wrapf(err, "Execute: converting input #%d to runtime format", i)
		}

		runtimeInputs[e.inputNames[i]] = data
	}

	// Execute the model
	runtimeOutputs, err := e.runtime.Run(runtimeInputs)
	if err != nil {
		return nil, errors.Wrap(err, "Execute: CoreML runtime execution failed")
	}

	// Convert outputs back to buffers
	outputs := make([]backends.Buffer, len(e.outputNames))
	for i, name := range e.outputNames {
		outputData, ok := runtimeOutputs[name]
		if !ok {
			return nil, errors.Errorf("Execute: output %q not found in CoreML results", name)
		}

		buffer, err := e.backend.runtimeDataToBuffer(outputData, e.outputShapes[i])
		if err != nil {
			return nil, errors.Wrapf(err, "Execute: converting output #%d from runtime format", i)
		}

		outputs[i] = buffer
	}

	return outputs, nil
}

// bufferToRuntimeData converts a CoreML Buffer to the data format expected by go-coreml runtime.
// The runtime expects slices of the appropriate type ([]float32, []int32, etc.).
func bufferToRuntimeData(buffer *Buffer) (interface{}, error) {
	if buffer.flat == nil {
		return nil, errors.New("buffer has nil data")
	}

	// The buffer.flat is already a slice of the correct type, so we can return it directly
	// However, we need to make a copy to avoid issues with buffer reuse
	switch data := buffer.flat.(type) {
	case []float32:
		dataCopy := make([]float32, len(data))
		copy(dataCopy, data)
		return dataCopy, nil
	case []float64:
		// CoreML typically uses float32, so convert if needed
		f32 := make([]float32, len(data))
		for i, v := range data {
			f32[i] = float32(v)
		}
		return f32, nil
	case []int32:
		dataCopy := make([]int32, len(data))
		copy(dataCopy, data)
		return dataCopy, nil
	case []int64:
		// CoreML typically uses int32, so convert if needed
		i32 := make([]int32, len(data))
		for i, v := range data {
			i32[i] = int32(v)
		}
		return i32, nil
	case []uint8:
		dataCopy := make([]uint8, len(data))
		copy(dataCopy, data)
		return dataCopy, nil
	case []int16:
		dataCopy := make([]int16, len(data))
		copy(dataCopy, data)
		return dataCopy, nil
	case []uint16:
		dataCopy := make([]uint16, len(data))
		copy(dataCopy, data)
		return dataCopy, nil
	case []uint32:
		dataCopy := make([]uint32, len(data))
		copy(dataCopy, data)
		return dataCopy, nil
	case []uint64:
		dataCopy := make([]uint64, len(data))
		copy(dataCopy, data)
		return dataCopy, nil
	default:
		return nil, errors.Errorf("unsupported buffer data type: %T", buffer.flat)
	}
}

// runtimeDataToBuffer converts data from go-coreml runtime format to a CoreML Buffer.
func (b *Backend) runtimeDataToBuffer(data interface{}, shape shapes.Shape) (*Buffer, error) {
	if data == nil {
		return nil, errors.New("runtime output data is nil")
	}

	// Create a new buffer with the expected shape
	buffer := b.NewBuffer(shape)
	if buffer == nil {
		return nil, errors.New("failed to allocate buffer")
	}

	// Copy data into the buffer
	switch srcData := data.(type) {
	case []float32:
		if dstData, ok := buffer.flat.([]float32); ok {
			if len(srcData) != len(dstData) {
				return nil, errors.Errorf("data size mismatch: expected %d elements, got %d", len(dstData), len(srcData))
			}
			copy(dstData, srcData)
		} else {
			return nil, errors.Errorf("type mismatch: expected []float32, buffer has %T", buffer.flat)
		}
	case []float64:
		if dstData, ok := buffer.flat.([]float64); ok {
			if len(srcData) != len(dstData) {
				return nil, errors.Errorf("data size mismatch: expected %d elements, got %d", len(dstData), len(srcData))
			}
			copy(dstData, srcData)
		} else {
			return nil, errors.Errorf("type mismatch: expected []float64, buffer has %T", buffer.flat)
		}
	case []int32:
		if dstData, ok := buffer.flat.([]int32); ok {
			if len(srcData) != len(dstData) {
				return nil, errors.Errorf("data size mismatch: expected %d elements, got %d", len(dstData), len(srcData))
			}
			copy(dstData, srcData)
		} else {
			return nil, errors.Errorf("type mismatch: expected []int32, buffer has %T", buffer.flat)
		}
	case []int64:
		if dstData, ok := buffer.flat.([]int64); ok {
			if len(srcData) != len(dstData) {
				return nil, errors.Errorf("data size mismatch: expected %d elements, got %d", len(dstData), len(srcData))
			}
			copy(dstData, srcData)
		} else {
			return nil, errors.Errorf("type mismatch: expected []int64, buffer has %T", buffer.flat)
		}
	case []uint8:
		if dstData, ok := buffer.flat.([]uint8); ok {
			if len(srcData) != len(dstData) {
				return nil, errors.Errorf("data size mismatch: expected %d elements, got %d", len(dstData), len(srcData))
			}
			copy(dstData, srcData)
		} else {
			return nil, errors.Errorf("type mismatch: expected []uint8, buffer has %T", buffer.flat)
		}
	case []int16:
		if dstData, ok := buffer.flat.([]int16); ok {
			if len(srcData) != len(dstData) {
				return nil, errors.Errorf("data size mismatch: expected %d elements, got %d", len(dstData), len(srcData))
			}
			copy(dstData, srcData)
		} else {
			return nil, errors.Errorf("type mismatch: expected []int16, buffer has %T", buffer.flat)
		}
	case []uint16:
		if dstData, ok := buffer.flat.([]uint16); ok {
			if len(srcData) != len(dstData) {
				return nil, errors.Errorf("data size mismatch: expected %d elements, got %d", len(dstData), len(srcData))
			}
			copy(dstData, srcData)
		} else {
			return nil, errors.Errorf("type mismatch: expected []uint16, buffer has %T", buffer.flat)
		}
	case []uint32:
		if dstData, ok := buffer.flat.([]uint32); ok {
			if len(srcData) != len(dstData) {
				return nil, errors.Errorf("data size mismatch: expected %d elements, got %d", len(dstData), len(srcData))
			}
			copy(dstData, srcData)
		} else {
			return nil, errors.Errorf("type mismatch: expected []uint32, buffer has %T", buffer.flat)
		}
	case []uint64:
		if dstData, ok := buffer.flat.([]uint64); ok {
			if len(srcData) != len(dstData) {
				return nil, errors.Errorf("data size mismatch: expected %d elements, got %d", len(dstData), len(srcData))
			}
			copy(dstData, srcData)
		} else {
			return nil, errors.Errorf("type mismatch: expected []uint64, buffer has %T", buffer.flat)
		}
	default:
		return nil, errors.Errorf("unsupported runtime data type: %T", data)
	}

	return buffer, nil
}

// Unused functions for reference (kept to show the pattern, but not needed for current implementation)
var _ = unsafe.Pointer(nil)
var _ = reflect.TypeOf(nil)
