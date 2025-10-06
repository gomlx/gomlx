// Package model provides helpers to build, execute, save, and load models with their weights and hyperparameters.
//
// Note: This package aims to replace the previous `ml/context` package.
// **It's still in beta**. Any feedback is very welcome.
//
// And this package defines:
//
//   - Variable: holds the model weights. It has a "concrete" view, with the actual value as a Tensor, and a graph node view
//     that can be used and updated during the building of the graph.
//   - Exec: it mimics the graph.Exec interface to execute computation graphs, but it also handles Variable objects: they are
//     automatically passed as side inputs to the graph when used, and side outputs to automatically update their values, if
//     they are updated in the graph.
//
// It hinges on the following abstraction of "model", which can be any user-defined struct (`any` type in Go), which can contain:
//
//   - Fields with static "hyperparameters" (e.g.: learning rate, batch size, number of layers, etc.)
//   - Fields of the type *Variable with model's weights (trainable or not).
//   - Slices, arrays, sub-structs, maps (with string or number keys) of "model" (a recursive definition).
//
// Example: A model that has a counter, and an increment function.
//
//	myModel := &struct{
//		counter *model.Variable
//	} {
//		counter: must.M1(model.VariableWithValue("counter", int32(0))),
//	}
//	incFn := func(g *graph.Graph) *graph.Node {
//		currentValue := myModel.counter.ValueGraph(g)
//		nextValue := graph.AddScalar(currentValue, 1)
//		myModel.counter.SetValueGraph(nextValue)  // Updates the counter.
//		return currentValue
//	}
//	inc := must.M1(model.NewExec(backend, incFn))  // Executor that increments the counter.
//	inc.Call1() // -> 0
//	inc.Call1() // -> 1
//	fmt.Printf("current myModel state: %s\n", myModel.counter.Value()) // -> 2
//	model.Save(myModel, "~/work/my_counter")  // Saved to files "~/work/my_counter.json" and "~/work/my_counter.bin".
package model

import (
	"encoding/gob"
	"encoding/json"
	"os"
	"sync"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/pkg/errors"
)

// Generate ExecFnSet interface, and a conversion tool to a canonical form.
//go:generate go run ../../../internal/cmd/builderiface/

var (
	// variablesBinEncoder is the file used to save the model's variables.
	//
	// It is a global resource because of how json.Encoder works, hopefully it can be eliminated when the
	// new json-encoder is generally available in Go 1.26: https://pkg.go.dev/encoding/json/v2
	variablesBinEncoder *gob.Encoder

	// variablesBinDecoder is the file used to load the model's variables from.
	variablesBinDecoder  *gob.Decoder
	variablesBinFileLock sync.Mutex
)

// VariableJSONMarker is a unique string to identify a JSON structure as being a variable.
const VariableJSONMarker = "#~github.com/gomlx/gomlx/pkg/ml/model.Variable"

// encodedVariable is used to encode/decode model's variables.
type encodedVariable struct {
	// Marker is used to detect that this object refers to a variable, for someone who doesn't link
	// the original model's struct.
	Marker string

	Name, Scope string
	Shape       shapes.Shape

	Trainable bool

	// HasValue is set to true if the variable's value was encoded.
	HasValue bool
}

// MarshalJSON implements json.Marshaler.
func (v *Variable) MarshalJSON() ([]byte, error) {
	encodedVariable := &encodedVariable{
		Marker:    VariableJSONMarker,
		Name:      v.name,
		Scope:     v.scope,
		Shape:     v.shape,
		Trainable: v.trainable,
	}
	if v.shape.Ok() && !v.shape.IsTuple() && v.value != nil {
		// Encode the variable's value on the ".bin" file.
		err := v.value.GobSerialize(variablesBinEncoder)
		if err != nil {
			return nil, err
		}
		encodedVariable.HasValue = true
	}
	return json.Marshal(encodedVariable)
}

// UnmarshalJSON implements json.Unmarshaler.
func (v *Variable) UnmarshalJSON(data []byte) error {
	var encodedVar encodedVariable
	err := json.Unmarshal(data, &encodedVar)
	if err != nil {
		return errors.Wrapf(err, "failed to decode variable")
	}
	if encodedVar.Marker != VariableJSONMarker {
		return errors.Errorf("invalid variable marker: %q", encodedVar.Marker)
	}
	v.name = encodedVar.Name
	v.scope = encodedVar.Scope
	v.shape = encodedVar.Shape
	v.graphToNodes.Clear()
	v.trainable = encodedVar.Trainable
	if encodedVar.HasValue {
		v.value, err = tensors.GobDeserialize(variablesBinDecoder)
		if err != nil {
			return errors.WithMessagef(err, "failed to decode variable %q value", encodedVar.Name)
		}
	}
	return nil
}

// Save the model to files "<basePath>.json" (static values, like hyperparameters) and
// "<basePath>.bin" (model's variables/weights).
//
// See package documentation for more details on valid model structures.
func Save(model any, basePath string) (err error) {
	basePath, err = fsutil.ReplaceTildeInDir(basePath)
	if err != nil {
		return err
	}
	variablesBinFileLock.Lock()
	defer func() {
		variablesBinEncoder = nil
		variablesBinFileLock.Unlock()
	}()

	// Open and registers the binary path.
	binPath := basePath + ".bin"
	var variablesBinFile *os.File
	variablesBinFile, err = os.Create(binPath)
	if err != nil {
		err = errors.Wrapf(err, "failed creating file %s", binPath)
		return
	}
	defer fsutil.ReportedClose(variablesBinFile, binPath, &err)
	variablesBinEncoder = gob.NewEncoder(variablesBinFile)
	variablesBinDecoder = nil

	// Encode model to JSON:
	jsonPath := basePath + ".json"
	var jsonFile *os.File
	jsonFile, err = os.Create(jsonPath)
	if err != nil {
		err = errors.Wrapf(err, "failed creating file %s", jsonPath)
		return
	}
	defer fsutil.ReportedClose(jsonFile, jsonPath, &err)
	enc := json.NewEncoder(jsonFile)
	err = enc.Encode(model)
	if err != nil {
		err = errors.Wrapf(err, "failed encoding model (%T) to JSON file %s", model, jsonPath)
		return
	}
	return nil
}

// Load a model from files "<basePath>.json" (static values, like hyperparameters) and
// "<basePath>.bin" (model's variables/weights).
//
// See package documentation for more details on valid model structures.
func Load(model any, basePath string) (err error) {
	basePath, err = fsutil.ReplaceTildeInDir(basePath)
	if err != nil {
		return err
	}
	variablesBinFileLock.Lock()
	defer func() {
		variablesBinDecoder = nil
		variablesBinFileLock.Unlock()
	}()

	// Open and registers the binary path.
	binPath := basePath + ".bin"
	var variablesBinFile *os.File
	variablesBinFile, err = os.Open(binPath)
	if err != nil {
		err = errors.Wrapf(err, "failed opening file %s", binPath)
		return
	}
	defer fsutil.ReportedClose(variablesBinFile, binPath, &err)
	variablesBinDecoder = gob.NewDecoder(variablesBinFile)
	variablesBinEncoder = nil

	// JSON decoder from "<basePath>.json":
	jsonPath := basePath + ".json"
	var jsonFile *os.File
	jsonFile, err = os.Open(jsonPath)
	if err != nil {
		err = errors.Wrapf(err, "failed opening file %s", jsonPath)
		return
	}
	defer fsutil.ReportedClose(jsonFile, jsonPath, &err)
	dec := json.NewDecoder(jsonFile)

	// Decode model along with variables:
	err = dec.Decode(model)
	if err != nil {
		err = errors.Wrapf(err, "failed decoding model (%T) from JSON file %s", model, jsonPath)
		return
	}
	return nil
}
