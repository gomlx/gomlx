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
//	model.Save(myModel, "~/work/my_counter")  // Saved to files "~/work/my_counter.jsonl" and "~/work/my_counter.bin".
package model

import (
	"encoding/gob"
	"encoding/json"
	"os"
	"sync"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/model/encoding"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/pkg/support/xsync"
	"github.com/pkg/errors"
)

// Generate ExecFnSet interface, and a conversion tool to a canonical form.
//go:generate go run ../../../internal/cmd/builderiface/

var (
	// VariablesDecodingMap maps a path to the variable during the decoding process.
	// It is used by Variable.UnmarshalJSON to find the preloaded variable to use.
	//
	// It is a global resource protected by VariablesDecodingLock because of how json.Encoder works.
	// Hopefully, this can be eliminated when the new json-encoder is generally available in Go 1.26: https://pkg.go.dev/encoding/json/v2
	VariablesDecodingMap map[string]*Variable

	// VariablesDecodingLock makes sure that only one JSON decoding with Variable objects is happening at a time:
	// it requires a global resource, hence the lock.
	// Hopefully, this can be eliminated when the new json-encoder is generally available in Go 1.26: https://pkg.go.dev/encoding/json/v2
	VariablesDecodingLock sync.Mutex
)

// MarshalJSON implements json.Marshaler.
//
// A variable is simply encoded as its path: it serves as a key to find the variable in the decoding process.
func (v *Variable) MarshalJSON() ([]byte, error) {
	return json.Marshal(&encoding.VarPath{Path: v.path})
}

// UnmarshalJSON implements json.Unmarshaler.
//
// The variable is decoded from its path, which serves as a key to find the already preloaded variable in
// VariablesDecodingMap.
func (v *Variable) UnmarshalJSON(data []byte) error {
	var vp encoding.VarPath
	err := json.Unmarshal(data, &vp)
	if err != nil {
		return errors.Wrapf(err, "failed to decode variable's path")
	}
	p := vp.Path
	vFrom, ok := VariablesDecodingMap[p]
	if !ok || vFrom == nil {
		return errors.Errorf("variable %q not found in VariablesDecodingMap: variables need to be preloaded into VariablesDecodingMap first", p)
	}
	delete(VariablesDecodingMap, p)
	*v = *vFrom
	v.graphToNodes = xsync.SyncMap[graph.GraphId, *variableNodes]{}
	vFrom.value = nil
	vFrom.shape = shapes.Invalid()
	return nil
}

// Save the model to files "<basePath>.jsonl" (JSON-streaming) with static values (like hyperparameters) and
// "<basePath>.bin" (model's variables/weights).
//
// See package documentation for more details on valid model structures.
//
// Note: variables initializers are not saved. So you attempt to save a model before initializing its variables,
// you will later load a model with zeroInitializer for the variables.
func Save(model any, basePath string) (err error) {
	basePath, err = fsutil.ReplaceTildeInDir(basePath)
	if err != nil {
		return err
	}

	// Open and registers the binary path.
	binPath := basePath + ".bin"
	var variablesBinFile *os.File
	variablesBinFile, err = os.Create(binPath)
	if err != nil {
		err = errors.Wrapf(err, "failed creating file %s", binPath)
		return
	}
	defer fsutil.ReportedClose(variablesBinFile, binPath, &err)
	variablesBinEncoder := gob.NewEncoder(variablesBinFile)

	// Encode model to JSON:
	jsonPath := basePath + ".jsonl"
	var jsonFile *os.File
	jsonFile, err = os.Create(jsonPath)
	if err != nil {
		err = errors.Wrapf(err, "failed creating file %s", jsonPath)
		return
	}
	defer fsutil.ReportedClose(jsonFile, jsonPath, &err)
	jsonEncoder := json.NewEncoder(jsonFile)

	// Build and encode the header first.
	header := encoding.Header{
		Version: encoding.Version1,
	}
	for pair, err := range IterVariables(model) {
		if err != nil {
			return err
		}
		v, p := pair.Variable, pair.Path
		v.path = p // This will be used as the key when decoding the variable.
		encodedVariable := encoding.EncodedVariable{
			Name:      v.name,
			Path:      p,
			Shape:     v.shape,
			Trainable: v.trainable,
		}
		if v.shape.Ok() && !v.shape.IsTuple() && v.value != nil {
			// Encode the variable's value into the ".bin" file.
			err := v.value.GobSerialize(variablesBinEncoder)
			if err != nil {
				return errors.WithMessagef(err, "while binary encoding variable %s", v)
			}
			encodedVariable.HasValue = true
		}
		header.Variables = append(header.Variables, encodedVariable)
	}
	err = jsonEncoder.Encode(header)
	if err != nil {
		err = errors.Wrapf(err, "failed encoding header to JSON file %s", jsonPath)
		return
	}

	err = jsonEncoder.Encode(model)
	if err != nil {
		err = errors.Wrapf(err, "failed encoding model (%T) to JSON file %s", model, jsonPath)
		return
	}
	return nil
}

// Load a model from files "<basePath>.jsonl" (JSON-streaming, with the static values, like hyperparameters) and
// "<basePath>.bin" (model's variables/weights).
//
// See package documentation for more details on valid model structures.
func Load(model any, basePath string) (err error) {
	basePath, err = fsutil.ReplaceTildeInDir(basePath)
	if err != nil {
		return err
	}

	// JSON decoder from "<basePath>.jsonl":
	jsonPath := basePath + ".jsonl"
	var jsonFile *os.File
	jsonFile, err = os.Open(jsonPath)
	if err != nil {
		err = errors.Wrapf(err, "failed opening file %s", jsonPath)
		return
	}
	defer fsutil.ReportedClose(jsonFile, jsonPath, &err)
	jsonDecoder := json.NewDecoder(jsonFile)

	// Decode the header first.
	var header encoding.Header
	err = jsonDecoder.Decode(&header)
	if err != nil {
		err = errors.Wrapf(err, "failed decoding header from JSON file %s", jsonPath)
		return
	}
	if header.Version != encoding.Version1 {
		return errors.Errorf("unsupported version %q", header.Version)
	}

	// Decode the variables from the binary file.
	varMap := make(map[string]*Variable, len(header.Variables))
	binPath := basePath + ".bin"
	var variablesBinFile *os.File
	variablesBinFile, err = os.Open(binPath)
	if err != nil {
		err = errors.Wrapf(err, "failed opening file %s", binPath)
		return
	}
	defer fsutil.ReportedClose(variablesBinFile, binPath, &err)
	binDecoder := gob.NewDecoder(variablesBinFile)

	for _, encV := range header.Variables {
		var v *Variable
		if encV.HasValue {
			// Value was encoded.
			value, err := tensors.GobDeserialize(binDecoder)
			if err != nil {
				return errors.Wrapf(err, "failed decoding variable %q in %s", encV.Name, encV.Path)
			}
			v, err = VariableWithValue(encV.Name, value)
			if err != nil {
				return errors.Wrapf(err, "failed creating variable %q in %s", encV.Name, encV.Path)
			}
		} else {
			// Variable value was not encoded, so it is created with a zero-initializer.
			v = VariableWithShape(encV.Name, encV.Shape)
		}
		v.path = encV.Path
		v.shape = encV.Shape
		v.trainable = encV.Trainable
		varMap[v.path] = v
	}

	// Lock the global variables decoding map.
	VariablesDecodingLock.Lock()
	defer func() {
		VariablesDecodingMap = nil
		VariablesDecodingLock.Unlock()
	}()
	VariablesDecodingMap = varMap

	// Decode the model structure: variables will be taken from VariablesDecodingMap.
	err = jsonDecoder.Decode(model)
	if err != nil {
		err = errors.Wrapf(err, "failed decoding model (%T) from JSON file %s", model, jsonPath)
		return
	}
	return nil
}
