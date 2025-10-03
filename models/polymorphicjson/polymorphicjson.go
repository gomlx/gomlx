/*
Package polymorphicjson provides a generic solution for serializing and deserializing
Go interfaces within parent structs using the standard encoding/json package.

It solves the common problem of polymorphism in JSON by injecting two discriminator fields
("json_type" and "interface_name") into the JSON payload, allowing the library to
instantiate the correct concrete struct during unmarshaling (decoding).

--- Usage Example ---

To use this package, you must follow three steps:
1. Define the interface contract (embedding JSONIdentifiable).
2. Define the clean, user-facing proxy type (embedding Wrapper[I]).
3. Define the concrete structs (implementing JSONTags()).

1. Define the Interface Contract (e.g., OptimizerIface)

The application interface must embed JSONIdentifiable.

	// OptimizerIface is the raw contract for all optimizer types.
	type OptimizerIface interface {
		polymorphicjson.JSONIdentifiable // MUST be embedded

		Tune(epochs int) error
	}

2. Define the Clean, User-Facing Proxy Type

This type wraps the generic polymorphicjson.Wrapper, inherits the JSON methods, and adds proxy methods
to eliminate the need for users to access the internal '.Value' field.

	// Optimizer holds a serializable OptimizerIface.
	type Optimizer polymorphicjson.Wrapper[OptimizerIface]

	// NewOptimizer from an OptimizerIface.
	func NewOptimizer(opt OptimizerIface) Optimizer {
		// This uses positional initialization for the anonymous embedded field.
		return Optimizer{polymorphicjson.Wrapper[OptimizerIface]{Value: opt}}
	}

	// Tune proxies the OptimizerIface method.
	func (o Optimizer) Tune(epochs int) error {
		// Safely check for nil before calling the method.
		if any(o.Value) == nil {
			return errors.New("cannot tune: optimizer is nil")
		}
		return o.Value.Tune(epochs)
	}

3. Define the Concrete Struct (e.g., AdagradOptimizer)

The concrete struct must implement all methods, including the required JSONTags().

	// AdagradOptimizer implements OptimizerIface.
	type AdagradOptimizer struct {
		// Fields required by the polymorphicjson package
		ConcreteType      string `json:"concrete_type"`
		InterfaceName string `json:"interface_name"`

		// Application fields
		LearningRate float64 `json:"learning_rate"`
	}

	func (a *AdagradOptimizer) JSONTags() (interfaceName, concreteType string) {
		// Report the unique type name and the interface name.
		return "OptimizerIface", "Adagrad",
	}

	func (a *AdagradOptimizer) Tune(epochs int) error { return nil } // Implementation goes here.

	func init() {
		// MANDATORY: Register the constructor once.
		polymorphicjson.Register(func() OptimizerIface { return NewAdagradOptimizer() })
	}

4. User Model Example

The user wanting to embed an optimizer in their model has a clean and simple API, with no reference to the underlying Wrapper.

	type Model struct {
		Name string         `json:"name"`
		Optimizer Optimizer `json:"optimizer"` // Clean type
	}

Usage example:

	model := ModelConfig{
		Name: "VGG",
		Optimizer: NewOptimizer(NewAdagradOptimizer()),
	}
	jsonData, _ := json.Marshal(model)
	// ... jsonData can be unmarshaled back into a Model struct.
*/
package polymorphicjson

import (
	"encoding/json"
	"fmt"
	"sync"

	"github.com/pkg/errors"
)

// JSONIdentifiable is the constraint interface. Any concrete type must implement
// this method to provide the unique tag for the concrete type and the name
// of the interface it satisfies.
type JSONIdentifiable interface {
	// JSONTags returns the unique name for the concrete type and the unique name for the interface.
	JSONTags() (interfaceName, concreteType string)
}

var (
	// Global registry. Maps the raw interface name (e.g., "InitializerRaw") to concrete type constructors.
	registry = make(map[string]map[string]func() JSONIdentifiable)

	// Global registry mutex.
	registryMu sync.RWMutex
)

// Register registers a concrete type T by using its JSONTags() method to determine
// its concrete type name and the interface it belongs to.
// T must be a pointer to a struct that implements JSONIdentifiable.
func Register[T JSONIdentifiable](constructor func() T) {
	registryMu.Lock()
	defer registryMu.Unlock()

	// Get names from the instance created by the constructor
	instance := constructor()
	typeName, interfaceName := instance.JSONTags()

	if _, exists := registry[interfaceName]; !exists {
		registry[interfaceName] = make(map[string]func() JSONIdentifiable)
	}

	// Register the constructor under the interface name and the concrete type name
	registry[interfaceName][typeName] = func() JSONIdentifiable {
		return constructor()
	}
}

// typeWrapper is a minimal struct used only to extract the type tags during the first
// pass of unmarshaling. It includes both the concrete type and the interface name.
type typeWrapper struct {
	InterfaceName string `json:"interface_name"`
	ConcreteType  string `json:"concrete_type"`
}

// internalWrapper is used to define the nested structure for Marshaling.
type internalWrapper[I JSONIdentifiable] struct {
	InterfaceName string `json:"interface_name"`
	ConcreteType  string `json:"concrete_type"`
	Value         I      `json:"value"` // The nested field holding the concrete data
}

func Wrap[I JSONIdentifiable](value I) Wrapper[I] {
	return Wrapper[I]{Value: value}
}

// Wrapper is the generic type wrapper that implements the standard
// json.Marshaler and json.Unmarshaler interfaces.
// The user places this type in their models:
// type ProjectModel { Initializer Wrapper[mylib.InitializerRaw] }
type Wrapper[I JSONIdentifiable] struct {
	Value I
}

// MarshalJSON implements json.Marshaler for the generic wrapper.
func (w Wrapper[I]) MarshalJSON() ([]byte, error) {
	// Check for nil using the `any` conversion, which correctly handles nil interfaces.
	if any(w.Value) == nil {
		return []byte("null"), nil
	}

	typeName, interfaceName := w.Value.JSONTags()

	// Create the nested structure for marshaling
	nested := internalWrapper[I]{
		InterfaceName: interfaceName,
		ConcreteType:  typeName,
		Value:         w.Value,
	}

	return json.Marshal(nested)
}

// UnmarshalJSON implements json.Unmarshaler for the generic wrapper.
func (w *Wrapper[I]) UnmarshalJSON(b []byte) error {
	if len(b) == 0 || string(b) == "null" {
		// Use local variable to safely assign nil to the interface type parameter I
		var nilI I
		w.Value = nilI
		return nil
	}

	// Pass 1: Extract the type tags
	var tags typeWrapper // Renamed to use the unexported type
	if err := json.Unmarshal(b, &tags); err != nil {
		return fmt.Errorf("polymorphic unmarshal failed to read type tags: %w", err)
	}

	// Lookup the concrete type constructor
	typeMap, ok := registry[tags.InterfaceName]
	if !ok {
		return fmt.Errorf("polymorphic unmarshal error: interface '%s' not registered", tags.InterfaceName)
	}

	constructor, ok := typeMap[tags.ConcreteType]
	if !ok {
		return fmt.Errorf("polymorphic unmarshal error: unknown concrete type '%s' for interface '%s'", tags.ConcreteType, tags.InterfaceName)
	}

	// Create an empty instance of the concrete type
	instance := constructor()

	// Pass 2: Extract the raw JSON bytes for the nested 'value' field.
	// We use a temporary struct and json.RawMessage to avoid a full decode.
	decoder := struct {
		Value json.RawMessage `json:"value"`
	}{}

	if err := json.Unmarshal(b, &decoder); err != nil {
		return fmt.Errorf("polymorphic unmarshal failed to extract nested value: %w", err)
	}

	if decoder.Value == nil {
		return errors.New("polymorphic unmarshal expected 'value' field, but it was missing or null")
	}

	// Pass 3: Decode the raw JSON value into the concrete instance.
	if err := json.Unmarshal(decoder.Value, instance); err != nil {
		return fmt.Errorf("polymorphic unmarshal failed to load concrete data into %T: %w", instance, err)
	}

	// Assign the concrete instance to the internal interface value
	w.Value = instance.(I)
	return nil
}
