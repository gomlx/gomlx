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

	// Optimizer is the clean type users embed in their models.
	type Optimizer polymorphicjson.Wrapper[OptimizerIface]

	// Tune proxies the call from the clean type to the underlying interface value.
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
		// Application fields
		LearningRate float64 `json:"learning_rate"`
	}

	func (a *AdagradOptimizer) JSONTags() (typeName string, interfaceName string) {
		// Report the unique type name and the interface name.
		return "Adagrad", "OptimizerIface"
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
		Optimizer: Optimizer{ NewAdagradOptimizer() },
	}
	jsonData, _ := json.Marshal(model)
	// ... jsonData can be unmarshaled back into a Model struct.
*/
package polymorphicjson

import (
	"encoding/json"
	"fmt"
	"sync"
)

// JSONIdentifiable is the constraint interface. Any concrete type must implement
// this method to provide the unique tag for the concrete type and the name
// of the interface it satisfies.
type JSONIdentifiable interface {
	// JSONTags returns the unique name for the concrete type and the unique name for the interface.
	JSONTags() (typeName string, interfaceName string)
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

// TypeWrapper is a minimal struct used only to extract the type tags during the first
// pass of unmarshaling. It includes both the concrete type and the interface name.
type TypeWrapper struct {
	JSONType      string `json:"json_type"`
	InterfaceName string `json:"interface_name"` // For validation
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
func (p Wrapper[I]) MarshalJSON() ([]byte, error) {
	return MarshalPolymorphic(p.Value)
}

// UnmarshalJSON implements json.Unmarshaler for the generic wrapper.
func (p *Wrapper[I]) UnmarshalJSON(b []byte) error {
	// UnmarshalPolymorphic populates p.Value using the two-pass logic.
	return UnmarshalPolymorphic(b, &p.Value)
}

// Get returns the wrapped value.
func (p *Wrapper[I]) Get() I {
	return p.Value
}

// UnmarshalPolymorphic performs the two-pass unmarshaling required for polymorphic types.
// 'I' is the interface type (e.g., InitializerRaw).
// 'target' is a pointer to the generic type's value field (e.g., *p.Value, which is **InitializerRaw).
func UnmarshalPolymorphic[I JSONIdentifiable](b []byte, target *I) error {
	if len(b) == 0 || string(b) == "null" {
		var nilI I
		*target = nilI
		return nil
	}

	// Pass 1: Extract the type tags
	var wrapper TypeWrapper
	if err := json.Unmarshal(b, &wrapper); err != nil {
		return fmt.Errorf("polymorphic unmarshal failed to read tags: %w", err)
	}

	// Look up the concrete type constructor using the extracted InterfaceName and JSONType.
	typeMap, ok := registry[wrapper.InterfaceName]
	if !ok {
		return fmt.Errorf("polymorphic unmarshal error: interface '%s' not registered", wrapper.InterfaceName)
	}

	constructor, ok := typeMap[wrapper.JSONType]
	if !ok {
		return fmt.Errorf("polymorphic unmarshal error: unknown concrete type '%s' for interface '%s'", wrapper.JSONType, wrapper.InterfaceName)
	}

	// Create an empty instance of the concrete type.
	instance := constructor()

	// Pass 2: Unmarshal the full JSON into the concrete instance.
	if err := json.Unmarshal(b, instance); err != nil {
		return fmt.Errorf("polymorphic unmarshal failed to load data into concrete type %T: %w", instance, err)
	}

	// Assign the concrete instance to the target pointer-to-interface.
	// This assignment inherently checks if the concrete type implements the interface I.
	*target = instance.(I)
	return nil
}

// MarshalPolymorphic handles marshaling. It relies on the concrete type containing
// the necessary `json_type` and `interface_name` fields for a flat output.
func MarshalPolymorphic[I JSONIdentifiable](value I) ([]byte, error) {
	if any(value) == nil {
		return []byte("null"), nil
	}
	// Simply marshal the concrete value (e.g., *SimpleInitializer)
	return json.Marshal(value)
}
