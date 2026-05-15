/*
Package pjson ("polymorphic-json") provides a generic, reflection-based solution
for JSON serializing and deserializing Go interfaces within parent structs using
the standard encoding/json package.

It solves the common problem of polymorphism in JSON by automatically injecting a
"type" discriminator field containing the fully qualified package path and struct
name, allowing the library to instantiate the correct concrete struct during
unmarshaling (decoding).

# How To Use It

This package requires zero modifications to your interfaces or concrete types.
Just follow these 3 steps:

1. Register each concrete type with the pjson.Register function during
   initialization.

    func init() {
        pjson.Register(func() OptimizerIface { return &AdagradOptimizer{} })
    }

2. Wrap the interface value using pjson.Wrapper[I] in your structs.

    type Trainer struct {
        Optimizer pjson.Wrapper[OptimizerIface] `json:"optimizer"`
    }
    
    myTrainer.Optimizer = pjson.Wrap[OptimizerIface](&AdagradOptimizer{LR: 0.005})

3. Optionally, define a proxy struct that embeds the Wrapper[I] to eliminate
   the need to access the internal '.Value' field.

    type Optimizer struct {
        pjson.Wrapper[OptimizerIface]
    }

    func (o Optimizer) Tune(epochs int) error {
        return o.Value.Tune(epochs)
    }

That's it! Marshaling automatically records the fully qualified type, and
unmarshaling automatically routes the payload to the correct constructor.
*/
package pjson

import (
	"encoding/json"
	"reflect"
	"sync"

	"github.com/pkg/errors"
)

var (
	// registry maps: Interface Type -> Fully Qualified Concrete Type Name -> Constructor
	registry   = make(map[reflect.Type]map[string]any)
	registryMu sync.RWMutex
)

// typeName generates a fully qualified name including the absolute package path
// e.g., "*github.com/user/repo/pkg.MyStruct"
func typeName(t reflect.Type) string {
	if t.Kind() == reflect.Pointer {
		return "*" + t.Elem().PkgPath() + "." + t.Elem().Name()
	}
	return t.PkgPath() + "." + t.Name()
}

// Register automatically extracts the interface type and the concrete type string.
func Register[I any](constructor func() I) {
	iType := reflect.TypeOf((*I)(nil)).Elem()
	instance := constructor()
	tName := typeName(reflect.TypeOf(instance))

	registryMu.Lock()
	defer registryMu.Unlock()

	if registry[iType] == nil {
		registry[iType] = make(map[string]any)
	}
	registry[iType][tName] = constructor
}

// Wrapper is the generic struct users place in their models.
type Wrapper[I any] struct {
	Value I
}

// envelope is used to cleanly marshal/unmarshal the discriminator.
type envelope struct {
	Type  string          `json:"type"`
	Value json.RawMessage `json:"value"`
}

// MarshalJSON implements json.Marshaler.
func (w Wrapper[I]) MarshalJSON() ([]byte, error) {
	if any(w.Value) == nil {
		return []byte("null"), nil
	}

	valBytes, err := json.Marshal(w.Value)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to marshal nested value of type %T", w.Value)
	}

	env := envelope{
		Type:  typeName(reflect.TypeOf(w.Value)),
		Value: valBytes,
	}
	return json.Marshal(env)
}

// UnmarshalJSON implements json.Unmarshaler.
func (w *Wrapper[I]) UnmarshalJSON(b []byte) error {
	if string(b) == "null" {
		var nilI I
		w.Value = nilI
		return nil
	}

	var env envelope
	if err := json.Unmarshal(b, &env); err != nil {
		return errors.Wrap(err, "polymorphic unmarshal failed to decode envelope")
	}

	iType := reflect.TypeOf((*I)(nil)).Elem()

	registryMu.RLock()
	typeMap := registry[iType]
	var constructorAny any
	if typeMap != nil {
		constructorAny = typeMap[env.Type]
	}
	registryMu.RUnlock()

	if constructorAny == nil {
		return errors.Errorf("unknown concrete type '%s' for interface '%v'", env.Type, iType)
	}

	// Create an empty instance using the typed constructor
	constructor := constructorAny.(func() I)
	instance := constructor()

	// Decode the raw value directly into the instantiated struct
	if err := json.Unmarshal(env.Value, &instance); err != nil {
		return errors.Wrapf(err, "failed to unmarshal nested value into %T", instance)
	}

	w.Value = instance
	return nil
}

// Wrap is a convenience function to instantiate the wrapper.
func Wrap[I any](value I) Wrapper[I] {
	return Wrapper[I]{Value: value}
}
