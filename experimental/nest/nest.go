/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

// Package nest implements the Nest generic container, used for generic inputs and outputs in GoMLX.
package nest

import (
	"fmt"
	"log"

	"github.com/pkg/errors"
	"github.com/gomlx/gomlx/types/slices"
)

// Nest is a generic container used in GoMLX in several of its libraries. It is a "sum type" (a union) of
// the value T itself, a map of string to T, a tuple of T values or an invalid Nest[T], which
// may contain an error message. It is only one of those and accessing the wrong instance will panic --
// except for Error, which may be checked at any time.
//
// Future Work:
//  1. Make new types for recursive Nest: if one wants a slice of maps, or a map of slices, etc. Not sure if
//     needed, but it could be expanded to that.
type Nest[T any] struct {
	nestType  Type
	slice     []T
	stringMap map[string]T
	value     T
	err       error
}

//go:generate stringer -type=Type

// Type of Nest.
type Type uint8

const (
	InvalidNest Type = iota
	ValueNest
	SliceNest
	MapNest
)

// Value creates a new Nest that contains the given value.
func Value[T any](value T) *Nest[T] {
	return &Nest[T]{
		nestType: ValueNest,
		value:    value,
	}
}

// Value returns the value stored in the Nest. It panics if the Nest is not a ValueNest.
func (n *Nest[T]) Value() T {
	if n.nestType != ValueNest {
		log.Panicf("Nest[T=%T].Value() called, but the Nest is a container of type %s", n.value, n.nestType)
	}
	return n.value
}

// IsValue returns whether the Nest holds a single value.
func (n *Nest[T]) IsValue() bool {
	return n.nestType == ValueNest
}

// Slice creates a new Nest that is a slice initialized with the given slice -- the exact same slice, it is not
// copied.
func Slice[T any](values ...T) *Nest[T] {
	return &Nest[T]{
		nestType: SliceNest,
		slice:    values,
	}
}

// IsSlice returns whether the Nest is storing a slice.
func (n *Nest[T]) IsSlice() bool {
	return n.nestType == SliceNest
}

// Slice returns the slice contained in the Nest. It panics if the Nest is not of SliceNest type.
func (n *Nest[T]) Slice() []T {
	if n.nestType != SliceNest {
		log.Panicf("Nest[T=%T].Slice() called, but the Nest is a container of type %s", n.value, n.nestType)
	}
	return n.slice
}

// Map creates a new Nest with the given valuesMap -- it is not copied, it's the same underlying map.
func Map[T any](stringMap map[string]T) *Nest[T] {
	return &Nest[T]{
		nestType:  MapNest,
		stringMap: stringMap,
	}
}

// IsMap returns whether the Nest is storing a map to T.
func (n *Nest[T]) IsMap() bool {
	return n.nestType == MapNest
}

// Map returns a reference to the underlying map. It panics if the Nest is not of MapNest type.
func (n *Nest[T]) Map() map[string]T {
	if n.nestType != MapNest {
		log.Panicf("Nest[T=%T].Map() called, but the Nest is a container of type %s", n.value, n.nestType)
	}
	return n.stringMap
}

// Enumerate all values stored in a Nest, in a deterministic order and call `fn` for each value. If fn returns an error,
// Enumerate will exit immediately and return the corresponding error.
func (n *Nest[T]) Enumerate(fn func(value T) error) error {
	switch n.nestType {
	case InvalidNest:
		return errors.Errorf("Nest[%T].Enumerate() of InvalidNest", n.value)
	case ValueNest:
		return fn(n.value)
	case SliceNest:
		for _, v := range n.slice {
			err := fn(v)
			if err != nil {
				return err
			}
		}
		return nil
	case MapNest:
		if n.stringMap == nil || len(n.stringMap) == 0 {
			return nil
		}
		// Range on sorted keys, to make it deterministic.
		for _, key := range slices.SortedKeys(n.stringMap) {
			err := fn(n.stringMap[key])
			if err != nil {
				return err
			}
		}
		return nil
	}
	return errors.Errorf("Nest[%T].Enumerate() of unknown type %s (%d)", n.value, n.nestType, n.nestType)
}

// EnumerateWithPath all values stored in a Nest, in a deterministic order and call `fn` with the path to the element and
// the corresponding value. If fn returns an error, Enumerate will exit immediately and return the corresponding error.
func (n *Nest[T]) EnumerateWithPath(fn func(path string, value T) error) error {
	switch n.nestType {
	case InvalidNest:
		return errors.Errorf("Nest[%T].Enumerate() of InvalidNest", n.value)
	case ValueNest:
		return fn(":", n.value)
	case SliceNest:
		for ii, v := range n.slice {
			err := fn(fmt.Sprintf("[%d]:", ii), v)
			if err != nil {
				return err
			}
		}
		return nil
	case MapNest:
		if n.stringMap == nil || len(n.stringMap) == 0 {
			return nil
		}
		// Range on sorted keys, to make it deterministic.
		for _, key := range slices.SortedKeys(n.stringMap) {
			err := fn(fmt.Sprintf(">%s:", key), n.stringMap[key])
			if err != nil {
				return err
			}
		}
		return nil
	}
	return errors.Errorf("Nest[%T].Enumerate() of unknown type %s (%d)", n.value, n.nestType, n.nestType)
}

// IsEmpty returns whether the Nest is empty: only happens if it's invalid, or an empty slice or empty map.
func (n *Nest[T]) IsEmpty() bool {
	switch n.nestType {
	case InvalidNest:
		return true
	case ValueNest:
		return false
	case SliceNest:
		return len(n.slice) == 0
	case MapNest:
		return len(n.stringMap) == 0
	}
	log.Panicf("Nest[T=%T].IsEmpty() called with Nest of unknown container of type %s (%d)", n.value, n.nestType, n.nestType)
	return true
}

// Flatten converts the Nest of any time to a slice, in a deterministic fashion. This is trivial if the Nest is already
// a slice, but this is a common interface for all Nest types.
//
// Notice the returned slice should not be changed, the underlying space is owned by the Nest.
//
// If the Nest is of an invalid type, it returns nil.
func (n *Nest[T]) Flatten() []T {
	switch n.nestType {
	case InvalidNest:
		return nil
	case ValueNest:
		return []T{n.value}
	case SliceNest:
		return n.slice
	case MapNest:
		if n.stringMap == nil || len(n.stringMap) == 0 {
			return nil
		}
		// Range on sorted keys, to make it deterministic.
		s := make([]T, 0, len(n.stringMap))
		for _, key := range slices.SortedKeys(n.stringMap) {
			s = append(s, n.stringMap[key])
		}
		return s
	}
	log.Panicf("Nest[T=%T].Flatten() called with Nest of unknown container of type %s (%d)", n.value, n.nestType, n.nestType)
	return nil
}

// Unflatten will create a Nest[T2], using the given flatten values, and the structure and NestType of nestBase.
func Unflatten[T1, T2 any](nestShape *Nest[T1], flatValues []T2) *Nest[T2] {
	switch nestShape.nestType {
	case InvalidNest:
		return &Nest[T2]{}
	case ValueNest:
		return Value(flatValues[0])
	case SliceNest:
		return Slice(flatValues...)
	case MapNest:
		stringMap := make(map[string]T2, len(nestShape.stringMap))
		if len(nestShape.stringMap) >= 0 {
			idx := 0
			for _, key := range slices.SortedKeys(nestShape.stringMap) {
				stringMap[key] = flatValues[idx]
				idx++
			}
		}
		return Map(stringMap)
	}
	var t2 T2
	log.Panicf("Unflatten[T1=%T, T2=%T]() called with nestShape of unknown container of type %s (%d)", nestShape.value, t2, nestShape.nestType, nestShape.nestType)
	return nil
}
