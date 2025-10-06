package model

import (
	"cmp"
	"fmt"
	"iter"
	"reflect"
	"slices"

	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

const modelsPkgPath = "github.com/gomlx/gomlx/pkg/ml/model"

// PathAndVariable refers to a variable within a model struct, at the "Path" location.
// See details in IterVariables.
type PathAndVariable struct {
	Path     string
	Variable *Variable
}

// IterVariables returns an iterator over the model's non-nil variables, performing a "depth first search" into the model,
// in a deterministic order (always the same for the same contents).
//
// Struct fields are iterated in the order they are defined in the struct. Maps are iterated in alphabetic order.
//
// It yields PathAndVariable objects, with the path to the variable within the model structure, and its variable
// pointer. See details on valid "model" objects in the package documentation.
//
// Example:
//
//	type A struct { a0, a1 *Variable }
//	type B struct { manyA []*A }
//	b := &B{manyA: []*A{&A{a0: v0, a1: v1}, &A{a0: v2, a1: v3}}}
//	IterVariables(b) -> { "manyA[0].a0", v0 }, { "manyA[0].a1", v1 }, { "manyA[1].a0", v2 }, { "manyA[1].a1", v3 }
//
// It may return an error if the model is invalid: Variable is included by value (as opposed to by reference/pointer);
// invalid map keys (only maps with string and numbers are accepted).
func IterVariables(model any) iter.Seq2[PathAndVariable, error] {
	return func(yield func(PathAndVariable, error) bool) {
		seen := sets.Make[uintptr]()
		var iterStruct func(v reflect.Value, path string) bool
		var iterSliceOrArray func(v reflect.Value, pathPrefix string) bool
		var iterMap func(v reflect.Value, pathPrefix string) bool
		var iterValue func(v reflect.Value, path string) bool

		// Helper to check if the pointer was already visited to avoid cycles.
		checkPtr := func(pointer uintptr) bool {
			if seen.Has(pointer) {
				return false
			}
			seen.Insert(pointer)
			return true
		}

		// Iterate over the values of a map, in alphabetic order.
		iterMap = func(v reflect.Value, pathPrefix string) bool {
			if v.IsNil() {
				return true
			}
			originalKeys := v.MapKeys()
			stringKeys := make([]string, len(originalKeys))
			for ii, k := range originalKeys {
				switch k.Kind() {
				case reflect.String:
					stringKeys[ii] = k.String()
				case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
					stringKeys[ii] = fmt.Sprintf("%d", k.Int())
				case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
					stringKeys[ii] = fmt.Sprintf("%d", k.Uint())
				default:
					return yield(PathAndVariable{}, fmt.Errorf("map key type %v not supported at path %q", k.Type(), pathPrefix))
				}
			}
			// Sort both keys at the same time, by the stringKeys.
			indices := xslices.Iota(0, len(originalKeys))
			slices.SortFunc(indices, func(i, j int) int { return cmp.Compare(stringKeys[i], stringKeys[j]) })
			for _, index := range indices {
				value := v.MapIndex(originalKeys[index])
				newPath := fmt.Sprintf("%s[%s]", pathPrefix, stringKeys[index])
				if !iterValue(value, newPath) {
					return false
				}
			}
			return true
		}

		// Iterate over the values of a slice or array.
		iterSliceOrArray = func(v reflect.Value, pathPrefix string) bool {
			for ii := 0; ii < v.Len(); ii++ {
				newPath := fmt.Sprintf("%s[%d]", pathPrefix, ii)
				if !iterValue(v.Index(ii), newPath) {
					return false
				}
			}
			return true
		}

		// Iterate over the values of a structure in alphabetic order.
		iterStruct = func(v reflect.Value, path string) bool {
			t := v.Type()
			numFields := t.NumField()
			for fieldIdx := range numFields {
				field := v.Field(fieldIdx)
				newPath := path
				if path != "" {
					newPath += "."
				}
				newPath += t.Field(fieldIdx).Name
				if !iterValue(field, newPath) {
					return false
				}
			}
			return true
		}

		// Iterate over one value.
		iterValue = func(v reflect.Value, path string) bool {
			switch v.Kind() {
			case reflect.Ptr:
				if v.IsNil() {
					return true
				}
				if !checkPtr(v.Pointer()) {
					return true
				}
				elem := v.Elem()
				if elem.Type().Name() == "Variable" && elem.Type().PkgPath() == modelsPkgPath {
					return yield(PathAndVariable{Path: path, Variable: v.Interface().(*Variable)}, nil)
				}
				return iterValue(elem, path)

			case reflect.Interface:
				if v.IsNil() {
					return true
				}
				return iterValue(v.Elem(), path)

			case reflect.Struct:
				if v.Type().Name() == "Variable" && v.Type().PkgPath() == modelsPkgPath {
					return yield(PathAndVariable{Path: path, Variable: nil},
						errors.Errorf("model has Variable passed by value, at path %q", path))
				}
				return iterStruct(v, path)

			case reflect.Slice, reflect.Array:
				return iterSliceOrArray(v, path)

			case reflect.Map:
				return iterMap(v, path)

			default:
				return true
			}
		}

		// Start recursive descent from root.
		iterValue(reflect.ValueOf(model), "")
	}
}
