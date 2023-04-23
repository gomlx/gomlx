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

package nest

import (
	"github.com/stretchr/testify/assert"
	"math"
	"testing"
)

func assertEnumeration[T any](t *testing.T, n *Nest[T], expectedPaths []string, expectedValues []T) {
	ii := 0
	n.EnumerateWithPath(func(path string, value T) error {
		if ii > len(expectedPaths) {
			t.Fatalf("Enumeration() returned too many elements: expected %d elements, but got element %d: %s %v", len(expectedPaths), ii, path, value)
		}
		assert.Equalf(t, expectedPaths[ii], path, "Unexpected path %s for element %d", path, ii)
		assert.Equalf(t, expectedValues[ii], value, "Unexpected value %v for element %d", value, ii)
		ii++
		return nil
	})
	if ii < len(expectedPaths) {
		t.Fatalf("Enumeration() returned too few elements: expected %d elements, but got only %d", len(expectedPaths), ii)
	}
}

func TestMap(t *testing.T) {
	n := Map(map[string]int{"a": 10, "b": 20})
	assert.True(t, n.IsMap())
	assert.False(t, n.IsValue())
	assert.False(t, n.IsSlice())
	assert.Equal(t, 20, n.Map()["b"])
	assert.Panics(t, func() { n.Value() })
	assert.Panics(t, func() { n.Slice() })
	assertEnumeration(t, n, []string{">a:", ">b:"}, []int{10, 20})
	assert.Equal(t, n.Flatten(), []int{10, 20})
	n2 := Unflatten(n, []float32{10.0, 20.0})
	assert.Equal(t, float32(20), n2.Map()["b"])
}

func TestSlice(t *testing.T) {
	n := Slice(10, 20)
	assert.False(t, n.IsMap())
	assert.False(t, n.IsValue())
	assert.True(t, n.IsSlice())
	assert.Equal(t, 20, n.Slice()[1])
	assert.Panics(t, func() { n.Map() })
	assert.Panics(t, func() { n.Value() })
	assertEnumeration(t, n, []string{"[0]:", "[1]:"}, []int{10, 20})
	assert.Equal(t, n.Flatten(), []int{10, 20})
	n2 := Unflatten(n, []float32{10.0, 20.0})
	assert.Equal(t, float32(20), n2.Slice()[1])
}

func TestValue(t *testing.T) {
	n := Value(math.Pi)
	assert.False(t, n.IsMap())
	assert.True(t, n.IsValue())
	assert.False(t, n.IsSlice())
	assert.Equal(t, math.Pi, n.Value())
	assert.Panics(t, func() { n.Map() })
	assert.Panics(t, func() { n.Slice() })
	assertEnumeration(t, n, []string{":"}, []float64{math.Pi})
	assert.Equal(t, n.Flatten(), []float64{math.Pi})
	n2 := Unflatten(n, []string{"pi"})
	assert.Equal(t, "pi", n2.Value())
}
