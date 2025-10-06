<<<<<<< HEAD:pkg/ml/variable/variablesiter_test.go
<<<<<<<< HEAD:pkg/ml/model/variableiter_test.go
package model
========
package variable
>>>>>>>> b1bebb70fa399f07e42a4df4a8b4735919894d8e:pkg/ml/variable/variablesiter_test.go
=======
package models
>>>>>>> parent of b1bebb7 (Middle of refactoring: package models -> packages exec and variable):pkg/ml/models/variablesiter_test.go

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestIterVariables(t *testing.T) {
	type SubStruct struct {
		V1 *Variable
		V2 *Variable
	}
	type TestStruct struct {
		StringMap map[string]*Variable
		IntMap    map[int]*Variable
		Array     [2]*Variable
		Slice     []*Variable
		Sub       *SubStruct
	}

	// Create variables and structure.
	v1 := &Variable{}
	v2 := &Variable{}
	v3 := &Variable{}
	v4 := &Variable{}
	v5 := &Variable{}
	v6 := &Variable{}
	v7 := &Variable{}
	v8 := &Variable{}

	test := &TestStruct{
		StringMap: map[string]*Variable{"a": v1, "b": v2},
		IntMap:    map[int]*Variable{1: v3, 2: v4},
		Array:     [2]*Variable{v5, v6},
		Slice:     []*Variable{v7},
		Sub:       &SubStruct{V1: v8},
	}

	// Collect all paths and variables.
	var got []PathAndVariable
	for pv := range IterVariables(test) {
		got = append(got, pv)
	}

	// Expected paths and variables -- sorted in the expected order.
	want := []PathAndVariable{
		{Path: "StringMap[a]", Variable: v1},
		{Path: "StringMap[b]", Variable: v2},
		{Path: "IntMap[1]", Variable: v3},
		{Path: "IntMap[2]", Variable: v4},
		{Path: "Array[0]", Variable: v5},
		{Path: "Array[1]", Variable: v6},
		{Path: "Slice[0]", Variable: v7},
		{Path: "Sub.V1", Variable: v8},
	}

	require.Equal(t, len(want), len(got), "different number of variables found")
	for ii := range want {
		require.Equal(t, want[ii].Path, got[ii].Path,
			"path at position %d different: want=%q got=%q", ii, want[ii].Path, got[ii].Path)
		require.Equal(t, want[ii].Variable, got[ii].Variable,
			"variable at position %d different", ii)
	}
}
