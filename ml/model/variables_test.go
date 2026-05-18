// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model_test

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/tensors"
	. "github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/require"
)

func TestVariable_CloneToContext(t *testing.T) {
	value := []float32{3, 5, 7, 11, 13}
	store0 := NewStore()
	root0 := store0.RootScope()
	sA := root0.In("a")
	v0x := sA.In("b").VariableWithValue("x", value)
	// Uninitialized variable:
	v0y := sA.Shared("b").VariableWithShape("y", shapes.Make(dtypes.Int8, 2, 3, 4))

	store1 := NewStore()
	v1x, err := v0x.CloneToStore(store1)
	require.NoError(t, err)
	fmt.Printf("Cloned variable %q: %s\n", v1x.Path(), v1x.MustValue())
	v1y, err := v0y.CloneToStore(store1)
	require.NoError(t, err)
	_, err = v1y.Value()
	require.Error(t, err, "/a/b/y was created uninitialized, it should have no value")
	fmt.Printf("Cloned variable %q, with no value, shape=%s\n", v1y.Path(), v1y.Shape())

	// Check the new variable has the right name, scope and was properly inserted in to the new model.
	if v1x.Path() != "/a/b/x" {
		fmt.Printf("Unexpeted scope/name of clone variable: %q\n", v1x.Path())
		t.Fail()
	}
	require.Equal(t, 2, store1.NumVariables())
	require.Equal(t, v1x, store1.GetVariable("/a/b/x"))
	require.Equal(t, v1y, store1.GetVariable("/a/b/y"))

	// Check the new variable value is independent of the old one.
	store0 = nil
	v0x.MustValue().MustFinalizeAll()
	for range 5 {
		runtime.GC()
	}
	require.Equal(t, value, tensors.MustCopyFlatData[float32](v1x.MustValue()))
}
