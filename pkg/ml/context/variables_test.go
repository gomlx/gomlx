package context_test

import (
	"fmt"
	"runtime"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	. "github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func TestVariable_CloneToContext(t *testing.T) {
	value := []float32{3, 5, 7, 11, 13}
	ctx0 := New()
	v0x := ctx0.In("a").In("b").VariableWithValue("x", value)
	// Uninitialized variable:
	v0y := ctx0.In("a").In("b").VariableWithShape("y", shapes.Make(dtypes.Int8, 2, 3, 4))

	ctx1 := New()
	v1x, err := v0x.CloneToContext(ctx1)
	require.NoError(t, err)
	fmt.Printf("Cloned variable %q: %s\n", v1x.ScopeAndName(), v1x.MustValue())
	v1y, err := v0y.CloneToContext(ctx1)
	require.NoError(t, err)
	_, err = v1y.Value()
	require.Error(t, err, "/a/b/y was created uninitialized, it should have no value")
	fmt.Printf("Cloned variable %q, with no value, shape=%s\n", v1y.ScopeAndName(), v1y.Shape())

	// Check the new variable has the right name, scope and was properly inserted in to the new context.
	if v1x.ScopeAndName() != "/a/b/x" {
		fmt.Printf("Unexpeted scope/name of clone variable: %q\n", v1x.ScopeAndName())
		t.Fail()
	}
	require.Equal(t, 2, ctx1.NumVariables())
	require.Equal(t, v1x, ctx1.GetVariableByScopeAndName("/a/b", "x"))
	require.Equal(t, v1y, ctx1.GetVariableByScopeAndName("/a/b", "y"))

	// Check the new variable value is independent of the old one.
	ctx0 = nil
	v0x.MustValue().MustFinalizeAll()
	for range 5 {
		runtime.GC()
	}
	require.Equal(t, value, tensors.MustCopyFlatData[float32](v1x.MustValue()))
}
