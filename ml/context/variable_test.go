package context_test

import (
	"fmt"
	. "github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/stretchr/testify/require"
	"runtime"
	"testing"
)

func TestVariable_CloneToContext(t *testing.T) {
	value := []float32{3, 5, 7, 11, 13}
	ctx0 := New()
	v0 := ctx0.In("a").In("b").VariableWithValue("x", value)

	ctx1 := New()
	v1 := v0.CloneToContext(ctx1)
	fmt.Printf("Cloned variable %q: %s\n", v1.ScopeAndName(), v1.Value())

	// Check the new variable has the right name, scope and was properly inserted in to the new context.
	if v1.ScopeAndName() != "/a/b/x" {
		fmt.Printf("Unexpeted scope/name of clone variable: %q\n", v1.ScopeAndName())
		t.Fail()
	}
	require.Equal(t, 1, ctx1.NumVariables())
	require.Equal(t, v1, ctx1.GetVariableByScopeAndName("/a/b", "x"))

	// Check the new variable value is independent of the old one.
	ctx0 = nil
	v0.Value().FinalizeAll()
	for _ = range 5 {
		runtime.GC()
	}
	require.Equal(t, value, tensors.CopyFlatData[float32](v1.Value()))
}
