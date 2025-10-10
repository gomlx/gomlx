package layers

import (
	"fmt"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/stretchr/testify/require"

	"testing"
)

func TestRMSNorm(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, x *Node) *Node {
		return RMSNorm(ctx, x).
			WithNormalizationAxes(-1, -2).
			Done()
	})

	x := [][][]float32{
		{{3, 4}, {5, 6}, {7, 8}},
		{{3 * 2, 4 * 2}, {5 * 2, 6 * 2}, {7 * 2, 8 * 2}},
	}
	got := exec.Call(x)[0]
	require.NoError(t, got.Shape().CheckDims(2, 3, 2), "shape should not have changed")
	want := [][][]float32{{
		{0.52091914, 0.6945589}, {0.86819863, 1.0418383}, {1.2154781, 1.3891178},
	}, {
		{0.52091914, 0.6945589}, {0.86819863, 1.0418383}, {1.2154781, 1.3891178},
	}}
	fmt.Printf("RMS(%v) = %s\n", x, got.GoStr())
	require.Equal(t, want, got.Value())

	scaleVar := ctx.GetVariableByScopeAndName("/rms_norm", "scale")
	require.NotNil(t, scaleVar)
	require.NoError(t, scaleVar.Shape().CheckDims(3, 2))
}
