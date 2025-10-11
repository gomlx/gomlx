package layers

import (
	"path"
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors/images"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"
)

func TestConvolution(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()

	t.Run("2D-PadSame", func(t *testing.T) {
		gotT := context.MustExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
			x := Ones(g, shapes.Make(dtypes.F32, 5, 4, 4, 3))
			ctx = ctx.In(path.Base(t.Name()))
			conv := Convolution(ctx, x).
				Channels(32).
				KernelSize(3).
				PadSame().
				Strides(1).Done()
			return conv
		})
		require.NoError(t, gotT.Shape().Check(dtypes.F32, 5, 4, 4, 32))
	})

	t.Run("3D-Strides", func(t *testing.T) {
		gotT := context.MustExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
			x := Ones(g, shapes.Make(dtypes.F32, 5, 3, 8, 8, 8))
			ctx = ctx.In(path.Base(t.Name()))
			conv := Convolution(ctx, x).
				ChannelsAxis(images.ChannelsFirst).
				Channels(32).
				KernelSizePerAxis(3, 2, 2).
				NoPadding().
				StridePerAxis(2, 3, 4).Done()
			return conv
		})
		require.NoError(t, gotT.Shape().Check(dtypes.F32, 5, 32, 3, 3, 2))
	})
}
