// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package layers

import (
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors/images"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

func TestConvolution(t *testing.T) {
	testutil.TestOfficialBackends(t, func(t *testing.T, backend compute.Backend) {
		scope := model.NewStore()
		defer scope.Finalize()

		t.Run("2D-PadSame", func(t *testing.T) {
			gotT := model.MustExecOnce(backend, scope, func(scope *model.Scope, g *Graph) *Node {
				x := Ones(g, shapes.Make(dtypes.F32, 5, 4, 4, 3))
				scope = scope.In("%s", model.BasePath(t.Name()))
				conv := Convolution(scope, x).
					Channels(32).
					KernelSize(3).
					PadSame().
					Strides(1).Done()
				return conv
			})
			require.NoError(t, gotT.Shape().Check(dtypes.F32, 5, 4, 4, 32))
		})

		t.Run("3D-Strides", func(t *testing.T) {
			gotT := model.MustExecOnce(backend, scope, func(scope *model.Scope, g *Graph) *Node {
				x := Ones(g, shapes.Make(dtypes.F32, 5, 3, 8, 8, 8))
				scope = scope.In("%s", model.BasePath(t.Name()))
				conv := Convolution(scope, x).
					ChannelsAxis(images.ChannelsFirst).
					Channels(32).
					KernelSizePerAxis(3, 2, 2).
					NoPadding().
					StridePerAxis(2, 3, 4).Done()
				return conv
			})
			require.NoError(t, gotT.Shape().Check(dtypes.F32, 5, 32, 3, 3, 2))
		})
	})
}
