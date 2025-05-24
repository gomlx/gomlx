package vnn

import (
	"fmt"
	"math"
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

// TestLinearLayer checks that the linear layer of the VNN is equivariant to rotation.
func TestLinearLayer(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.RngStateFromSeed(42)
	y0 := context.ExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
		pi2 := math.Pi * 2.0

		// Random inputs and rotations:
		// - Inputs has extra batch dimensions (1, 1, 1), we are also testing that they are preserved.
		input := ctx.RandomUniform(g, shapes.Make(dtypes.Float64, 1, 1, 1, 10, 3))
		roll := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		pitch := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		yaw := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)

		// Linear function: fix seed so we always have the same values.
		ctx.SetParam(initializers.ParamInitialSeed, 42)
		ctx = ctx.Checked(false).WithInitializer(initializers.HeFn(ctx))
		linearFn := func(x *Node) *Node {
			return New(ctx, x, 2).
				NumHiddenLayers(0, 0).
				Activation("").Regularizer(nil).Done()
		}

		// Outputs: out1 rotates after linear transformation, out2 rotates before linear transformation.
		out1 := RotateOnOrigin(linearFn(input), roll, pitch, yaw)
		require.NoError(t, out1.Shape().CheckDims(1, 1, 1, 2, 3))
		out2 := linearFn(RotateOnOrigin(input, roll, pitch, yaw))
		require.NoError(t, out2.Shape().CheckDims(1, 1, 1, 2, 3))
		diff := Abs(Sub(out1, out2))
		diff.SetLogged("Difference of rotation before/after linear transformation")
		return ReduceAllMean(diff)
	})
	fmt.Printf("\tMean absolute difference: %s\n", y0.GoStr())
	require.Less(t, tensors.ToScalar[float64](y0), 1e-3)
}

// TestRelu checks that the Relu activation with a learned projection is equivariant to rotation.
func TestRelu(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.RngStateFromSeed(42)
	outputs := context.ExecOnceN(backend, ctx, func(ctx *context.Context, g *Graph) []*Node {
		pi2 := math.Pi * 2.0

		// Random inputs and rotations:
		// - Inputs has extra batch dimensions (1, 1, 1), we are also testing that they are preserved.
		input := ctx.RandomUniform(g, shapes.Make(dtypes.Float64, 20, 2, 3))
		roll := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		pitch := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		yaw := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)

		// Linear function: fix seed so we always have the same values.
		ctx.SetParam(initializers.ParamInitialSeed, 42)
		ctx = ctx.Checked(false).WithInitializer(initializers.HeFn(ctx))

		// Outputs: out1 rotates after linear transformation, out2 rotates before linear transformation.
		out1 := Relu(ctx, input)
		diffRelu := ReduceAllMax(Abs(Sub(input, out1)))
		out1 = RotateOnOrigin(out1, roll, pitch, yaw)
		require.NoError(t, out1.Shape().CheckDims(20, 2, 3))
		out2 := Relu(ctx, RotateOnOrigin(input, roll, pitch, yaw))
		require.NoError(t, out2.Shape().CheckDims(20, 2, 3))
		diff := Abs(Sub(out1, out2))
		return []*Node{diffRelu, ReduceAllMean(diff)}
	})
	reluDiff, rotDiff := outputs[0], outputs[1]
	fmt.Printf("\tBefore/after relu abs difference: %s\n", reluDiff.GoStr())
	fmt.Printf("\tRotation (before/after relu) abs difference: %s\n", rotDiff.GoStr())
	require.Greater(t, tensors.ToScalar[float64](reluDiff), 1e-3)
	require.Less(t, tensors.ToScalar[float64](rotDiff), 1e-3)
}

// TestLayerNormalization checks that the LayerNormalization normalizes properly -- mean close to
// the origin -- and that it is equivariant to rotation.
func TestLayerNormalization(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.RngStateFromSeed(42)
	outputs := context.ExecOnceN(backend, ctx, func(ctx *context.Context, g *Graph) []*Node {
		pi2 := math.Pi * 2.0

		// Random inputs and rotations:
		// - Inputs has extra batch dimensions (1, 1, 1), we are also testing that they are preserved.
		input := ctx.RandomUniform(g, shapes.Make(dtypes.Float64, 1, 1_000, 3))
		roll := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		pitch := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		yaw := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)

		// Linear function: fix seed so we always have the same values.
		ctx.SetParam(initializers.ParamInitialSeed, 42)
		ctx = ctx.Checked(false).WithInitializer(initializers.HeFn(ctx))

		// Outputs: out1 rotates after linear transformation, out2 rotates before linear transformation.

		epsilon := 1.0e-8
		out1 := RotateOnOrigin(LayerNormalization(input, epsilon), roll, pitch, yaw)
		out1Mean := ReduceMean(out1, 1)
		meanAbsDiff := ReduceAllSum(Abs(out1Mean))
		require.True(t, out1.Shape().Equal(input.Shape()))
		out2 := LayerNormalization(RotateOnOrigin(input, roll, pitch, yaw), epsilon)
		require.True(t, out2.Shape().Equal(input.Shape()))
		diff := Abs(Sub(out1, out2))
		return []*Node{meanAbsDiff, ReduceAllMean(diff)}
	})
	meanAbsDiff, rotDiff := outputs[0], outputs[1]
	fmt.Printf("\tMean diff to origin: %s\n", meanAbsDiff.GoStr())
	fmt.Printf("\tRotation (before/after relu) abs difference: %s\n", rotDiff.GoStr())
	require.Less(t, tensors.ToScalar[float64](meanAbsDiff), 1e-3)
	require.Less(t, tensors.ToScalar[float64](rotDiff), 1e-3)
}

// TestVNN checks that a fully configured VNN is SO(3) equivariant for rotations.
func TestVNN(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ctx := context.New()
	ctx.RngStateFromSeed(42)
	rotDiff := context.ExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
		pi2 := math.Pi * 2.0

		// Random inputs and rotations:
		// - Inputs has extra batch dimensions (1, 1, 1), we are also testing that they are preserved.
		input := ctx.RandomUniform(g, shapes.Make(dtypes.Float64, 2, 3, 20, 3))
		roll := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		pitch := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)
		yaw := MulScalar(ctx.RandomUniform(g, shapes.Make(dtypes.Float64)), pi2)

		// vnn layer: fix seed so we always have the same values.
		ctx.SetParam(initializers.ParamInitialSeed, 42)
		ctx = ctx.Checked(false).WithInitializer(initializers.HeFn(ctx))
		vnnFn := func(x *Node) *Node {
			return New(ctx, x, 5).
				NumHiddenLayers(3, 10).
				Activation("relu").
				Normalization("layer").
				Regularizer(nil).
				Done()
		}

		// Outputs: out1 rotates after linear transformation, out2 rotates before linear transformation.
		out1 := RotateOnOrigin(vnnFn(input), roll, pitch, yaw)
		require.NoError(t, out1.Shape().CheckDims(2, 3, 5, 3))
		out2 := vnnFn(RotateOnOrigin(input, roll, pitch, yaw))
		require.NoError(t, out2.Shape().CheckDims(2, 3, 5, 3))
		diff := Abs(Sub(out1, out2))
		return ReduceAllMean(diff)
	})
	fmt.Printf("\tRotation (before/after relu) abs difference: %s\n", rotDiff.GoStr())
	require.Less(t, tensors.ToScalar[float64](rotDiff), 1e-3)
}
