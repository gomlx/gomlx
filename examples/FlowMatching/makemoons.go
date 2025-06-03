package fm

import (
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"math"
)

var (
	backend = backends.MustNew()
	DType   = dtypes.Float32
)

// MakeMoons returns a collection of n points sampled from two interleaved half circles.
// This is a toy dataset to visualize clustering and classification algorithms.
//
// Modeled after scikit-learn make_moons function.
//
// It returns a tensor of the given shaped [n, 2].
func MakeMoons(ctx *context.Context, g *Graph, n int) *Node {
	angles := ctx.RandomUniform(g, shapes.Make(DType, n))
	angles = MulScalar(angles, math.Pi)
	outerMoonX := Cos(angles)
	outerMoonY := Sin(angles)
	innerMoonX := OneMinus(outerMoonX)
	innerMoonY := AddScalar(OneMinus(outerMoonY), -0.5)

	coinFlip := ctx.RandomUniform(g, shapes.Make(DType, n))
	coinFlip = GreaterThan(AddScalar(coinFlip, -0.5), ScalarZero(g, DType))
	xs := Where(coinFlip, innerMoonX, outerMoonX)
	ys := Where(coinFlip, innerMoonY, outerMoonY)
	return Stack([]*Node{xs, ys}, -1)
}
