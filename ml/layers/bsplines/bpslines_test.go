package bsplines

import (
	"fmt"
	"github.com/gomlx/bsplines"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math/rand/v2"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

func TestEvaluateSimple(t *testing.T) {
	const (
		epsilon       = 1e-4
		numTestPoints = 30
		margin        = 0.1
	)

	controlPoints := [][]float64{
		{1.0, 0.7, -0.7, -1.0, -0.7, 0.7, 1.0, 0.7},
		{1.0, 0.7, -0.7, -1.0, -0.7, 0.7, 1.0, 0.7},
	}
	b := bsplines.NewRegular(3, len(controlPoints[0])).WithExtrapolation(bsplines.ExtrapolateLinear)

	x := make([][]float64, numTestPoints)
	want := make([][]float64, numTestPoints)
	for ii := range numTestPoints {
		x[ii] = make([]float64, 2)
		x[ii][0] = (1.0+2.0*margin)*float64(ii)/(numTestPoints-1)/2 - margin // First input takes the first half, from -margin to 0.5
		x[ii][1] = x[ii][0] + 0.5 + margin                                   // Second half, from 0.5 to 1.0+margin
		want[ii] = make([]float64, 2)
		for cc, control := range controlPoints {
			b = b.WithControlPoints(control)
			want[ii][cc] = b.Evaluate(x[ii][cc])
		}
	}

	backend := graphtest.BuildTestBackend()
	exec := NewExec(backend, func(x, controlPoints *Node) *Node {
		values := Evaluate(b,
			x,
			InsertAxes(controlPoints, 1))
		fmt.Printf("output.shape=%s\n", values.Shape())
		return Reshape(values, -1, 2)
	})
	got := exec.Call(x, controlPoints)[0].Value().([][]float64)
	fmt.Printf("\nB-spline(%v):\n> want=%v\n>  got=%v\n\n", x, want, got)
	for ii := range numTestPoints {
		assert.InDeltaSlicef(t, want[ii], got[ii], epsilon, "Got wrong value for example %d: want=%v, got=%v", ii, want[ii], got[ii])
	}
}

func TestEvaluateBatchMultiInputsAndOutputs(t *testing.T) {
	const (
		// Choose some unique prime numbers, so shapes won't get mixed up.
		batchSize        = 11
		numInputs        = 3
		numOutputs       = 5
		numControlPoints = 7
		margin           = 0.1 // So we get some extrapolated points.
	)
	b := bsplines.NewRegular(0, numControlPoints)
	rng := rand.New(rand.NewPCG(42, 42))

	inputs := make([][]float32, batchSize)
	for ee := range batchSize {
		inputs[ee] = make([]float32, numInputs)
		for ii := range numInputs {
			inputs[ee][ii] = rng.Float32()*(1+2*margin) - margin
		}
	}

	controlPoints := make([][][]float64, numInputs) // Need to be float64 to use the normal B-spline implementation.
	for ii := range numInputs {
		controlPoints[ii] = make([][]float64, numOutputs)
		for oo := range numOutputs {
			controlPoints[ii][oo] = make([]float64, numControlPoints)
			for cc := range numControlPoints {
				controlPoints[ii][oo][cc] = rng.NormFloat64()
			}
		}
	}

	want := make([][][]float32, batchSize)
	for ee := range batchSize {
		want[ee] = make([][]float32, numOutputs)
		for oo := range numOutputs {
			want[ee][oo] = make([]float32, numInputs)
			for ii := range numInputs {
				b.WithControlPoints(controlPoints[ii][oo])
				want[ee][oo][ii] = float32(b.Evaluate(float64(inputs[ee][ii])))
			}
		}
	}
	fmt.Printf("\ninput=%v\n\nwant=%v\n\n", inputs, want)

	graphtest.RunTestGraphFn(t, "B-spline batched, multi-inputs, multi-outputs", func(g *Graph) ([]*Node, []*Node) {
		nodeInputs := Const(g, inputs)
		nodeControlPoints := ConvertDType(Const(g, controlPoints), nodeInputs.DType())
		outputs := Evaluate(b, nodeInputs, nodeControlPoints)
		return []*Node{nodeInputs, nodeControlPoints}, []*Node{outputs}
	}, []any{want},
		1e-4)
}

func TestExtrapolation(t *testing.T) {
	controlPoints := []float64{1.0, 0.7, -0.7, -1.0, -0.7, 0.7, 1.0, 0.7}
	b := bsplines.NewRegular(3, len(controlPoints)).WithControlPoints(controlPoints)

	backend := graphtest.BuildTestBackend()

	xs := []float64{-0.1, 0.0, 1.0, 1.1}
	evalFn := func() []float64 {
		got := make([]float64, len(xs))
		exec := NewExec(backend, func(x, controlPoints *Node) *Node {
			return Evaluate(b, x, controlPoints)
		})
		for ii, x := range xs {
			got[ii] = exec.Call(x, controlPoints)[0].Value().(float64)
		}
		return got
	}

	for _, extrapolation := range []bsplines.ExtrapolationType{bsplines.ExtrapolateZero, bsplines.ExtrapolateConstant, bsplines.ExtrapolateLinear} {
		fmt.Printf("%s:\n", extrapolation)
		b.WithExtrapolation(extrapolation)
		want := make([]float64, len(xs))
		for ii, x := range xs {
			want[ii] = b.Evaluate(x)
		}
		got := evalFn()
		require.InDeltaSlicef(t, want, got, 1e-4, "Testing %s", extrapolation)
		fmt.Printf("\tOk.\n")
	}
}
