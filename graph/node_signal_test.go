package graph_test

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"math"
	"testing"
)

func TestFFT(t *testing.T) {
	graphtest.RunTestGraphFn(t, "FFT and InverseFFT", func(g *Graph) (inputs, outputs []*Node) {
		const numPoints = 1000
		x := Iota(g, shapes.Make(shapes.F32, numPoints), 0)
		const numPeriods = 10
		y := Sin(MulScalar(x, 2*math.Pi*numPeriods/numPoints))
		y.AssertDims(numPoints)
		inputs = []*Node{y}
		yC := ConvertType(y, shapes.Complex128)
		yC.AssertDims(numPoints)
		fft := FFT(yC)
		yHat := InverseFFT(fft)
		diff := ReduceAllSum(Abs(Sub(yC, yHat)))
		outputs = []*Node{Real(diff), Imag(diff)}
		return
	}, []any{
		0.0,
		0.0,
	}, 1.0)

	graphtest.RunTestGraphFn(t, "RealFFT and InverseRealFFT", func(g *Graph) (inputs, outputs []*Node) {
		const batchDim = 2
		const numPoints = 1000
		x := Iota(g, shapes.Make(shapes.F32, batchDim, numPoints), 0)
		const numPeriods = 10
		y := Sin(MulScalar(x, 2*math.Pi*numPeriods/numPoints))
		y.AssertDims(batchDim, numPoints)
		inputs = []*Node{y}
		fft := RealFFT(y)
		fft.AssertDims(batchDim, numPoints/2+1)
		yHat := InverseRealFFT(fft)
		yHat.AssertDims(batchDim, numPoints)
		diff := ReduceAllSum(Abs(Sub(y, yHat)))
		outputs = []*Node{diff, diff}
		return
	}, []any{
		float32(0.0),
		float32(0.0),
	}, 1.0)
}
