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

func TestGradientFFT(t *testing.T) {
	// Create a small sine curve using 11 numbers (scaled from 0 to 2π),
	// calculate the FFT, takes the gradient of that w.r.t input.
	// Also checks that the InverseFFT takes it back to the input.
	graphtest.RunTestGraphFn(t, "GradientFFT", func(g *Graph) (inputs, outputs []*Node) {
		x := Iota(g, shapes.Make(shapes.F64, 11), 0)
		x = MulScalar(x, 2.0*math.Pi/11)
		x = Sin(x)
		x = ConvertType(x, shapes.Complex128)
		fft := FFT(x)
		output := ReduceAllSum(Abs(fft))
		argmax := ArgMax(Abs(fft), -1)
		inv := InverseFFT(fft)
		diff := ReduceAllSum(Abs(Sub(inv, x)))
		grad := Gradient(output, x)[0]
		return []*Node{x}, []*Node{output, argmax, diff, grad}
	}, []any{
		11.0, int32(10), 0.0,
		// Notice that GPUs FFT implementation yields significantly different results, so this only works on
		// the CPU (Eigen presumably) implementation. The results matches results yielded by similar code in
		// TensorFlow (notice that if using complex64 the numbers differ).
		[]complex128{
			-0.7126987103881084 + 2.050415210607949i,
			-0.34708456852657743 + 2.1129871487498937i,
			6.221747438979876 - 0.7815639565770847i,
			1.2212951935321779 - 0.5731899738198121i,
			2.737096425081983 + 0.5834007264215436i,
			2.8156998840012886 - 2.042120467795749i,
			0.7909217660400658 + 2.4657154202830442i,
			-4.848589784434233 + 2.175014774879141i,
			2.2094212306941765 - 0.2007046968306172i,
			-1.3018177490725729 - 2.3496106014958604i,
			1.815039114738962 - 0.5046736716278869i},
	}, 1e-3)

	// Similar to previous test, same input, but now we check the gradient of the diff of the reversed sequence
	// w.r.t the FFT values.
	// It should be close to 0, since the diff should be close to 0 -- it would be if the FFT were perfect.
	graphtest.RunTestGraphFn(t, "GradientFFT", func(g *Graph) (inputs, outputs []*Node) {
		x := Iota(g, shapes.Make(shapes.F64, 11), 0)
		x = MulScalar(x, 2.0*math.Pi/11)
		x = Sin(x)
		x = ConvertType(x, shapes.Complex128)
		fft := FFT(x)
		sumFft := ReduceAllSum(Abs(fft))
		argmax := ArgMax(Abs(fft), -1)
		inv := InverseFFT(fft)
		diff := ReduceAllSum(Abs(Sub(inv, x)))
		grad := Gradient(diff, inv)[0]
		return []*Node{x}, []*Node{sumFft, argmax, diff, grad}
	}, []any{
		11.0, int32(10), 0.0,
		// Notice that GPUs FFT implementation yields significantly different results, so this only works on
		// the CPU (Eigen presumably) implementation. The results matches results yielded by similar code in
		// TensorFlow (notice that if using complex64 the numbers differ).
		[]complex128{
			0.21951219512195125 + 0.9756097560975611i,
			-0.043097175436008814 + 0.9990708851074771i,
			0.8087360843031886 + 0.5881716976750463i,
			0.11383845836148958 + 0.9934992729729999i,
			0.15968884391337387 + 0.9871673987372204i,
			0.656144370283719 - 0.7546353856962857i,
			-0.9991847547814338 + 0.04037110120329026i,
			-0.9549934880178522 - 0.29662676521766623i,
			-0.6288692494311636 - 0.7775110720239841i,
			0.13621834047872816 - 0.9906788398452958i,
			0.9680364763016536 + 0.2508094506781557i},
	}, 1e-3)
}
