package lstm

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

func TestLSTM(t *testing.T) {
	batchSize := 2
	seqLen := 4
	featuresSize := 2
	hiddenSize := 3
	dtype := dtypes.Float32

	// The batch has 2 examples: the second example is the reverse of the first.
	// So when we reverse the direction of the LSTM we expect the results to just switch.
	hiddenForward := []float32{0.0369643, 0.09801698, 0.21440339}
	hiddenReverse := []float32{0.07546426, 0.1404648, 0.23588456}
	cellForward := []float32{0.16458873, 0.31134608, 0.53026175}
	cellReverse := []float32{0.23274383, 0.37023422, 0.5559477}
	want := [3][]any{
		{ // Forward
			// Last hidden state: (Float32)[numDirections=1, batchSize=2, featuresSize=3]
			[][][]float32{{hiddenForward, hiddenReverse}},
			// Last cell state: (Float32)[numDirections=1, batchSize=2, featuresSize=3]
			[][][]float32{{cellForward, cellReverse}},
		},
		{ // Reverse
			// Last hidden state: (Float32)[numDirections=1, batchSize=2, featuresSize=3]
			[][][]float32{{hiddenReverse, hiddenForward}},
			// Last cell state: (Float32)[numDirections=1, batchSize=2, featuresSize=3]
			[][][]float32{{cellReverse, cellForward}},
		},
		{ // Bidirectional: forward results and then results results.
			// Last hidden state: (Float32)[numDirections=2, batchSize=2, featuresSize=3]
			[][][]float32{{hiddenForward, hiddenReverse}, {hiddenReverse, hiddenForward}},
			[][][]float32{{cellForward, cellReverse}, {cellReverse, cellForward}},
		},
	}

	for dirIdx, dir := range []DirectionType{DirForward, DirReverse, DirBidirectional} {
		numDirections := 1
		if dir == DirBidirectional {
			numDirections = 2
		}
		graphtest.RunTestGraphFn(t, fmt.Sprintf("LSTM: %s", dir),
			func(g *Graph) (inputs, outputs []*Node) {
				// x shaped [batchSize, seqLen, featuresSize]
				// We create the first example first, and make the second example the reverse of the first, so
				// we can test that the directions work.
				x := IotaFull(g, shapes.Make(dtype, seqLen, featuresSize))
				x = MulScalar(OnePlus(x), 0.1)
				x = Stack([]*Node{x, Reverse(x, 0)}, 0) // Create batch axes, size = 2.

				// Create values from -1.0... to 1.0.
				initializeFn := func(dims ...int) *Node {
					v := IotaFull(g, shapes.Make(dtype, dims...))
					v = MulScalar(v, 2.0/float64(v.Shape().Size()-1))
					v = AddScalar(v, -1)
					if numDirections == 2 {
						// Same weights for both directions.
						v = Concatenate([]*Node{v, v}, 0)
					}
					return v
				}
				inputsW := initializeFn(1, 4, hiddenSize, featuresSize)
				recurrentW := initializeFn(1, 4, hiddenSize, hiddenSize)
				biasW := initializeFn(1, 8, hiddenSize)

				// Run LSTM:
				lstmLayer := NewWithWeights(x, inputsW, recurrentW, biasW, nil).Direction(dir)
				allHiddenStates, lastHiddenState, lastCellState := lstmLayer.Done()
				allHiddenStates.AssertDims(seqLen, numDirections, batchSize, hiddenSize)
				lastHiddenState.AssertDims(numDirections, batchSize, hiddenSize)
				lastCellState.AssertDims(numDirections, batchSize, hiddenSize)

				inputs = []*Node{x}
				outputs = []*Node{lastHiddenState, lastCellState}
				return
			}, want[dirIdx], 1e-4)
	}
}
