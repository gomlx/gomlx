// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package lstm provides a minimal "Long Short-Term Memory RNN" (LSTM) [1] implementation.
//
// An LSTM is a type of recurrent neural network that addresses the vanishing gradient problem in vanilla RNNs through
// additional cells, input and output gates. Intuitively, vanishing gradients are solved through additional additive
// components, and forget gate activations, that allow the gradients to flow through the network without vanishing
// as quickly.
//
// Since GoMLX doesn't implement loops, the size of the graph will be O(N) on the size of the sequence -- each
// step of the LSTM is instantiated as its own graph nodes.
//
// In any case, if not for educational or historical reasons, consider using transformer or (dilated) convolution layers
// instead.
//
// It was created to allow conversion of ONNX model, but it's fully differentiable and can be used to train models.
//
// See discussions in [2], and specification of ONNX LSTM which this was created to support in [3].
//
// [1] https://www.bioinf.jku.at/publications/older/2604.pdf, Hochreiter & Schmidhuber, 1997
// [2] https://colah.github.io/posts/2015-08-Understanding-LSTMs/
// [3] https://onnx.ai/onnx/operators/onnx__LSTM.html
package lstm

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// LSTM holds an LSTM configuration. It can be created with New (or NewWithWeights),
// and once finished to be configured, can be applied to x with Done.
type LSTM struct {
	ctx                                  *context.Context
	x                                    *Node
	xLengths                             *Node
	initialHiddenState, initialCellState *Node
	direction                            DirectionType
	batchSize, featuresSize, hiddenSize  int
	usePeephole                          bool

	// Model weights: see NewWithWeights for specification.
	inputsW, recurrentW, biasesW, peepholeW *Node

	// Activation functions: default to Sigmoid, Tanh and Tanh.
	// One for each combination of (forward, backward) x (forget (f), cellStateUpdate (g), hiddenStateUpdate (h)).
	// Defaults to 2 x [sigmoid, tanh, tanh].
	activations [2][3]ActivationFn
}

// ActivationFn defines an activation function used by the LSTM.
type ActivationFn func(x *Node) *Node

// New creates a new LSTM layer to be configured and then applied to x.
// x should be shaped [batchSize, sequenceSize, featuresSize].
//
// See LSTM.Ragged if x is not densely used: a more compact version to padding or masking.
//
// Once finished configuring, call LSTM.Done and it will return the final state of the LSTM.
func New(ctx *context.Context, x *Node, hiddenSize int) *LSTM {
	return &LSTM{
		ctx:          ctx,
		x:            x,
		direction:    DirForward,
		batchSize:    x.Shape().Dim(0),
		featuresSize: x.Shape().Dim(2),
		hiddenSize:   hiddenSize,
		activations: [2][3]ActivationFn{
			{Sigmoid, Tanh, Tanh},
			{Sigmoid, Tanh, Tanh},
		},
	}
}

// NewWithWeights creates a new LSTM layer using the given weights -- as opposed to creating them
// on-the-fly.
//
// Args:
//   - x: shaped [batchSize, sequenceSize, featuresSize]
//   - inputsW: shaped [numDirections, 4, hiddenSize, featuresSize]
//   - recurrentW: shaped [numDirections, 4, hiddenSize, hiddenSize]
//   - biases: for both gates and cell updates, shaped [numDirections, 8, hiddenSize].
//   - peepholeW: optional (can be nil), shaped [numDirections, 3, hiddenSize].
//
// See details in [3]
func NewWithWeights(x *Node, inputsW, recurrentW, biases, peepholeW *Node) *LSTM {
	l := New(nil, x, inputsW.Shape().Dim(2))
	l.inputsW = inputsW
	l.recurrentW = recurrentW
	l.biasesW = biases
	l.peepholeW = peepholeW
	if inputsW.Shape().Dim(0) == 2 {
		l.direction = DirBidirectional
	}
	inputsW.AssertDims(l.NumDirections(), 4, l.hiddenSize, l.featuresSize)
	recurrentW.AssertDims(l.NumDirections(), 4, l.hiddenSize, l.hiddenSize)
	biases.AssertDims(l.NumDirections(), 8, l.hiddenSize)
	if peepholeW != nil {
		l.usePeephole = true
		peepholeW.AssertDims(l.NumDirections(), 3, l.hiddenSize)
	}

	return l
}

// DirectionType defines the direction to run the LSTM.
type DirectionType int

const (
	DirForward DirectionType = iota
	DirReverse
	DirBidirectional
)

//go:generate enumer -trimprefix Dir -type=DirectionType -transform=snake -values -text -json -yaml lstm.go

// Direction configures in which direction to run the LSTM: DirForward, DirReverse or both.
func (l *LSTM) Direction(dir DirectionType) *LSTM {
	l.direction = dir
	return l
}

// Ragged indicates that x is "ragged" (the sequences are not used to the end), and its lengths are
// given by sequenceLengths, which must be shaped [batchSize].
// It is a more compact version of padding.
//
// The default is to assume all sequences are dense -- used to the end.
func (l *LSTM) Ragged(sequencesLengths *Node) *LSTM {
	l.xLengths = sequencesLengths
	return l
}

// UsePeephole configures whether to use a "peephole" to the "cell state" (c_i) when calculating
// values that usually only depend on the hidden state (h_i).
//
// Default to false.
func (l *LSTM) UsePeephole(usePeephole bool) *LSTM {
	l.usePeephole = usePeephole
	return l
}

// InitialStates configures the LSTM initial hidden state and cell state (h_0 and c_0 in the literature).
// If not set it defaults to 0.
//
// Both must be shaped [numDirections, batchSize, hiddenSize].
//
// This is useful if concatenating the output of the LSTM to another instance of the (same?) LSTM.
// That is, you can feed here the output values from LSTM.Done of a previous call.
func (l *LSTM) InitialStates(initialHiddenState, initialCellState *Node) *LSTM {
	l.initialHiddenState = initialHiddenState
	l.initialCellState = initialCellState
	return l
}

// NumDirections based on the direction information selected.
// See LSTM.Direction to configure the direction.
func (l *LSTM) NumDirections() int {
	if l.direction == DirBidirectional {
		return 2
	}
	return 1
}

// Done should be called once the LSTM is configured.
// It will apply the LSTM layer to the sequence in X.
// - allHiddenStates: [sequenceSize, numDirections, batchSize, hiddenSize]
// - lastHiddenState and lastCellState: [numDirections, batchSize, hiddenSize]
func (l *LSTM) Done() (allHiddenStates, lastHiddenState, lastCellState *Node) {
	// "Mis en place": everything we need in local variables.
	ctx := l.ctx
	x := l.x
	g := l.x.Graph()
	dtype := x.DType()
	numDirections := l.NumDirections()
	batchSize := l.batchSize
	sequenceSize := x.Shape().Dim(1)
	featuresSize := l.featuresSize
	hiddenSize := l.hiddenSize
	xLengths := l.xLengths
	inputsW := l.inputsW
	recurrentW := l.recurrentW
	biasesW := l.biasesW
	peepholeW := l.peepholeW

	// If model weights were not given, create them here.
	if inputsW == nil {
		//   - inputsW: shaped [numDirections, 4, hiddenSize, featuresSize]
		//   - recurrentW: shaped [numDirections, 4, hiddenSize, hiddenSize]
		//   - biases: for both gates and cell updates, shaped [numDirections, 8, hiddenSize].
		//   - peepholeW: optional (can be nil), shaped [numDirections, 3, hiddenSize].
		inputsW = ctx.VariableWithShape("inputsW", shapes.Make(dtype, numDirections, 4, hiddenSize, featuresSize)).ValueGraph(g)
		recurrentW = ctx.VariableWithShape("recurrentW", shapes.Make(dtype, numDirections, 4, hiddenSize, hiddenSize)).ValueGraph(g)
		biasesW = ctx.VariableWithShape("biasesW", shapes.Make(dtype, numDirections, 8, hiddenSize)).ValueGraph(g)
		if l.usePeephole {
			peepholeW = ctx.VariableWithShape("peepholeW", shapes.Make(dtype, numDirections, 3, hiddenSize)).ValueGraph(g)
		}
	}

	// Calculate all linear projections of x.
	// b->batchSize, s->sequenceSize, f->featuresSize, d->numDirections, n=4, h->hiddenSize.
	projX := Einsum("bsf,dnhf->dnbsh", x, inputsW)
	{
		biasX := Slice(biasesW, AxisRange(), AxisRangeFromStart(4)) // 4 first biases.
		biasX = ExpandAxes(biasX, 2, 3)                             // Create batchSize and seqLen axes.
		projX = Add(projX, biasX)
	}

	// Starting states: h_{i-1} and c_{i-1} so to say.
	prevHidden, prevCell := make([]*Node, numDirections), make([]*Node, numDirections) // One for each direction.
	for dirIdx := range numDirections {
		if l.initialHiddenState == nil {
			prevHidden[dirIdx] = Zeros(g, shapes.Make(dtype, batchSize, hiddenSize))
		} else {
			// Notice we can't check it earlier, in LSTM.InitialStates, because the user could change the
			// number of directions after setting the LSTM.InitialStates.
			l.initialHiddenState.AssertDims(numDirections, batchSize, hiddenSize)
			prevHidden[dirIdx] = Squeeze(Slice(l.initialHiddenState, AxisElem(dirIdx)), 0)
		}
		if l.initialCellState == nil {
			prevCell[dirIdx] = Zeros(g, shapes.Make(dtype, batchSize, hiddenSize))
		} else {
			l.initialCellState.AssertDims(numDirections, batchSize, hiddenSize)
			prevCell[dirIdx] = Squeeze(Slice(l.initialCellState, AxisElem(dirIdx)), 0)
		}
	}

	// Collect hidden states of each step, to be returned later.
	seqHiddenStates := make([][]*Node, numDirections)
	for ii := range numDirections {
		seqHiddenStates[ii] = make([]*Node, sequenceSize)
	}

	// Loop over each position of the sequence.
	for seqIdx := range sequenceSize {
		// Loop over directions.
		for dirIdx := range numDirections {
			seqPos := seqIdx
			if dirIdx == 1 || l.direction == DirReverse {
				// DirReverse:
				seqPos = sequenceSize - 1 - seqIdx
			}

			// Recurrent projection. recurrentW: [numDirections, 4, hiddenSize (j), hiddenSize(h)]
			dirRecurrentW := Squeeze(Slice(recurrentW, AxisElem(dirIdx)), 0)
			projState := Einsum("bh,njh->nbj", prevHidden[dirIdx], dirRecurrentW) // [4, batchSize, hiddenSize]
			{
				biasState := Slice(biasesW, AxisElem(dirIdx), AxisRangeToEnd(4)) // 4 last biases.
				biasState = Reshape(biasState, 4, 1, hiddenSize)                 // Remove direction axis, and add a batchSize axis.
				projState = Add(projState, biasState)
			}

			// See [3] for details on the inner values of the LSTM cell:
			// elemIdx: 0 input; 1 output; 2 forget; 3 cell, where 3 (cell) doesn't take peephole.
			sliceFn := func(elemIdx int) *Node {
				proj := Slice(projX, AxisElem(dirIdx), AxisElem(elemIdx), AxisRange() /*batch*/, AxisElem(seqPos))
				proj = Reshape(proj, batchSize, hiddenSize)
				recurrentProj := Squeeze(Slice(projState, AxisElem(elemIdx)), 0)
				proj = Add(proj, recurrentProj)
				if l.usePeephole && elemIdx <= 2 {
					// peepholeW: [numDirections, 3, hiddenSize].
					peepValue := Slice(peepholeW, AxisElem(dirIdx), AxisElem(elemIdx))
					peepValue = Reshape(peepValue, 1, hiddenSize) // Leave a batch-dimension to broadcast.
					proj = Add(proj, peepValue)
				}
				return proj
			}

			// Calculate new states (hidden and cell) for this direction.
			iT := l.activations[dirIdx][0](sliceFn(0)) // Shape [batchSize, hiddenSize]
			oT := l.activations[dirIdx][0](sliceFn(1))
			fT := l.activations[dirIdx][0](sliceFn(2))
			cT := l.activations[dirIdx][1](sliceFn(3))
			cellState := Add(
				Mul(prevCell[dirIdx], fT),
				Mul(cT, iT))
			hiddenState := Mul(oT, l.activations[dirIdx][2](cellState))

			// Mask results after the sentence end: if position is after the end of the sentence, just
			// use the prevHidden/prevCell unchanged -- notice it works in both directions.
			if l.xLengths != nil {
				masked := GreaterOrEqual(Scalar(g, xLengths.DType(), seqPos), xLengths)
				masked = ExpandAxes(masked, -1)
				hiddenState = Where(masked, prevHidden[dirIdx], hiddenState)
				cellState = Where(masked, prevCell[dirIdx], cellState)
			}

			// Save hidden state and move to next.
			seqHiddenStates[dirIdx][seqPos] = hiddenState
			prevHidden[dirIdx] = hiddenState
			prevCell[dirIdx] = cellState
		}
	}

	lastCellState = Stack(prevCell, 0)
	lastHiddenState = Stack(prevHidden, 0)
	if numDirections == 2 {
		allHiddenStates = Stack([]*Node{
			Stack(seqHiddenStates[0], 0),
			Stack(seqHiddenStates[1], 0)}, 1)
	} else {
		// Only one direction to stack.
		allHiddenStates = Stack(seqHiddenStates[0], 0)
		allHiddenStates = Reshape(allHiddenStates, sequenceSize, 1, batchSize, hiddenSize)
	}
	return
}
