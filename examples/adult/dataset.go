/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package adult

import (
	"fmt"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/google/uuid"
	"github.com/pkg/errors"
	"io"
	"log"
	"math/rand"
)

// Dataset implements a train.Dataset that samples random batches.
//
// If a batch size is given it draws batches randomly, with replacement.
//
// If a batch size is not given, it returns a batch with the full epoch in one read
// and after that returns io.EOF, for evaluation. Notice the full epoch is small for
// this dataset.
//
// Notice that we chose to implement the Dataset using a computation graph to
// gather the examples (see Dataset.sampleGraph). This is simply to exemplify
// what can be done. Definitely an overkill here, since the data is small, and it could
// easierly have been done in Go. Notice an advantage of this, specially for large
// inputs, is that the data never leaves the acceleartor (GPU / TPU) -- again not the case
// for this trivial dataset.
type Dataset struct {
	// name of the dataset.
	name string

	id string // Some unique id.

	// data is a reference to the object containing the tensors with the whole dataset.
	data *TensorData

	batchSize     int // if < 0, this is used for batch.
	indicesShape  shapes.Shape
	evalExhausted bool // if true, batch (the only one for eval) has already been yielded.

	// Executes Dataset.
	exec *context.Exec
}

var (
	_ train.Dataset = &Dataset{} // Assert BatchSampelr is a dataset.
)

// IsEval returns whether Dataset is being used for evaluation, in which case it returns a batch
// with the whole data.
func (b *Dataset) IsEval() bool {
	return b.batchSize < 0
}

// sampleGraph builds the computation graph for the sampler. The indices param is a `[batch_size, 1]` shaped
// tensor, holding the (presumably random) indices to the examples to go into the batch.
//
// It returns 4 results: categorical, continuous and weights inputs, and the labels.
func (b *Dataset) sampleGraph(ctx *context.Context, indices *Node) (inputsAndLabels []*Node) {
	g := indices.Graph()
	if !g.Ok() {
		return []*Node{g.InvalidNode(), g.InvalidNode(), g.InvalidNode(), g.InvalidNode()}
	}

	// Create static variables the contains the whole dataset, from which we sample. This accelerates things
	// because the whole dataset can be placed on device (e.g: GPU) and there is no need for data transfer.
	// (The UCI-Adult dataset is so small that this doesn't matter, but serves as an example).
	ctx = ctx.Checked(false).In(b.id)
	if !b.data.CategoricalTensor.Ok() || !b.data.ContinuousTensor.Ok() || !b.data.WeightsTensor.Ok() || !b.data.LabelsTensor.Ok() {
		g.SetErrorf("sampleGraph: data is not set as a tensor, something went wrong?")
		return []*Node{g.InvalidNode(), g.InvalidNode(), g.InvalidNode(), g.InvalidNode()}
	}
	categoricalVar := ctx.VariableWithValue("RawData.Categorical", b.data.CategoricalTensor).SetTrainable(false)
	continuousVar := ctx.VariableWithValue("RawData.Continuous", b.data.ContinuousTensor).SetTrainable(false)
	weightsVar := ctx.VariableWithValue("RawData.Weights", b.data.WeightsTensor).SetTrainable(false)
	labelsVar := ctx.VariableWithValue("RawData.Labels", b.data.LabelsTensor).SetTrainable(false)
	if !ctx.Ok() {
		g.SetError(ctx.Error())
		return []*Node{g.InvalidNode(), g.InvalidNode(), g.InvalidNode(), g.InvalidNode()}
	}
	if b.IsEval() {
		// Ignore indices and return whole dataset.
		inputsAndLabels = []*Node{
			categoricalVar.ValueGraph(g),
			continuousVar.ValueGraph(g),
			weightsVar.ValueGraph(g),
			labelsVar.ValueGraph(g),
		}
	} else {
		inputsAndLabels = []*Node{
			Gather(categoricalVar.ValueGraph(g), indices),
			Gather(continuousVar.ValueGraph(g), indices),
			Gather(weightsVar.ValueGraph(g), indices),
			Gather(labelsVar.ValueGraph(g), indices),
		}
	}
	return
}

func (b *Dataset) Name() string {
	return b.name
}

// Yield implements train.Dataset.
func (b *Dataset) Yield() (spec any, inputs, labels []tensor.Tensor, err error) {
	var indicesT *tensor.Local
	if b.IsEval() {
		// Eval: check that it is not exhausted yet, and indices can be anything.
		if b.evalExhausted {
			return nil, nil, nil, io.EOF
		}
		indicesT = tensor.Zeros(shapes.Make(shapes.I64))
		b.evalExhausted = true

	} else {
		// Indices are a random samples with replacement.
		indicesT = tensor.FromShape(b.indicesShape)
		indices := tensor.Data[int](indicesT)
		numRows := b.data.LabelsTensor.Shape().Dimensions[0]
		for ii := range indices {
			indices[ii] = rand.Intn(numRows)
		}
	}

	// Execute to generate data.
	var inputsAndLabels []tensor.Tensor
	inputsAndLabels, err = b.exec.Call(indicesT)
	if err != nil {
		err = errors.WithMessage(err, "While generating a batch of data")
	}
	inputs = inputsAndLabels[:len(inputsAndLabels)-1]
	labels = inputsAndLabels[len(inputsAndLabels)-1:]
	return
}

// Reset implements train.Dataset
func (b *Dataset) Reset() {
	b.evalExhausted = false
	return
}

// NewDataset creates a batch of numExamples.
func NewDataset(name string, data *RawData, manager *Manager, batchSize int) *Dataset {
	b := &Dataset{
		name:          name,
		id:            fmt.Sprintf("Dataset_%s_%s", name, uuid.NewString()),
		data:          data.CreateTensors(manager),
		batchSize:     batchSize,
		evalExhausted: false,
	}
	if batchSize > 0 {
		b.indicesShape = shapes.Make(shapes.I64, batchSize, 1)
	}
	b.exec = context.NewExec(manager, nil, b.sampleGraph)
	return b
}

// NewDatasetForEval does not loop, nad returns whole dataset (not very large) at once --
// one batch is the full epoch.
func NewDatasetForEval(name string, data *RawData, manager *Manager) *Dataset {
	return NewDataset(name, data, manager, -1)
}

// PrintBatchSamples just generate a couple of batches of size 3 and print on the output.
// Just for debugging.
func PrintBatchSamples(data *RawData, manager *Manager) {
	sampler := NewDataset("batched train", data, manager, 3)
	for ii := 0; ii < 2; ii++ {
		fmt.Printf("\nSample batch %d:\n", ii)
		_, inputs, labels, err := sampler.Yield()
		if err != nil {
			log.Fatalf("Failed to sample batches: %+v", err)
		}
		fmt.Printf("\tcategorical=%v\n", inputs[0])
		fmt.Printf("\tcontinuos=%v\n", inputs[1])
		fmt.Printf("\tweights=%v\n", inputs[2])
		fmt.Printf("\tlabels=%v\n", labels)
	}
}
