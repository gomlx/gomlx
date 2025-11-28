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

package datasets

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io"
	"math"
	"math/rand"
	"sync/atomic"
	"testing"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

type testDS struct {
	count atomic.Int64
}

var (
	testDSMaxValue = int64(10000)
)

func (ds *testDS) Name() string { return "testDS" }
func (ds *testDS) Reset()       { ds.count.Store(0) }
func (ds *testDS) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	value := ds.count.Add(1)
	if value > testDSMaxValue {
		err = io.EOF
		return
	}
	inputs = []*tensors.Tensor{tensors.FromAnyValue(int(value))} // One nil element.
	return                                                       // As if a batch was returned.
}

// TestNewParallelDataset with and without buffer.
func TestParallelDataset(t *testing.T) {
	for _, cacheSize := range []int{0, 10} {
		ds := &testDS{}
		pDS := CustomParallel(ds).Parallelism(0).Buffer(cacheSize).Start()
		count := int64(0)
		for {
			_, inputs, _, err := pDS.Yield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err, "Test failed with unexpected error")
			require.Len(t, inputs, 1, "Expected Dataset to yield 1 input tensor")
			count++
		}
		require.Equalf(t, testDSMaxValue, count, "Number of yielded batches first loop, cacheSize=%d.", cacheSize)
		count = 0
		pDS.Reset()
		for {
			_, inputs, _, err := pDS.Yield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err, "Test failed with unexpected error")
			require.Len(t, inputs, 1, "Expected Dataset to yield 1 input tensor")
			count++
		}
		require.Equal(t, testDSMaxValue, count, "Number of yielded batches at second loop, cacheSize=%d.", cacheSize)
	}
}

func TestBatchedDataset(t *testing.T) {
	manager := graphtest.BuildTestBackend()
	ds := &testDS{}
	batchSize := 3
	numFullBatches := int(testDSMaxValue) / batchSize
	for ii := 0; ii < 2; ii++ {
		dropIncompleteBatch := ii == 0
		batched := Batch(manager, ds, batchSize, true, dropIncompleteBatch)
		wantNumBatches := numFullBatches + 1
		if dropIncompleteBatch {
			wantNumBatches--
		}
		count := 0
		for {
			_, inputs, _, err := batched.Yield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err, "Test failed with unexpected error")
			require.Len(t, inputs, 1, "Expected Dataset to yield 1 input tensor")
			if dropIncompleteBatch || count < numFullBatches {
				require.Equalf(t, batchSize, inputs[0].Shape().Dimensions[0], "Batch #%d has shape %s",
					count, inputs[0].Shape())
			}
			count++
			require.LessOrEqualf(
				t,
				count,
				wantNumBatches,
				"Expected at most %d batches in epoch (dropIncompleteBatch=%v), dataset yielding more than that",
				wantNumBatches,
				dropIncompleteBatch,
			)
		}
		ds.Reset()
	}
}

type testSlicesDS struct {
	numExamples, next int
}

func (ds *testSlicesDS) Name() string { return "testSlicesDS" }
func (ds *testSlicesDS) Reset()       { ds.next = 0 }
func (ds *testSlicesDS) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	if ds.next >= ds.numExamples {
		err = io.EOF
		return
	}
	spec = ds
	input := make([]int, 3)
	label := make([]int, 3)
	for ii := range input {
		input[ii] = ds.next*len(input) + ii
		label[ii] = -input[ii]
	}
	inputs = []*tensors.Tensor{tensors.FromValue(input)}
	labels = []*tensors.Tensor{tensors.FromValue(label)}
	ds.next += 1
	return
}

func TestInMemoryDataset(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	ds := &testSlicesDS{numExamples: 17}
	const bytesPerValue = 8 // int uses dtypes.Int64, 8 bytes per value.
	const valuesPerExample = 3

	// Test as if each element is int[3]
	mds, err := InMemory(backend, ds, false)
	require.NoError(t, err)
	require.Equal(t, 2, len(mds.inputsAndLabelsData))
	require.Equal(t, 1, mds.numInputsTensors)
	require.Equal(t, ds.numExamples, mds.numExamples)
	require.Equal(t, uintptr(2*ds.numExamples*valuesPerExample*bytesPerValue), mds.Memory())
	require.True(
		t,
		shapes.Make(dtypes.Int64, ds.numExamples, valuesPerExample).Equal(mds.inputsAndLabelsData[0].Shape()),
	)
	require.True(
		t,
		shapes.Make(dtypes.Int64, ds.numExamples, valuesPerExample).Equal(mds.inputsAndLabelsData[0].Shape()),
	)

	// Test as if ds provided a batch of 3 elements each time.
	ds.Reset()
	mds, err = InMemory(backend, ds, true)
	require.NoError(t, err)
	require.Equal(t, 2, len(mds.inputsAndLabelsData))
	require.Equal(t, 1, mds.numInputsTensors)
	require.Equal(t, ds.numExamples*valuesPerExample, mds.numExamples)
	require.Equal(t, uintptr(2*ds.numExamples*valuesPerExample*bytesPerValue), mds.Memory())
	require.True(
		t,
		shapes.Make(dtypes.Int64, ds.numExamples*valuesPerExample).Equal(mds.inputsAndLabelsData[0].Shape()),
	)
	require.True(
		t,
		shapes.Make(dtypes.Int64, ds.numExamples*valuesPerExample).Equal(mds.inputsAndLabelsData[0].Shape()),
	)

	// Read one element at a time: repeat 4 times, the last two are randomized.
	for repeat := 0; repeat < 4; repeat++ {
		//fmt.Printf("\tRepeat %d:\n", repeat)
		count := 0
		if repeat == 2 {
			mds.RandomWithReplacement()
		} else if repeat == 3 {
			mds.Shuffle()
		}
		isRandomized := false
		for {
			_, inputs, labels, err := mds.Yield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err)
			require.Less(t, count, ds.numExamples*valuesPerExample)

			input := int(inputs[0].Value().(int64))
			label := int(labels[0].Value().(int64))
			if repeat < 2 {
				// In-order:
				require.Equal(t, count, input)
				require.Equal(t, -count, label)
			} else {
				// Randomized:
				isRandomized = isRandomized || ((count != input) || (-count != label))
			}
			count++
		}
		if repeat >= 2 {
			// Check that it was randomized: chances of happening in order are astronomically low (mds.NumExamples() factorial).
			require.True(t, isRandomized)
		}
		require.Equal(t, count, ds.numExamples*valuesPerExample)

		// Test that mds keeps exhausted.
		_, _, _, err = mds.Yield()
		require.True(t, io.EOF == err)
		mds.Reset()
	}

	// Read in-memory dataset in batches: there are 51 examples, a batch of 50 should return only one batch if
	// dropping incomplete.
	mds = mds.Copy().BatchSize(50, true) // This should also reset shuffling/random sampling.
	_, inputs, labels, err := mds.Yield()
	require.NoError(t, err)
	input := inputs[0].Value().([]int64)
	label := labels[0].Value().([]int64)
	want := xslices.Iota(int64(0), 50)
	require.Equal(t, want, input)
	for ii := range want {
		want[ii] = -want[ii]
	}
	require.Equal(t, want, label)
	_, inputs, labels, err = mds.Yield()
	require.True(t, err == io.EOF) // Not enough examples for a second batch.

	// Restart batch reading, this time allowing incomplete batches.
	mds.Reset()
	mds.BatchSize(50, false)
	_, _, _, err = mds.Yield()
	require.NoError(t, err)
	_, inputs, labels, err = mds.Yield() // Second batch will have size 1.
	require.NoError(t, err)
	input = inputs[0].Value().([]int64)
	label = labels[0].Value().([]int64)
	require.Equal(t, []int64{50}, input)
	require.Equal(t, []int64{-50}, label)

	// Serialize and deserialize, check that we recover it.
	require.NoError(t, err)
	buf := &bytes.Buffer{}
	enc := gob.NewEncoder(buf)
	require.NoError(t, mds.GobSerialize(enc))

	// Deserialization:
	deviceNum := backends.DeviceNum(0)
	dec := gob.NewDecoder(buf)
	mds, err = GobDeserializeInMemoryToDevice(backend, deviceNum, dec)
	require.NoError(t, err)

	// Check that the recovered InMemoryDataset yields the same.
	mds = mds.BatchSize(50, true)
	_, inputs, labels, err = mds.Yield()
	require.NoError(t, err)
	input = inputs[0].Value().([]int64)
	label = labels[0].Value().([]int64)
	want = xslices.Iota(int64(0), 50)
	require.Equal(t, want, input)
	for ii := range want {
		want[ii] = -want[ii]
	}
	require.Equal(t, want, label)
	_, inputs, labels, err = mds.Yield()
	require.True(t, err == io.EOF) // Not enough examples for a second batch.
}

func TestInMemoryFromData(t *testing.T) {
	manager := graphtest.BuildTestBackend()
	mds, err := InMemoryFromData(manager, "test",
		[]any{[][]float32{{1, 2}, {3, 4}}},
		[]any{[][]float32{{3}, {7}}})
	require.NoError(t, err)

	_, inputs, labels, err := mds.Yield()
	require.NoError(t, err)
	input, ok := inputs[0].Value().([]float32)
	require.True(t, ok, "Could not convert input to the expected []float32")
	label := labels[0].Value().([]float32)
	require.Equal(t, []float32{1, 2}, input)
	require.Equal(t, []float32{3}, label)

	_, inputs, labels, err = mds.Yield()
	require.NoError(t, err)
	input = inputs[0].Value().([]float32)
	label = labels[0].Value().([]float32)
	require.Equal(t, []float32{3, 4}, input)
	require.Equal(t, []float32{7}, label)

	_, _, _, err = mds.Yield()
	require.Equal(t, io.EOF, err)

	mds.Reset()
	mds.BatchSize(2, true)
	_, inputs, labels, err = mds.Yield()
	require.NoError(t, err)
	batchInput, ok := inputs[0].Value().([][]float32)
	require.True(t, ok, "Could not convert batched input to the expected [][]float32")
	require.Equal(t, [][]float32{{1, 2}, {3, 4}}, batchInput)
}

func TestNormalization(t *testing.T) {
	manager := graphtest.BuildTestBackend()

	// Create dataset with mean `(pi + featureNum)` and stddev `(e + featureNum)`.
	rng := rand.New(rand.NewSource(42))
	baseMean := math.Pi
	baseStddev := math.E
	const (
		numExamples = 10000
		midDim      = 3
		numFeatures = 5
	)
	wantMean := make([]float64, numFeatures)
	wantStddev := make([]float64, numFeatures)
	for featureIdx := 0; featureIdx < numFeatures; featureIdx++ {
		wantMean[featureIdx] = baseMean + float64(featureIdx)
		wantStddev[featureIdx] = baseStddev + float64(featureIdx)
	}
	input := tensors.FromShape(shapes.Make(dtypes.Float64, numExamples, midDim, numFeatures))
	input.MustMutableFlatData(func(flat any) {
		data := flat.([]float64)
		for ii := range data {
			featureIdx := ii % numFeatures
			data[ii] = rng.NormFloat64()*wantStddev[featureIdx] + wantMean[featureIdx]
		}
	})

	const batchSize = 32
	mds, err := InMemoryFromData(manager, "test", []any{input}, nil)
	require.NoError(t, err)
	mds.BatchSize(batchSize, true)

	meanT, stddevT, err := Normalization(manager, mds, 0, -1)
	require.NoError(t, err)
	mean, stddev := meanT.Value().([][][]float64), stddevT.Value().([][][]float64)
	fmt.Printf("\tmean=%v\n\tstddev=%v\n", mean, stddev)
	assert.InDeltaSlicef(t, wantMean, mean[0][0], 0.1, "mean pi+featureNum does not match")
	assert.InDeltaSlicef(t, wantStddev, stddev[0][0], 0.1, "stddev e+featureNum does not match")
}

func TestReplaceZerosByOnes(t *testing.T) {
	graphtest.RunTestGraphFn(t, "ReplaceZerosByOnes", func(g *Graph) (inputs, outputs []*Node) {
		inputs = []*Node{Const(g, []float32{1, 0, 3})}
		outputs = []*Node{ReplaceZerosByOnes(inputs[0])}
		return
	}, []any{
		[]float32{1, 1, 3},
	}, 0.1)
}

func TestMap(t *testing.T) {
	manager := graphtest.BuildTestBackend()
	ds, err := InMemoryFromData(manager, "test",
		[]any{[][]float32{{1, 2}, {3, 4}}},
		[]any{[][]float32{{3}, {7}}})
	require.NoError(t, err)
	ds.BatchSize(2, true)
	mapDS := MapWithGraphFn(
		manager,
		nil,
		ds,
		func(_ *context.Context, inputs, labels []*Node) (mappedInputs, mappedLabels []*Node) {
			// Add 1 to the inputs[0], drop the labels.
			return []*Node{AddScalar(inputs[0], 1)}, nil
		},
	)

	_, inputs, labels, err := mapDS.Yield()
	require.NoError(t, err)
	batchInput, ok := inputs[0].Value().([][]float32)
	require.True(t, ok, "Could not convert batched input to the expected [][]float32")
	require.Equal(t, [][]float32{{2, 3}, {4, 5}}, batchInput)
	require.Empty(t, labels, "MapGraphFn provided should have dropped the labels")

	_, _, _, err = mapDS.Yield()
	require.Equal(t, io.EOF, err)
}
