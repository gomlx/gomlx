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

package data

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"io"
	"sync/atomic"
	"testing"

	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/require"
)

type testDS struct {
	count atomic.Int64
}

var (
	testDSMaxValue = int64(10000)
)

func (ds *testDS) Name() string { return "testDS" }
func (ds *testDS) Reset()       { ds.count.Store(0) }
func (ds *testDS) Yield() (spec any, inputs []tensor.Tensor, labels []tensor.Tensor, err error) {
	value := ds.count.Add(1)
	if value > testDSMaxValue {
		err = io.EOF
		return
	}
	inputs = []tensor.Tensor{tensor.FromAnyValue(int(value))} // One nil element.
	return                                                    // As if a batch was returned.
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
	manager := graphtest.BuildTestManager()
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
			require.LessOrEqualf(t, count, wantNumBatches, "Expected at most %d batches in epoch (dropIncompleteBatch=%v), dataset yielding more than that",
				wantNumBatches, dropIncompleteBatch)
		}
		ds.Reset()
	}
}

type testSlicesDS struct {
	numExamples, next int
}

func (ds *testSlicesDS) Name() string { return "testSlicesDS" }
func (ds *testSlicesDS) Reset()       { ds.next = 0 }
func (ds *testSlicesDS) Yield() (spec any, inputs []tensor.Tensor, labels []tensor.Tensor, err error) {
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
	inputs = []tensor.Tensor{tensor.FromValue(input)}
	labels = []tensor.Tensor{tensor.FromValue(label)}
	ds.next += 1
	return
}

func TestInMemoryDataset(t *testing.T) {
	manager := graphtest.BuildTestManager()
	ds := &testSlicesDS{numExamples: 17}
	const bytesPerValue = 8 // int uses shapes.Int64, 8 bytes per value.
	const valuesPerExample = 3

	// Test as if each element is int[3]
	mds, err := InMemory(manager, ds, false)
	require.NoError(t, err)
	require.Equal(t, 2, len(mds.inputsAndLabelsData))
	require.Equal(t, 1, mds.numInputsTensors)
	require.Equal(t, ds.numExamples, mds.numExamples)
	require.Equal(t, int64(2*ds.numExamples*valuesPerExample*bytesPerValue), mds.Memory())
	require.True(t, shapes.Make(shapes.I64, ds.numExamples, valuesPerExample).Eq(mds.inputsAndLabelsData[0].Shape()))
	require.True(t, shapes.Make(shapes.I64, ds.numExamples, valuesPerExample).Eq(mds.inputsAndLabelsData[0].Shape()))

	// Test as if ds provided a batch of 3 elements each time.
	ds.Reset()
	mds, err = InMemory(manager, ds, true)
	require.NoError(t, err)
	require.Equal(t, 2, len(mds.inputsAndLabelsData))
	require.Equal(t, 1, mds.numInputsTensors)
	require.Equal(t, ds.numExamples*valuesPerExample, mds.numExamples)
	require.Equal(t, int64(2*ds.numExamples*valuesPerExample*bytesPerValue), mds.Memory())
	require.True(t, shapes.Make(shapes.I64, ds.numExamples*valuesPerExample).Eq(mds.inputsAndLabelsData[0].Shape()))
	require.True(t, shapes.Make(shapes.I64, ds.numExamples*valuesPerExample).Eq(mds.inputsAndLabelsData[0].Shape()))

	// Read one element at a time: repeat 4 times, the last two are randomized.
	for repeat := 0; repeat < 4; repeat++ {
		fmt.Printf("\tRepeat %d:\n", repeat)
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

			input := inputs[0].Value().(int)
			label := labels[0].Value().(int)
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
	input := inputs[0].Local().Value().([]int)
	label := labels[0].Local().Value().([]int)
	want := slices.IotaSlice(0, 50)
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
	input = inputs[0].Local().Value().([]int)
	label = labels[0].Local().Value().([]int)
	require.Equal(t, []int{50}, input)
	require.Equal(t, []int{-50}, label)

	// Serialize and deserialize, check that we recover it.
	buf := &bytes.Buffer{}
	enc := gob.NewEncoder(buf)
	require.NoError(t, mds.GobSerialize(enc))
	dec := gob.NewDecoder(buf)
	mds, err = GobDeserializeInMemory(manager, dec)
	require.NoError(t, err)

	// Check that the recovered InMemoryDataset yields the same.
	mds = mds.BatchSize(50, true)
	_, inputs, labels, err = mds.Yield()
	require.NoError(t, err)
	input = inputs[0].Local().Value().([]int)
	label = labels[0].Local().Value().([]int)
	want = slices.IotaSlice(0, 50)
	require.Equal(t, want, input)
	for ii := range want {
		want[ii] = -want[ii]
	}
	require.Equal(t, want, label)
	_, inputs, labels, err = mds.Yield()
	require.True(t, err == io.EOF) // Not enough examples for a second batch.
}

func TestInMemoryFromData(t *testing.T) {
	manager := graphtest.BuildTestManager()
	mds, err := InMemoryFromData(manager, "test",
		[]any{[][]float32{{1, 2}, {3, 4}}},
		[]any{[][]float32{{3}, {7}}})
	require.NoError(t, err)

	_, inputs, labels, err := mds.Yield()
	require.NoError(t, err)
	input, ok := inputs[0].Local().Value().([]float32)
	require.True(t, ok, "Could not convert input to the expected []float32")
	label := labels[0].Local().Value().([]float32)
	require.Equal(t, []float32{1, 2}, input)
	require.Equal(t, []float32{3}, label)

	_, inputs, labels, err = mds.Yield()
	require.NoError(t, err)
	input = inputs[0].Local().Value().([]float32)
	label = labels[0].Local().Value().([]float32)
	require.Equal(t, []float32{3, 4}, input)
	require.Equal(t, []float32{7}, label)

	_, _, _, err = mds.Yield()
	require.Equal(t, io.EOF, err)

	mds.Reset()
	mds.BatchSize(2, true)
	_, inputs, labels, err = mds.Yield()
	require.NoError(t, err)
	batchInput, ok := inputs[0].Local().Value().([][]float32)
	require.True(t, ok, "Could not convert batched input to the expected [][]float32")
	require.Equal(t, [][]float32{{1, 2}, {3, 4}}, batchInput)
}
