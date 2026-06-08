// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dataset

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"iter"
	"math"
	"math/rand"
	"sync/atomic"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/graph/graphtest"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/support/testutil"
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

func (ds *testDS) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		ds.count.Store(0)
		for {
			value := ds.count.Add(1)
			if value > testDSMaxValue {
				return
			}
			batch := train.Batch{
				Inputs: []*tensors.Tensor{tensors.MustFromAnyValue(int(value))},
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

// TestNewParallelDataset with and without buffer.
func TestParallelDataset(t *testing.T) {
	for _, cacheSize := range []int{0, 10} {
		ds := &testDS{}
		pDS := CustomParallel(ds).Parallelism(0).Buffer(cacheSize).Start()
		count := int64(0)
		for batch, err := range pDS.Iter() {
			require.NoError(t, err, "Test failed with unexpected error")
			require.Len(t, batch.Inputs, 1, "Expected Dataset to yield 1 input tensor")
			count++
			_ = batch.Finalize()
		}
		require.Equalf(t, testDSMaxValue, count, "Number of yielded batches first loop, cacheSize=%d.", cacheSize)
		count = 0
		for batch, err := range pDS.Iter() {
			require.NoError(t, err, "Test failed with unexpected error")
			require.Len(t, batch.Inputs, 1, "Expected Dataset to yield 1 input tensor")
			count++
			_ = batch.Finalize()
		}
		require.Equal(t, testDSMaxValue, count, "Number of yielded batches at second loop, cacheSize=%d.", cacheSize)
	}
}

func TestBatchedDataset(t *testing.T) {
	manager := testutil.BuildTestBackend()
	ds := &testDS{}
	batchSize := 3
	numFullBatches := int(testDSMaxValue) / batchSize
	for ii := range 2 {
		dropIncompleteBatch := ii == 0
		batched := Batch(manager, ds, batchSize, true, dropIncompleteBatch)
		wantNumBatches := numFullBatches + 1
		if dropIncompleteBatch {
			wantNumBatches--
		}
		count := 0
		for batch, err := range batched.Iter() {
			require.NoError(t, err, "Test failed with unexpected error")
			require.Len(t, batch.Inputs, 1, "Expected Dataset to yield 1 input tensor")
			if dropIncompleteBatch || count < numFullBatches {
				require.Equalf(t, batchSize, batch.Inputs[0].Shape().Dimensions[0], "Batch #%d has shape %s",
					count, batch.Inputs[0].Shape())
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
			_ = batch.Finalize()
		}
	}
}

type testSlicesDS struct {
	numExamples int
}

func (ds *testSlicesDS) Name() string { return "testSlicesDS" }

func (ds *testSlicesDS) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for next := 0; next < ds.numExamples; next++ {
			input := make([]int, 3)
			label := make([]int, 3)
			for ii := range input {
				input[ii] = next*len(input) + ii
				label[ii] = -input[ii]
			}
			batch := train.Batch{
				Spec:   ds,
				Inputs: []*tensors.Tensor{tensors.FromValue(input)},
				Labels: []*tensors.Tensor{tensors.FromValue(label)},
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

func TestInMemoryDataset(t *testing.T) {
	backend := testutil.BuildTestBackend()
	ds := &testSlicesDS{numExamples: 17}
	const bytesPerValue = 8 // int uses dtypes.Int64, 8 bytes per value.
	const valuesPerExample = 3

	// Test as if each element is int[3]
	mds, err := InMemory(backend, ds, false)
	require.NoError(t, err)
	require.Equal(t, 2, len(mds.inputsAndLabelsData))
	require.Equal(t, 1, mds.numInputsTensors)
	require.Equal(t, ds.numExamples, mds.numExamples)
	require.Equal(t, int64(2*ds.numExamples*valuesPerExample*bytesPerValue), mds.ByteSize())
	require.True(
		t,
		shapes.Make(dtypes.Int64, ds.numExamples, valuesPerExample).Equal(mds.inputsAndLabelsData[0].Shape()),
	)
	require.True(
		t,
		shapes.Make(dtypes.Int64, ds.numExamples, valuesPerExample).Equal(mds.inputsAndLabelsData[0].Shape()),
	)

	// Test as if ds provided a batch of 3 elements each time.
	mds, err = InMemory(backend, ds, true)
	require.NoError(t, err)
	require.Equal(t, 2, len(mds.inputsAndLabelsData))
	require.Equal(t, 1, mds.numInputsTensors)
	require.Equal(t, ds.numExamples*valuesPerExample, mds.numExamples)
	require.Equal(t, int64(2*ds.numExamples*valuesPerExample*bytesPerValue), mds.ByteSize())
	require.True(
		t,
		shapes.Make(dtypes.Int64, ds.numExamples*valuesPerExample).Equal(mds.inputsAndLabelsData[0].Shape()),
	)
	require.True(
		t,
		shapes.Make(dtypes.Int64, ds.numExamples*valuesPerExample).Equal(mds.inputsAndLabelsData[0].Shape()),
	)

	// Read one element at a time: repeat 4 times, the last two are randomized.
	for repeat := range 4 {
		count := 0
		if repeat == 2 {
			mds.RandomWithReplacement()
		} else if repeat == 3 {
			mds.Shuffle()
		}
		isRandomized := false
		for batch, err := range mds.Iter() {
			require.NoError(t, err)
			require.Less(t, count, ds.numExamples*valuesPerExample)

			input := int(batch.Inputs[0].Value().(int64))
			label := int(batch.Labels[0].Value().(int64))
			if repeat < 2 {
				// In-order:
				require.Equal(t, count, input)
				require.Equal(t, -count, label)
			} else {
				// Randomized:
				isRandomized = isRandomized || ((count != input) || (-count != label))
			}
			count++
			_ = batch.Finalize()
		}
		if repeat >= 2 {
			// Check that it was randomized: chances of happening in order are astronomically low (mds.NumExamples() factorial).
			require.True(t, isRandomized)
		}
		require.Equal(t, count, ds.numExamples*valuesPerExample)
	}

	// Read in-memory dataset in batches: there are 51 examples, a batch of 50 should return only one batch if
	// dropping incomplete.
	mds = mds.Copy().BatchSize(50, true) // This should also reset shuffling/random sampling.

	next, stop := iter.Pull2(mds.Iter())
	batch, err, ok := next()
	require.True(t, ok)
	require.NoError(t, err)
	input := batch.Inputs[0].Value().([]int64)
	label := batch.Labels[0].Value().([]int64)
	want := xslices.Iota(int64(0), 50)
	require.Equal(t, want, input)
	for ii := range want {
		want[ii] = -want[ii]
	}
	require.Equal(t, want, label)
	_ = batch.Finalize()

	_, _, ok = next()
	require.False(t, ok) // Not enough examples for a second batch.
	stop()

	// Restart batch reading, this time allowing incomplete batches.
	mds.BatchSize(50, false)
	next, stop = iter.Pull2(mds.Iter())
	batch, err, ok = next()
	require.True(t, ok)
	require.NoError(t, err)
	_ = batch.Finalize()

	batch, err, ok = next() // Second batch will have size 1.
	require.True(t, ok)
	require.NoError(t, err)
	input = batch.Inputs[0].Value().([]int64)
	label = batch.Labels[0].Value().([]int64)
	require.Equal(t, []int64{50}, input)
	require.Equal(t, []int64{-50}, label)
	_ = batch.Finalize()
	stop()

	// Serialize and deserialize, check that we recover it.
	buf := &bytes.Buffer{}
	enc := gob.NewEncoder(buf)
	require.NoError(t, mds.GobSerialize(enc))

	// Deserialization:
	deviceNum := compute.DeviceNum(0)
	dec := gob.NewDecoder(buf)
	mds, err = GobDeserializeInMemoryToDevice(backend, deviceNum, dec)
	require.NoError(t, err)

	// Check that the recovered InMemoryDataset yields the same.
	mds = mds.BatchSize(50, true)
	next, stop = iter.Pull2(mds.Iter())
	batch, err, ok = next()
	require.True(t, ok)
	require.NoError(t, err)
	input = batch.Inputs[0].Value().([]int64)
	label = batch.Labels[0].Value().([]int64)
	want = xslices.Iota(int64(0), 50)
	require.Equal(t, want, input)
	for ii := range want {
		want[ii] = -want[ii]
	}
	require.Equal(t, want, label)
	_ = batch.Finalize()

	_, _, ok = next()
	require.False(t, ok) // Not enough examples for a second batch.
	stop()
}

func TestInMemoryFromData(t *testing.T) {
	manager := testutil.BuildTestBackend()
	mds, err := InMemoryFromData(manager, "test",
		[]any{[][]float32{{1, 2}, {3, 4}}},
		[]any{[][]float32{{3}, {7}}})
	require.NoError(t, err)

	next, stop := iter.Pull2(mds.Iter())
	batch, err, ok := next()
	require.True(t, ok)
	require.NoError(t, err)
	input, ok := batch.Inputs[0].Value().([]float32)
	require.True(t, ok, "Could not convert input to the expected []float32")
	label := batch.Labels[0].Value().([]float32)
	require.Equal(t, []float32{1, 2}, input)
	require.Equal(t, []float32{3}, label)
	_ = batch.Finalize()

	batch, err, ok = next()
	require.True(t, ok)
	require.NoError(t, err)
	input = batch.Inputs[0].Value().([]float32)
	label = batch.Labels[0].Value().([]float32)
	require.Equal(t, []float32{3, 4}, input)
	require.Equal(t, []float32{7}, label)
	_ = batch.Finalize()

	_, _, ok = next()
	require.False(t, ok)
	stop()

	mds.BatchSize(2, true)
	next, stop = iter.Pull2(mds.Iter())
	batch, err, ok = next()
	require.True(t, ok)
	require.NoError(t, err)
	batchInput, ok := batch.Inputs[0].Value().([][]float32)
	require.True(t, ok, "Could not convert batched input to the expected [][]float32")
	require.Equal(t, [][]float32{{1, 2}, {3, 4}}, batchInput)
	_ = batch.Finalize()
	stop()
}

func TestNormalization(t *testing.T) {
	manager := testutil.BuildTestBackend()

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
	for featureIdx := range numFeatures {
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
	graphtest.RunTestGraphFn(t, "ReplaceZerosByOnes", func(g *graph.Graph) (inputs, outputs []*graph.Node) {
		inputs = []*graph.Node{graph.Const(g, []float32{1, 0, 3})}
		outputs = []*graph.Node{ReplaceZerosByOnes(inputs[0])}
		return
	}, []any{
		[]float32{1, 1, 3},
	}, 0.1)
}

func TestMap(t *testing.T) {
	manager := testutil.BuildTestBackend()
	ds, err := InMemoryFromData(manager, "test",
		[]any{[][]float32{{1, 2}, {3, 4}}},
		[]any{[][]float32{{3}, {7}}})
	require.NoError(t, err)
	ds.BatchSize(2, true)
	mapDS := Map(
		manager,
		ds,
		func(graphBatch GraphBatch) GraphBatch {
			// Add 1 to the inputs[0], drop the labels.
			return GraphBatch{
				Inputs: []*graph.Node{graph.AddScalar(graphBatch.Inputs[0], 1)},
				Labels: nil,
				Spec:   graphBatch.Spec,
			}
		},
	)

	next, stop := iter.Pull2(mapDS.Iter())
	batch, err, ok := next()
	require.True(t, ok)
	require.NoError(t, err)
	batchInput, ok := batch.Inputs[0].Value().([][]float32)
	require.True(t, ok, "Could not convert batched input to the expected [][]float32")
	require.Equal(t, [][]float32{{2, 3}, {4, 5}}, batchInput)
	require.Empty(t, batch.Labels, "MapFn provided should have dropped the labels")
	_ = batch.Finalize()

	_, _, ok = next()
	require.False(t, ok)
	stop()
}

func TestModelMap(t *testing.T) {
	manager := testutil.BuildTestBackend()
	ds, err := InMemoryFromData(manager, "test",
		[]any{[][]float32{{1, 2}, {3, 4}}},
		[]any{[][]float32{{3}, {7}}})
	require.NoError(t, err)
	ds.BatchSize(2, true)

	// Create a model store and add a variable we can modify/use.
	store := model.NewStore()

	mapDS := ModelMap(
		manager,
		store,
		ds,
		func(scope *model.Scope, graphBatch GraphBatch) GraphBatch {
			// Get or create variable 'w', initialized to 1.0.
			wVar := scope.VariableWithValue("w", float32(1.0))
			w := wVar.NodeValue(graphBatch.Inputs[0].Graph())
			// Add w to the inputs[0].
			return GraphBatch{
				Inputs: []*graph.Node{graph.Add(graphBatch.Inputs[0], w)},
				Labels: nil,
				Spec:   graphBatch.Spec,
			}
		},
	)

	next, stop := iter.Pull2(mapDS.Iter())
	batch, err, ok := next()
	require.True(t, ok)
	require.NoError(t, err)
	batchInput, ok := batch.Inputs[0].Value().([][]float32)
	require.True(t, ok, "Could not convert batched input to the expected [][]float32")
	require.Equal(t, [][]float32{{2, 3}, {4, 5}}, batchInput)
	require.Empty(t, batch.Labels, "ModelMap provided should have dropped the labels")
	_ = batch.Finalize()

	_, _, ok = next()
	require.False(t, ok)
	stop()
}

func TestInMemoryDatasetConcurrent(t *testing.T) {
	backend := testutil.BuildTestBackend()
	ds := &testSlicesDS{numExamples: 100}
	mds, err := InMemory(backend, ds, false)
	require.NoError(t, err)
	mds.Shuffle()

	// Run two iterators concurrently and verify they get different sequences
	// but both exhaust the dataset correctly without racing.
	type seqResult struct {
		values []int
		err    error
	}

	runIterator := func() seqResult {
		var vals []int
		for batch, err := range mds.Iter() {
			if err != nil {
				return seqResult{err: err}
			}
			vals = append(vals, int(batch.Inputs[0].Value().([]int64)[0]))
			_ = batch.Finalize()
		}
		return seqResult{values: vals}
	}

	ch1 := make(chan seqResult, 1)
	ch2 := make(chan seqResult, 1)

	go func() { ch1 <- runIterator() }()
	go func() { ch2 <- runIterator() }()

	res1 := <-ch1
	res2 := <-ch2

	require.NoError(t, res1.err)
	require.NoError(t, res2.err)

	require.Equal(t, 100, len(res1.values))
	require.Equal(t, 100, len(res2.values))

	// Verify they are different orderings due to independent shuffling
	assert.NotEqual(t, res1.values, res2.values, "Both iterators should have generated different random shufflings")
}

func TestMapOnHost(t *testing.T) {
	manager := testutil.BuildTestBackend()
	ds, err := InMemoryFromData(manager, "test",
		[]any{[][]float32{{1, 2}, {3, 4}}},
		[]any{[][]float32{{3}, {7}}})
	require.NoError(t, err)
	ds.BatchSize(2, true)

	mapDS := MapOnHost(ds, func(batch train.Batch) train.Batch {
		// Just add 10 to inputs[0] and return a new batch
		tIn := batch.Inputs[0]
		inVal := tIn.Value().([][]float32)
		mappedVal := [][]float32{
			{inVal[0][0] + 10, inVal[0][1] + 10},
			{inVal[1][0] + 10, inVal[1][1] + 10},
		}

		mappedBatch := train.Batch{
			Spec:   batch.Spec,
			Inputs: []*tensors.Tensor{tensors.FromValue(mappedVal)},
			Labels: nil,
		}
		_ = batch.Finalize()
		return mappedBatch
	})

	next, stop := iter.Pull2(mapDS.Iter())
	batch, err, ok := next()
	require.True(t, ok)
	require.NoError(t, err)
	batchInput, ok := batch.Inputs[0].Value().([][]float32)
	require.True(t, ok, "Could not convert batched input to the expected [][]float32")
	require.Equal(t, [][]float32{{11, 12}, {13, 14}}, batchInput)
	require.Empty(t, batch.Labels, "MapOnHost provided should have dropped the labels")
	_ = batch.Finalize()

	_, _, ok = next()
	require.False(t, ok)
	stop()
}

func TestConstAndZero(t *testing.T) {
	// 1. Test Const dataset
	tVal := tensors.FromValue(float32(42.0))
	constDS := Const(train.Batch{
		Inputs: []*tensors.Tensor{tVal},
	})

	nextConst, stopConst := iter.Pull2(constDS.Iter())
	defer stopConst()

	for i := 0; i < 3; i++ {
		batch, err, ok := nextConst()
		require.True(t, ok)
		require.NoError(t, err)
		val := batch.Inputs[0].Value().(float32)
		require.Equal(t, float32(42.0), val)
		_ = batch.Finalize()
	}

	// 2. Test Zero dataset
	zeroDS := Zero()
	nextZero, stopZero := iter.Pull2(zeroDS.Iter())
	defer stopZero()

	for i := 0; i < 3; i++ {
		batch, err, ok := nextZero()
		require.True(t, ok)
		require.NoError(t, err)
		require.Len(t, batch.Inputs, 1)
		require.Len(t, batch.Labels, 1)
		valIn := batch.Inputs[0].Value().(int32)
		valLabel := batch.Labels[0].Value().(int32)
		require.Equal(t, int32(0), valIn)
		require.Equal(t, int32(0), valLabel)
		_ = batch.Finalize()
	}
}
