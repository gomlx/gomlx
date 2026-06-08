// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dataset

import (
	"errors"
	"iter"
	"testing"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

// fakeSourceDataset is a simple dataset that yields a fixed number of batches with dummy values.
type fakeSourceDataset struct {
	name       string
	numBatches int
}

func (ds *fakeSourceDataset) Name() string {
	return ds.name
}

func (ds *fakeSourceDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for currentBatch := range ds.numBatches {
			batchIdx := float32(currentBatch)
			input := tensors.MustFromAnyValue([]float32{batchIdx, batchIdx + 1.0, batchIdx + 2.0})
			label := tensors.MustFromAnyValue(batchIdx)
			batch := train.Batch{
				Inputs: []*tensors.Tensor{input},
				Labels: []*tensors.Tensor{label},
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

func TestOnDevice(t *testing.T) {
	backend := testutil.BuildTestBackend()
	require.NotNil(t, backend)

	// Create a fake source dataset that yields 100 times
	sourceDS := &fakeSourceDataset{
		name:       "fakeSource",
		numBatches: 100,
	}

	// Create OnDevice dataset wrapping the source
	targetDevice := compute.DeviceNum(0)
	bufferSize := 1
	onDeviceDS, err := NewOnDevice(backend, sourceDS, bufferSize, targetDevice)
	require.NoError(t, err)
	require.NotNil(t, onDeviceDS)

	// Verify we can yield 100 batches
	yieldedValues := make(map[float32]bool)
	next, stop := iter.Pull2(onDeviceDS.Iter())
	for i := range 100 {
		batch, err, ok := next()
		require.True(t, ok, "should yield batch #%d", i)
		require.NoError(t, err, "yield #%d should not error", i)
		require.NotNil(t, batch.Inputs)
		require.NotNil(t, batch.Labels)
		require.Len(t, batch.Inputs, 1, "should have 1 input tensor")
		require.Len(t, batch.Labels, 1, "should have 1 label tensor")

		// Verify tensors are on device (or have shared buffer)
		assert.True(t, batch.Inputs[0].IsOnAnyDevice(), "input tensor #%d should be on device", i)
		assert.True(t, batch.Labels[0].IsOnAnyDevice(), "label tensor #%d should be on device", i)

		// Verify they're on the correct device
		assert.True(t, batch.Inputs[0].IsOnDevice(targetDevice), "input tensor #%d should be on device %d", i, targetDevice)
		assert.True(t, batch.Labels[0].IsOnDevice(targetDevice), "label tensor #%d should be on device %d", i, targetDevice)

		// Extract values to verify later (order may vary due to buffering)
		labelValue := batch.Labels[0].Value().(float32)
		yieldedValues[labelValue] = true

		// Verify label value is in expected range (0-99)
		assert.GreaterOrEqual(t, labelValue, float32(0), "label value should be >= 0")
		assert.Less(t, labelValue, float32(100), "label value should be < 100")

		// Clean up
		_ = batch.Finalize()
	}
	stop()

	// Verify we got all 100 unique values
	assert.Equal(t, 100, len(yieldedValues), "should have received 100 unique batches")

	// Verify we get EOF after 100 batches
	next, stop = iter.Pull2(onDeviceDS.Iter())
	for range 100 {
		_, _, ok := next()
		require.True(t, ok)
	}
	_, err, ok := next()
	assert.False(t, ok, "should get EOF after 100 batches")
	assert.NoError(t, err)
	stop()

	// Resetting is done simply by starting a new iterator:
	next, stop = iter.Pull2(onDeviceDS.Iter())
	defer stop()

	// Verify we get the same 100 batches again
	resetValues := make(map[float32]bool)
	for i := range 100 {
		batch, err, ok := next()
		require.True(t, ok, "yield #%d after reset should not error", i)
		require.NoError(t, err)
		require.NotNil(t, batch.Inputs)
		require.NotNil(t, batch.Labels)
		require.Len(t, batch.Inputs, 1, "should have 1 input tensor")
		require.Len(t, batch.Labels, 1, "should have 1 label tensor")

		// Verify tensors are on device
		assert.True(t, batch.Inputs[0].IsOnAnyDevice(), "input tensor #%d after reset should be on device", i)
		assert.True(t, batch.Labels[0].IsOnAnyDevice(), "label tensor #%d after reset should be on device", i)
		assert.True(t, batch.Inputs[0].IsOnDevice(targetDevice), "input tensor #%d after reset should be on device %d", i, targetDevice)
		assert.True(t, batch.Labels[0].IsOnDevice(targetDevice), "label tensor #%d after reset should be on device %d", i, targetDevice)

		// Extract values (order may vary due to buffering)
		labelValue := batch.Labels[0].Value().(float32)
		resetValues[labelValue] = true

		// Verify label value is in expected range (0-99)
		assert.GreaterOrEqual(t, labelValue, float32(0), "label value after reset should be >= 0")
		assert.Less(t, labelValue, float32(100), "label value after reset should be < 100")

		// Clean up
		_ = batch.Finalize()
	}

	// Verify we got all 100 unique values after reset
	assert.Equal(t, 100, len(resetValues), "should have received 100 unique batches after reset")

	// Verify the values are the same before and after reset
	assert.Equal(t, yieldedValues, resetValues, "values after reset should match original values")
}

func TestOnDeviceEarlyStop(t *testing.T) {
	backend := testutil.BuildTestBackend()
	require.NotNil(t, backend)

	sourceDS := &fakeSourceDataset{
		name:       "fakeSource",
		numBatches: 100,
	}

	targetDevice := compute.DeviceNum(0)
	bufferSize := 5
	onDeviceDS, err := NewOnDevice(backend, sourceDS, bufferSize, targetDevice)
	require.NoError(t, err)

	next, stop := iter.Pull2(onDeviceDS.Iter())
	for i := range 5 {
		batch, err, ok := next()
		require.True(t, ok)
		require.NoError(t, err)
		assert.True(t, batch.Inputs[0].IsOnDevice(targetDevice))
		_ = batch.Finalize()
		_ = i
	}
	stop()
}

type errorSourceDataset struct {
	name   string
	failAt int
}

func (ds *errorSourceDataset) Name() string {
	return ds.name
}

func (ds *errorSourceDataset) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for i := range 10 {
			if i == ds.failAt {
				yield(train.Batch{}, errors.New("fake error"))
				return
			}
			batch := train.Batch{
				Inputs: []*tensors.Tensor{tensors.MustFromAnyValue(float32(i))},
			}
			if !yield(batch, nil) {
				return
			}
		}
	}
}

func TestOnDeviceError(t *testing.T) {
	backend := testutil.BuildTestBackend()
	require.NotNil(t, backend)

	sourceDS := &errorSourceDataset{
		name:   "errorSource",
		failAt: 3,
	}

	onDeviceDS, err := NewOnDevice(backend, sourceDS, 2, compute.DeviceNum(0))
	require.NoError(t, err)

	count := 0
	for batch, err := range onDeviceDS.Iter() {
		if err != nil {
			assert.Contains(t, err.Error(), "fake error")
			break
		}
		assert.True(t, batch.Inputs[0].IsOnDevice(compute.DeviceNum(0)))
		_ = batch.Finalize()
		count++
	}
	assert.Equal(t, 3, count, "should yield 3 successful batches before error")
}
