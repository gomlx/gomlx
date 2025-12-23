package datasets

import (
	"io"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

// fakeSourceDataset is a simple dataset that yields a fixed number of batches with dummy values.
type fakeSourceDataset struct {
	name         string
	numBatches   int
	currentBatch int
}

func (ds *fakeSourceDataset) Name() string {
	return ds.name
}

func (ds *fakeSourceDataset) Reset() {
	ds.currentBatch = 0
}

func (ds *fakeSourceDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	if ds.currentBatch >= ds.numBatches {
		return nil, nil, nil, io.EOF
	}

	// Create dummy tensors with batch index as value
	batchIdx := float32(ds.currentBatch)
	input := tensors.FromAnyValue([]float32{batchIdx, batchIdx + 1.0, batchIdx + 2.0})
	label := tensors.FromAnyValue(batchIdx)

	inputs = []*tensors.Tensor{input}
	labels = []*tensors.Tensor{label}

	ds.currentBatch++
	return nil, inputs, labels, nil
}

func TestOnDevice(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	require.NotNil(t, backend)

	// Create a fake source dataset that yields 100 times
	sourceDS := &fakeSourceDataset{
		name:       "fakeSource",
		numBatches: 100,
	}

	// Create OnDevice dataset wrapping the source
	//
	// Notice for bufferSize > 1 the ordder is not preserved, and the IOF may be queued before a valid batch
	// so we use bufferSize = 1 here.
	targetDevice := backends.DeviceNum(0)
	bufferSize := 1
	onDeviceDS, err := NewOnDevice(backend, sourceDS, false, bufferSize, targetDevice)
	require.NoError(t, err)
	require.NotNil(t, onDeviceDS)

	// Verify we can yield 100 batches
	yieldedValues := make(map[float32]bool)
	for i := 0; i < 100; i++ {
		_, inputs, labels, err := onDeviceDS.Yield()
		require.NoError(t, err, "yield #%d should not error", i)
		require.NotNil(t, inputs)
		require.NotNil(t, labels)
		require.Len(t, inputs, 1, "should have 1 input tensor")
		require.Len(t, labels, 1, "should have 1 label tensor")

		// Verify tensors are on device (or have shared buffer)
		assert.True(t, inputs[0].IsOnAnyDevice(), "input tensor #%d should be on device", i)
		assert.True(t, labels[0].IsOnAnyDevice(), "label tensor #%d should be on device", i)

		// Verify they're on the correct device
		assert.True(t, inputs[0].IsOnDevice(targetDevice), "input tensor #%d should be on device %d", i, targetDevice)
		assert.True(t, labels[0].IsOnDevice(targetDevice), "label tensor #%d should be on device %d", i, targetDevice)

		// Extract values to verify later (order may vary due to buffering)
		labelValue := labels[0].Value().(float32)
		yieldedValues[labelValue] = true

		// Verify label value is in expected range (0-99)
		assert.GreaterOrEqual(t, labelValue, float32(0), "label value should be >= 0")
		assert.Less(t, labelValue, float32(100), "label value should be < 100")

		// Clean up
		inputs[0].FinalizeAll()
		labels[0].FinalizeAll()
	}

	// Verify we got all 100 unique values
	assert.Equal(t, 100, len(yieldedValues), "should have received 100 unique batches")

	// Verify we get EOF after 100 batches
	_, _, _, err = onDeviceDS.Yield()
	assert.Equal(t, io.EOF, err, "should get EOF after 100 batches")

	// Reset the dataset
	onDeviceDS.Reset()

	// Verify we get the same 100 batches again
	resetValues := make(map[float32]bool)
	for i := 0; i < 100; i++ {
		_, inputs, labels, err := onDeviceDS.Yield()
		require.NoError(t, err, "yield #%d after reset should not error", i)
		require.NotNil(t, inputs)
		require.NotNil(t, labels)
		require.Len(t, inputs, 1, "should have 1 input tensor")
		require.Len(t, labels, 1, "should have 1 label tensor")

		// Verify tensors are on device
		assert.True(t, inputs[0].IsOnAnyDevice(), "input tensor #%d after reset should be on device", i)
		assert.True(t, labels[0].IsOnAnyDevice(), "label tensor #%d after reset should be on device", i)
		assert.True(t, inputs[0].IsOnDevice(targetDevice), "input tensor #%d after reset should be on device %d", i, targetDevice)
		assert.True(t, labels[0].IsOnDevice(targetDevice), "label tensor #%d after reset should be on device %d", i, targetDevice)

		// Extract values (order may vary due to buffering)
		labelValue := labels[0].Value().(float32)
		resetValues[labelValue] = true

		// Verify label value is in expected range (0-99)
		assert.GreaterOrEqual(t, labelValue, float32(0), "label value after reset should be >= 0")
		assert.Less(t, labelValue, float32(100), "label value after reset should be < 100")

		// Clean up
		inputs[0].FinalizeAll()
		labels[0].FinalizeAll()
	}

	// Verify we got all 100 unique values after reset
	assert.Equal(t, 100, len(resetValues), "should have received 100 unique batches after reset")

	// Verify the values are the same before and after reset
	assert.Equal(t, yieldedValues, resetValues, "values after reset should match original values")

	// Verify EOF again
	_, _, _, err = onDeviceDS.Yield()
	assert.Equal(t, io.EOF, err, "should get EOF after reset and 100 batches")
}
