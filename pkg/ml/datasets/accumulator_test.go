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
	"io"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

// mockDataset is a simple mock dataset that yields a fixed number of batches.
type mockDataset struct {
	name         string
	numBatches   int
	currentBatch int
	inputShape   shapes.Shape
	labelShape   shapes.Shape
}

func (ds *mockDataset) Name() string {
	return ds.name
}

func (ds *mockDataset) Reset() {
	ds.currentBatch = 0
}

func (ds *mockDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	if ds.currentBatch >= ds.numBatches {
		return nil, nil, nil, io.EOF
	}

	// Create trivial input and label tensors
	input := tensors.FromShape(ds.inputShape)
	label := tensors.FromShape(ds.labelShape)

	// Fill with trivial values (all zeros, but shape is what matters)
	inputs = []*tensors.Tensor{input}
	labels = []*tensors.Tensor{label}

	ds.currentBatch++
	return nil, inputs, labels, nil
}

// heterogeneousMockDataset yields batches with different shapes.
type heterogeneousMockDataset struct {
	name                                   string
	numBatchesShape1, numBatchesShape2     int
	currentBatchShape1, currentBatchShape2 int
	inputShape1, inputShape2               shapes.Shape
	labelShape1, labelShape2               shapes.Shape
}

func (ds *heterogeneousMockDataset) Name() string {
	return ds.name
}

func (ds *heterogeneousMockDataset) Reset() {
	ds.currentBatchShape1 = 0
	ds.currentBatchShape2 = 0
}

func (ds *heterogeneousMockDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	if ds.currentBatchShape1 >= ds.numBatchesShape1 && ds.currentBatchShape2 >= ds.numBatchesShape2 {
		return nil, nil, nil, io.EOF
	}

	var inputShape, labelShape shapes.Shape
	if ds.currentBatchShape1 < ds.numBatchesShape1 &&
		(ds.currentBatchShape1 <= ds.currentBatchShape2 || ds.currentBatchShape2 >= ds.numBatchesShape2) {
		inputShape = ds.inputShape1
		labelShape = ds.labelShape1
		ds.currentBatchShape1++
	} else {
		inputShape = ds.inputShape2
		labelShape = ds.labelShape2
		ds.currentBatchShape2++
	}

	input := tensors.FromShape(inputShape)
	label := tensors.FromShape(labelShape)

	inputs = []*tensors.Tensor{input}
	labels = []*tensors.Tensor{label}

	return nil, inputs, labels, nil
}

func TestDistributedAccumulator(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	if backend.NumDevices() < 4 {
		t.Skipf("Skipping distributed test because backend only has %d devices, need at least 4", backend.NumDevices())
	}

	t.Run("Mesh({2}, {\"shards\"})", func(t *testing.T) {
		// Simple data distribution with only two shards
		mesh, err := distributed.NewDeviceMesh([]int{2}, []string{"shards"})
		require.NoError(t, err)

		inputSpec, err := distributed.BuildSpec(mesh).S("shards").Done()
		require.NoError(t, err)
		labelSpec, err := distributed.BuildSpec(mesh).S("shards").Done()
		require.NoError(t, err)

		// Create mock source dataset that yields 100 batches
		sourceShape := shapes.Make(dtypes.Float32, 10, 5)
		labelShape := shapes.Make(dtypes.Float32, 10)
		source := &mockDataset{
			name:       "mockSource",
			numBatches: 100,
			inputShape: sourceShape,
			labelShape: labelShape,
		}

		// Create distributed dataset
		distDS, err := NewDistributedAccumulator(
			backend,
			source,
			distributed.AutoSharding,
			[]*distributed.ShardingSpec{inputSpec},
			[]*distributed.ShardingSpec{labelSpec},
			nil, // deviceAssignment - use default
		)
		require.NoError(t, err)

		// Check that Distributed yields only 50 batches before EOF
		batchCount := 0
		for {
			_, inputs, labels, err := distDS.DistributedYield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err)
			require.Len(t, inputs, 1)
			require.Len(t, labels, 1)

			// Check that the distributed tensors have 2 shards
			require.Len(t, inputs[0].Shards(), 2, "Input should have 2 shards")
			require.Len(t, labels[0].Shards(), 2, "Label should have 2 shards")

			// Check that shards are on correct devices (0 and 1)
			for i, shard := range inputs[0].Shards() {
				deviceNum, err := shard.Device()
				require.NoError(t, err)
				assert.Equal(t, backends.DeviceNum(i), deviceNum, "Input shard %d should be on device %d", i, i)
			}
			for i, shard := range labels[0].Shards() {
				deviceNum, err := shard.Device()
				require.NoError(t, err)
				assert.Equal(t, backends.DeviceNum(i), deviceNum, "Label shard %d should be on device %d", i, i)
			}

			batchCount++
		}
		assert.Equal(t, 50, batchCount, "Should yield 50 distributed batches from 100 source batches")

		// Reset and restart once
		distDS.Reset()
		batchCount = 0
		for {
			_, inputs, labels, err := distDS.DistributedYield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err)
			require.Len(t, inputs, 1)
			require.Len(t, labels, 1)
			require.Len(t, inputs[0].Shards(), 2)
			require.Len(t, labels[0].Shards(), 2)
			batchCount++
		}
		assert.Equal(t, 50, batchCount, "After reset, should still yield 50 distributed batches")
	})

	t.Run("Mesh({2}, {\"replicas\"})", func(t *testing.T) {
		// Simple replication with 2 devices, every input is replicated
		mesh, err := distributed.NewDeviceMesh([]int{2}, []string{"replicas"})
		require.NoError(t, err)

		// Both input and label are replicated
		inputSpec, err := distributed.BuildSpec(mesh).R().Done()
		require.NoError(t, err)
		labelSpec, err := distributed.BuildSpec(mesh).R().Done()
		require.NoError(t, err)

		// Create mock source dataset that yields 100 batches
		sourceShape := shapes.Make(dtypes.Float32, 10, 5)
		labelShape := shapes.Make(dtypes.Float32, 10)
		source := &mockDataset{
			name:       "mockSource",
			numBatches: 100,
			inputShape: sourceShape,
			labelShape: labelShape,
		}

		// Create distributed dataset
		distDS, err := NewDistributedAccumulator(
			backend,
			source,
			distributed.AutoSharding,
			[]*distributed.ShardingSpec{inputSpec},
			[]*distributed.ShardingSpec{labelSpec},
			nil, // deviceAssignment - use default
		)
		require.NoError(t, err)

		// Check that Distributed yields 100 batches (same as source, since numInputShards = 1)
		batchCount := 0
		for {
			_, inputs, labels, err := distDS.DistributedYield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err)
			require.Len(t, inputs, 1)
			require.Len(t, labels, 1)

			// Check that the distributed tensors have 2 shards (replicated across 2 devices)
			require.Len(t, inputs[0].Shards(), 2, "Input should have 2 shards (replicated)")
			require.Len(t, labels[0].Shards(), 2, "Label should have 2 shards (replicated)")

			// Check that shards are on correct devices (0 and 1)
			for i, shard := range inputs[0].Shards() {
				deviceNum, err := shard.Device()
				require.NoError(t, err)
				assert.Equal(t, backends.DeviceNum(i), deviceNum, "Input shard %d should be on device %d", i, i)
			}
			for i, shard := range labels[0].Shards() {
				deviceNum, err := shard.Device()
				require.NoError(t, err)
				assert.Equal(t, backends.DeviceNum(i), deviceNum, "Label shard %d should be on device %d", i, i)
			}

			batchCount++
		}
		assert.Equal(t, 100, batchCount, "Should yield 100 distributed batches from 100 source batches (numInputShards = 1)")
	})

	t.Run("Mesh({2, 2}, {\"shards\", \"replicas\"})", func(t *testing.T) {
		// Mesh with 2x2 = 4 devices, with shards and replicas
		// Input is sharded on "shards" axis (size 2), so we need 2 input shards
		// Each shard is then replicated across "replicas" axis (size 2), giving 4 total shards
		mesh, err := distributed.NewDeviceMesh([]int{2, 2}, []string{"shards", "replicas"})
		require.NoError(t, err)

		// Input is sharded on "shards" axis, label is replicated
		inputSpec, err := distributed.BuildSpec(mesh).S("shards").Done()
		require.NoError(t, err)
		labelSpec, err := distributed.BuildSpec(mesh).R().Done()
		require.NoError(t, err)

		// Create mock source dataset that yields 100 batches
		sourceShape := shapes.Make(dtypes.Float32, 10, 5)
		labelShape := shapes.Make(dtypes.Float32, 10)
		source := &mockDataset{
			name:       "mockSource",
			numBatches: 100,
			inputShape: sourceShape,
			labelShape: labelShape,
		}

		// Create distributed dataset
		distDS, err := NewDistributedAccumulator(
			backend,
			source,
			distributed.AutoSharding,
			[]*distributed.ShardingSpec{inputSpec},
			[]*distributed.ShardingSpec{labelSpec},
			nil, // deviceAssignment - use default
		)
		require.NoError(t, err)

		// Check that Distributed yields 50 batches (100 / 2 input shards)
		// User requirement: "the Distributed dataset should consume two batches, and yield a distributed batch with 4 shards"
		// This suggests numInputShards should be 2 (size of "shards" axis), not 4 (total devices)
		batchCount := 0
		for {
			_, inputs, labels, err := distDS.DistributedYield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err)
			require.Len(t, inputs, 1)
			require.Len(t, labels, 1)

			// Check that the distributed batch has 4 shards (2 shards x 2 replicas)
			require.Len(t, inputs[0].Shards(), 4, "Input should have 4 shards (2 shards x 2 replicas)")
			require.Len(t, labels[0].Shards(), 4, "Label should have 4 shards (replicated across 4 devices)")

			// Check that shards are on correct devices (0, 1, 2, 3)
			for i, shard := range inputs[0].Shards() {
				deviceNum, err := shard.Device()
				require.NoError(t, err)
				assert.Equal(t, backends.DeviceNum(i), deviceNum, "Input shard %d should be on device %d", i, i)
			}
			for i, shard := range labels[0].Shards() {
				deviceNum, err := shard.Device()
				require.NoError(t, err)
				assert.Equal(t, backends.DeviceNum(i), deviceNum, "Label shard %d should be on device %d", i, i)
			}

			batchCount++
		}
		assert.Equal(t, 50, batchCount,
			"Should yield 50 distributed batches from 100 source batches (consuming 2 batches per distributed batch)")
	})

	t.Run("Heterogeneous", func(t *testing.T) {
		// Mesh with 2x2 = 4 devices
		mesh, err := distributed.NewDeviceMesh([]int{2, 2}, []string{"shards", "replicas"})
		require.NoError(t, err)

		// Input is sharded, label is replicated
		inputSpec, err := distributed.BuildSpec(mesh).S("shards").Done()
		require.NoError(t, err)
		labelSpec, err := distributed.BuildSpec(mesh).R().Done()
		require.NoError(t, err)

		// Create heterogeneous mock dataset: 81 batches of one shape, 19 of another
		sourceShape1 := shapes.Make(dtypes.Float32, 10, 5)
		labelShape1 := shapes.Make(dtypes.Float32, 10)
		sourceShape2 := shapes.Make(dtypes.Float32, 8, 3) // Different shape
		labelShape2 := shapes.Make(dtypes.Float32, 8)     // Different shape
		source := &heterogeneousMockDataset{
			name:             "heterogeneousMockSource",
			numBatchesShape1: 81,
			numBatchesShape2: 19,
			inputShape1:      sourceShape1,
			labelShape1:      labelShape1,
			inputShape2:      sourceShape2,
			labelShape2:      labelShape2,
		}

		// Create distributed dataset
		distDS, err := NewDistributedAccumulator(
			backend,
			source,
			distributed.AutoSharding,
			[]*distributed.ShardingSpec{inputSpec},
			[]*distributed.ShardingSpec{labelSpec},
			nil, // deviceAssignment - use default
		)
		require.NoError(t, err)

		// Count batches by shape
		batchCountShape1 := 0
		batchCountShape2 := 0

		for {
			_, inputs, labels, err := distDS.DistributedYield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err)
			require.Len(t, inputs, 1)
			require.Len(t, labels, 1)

			// Check that we have 4 shards
			require.Len(t, inputs[0].Shards(), 4)
			require.Len(t, labels[0].Shards(), 4)

			// Determine which shape this batch has by checking shard shape
			// The shard shape matches the source tensor shape since we're providing full tensors as shards
			shardShape := inputs[0].ShardShape()

			if shardShape.Equal(sourceShape1) {
				batchCountShape1++
			} else if shardShape.Equal(sourceShape2) {
				batchCountShape2++
			} else {
				t.Fatalf("Unexpected shard shape: %s (expected either %s or %s)", shardShape, sourceShape1, sourceShape2)
			}
		}

		// Check counts: 81 batches -> 40 distributed batches (81/2 = 40.5, rounded down)
		// 19 batches -> 9 distributed batches (19/2 = 9.5, rounded down)
		// Note: This assumes numInputShards = 2 (size of "shards" axis), not 4 (total devices)
		// If the code uses numInputShards = 4, we'd get 20 and 4 batches instead
		assert.Equal(t, 40, batchCountShape1, "Should yield 40 distributed batches of first shape (81/2 input shards)")
		assert.Equal(t, 9, batchCountShape2, "Should yield 9 distributed batches of second shape (19/2 input shards)")
	})
}
