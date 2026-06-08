// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package dataset

import (
	"iter"
	"testing"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

type mockMultiBatchDS struct {
	batches []train.Batch
}

func (m *mockMultiBatchDS) Name() string { return "mockMultiBatch" }
func (m *mockMultiBatchDS) Iter() iter.Seq2[train.Batch, error] {
	return func(yield func(train.Batch, error) bool) {
		for _, batch := range m.batches {
			if !yield(batch, nil) {
				return
			}
		}
	}
}

type dummySpec string

func (d dummySpec) String() string { return string(d) }

func TestMapExecutorsCaching(t *testing.T) {
	backend := testutil.BuildTestBackend()

	// Create batches with different configurations:
	// Batch 0: 1 Input, 1 Label, Spec "A"
	// Batch 1: 2 Inputs, 1 Label, Spec "A"
	// Batch 2: 1 Input, 1 Label, Spec "B"
	// Batch 3: 1 Input, 1 Label, Spec "A" (reused)
	// Batch 4: 2 Inputs, 1 Label, Spec "A" (reused)

	t1 := tensors.FromValue(float32(1.0))
	t2 := tensors.FromValue(float32(2.0))
	t3 := tensors.FromValue(float32(3.0))

	batches := []train.Batch{
		{
			Inputs: []*tensors.Tensor{t1},
			Labels: []*tensors.Tensor{t2},
			Spec:   dummySpec("A"),
		},
		{
			Inputs: []*tensors.Tensor{t1, t2},
			Labels: []*tensors.Tensor{t3},
			Spec:   dummySpec("A"),
		},
		{
			Inputs: []*tensors.Tensor{t1},
			Labels: []*tensors.Tensor{t2},
			Spec:   dummySpec("B"),
		},
		{
			Inputs: []*tensors.Tensor{t1},
			Labels: []*tensors.Tensor{t2},
			Spec:   dummySpec("A"),
		},
		{
			Inputs: []*tensors.Tensor{t1, t2},
			Labels: []*tensors.Tensor{t3},
			Spec:   dummySpec("A"),
		},
	}

	ds := &mockMultiBatchDS{batches: batches}

	// Map dataset that simply passes through inputs and labels.
	// But we will inspect the executors map to verify caching.
	mapDS := Map(backend, ds, func(graphBatch GraphBatch) GraphBatch {
		return graphBatch
	})

	// We cast to get access to internal executors map.
	mDS := mapDS.(*mapDataset)

	// Let's iterate and check the executors count and reuse.
	next, stop := iter.Pull2(mDS.Iter())
	defer stop()

	// Batch 0 (1 input, 1 label, Spec A) -> should compile Executor 1
	b0, err, ok := next()
	require.True(t, ok)
	require.NoError(t, err)
	_ = b0.Finalize()
	require.Len(t, mDS.executors, 1)

	// Get key and reference for first executor
	k0 := mapDatasetExecutorKey{numInputs: 1, numLabels: 1, specString: "A"}
	exec0, found := mDS.executors[k0]
	require.True(t, found)
	require.NotNil(t, exec0)

	// Batch 1 (2 inputs, 1 label, Spec A) -> should compile Executor 2
	b1, err, ok := next()
	require.True(t, ok)
	require.NoError(t, err)
	_ = b1.Finalize()
	require.Len(t, mDS.executors, 2)

	k1 := mapDatasetExecutorKey{numInputs: 2, numLabels: 1, specString: "A"}
	exec1, found := mDS.executors[k1]
	require.True(t, found)
	require.NotNil(t, exec1)
	require.NotEqual(t, exec0, exec1)

	// Batch 2 (1 input, 1 label, Spec B) -> should compile Executor 3
	b2, err, ok := next()
	require.True(t, ok)
	require.NoError(t, err)
	_ = b2.Finalize()
	require.Len(t, mDS.executors, 3)

	k2 := mapDatasetExecutorKey{numInputs: 1, numLabels: 1, specString: "B"}
	exec2, found := mDS.executors[k2]
	require.True(t, found)
	require.NotNil(t, exec2)
	require.NotEqual(t, exec0, exec2)

	// Batch 3 (1 input, 1 label, Spec A) -> should reuse Executor 1
	b3, err, ok := next()
	require.True(t, ok)
	require.NoError(t, err)
	_ = b3.Finalize()
	require.Len(t, mDS.executors, 3) // No new executor created

	exec3, found := mDS.executors[k0]
	require.True(t, found)
	require.Equal(t, exec0, exec3)

	// Batch 4 (2 inputs, 1 label, Spec A) -> should reuse Executor 2
	b4, err, ok := next()
	require.True(t, ok)
	require.NoError(t, err)
	_ = b4.Finalize()
	require.Len(t, mDS.executors, 3) // No new executor created

	exec4, found := mDS.executors[k1]
	require.True(t, found)
	require.Equal(t, exec1, exec4)
}
