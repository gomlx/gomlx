// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package kvcache

import (
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
)

// TestKVCacheConfig tests KVCache config options, GetLayerType, and CacheSeqLen logic.
func TestKVCacheConfig(t *testing.T) {
	k := NewKVCache().
		WithLayerTypes(map[string]attention.LayerType{
			"/transformer/layer_0": attention.GlobalLayer,
			"/transformer/layer_1": attention.LocalLayer,
		}).
		WithMaxSeqLenPerLayerType([]int{1024, 256}).
		WithMinSeqLenPerLayerType([]int{64, 32})

	assert.Equal(t, attention.GlobalLayer, k.GetLayerType("/transformer/layer_0/attention"))
	assert.Equal(t, attention.LocalLayer, k.GetLayerType("/transformer/layer_1/attention"))
	assert.Equal(t, attention.GlobalLayer, k.GetLayerType("/transformer/layer_2/attention")) // defaults to Global

	// CacheSeqLen testing:
	// For Global (index 0, min=64, max=1024):
	assert.Equal(t, 64, k.CacheSeqLen(attention.GlobalLayer, 10))
	assert.Equal(t, 64, k.CacheSeqLen(attention.GlobalLayer, 64))
	assert.Equal(t, 128, k.CacheSeqLen(attention.GlobalLayer, 65))
	assert.Equal(t, 1024, k.CacheSeqLen(attention.GlobalLayer, 1024))
	assert.Equal(t, 1024, k.CacheSeqLen(attention.GlobalLayer, 2000))

	// For Local (index 1, min=32, max=256):
	assert.Equal(t, 32, k.CacheSeqLen(attention.LocalLayer, 10))
	assert.Equal(t, 64, k.CacheSeqLen(attention.LocalLayer, 33))
	assert.Equal(t, 256, k.CacheSeqLen(attention.LocalLayer, 300))
}

// TestKVCacheSerialization tests SerializeTensors, DeserializeTensors, SerializeNodes, and DeserializeNodes.
func TestKVCacheSerialization(t *testing.T) {
	k := NewKVCache().WithOrderedScopes([]string{"/layer_0", "/layer_1"})

	// Create some dummy tensors
	t0Key := tensors.FromValue(float32(1.0))
	t0Val := tensors.FromValue(float32(2.0))
	t1Key := tensors.FromValue(float32(3.0))
	t1Val := tensors.FromValue(float32(4.0))

	kvTensors := KVCacheTensors{
		"/layer_0" + KeySuffix:   t0Key,
		"/layer_0" + ValueSuffix: t0Val,
		"/layer_1" + KeySuffix:   t1Key,
		"/layer_1" + ValueSuffix: t1Val,
	}

	serialized, err := k.SerializeTensors(kvTensors)
	assert.NoError(t, err)
	assert.Len(t, serialized, 4)
	assert.Equal(t, t0Key, serialized[0])
	assert.Equal(t, t0Val, serialized[1])
	assert.Equal(t, t1Key, serialized[2])
	assert.Equal(t, t1Val, serialized[3])

	deserialized := k.DeserializeTensors(serialized)
	assert.Equal(t, t0Key, deserialized["/layer_0"+KeySuffix])
	assert.Equal(t, t0Val, deserialized["/layer_0"+ValueSuffix])
	assert.Equal(t, t1Key, deserialized["/layer_1"+KeySuffix])
	assert.Equal(t, t1Val, deserialized["/layer_1"+ValueSuffix])

	// Test error case
	invalidK := NewKVCache().WithOrderedScopes([]string{"/layer_0", "/layer_missing"})
	_, err = invalidK.SerializeTensors(kvTensors)
	assert.Error(t, err)

	// Test SerializeNodes / DeserializeNodes
	backend := testutil.BuildTestBackend()
	g := NewGraph(backend, "test")
	n0Key := Const(g, float32(1.0))
	n0Val := Const(g, float32(2.0))
	n1Key := Const(g, float32(3.0))
	n1Val := Const(g, float32(4.0))

	kvNodes := KVCacheNodes{
		"/layer_0" + KeySuffix:   n0Key,
		"/layer_0" + ValueSuffix: n0Val,
		"/layer_1" + KeySuffix:   n1Key,
		"/layer_1" + ValueSuffix: n1Val,
	}

	serializedNodes, err := k.SerializeNodes(kvNodes)
	assert.NoError(t, err)
	assert.Len(t, serializedNodes, 4)
	assert.Equal(t, n0Key, serializedNodes[0])
	assert.Equal(t, n0Val, serializedNodes[1])
	assert.Equal(t, n1Key, serializedNodes[2])
	assert.Equal(t, n1Val, serializedNodes[3])

	deserializedNodes := k.DeserializeNodes(serializedNodes)
	assert.Equal(t, n0Key, deserializedNodes["/layer_0"+KeySuffix])
	assert.Equal(t, n0Val, deserializedNodes["/layer_0"+ValueSuffix])
	assert.Equal(t, n1Key, deserializedNodes["/layer_1"+KeySuffix])
	assert.Equal(t, n1Val, deserializedNodes["/layer_1"+ValueSuffix])
}

// TestKVCacheTensorsInitAndPad tests InitializeTensors and PadTensors (including padTensor4D) functionality.
func TestKVCacheTensorsInitAndPad(t *testing.T) {
	k := NewKVCache().
		WithOrderedScopes([]string{"/layer_0", "/layer_1"}).
		WithLayerTypes(map[string]attention.LayerType{
			"/layer_0": attention.GlobalLayer,
			"/layer_1": attention.LocalLayer,
		}).
		WithMaxSeqLenPerLayerType([]int{1024, 256}).
		WithMinSeqLenPerLayerType([]int{64, 32})

	batchSize := 2
	numKVHeads := 4
	headDim := 8
	dtype := dtypes.Float32

	// Initialize tensors at currentSeqLen = 10
	// Global: cache length = 64 (min)
	// Local: cache length = 32 (min)
	kv := k.InitializeTensors(batchSize, numKVHeads, headDim, dtype, 10)

	assert.Len(t, kv, 4)
	assert.Equal(t, []int{2, 4, 64, 8}, kv["/layer_0"+KeySuffix].Shape().Dimensions)
	assert.Equal(t, []int{2, 4, 32, 8}, kv["/layer_1"+KeySuffix].Shape().Dimensions)

	// Pad tensors to currentSeqLen = 80
	// Global: CacheSeqLen(Global, 80) = 128
	// Local: CacheSeqLen(Local, 80) = 128
	padded, err := k.PadTensors(kv, 80)
	assert.NoError(t, err)
	assert.Len(t, padded, 4)
	assert.Equal(t, []int{2, 4, 128, 8}, padded["/layer_0"+KeySuffix].Shape().Dimensions)
	assert.Equal(t, []int{2, 4, 128, 8}, padded["/layer_1"+KeySuffix].Shape().Dimensions)
}

// TestKVCacheUpdateAndMask tests rolling cache updates, sink position preservation, and attention mask building.
func TestKVCacheUpdateAndMask(t *testing.T) {
	backend := testutil.BuildTestBackend()
	store := model.NewStore()

	// Config
	k := NewKVCache().
		WithSinkPositions(2).
		WithMaxSeqLenPerLayerType([]int{8, 8}).
		WithMinSeqLenPerLayerType([]int{8, 8}).
		WithOrderedScopes([]string{"/layer_0"})

	batchSize := 1
	numKVHeads := 1
	headDim := 2
	dtype := dtypes.Float32

	exec := model.MustNewExec(backend, store, func(testScope *model.Scope, prevK, prevV, nextK, nextV, position *Node) (*Node, *Node) {
		cache := KVCacheNodes{
			"/layer_0" + KeySuffix:   prevK,
			"/layer_0" + ValueSuffix: prevV,
		}
		// Update cache
		k.Update(testScope.Store().Scope("/layer_0"), cache, nextK, nextV, position)

		updatedK, updatedV := k.Get(cache, "/layer_0")
		return updatedK, updatedV
	})

	// Initial cache tensors (zero initialized, shape [1, 1, 8, 2])
	tK := tensors.FromShape(shapes.Make(dtype, batchSize, numKVHeads, 8, headDim))
	tV := tensors.FromShape(shapes.Make(dtype, batchSize, numKVHeads, 8, headDim))

	// Step 1: Update at position 0 with 3 tokens
	nextK1 := tensors.FromValue([][][][]float32{{{{1, 1}, {2, 2}, {3, 3}}}})
	nextV1 := tensors.FromValue([][][][]float32{{{{10, 10}, {20, 20}, {30, 30}}}})
	pos1 := tensors.FromValue(int32(0))

	results := exec.MustCall(tK, tV, nextK1, nextV1, pos1)
	tK = results[0]
	tV = results[1]

	// Check that slots 0, 1, 2 were written
	valK := tK.Value().([][][][]float32)
	assert.InDelta(t, 1.0, valK[0][0][0][0], 1e-5)
	assert.InDelta(t, 2.0, valK[0][0][1][0], 1e-5)
	assert.InDelta(t, 3.0, valK[0][0][2][0], 1e-5)
	assert.InDelta(t, 0.0, valK[0][0][3][0], 1e-5)

	// Step 2: Update at position 3 with 4 tokens (total tokens: 7, no wrap-around yet)
	nextK2 := tensors.FromValue([][][][]float32{{{{4, 4}, {5, 5}, {6, 6}, {7, 7}}}})
	nextV2 := tensors.FromValue([][][][]float32{{{{40, 40}, {50, 50}, {60, 60}, {70, 70}}}})
	pos2 := tensors.FromValue(int32(3))

	results = exec.MustCall(tK, tV, nextK2, nextV2, pos2)
	tK = results[0]
	tV = results[1]

	valK = tK.Value().([][][][]float32)
	assert.InDelta(t, 4.0, valK[0][0][3][0], 1e-5)
	assert.InDelta(t, 7.0, valK[0][0][6][0], 1e-5)
	assert.InDelta(t, 0.0, valK[0][0][7][0], 1e-5) // slot 7 is still 0

	// Step 3: Update at position 7 with 2 tokens (total tokens: 9, max size is 8. Should wrap around!)
	// With SinkPositions: 2, the rolling buffer starts at index 2 (size: 8 - 2 = 6).
	// Tokens 0, 1 (slots 0, 1) must be preserved.
	// Index 7 (8th token) goes to slot 7.
	// Index 8 (9th token) should wrap to slot 2 (since 8 >= 8, wrappedVal = (8 - 2) % 6 + 2 = 6 % 6 + 2 = 2).
	nextK3 := tensors.FromValue([][][][]float32{{{{8, 8}, {9, 9}}}})
	nextV3 := tensors.FromValue([][][][]float32{{{{80, 80}, {90, 90}}}})
	pos3 := tensors.FromValue(int32(7))

	results = exec.MustCall(tK, tV, nextK3, nextV3, pos3)
	tK = results[0]
	tV = results[1]

	valK = tK.Value().([][][][]float32)
	// Slot 0 and 1 must be preserved (1.0 and 2.0)
	assert.InDelta(t, 1.0, valK[0][0][0][0], 1e-5)
	assert.InDelta(t, 2.0, valK[0][0][1][0], 1e-5)
	// Slot 2 should have wrapped token 9 (value 9.0)
	assert.InDelta(t, 9.0, valK[0][0][2][0], 1e-5)
	// Slot 3 should still have 4.0
	assert.InDelta(t, 4.0, valK[0][0][3][0], 1e-5)
	// Slot 7 should have token 8 (value 8.0)
	assert.InDelta(t, 8.0, valK[0][0][7][0], 1e-5)

	// --- Test Attention Mask ---
	execMask := model.MustNewExec(backend, store, func(testScope *model.Scope, keyCacheNode, queryNode, positionNode *Node) *Node {
		cache := KVCacheNodes{
			"/layer_0" + KeySuffix: keyCacheNode,
		}
		mask := k.BuildAttentionMask(testScope.Store().Scope("/layer_0"), cache, queryNode, positionNode, true, 4)
		return mask
	})

	// Test case 1: Non-wrapped state (position = 3, querySeqLen = 2)
	query1 := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 2, 2))
	pos1 = tensors.FromValue(int32(3))

	mask1 := execMask.MustCall(tK, query1, pos1)[0]
	maskVal1 := mask1.Value().([][][]bool)

	// query 0 (pos 3) attending to keys 0..7
	assert.Equal(t, []bool{true, true, true, true, false, false, false, false}, maskVal1[0][0])
	// query 1 (pos 4) attending to keys 0..7
	assert.Equal(t, []bool{true, true, true, true, true, false, false, false}, maskVal1[0][1])

	// Test case 2: Wrapped state (position = 8, querySeqLen = 1)
	query2 := tensors.FromShape(shapes.Make(dtypes.Float32, 1, 1, 2))
	pos2 = tensors.FromValue(int32(8))

	mask2 := execMask.MustCall(tK, query2, pos2)[0]
	maskVal2 := mask2.Value().([][][]bool)

	// Query at pos 8 attending to slots 0..7:
	// Slot 0 (Sink): true
	// Slot 1 (Sink): true
	// Slot 2 (pos 8): true
	// Slot 3 (pos 3): false
	// Slot 4 (pos 4): false
	// Slot 5 (pos 5): true
	// Slot 6 (pos 6): true
	// Slot 7 (pos 7): true
	expectedMask := []bool{true, true, true, false, false, true, true, true}
	assert.Equal(t, expectedMask, maskVal2[0][0])
}
