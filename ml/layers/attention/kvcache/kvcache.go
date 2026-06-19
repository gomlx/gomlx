// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package kvcache implements a flexible key-value (KV) cache for transformer models.
// It supports rolling (circular) cache, attention sinks (keeping the first N tokens
// always in the cache), dynamic shape handling (allocating cache with powers-of-two
// sequence lengths to minimize JIT compilations), and serialization/deserialization
// of cache nodes and tensors.
package kvcache

import (
	"fmt"
	"strings"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/attention"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/pkg/errors"
)

// KVCacheNodes maps a scope-specific cache key/value path to its corresponding graph *Node.
// Keys are formatted as "<scope_path>/key" and "<scope_path>/value".
type KVCacheNodes map[string]*Node

// KVCacheTensors maps a scope-specific cache key/value path to its corresponding concrete *tensors.Tensor.
// Keys are formatted as "<scope_path>/key" and "<scope_path>/value".
type KVCacheTensors map[string]*tensors.Tensor

const (
	// KeySuffix is appended to a layer's scope path to identify its key cache.
	KeySuffix = "/key"
	// ValueSuffix is appended to a layer's scope path to identify its value cache.
	ValueSuffix = "/value"
)

// KVCache configures and manages the attention Key-Value cache for transformer layers.
// It supports attention sinks, bucketing sequence lengths (powers-of-two padding) to avoid
// recompiling graphs for each new sequence length, and sliding-window layouts.
type KVCache struct {
	// SinkPositions is the number of initial tokens to keep fixed at the start of the cache
	// (also known as attention sinks, e.g. for streaming LLMs).
	SinkPositions int
	// MaxSeqLenPerLayerType defines the maximum sequence length per LayerType.
	// Indexed by attention.LayerType (GlobalLayer=0, LocalLayer=1).
	MaxSeqLenPerLayerType []int
	// MinSeqLenPerLayerType defines the minimum sequence length per LayerType.
	// Indexed by attention.LayerType (GlobalLayer=0, LocalLayer=1).
	MinSeqLenPerLayerType []int
	// OrderedScopes defines the execution order of attention layer scopes.
	// This is required to serialize and deserialize the cache tensors/nodes deterministically.
	OrderedScopes []string
	// LayerTypes maps scope paths (or prefixes) to their attention.LayerType (Global vs Local).
	LayerTypes map[string]attention.LayerType
	// GlobalHeadDim specifies a custom key/value head dimension for global layers if different.
	GlobalHeadDim int
}

// NewKVCache creates a new KVCache configuration builder with default settings:
//   - SinkPositions: 4
//   - MaxSeqLenPerLayerType: [64k, 1k] (Global/Local)
//   - MinSeqLenPerLayerType: [32, 32]
func NewKVCache() *KVCache {
	return &KVCache{
		SinkPositions:         4,
		MaxSeqLenPerLayerType: []int{64 * 1024, 1024},
		MinSeqLenPerLayerType: []int{32, 32},
		LayerTypes:            make(map[string]attention.LayerType),
	}
}

// WithGlobalHeadDim configures the key/value head dimension for global layers if different from the default.
func (k *KVCache) WithGlobalHeadDim(dim int) *KVCache {
	k.GlobalHeadDim = dim
	return k
}

// WithSinkPositions configures the number of initial tokens to keep in the cache (attention sinks).
func (k *KVCache) WithSinkPositions(n int) *KVCache {
	k.SinkPositions = n
	return k
}

// WithMaxSeqLenPerLayerType sets the maximum sequence lengths per layer type.
func (k *KVCache) WithMaxSeqLenPerLayerType(maxLens []int) *KVCache {
	k.MaxSeqLenPerLayerType = maxLens
	return k
}

// WithMinSeqLenPerLayerType sets the minimum sequence lengths per layer type.
// The KVCache sequence length grows in powers of two (16, 32, 64, 128, etc.) to minimize JIT
// compilation costs, from MinSeqLen up to the configured maximum.
func (k *KVCache) WithMinSeqLenPerLayerType(minLens []int) *KVCache {
	k.MinSeqLenPerLayerType = minLens
	return k
}

// WithOrderedScopes configures the ordered list of scope paths for all layers.
// This is required for serialization/deserialization to guarantee deterministic ordering.
func (k *KVCache) WithOrderedScopes(scopes []string) *KVCache {
	k.OrderedScopes = scopes
	return k
}

// WithLayerTypes configures the mapping from scope prefixes to their attention layer type (Local/Global).
func (k *KVCache) WithLayerTypes(layerTypes map[string]attention.LayerType) *KVCache {
	k.LayerTypes = layerTypes
	return k
}

// GetLayerType returns the layer type for the given scope path by finding the longest matching prefix
// in the configured LayerTypes map. Defaults to attention.GlobalLayer if no prefix matches.
func (k *KVCache) GetLayerType(scopePath string) attention.LayerType {
	var bestPrefix string
	var bestType attention.LayerType = attention.GlobalLayer
	for prefix, lt := range k.LayerTypes {
		if strings.HasPrefix(scopePath, prefix) && len(prefix) > len(bestPrefix) {
			bestPrefix = prefix
			bestType = lt
		}
	}
	return bestType
}

// CacheSeqLen calculates the rounded-up cache sequence length for a given layer type.
// It rounds the sequence length up to the next power of 2 (between min and max limits)
// to reduce JIT graph recompilations.
func (k *KVCache) CacheSeqLen(lt attention.LayerType, currentSeqLen int) int {
	maxLen := 4096
	if int(lt) < len(k.MaxSeqLenPerLayerType) {
		maxLen = k.MaxSeqLenPerLayerType[int(lt)]
	}
	if currentSeqLen >= maxLen {
		return maxLen
	}
	minLen := 32
	if int(lt) < len(k.MinSeqLenPerLayerType) {
		minLen = k.MinSeqLenPerLayerType[int(lt)]
	}
	if currentSeqLen <= minLen {
		return minLen
	}
	size := minLen
	for size < currentSeqLen && size < maxLen {
		size *= 2
	}
	if size > maxLen {
		size = maxLen
	}
	return size
}

// SerializeTensors packs the map of KV cache tensors into a slice in a deterministic order
// defined by OrderedScopes. Returns an error if any scope's key or value tensor is missing.
func (k *KVCache) SerializeTensors(kv KVCacheTensors) ([]*tensors.Tensor, error) {
	result := make([]*tensors.Tensor, 0, 2*len(k.OrderedScopes))
	for _, scope := range k.OrderedScopes {
		kKey := scope + KeySuffix
		vKey := scope + ValueSuffix

		kTensor, ok := kv[kKey]
		if !ok {
			return nil, errors.Errorf("missing key tensor for scope %q", scope)
		}
		vTensor, ok := kv[vKey]
		if !ok {
			return nil, errors.Errorf("missing value tensor for scope %q", scope)
		}

		result = append(result, kTensor, vTensor)
	}
	return result, nil
}

// DeserializeTensors unpacks a slice of KV cache tensors back into a KVCacheTensors map
// using the ordering defined by OrderedScopes. Panics if the number of tensors is incorrect.
func (k *KVCache) DeserializeTensors(tensors []*tensors.Tensor) KVCacheTensors {
	expectedLen := 2 * len(k.OrderedScopes)
	if len(tensors) != expectedLen {
		panic(fmt.Sprintf("DeserializeTensors: expected %d tensors, got %d", expectedLen, len(tensors)))
	}

	kv := make(KVCacheTensors)
	for i, scope := range k.OrderedScopes {
		kv[scope+KeySuffix] = tensors[2*i]
		kv[scope+ValueSuffix] = tensors[2*i+1]
	}
	return kv
}

// DeserializeNodes unpacks a slice of KV cache graph nodes back into a KVCacheNodes map
// using the ordering defined by OrderedScopes. Panics if the number of nodes is incorrect.
func (k *KVCache) DeserializeNodes(nodes []*Node) KVCacheNodes {
	expectedLen := 2 * len(k.OrderedScopes)
	if len(nodes) != expectedLen {
		panic(fmt.Sprintf("DeserializeNodes: expected %d nodes, got %d", expectedLen, len(nodes)))
	}

	cache := make(KVCacheNodes)
	for i, scope := range k.OrderedScopes {
		cache[scope+KeySuffix] = nodes[2*i]
		cache[scope+ValueSuffix] = nodes[2*i+1]
	}
	return cache
}

// SerializeNodes packs the map of KV cache nodes into a slice in a deterministic order
// defined by OrderedScopes. Returns an error if any scope's key or value node is missing.
func (k *KVCache) SerializeNodes(kv KVCacheNodes) ([]*Node, error) {
	result := make([]*Node, 0, 2*len(k.OrderedScopes))
	for _, scope := range k.OrderedScopes {
		kKey := scope + KeySuffix
		vKey := scope + ValueSuffix

		kNode, ok := kv[kKey]
		if !ok {
			return nil, errors.Errorf("missing key Node for scope %q", scope)
		}
		vNode, ok := kv[vKey]
		if !ok {
			return nil, errors.Errorf("missing value Node for scope %q", scope)
		}

		result = append(result, kNode, vNode)
	}
	return result, nil
}

// InitializeTensors allocates and returns a new zero-initialized KVCacheTensors map
// based on the configuration and current sequence length. Tensors are allocated with shape
// [batchSize, numKVHeads, cacheSeqLen, headDim].
func (k *KVCache) InitializeTensors(batchSize, numKVHeads, headDim int, dtype dtypes.DType, currentSeqLen int) KVCacheTensors {
	kv := make(KVCacheTensors)
	for _, scope := range k.OrderedScopes {
		lt := k.GetLayerType(scope)
		layerHeadDim := headDim
		if lt == attention.GlobalLayer && k.GlobalHeadDim > 0 {
			layerHeadDim = k.GlobalHeadDim
		}
		seqLen := k.CacheSeqLen(lt, currentSeqLen)
		shape := shapes.Make(dtype, batchSize, numKVHeads, seqLen, layerHeadDim)

		kv[scope+KeySuffix] = tensors.FromShape(shape)
		kv[scope+ValueSuffix] = tensors.FromShape(shape)
	}
	return kv
}

// PadTensors pads the sequence length dimension of the KV cache tensors to match the target
// sequence length calculated by CacheSeqLen. This is used when the sequence length grows
// past the current bucket size during generation.
func (k *KVCache) PadTensors(kv KVCacheTensors, currentSeqLen int) (KVCacheTensors, error) {
	padded := make(KVCacheTensors)
	for _, scope := range k.OrderedScopes {
		lt := k.GetLayerType(scope)
		targetSeqLen := k.CacheSeqLen(lt, currentSeqLen)

		for _, suffix := range []string{KeySuffix, ValueSuffix} {
			path := scope + suffix
			tensor, ok := kv[path]
			if !ok {
				continue
			}

			shape := tensor.Shape()
			if shape.Rank() != 4 {
				return nil, errors.Errorf("invalid KV cache tensor rank: %d (expected 4)", shape.Rank())
			}

			currentLen := shape.Dimensions[2]
			if currentLen == targetSeqLen {
				padded[path] = tensor
				continue
			}
			if currentLen > targetSeqLen {
				return nil, errors.Errorf("KV cache tensor %q has sequence length %d, larger than target %d", path, currentLen, targetSeqLen)
			}

			paddedTensor, err := padTensor4D(tensor, targetSeqLen)
			if err != nil {
				return nil, errors.Wrapf(err, "failed to pad KV cache tensor %q", path)
			}
			padded[path] = paddedTensor
		}
	}
	return padded, nil
}

// padTensor4D is a helper that performs 4D tensor padding along the sequence dimension (axis 2).
func padTensor4D(tensor *tensors.Tensor, targetSeqLen int) (*tensors.Tensor, error) {
	shape := tensor.Shape()
	B := shape.Dimensions[0]
	H := shape.Dimensions[1]
	S := shape.Dimensions[2]
	D := shape.Dimensions[3]

	if S == targetSeqLen {
		return tensor, nil
	}

	newShape := shapes.Make(shape.DType, B, H, targetSeqLen, D)
	paddedTensor := tensors.FromShape(newShape)

	elemSize := int(shape.ByteSize() / int64(shape.Size()))
	chunkSizeBytes := S * D * elemSize
	dstChunkSizeBytes := targetSeqLen * D * elemSize

	var copyErr error
	err := tensor.ConstBytes(func(srcBytes []byte) {
		copyErr = paddedTensor.MutableBytes(func(dstBytes []byte) {
			for b := 0; b < B; b++ {
				for h := 0; h < H; h++ {
					srcStart := (b*H + h) * chunkSizeBytes
					srcEnd := srcStart + chunkSizeBytes

					dstStart := (b*H + h) * dstChunkSizeBytes
					dstEnd := dstStart + chunkSizeBytes

					copy(dstBytes[dstStart:dstEnd], srcBytes[srcStart:srcEnd])
				}
			}
		})
	})
	if err != nil {
		return nil, err
	}
	if copyErr != nil {
		return nil, copyErr
	}
	return paddedTensor, nil
}

// Update inserts the new key/value nodes (nextK, nextV) into the cache at the specified position.
// It handles transpose if the layout is different (e.g. BSHD vs BHSD) and implements the rolling
// cache mechanism with attention sink preservation (SinkPositions) when sequence length exceeds maxSeqLen.
func (k *KVCache) Update(scope *model.Scope, cache KVCacheNodes, nextK, nextV *Node, position *Node) {
	path := scope.Scope()
	kKey := path + KeySuffix
	vKey := path + ValueSuffix

	prevK, ok1 := cache[kKey]
	prevV, ok2 := cache[vKey]
	if !ok1 || !ok2 {
		panic(fmt.Sprintf("KVCache.Update: cache nodes not found for path %q", path))
	}

	g := nextK.Graph()
	lt := k.GetLayerType(path)
	maxSeqLen := k.MaxSeqLenPerLayerType[int(lt)]

	if nextK.Shape().Dimensions[1] != prevK.Shape().Dimensions[1] {
		nextK = TransposeAllDims(nextK, 0, 2, 1, 3)
		nextV = TransposeAllDims(nextV, 0, 2, 1, 3)
	}

	updateSeqLen := nextK.Shape().Dimensions[2]

	batchIdx := Const(g, int32(0))
	headsIdx := Const(g, int32(0))
	dimIdx := Const(g, int32(0))

	updatedK := prevK
	updatedV := prevV

	for i := 0; i < updateSeqLen; i++ {
		pVal := AddScalar(position, i)

		var writeIdx *Node
		if k.SinkPositions > 0 && maxSeqLen > k.SinkPositions {
			rollingSize := maxSeqLen - k.SinkPositions
			wrappedVal := AddScalar(ModScalar(SubScalar(pVal, k.SinkPositions), rollingSize), k.SinkPositions)
			isBelowMax := LessThan(pVal, Const(g, int32(maxSeqLen)))
			writeIdx = Where(isBelowMax, pVal, wrappedVal)
		} else {
			writeIdx = ModScalar(pVal, maxSeqLen)
		}

		tokenK := Slice(nextK, AxisRange(), AxisRange(), AxisElem(i), AxisRange())
		tokenV := Slice(nextV, AxisRange(), AxisRange(), AxisElem(i), AxisRange())

		updatedK = DynamicUpdateSlice(updatedK, tokenK, []*Node{batchIdx, headsIdx, writeIdx, dimIdx})
		updatedV = DynamicUpdateSlice(updatedV, tokenV, []*Node{batchIdx, headsIdx, writeIdx, dimIdx})
	}

	cache[kKey] = updatedK
	cache[vKey] = updatedV
}

// Get retrieves the key and value nodes for the specified scope path from the cache map.
func (k *KVCache) Get(cache KVCacheNodes, path string) (key, value *Node) {
	return cache[path+KeySuffix], cache[path+ValueSuffix]
}

// BuildAttentionMask constructs the attention mask for the query at the current sequence position.
// It properly handles causal masking, sliding windows, and rolling cache/attention sinks, returning
// a boolean mask of shape [batchSize, qSeqLen, cacheSeqLen] where true means attend, false means ignore.
func (k *KVCache) BuildAttentionMask(scope *model.Scope, cache KVCacheNodes, query *Node, position *Node, useCausalMask bool, slidingWindow int) *Node {
	path := scope.Scope()
	keyCache := cache[path+KeySuffix]

	g := query.Graph()
	shape := keyCache.Shape()
	batchSize := shape.Dimensions[0]
	cacheSeqLen := shape.Dimensions[2]

	qSeqLen := query.Shape().Dimensions[1]

	currentSeqLen := AddScalar(position, qSeqLen)
	effectivePosition := MinScalar(currentSeqLen, cacheSeqLen)

	linearPositions := Iota(g, shapes.Make(dtypes.Int32, cacheSeqLen), 0)
	validMask := LessThan(linearPositions, effectivePosition)

	validMask3D := ExpandDims(validMask, 0)
	validMask3D = ExpandDims(validMask3D, 0)
	validMask3D = BroadcastToDims(validMask3D, batchSize, qSeqLen, cacheSeqLen)

	var mask *Node = validMask3D

	if useCausalMask {
		isWrapped := GreaterThan(currentSeqLen, Const(g, int32(cacheSeqLen)))

		var wrappedPositions *Node
		if k.SinkPositions > 0 && cacheSeqLen > k.SinkPositions {
			rollingSize := cacheSeqLen - k.SinkPositions
			lastWriteIdx := AddScalar(ModScalar(SubScalar(SubScalar(currentSeqLen, 1), k.SinkPositions), rollingSize), k.SinkPositions)

			isSink := LessThan(linearPositions, Const(g, int32(k.SinkPositions)))

			lastWriteIdxExpanded := BroadcastToShape(lastWriteIdx, linearPositions.Shape())
			currentSeqLenExpanded := BroadcastToShape(currentSeqLen, linearPositions.Shape())

			diff := Sub(lastWriteIdxExpanded, linearPositions)
			valIfBelow := Sub(SubScalar(currentSeqLenExpanded, 1), diff)
			valIfAbove := Sub(SubScalar(currentSeqLenExpanded, 1), AddScalar(diff, rollingSize))

			isBelowOrEqual := LessOrEqual(linearPositions, lastWriteIdxExpanded)
			rollingPositions := Where(isBelowOrEqual, valIfBelow, valIfAbove)

			wrappedPositions = Where(isSink, linearPositions, rollingPositions)
		} else {
			lastWriteIdx := ModScalar(SubScalar(currentSeqLen, 1), cacheSeqLen)
			lastWriteIdxExpanded := BroadcastToShape(lastWriteIdx, linearPositions.Shape())
			currentSeqLenExpanded := BroadcastToShape(currentSeqLen, linearPositions.Shape())

			diff := Sub(lastWriteIdxExpanded, linearPositions)
			valIfBelow := Sub(SubScalar(currentSeqLenExpanded, 1), diff)
			valIfAbove := Sub(SubScalar(currentSeqLenExpanded, 1), AddScalar(diff, cacheSeqLen))

			isBelowOrEqual := LessOrEqual(linearPositions, lastWriteIdxExpanded)
			wrappedPositions = Where(isBelowOrEqual, valIfBelow, valIfAbove)
		}

		absPositions := Where(isWrapped, wrappedPositions, linearPositions)

		queryPositions := Iota(g, shapes.Make(dtypes.Int32, qSeqLen), 0)
		queryPositions = Add(queryPositions, BroadcastToShape(position, queryPositions.Shape()))

		absQ := ExpandDims(queryPositions, -1)
		absK := ExpandDims(absPositions, 0)

		causalCheck := GreaterOrEqual(absQ, absK)

		causalCheck3D := ExpandDims(causalCheck, 0)
		causalCheck3D = BroadcastToDims(causalCheck3D, batchSize, qSeqLen, cacheSeqLen)

		mask = LogicalAnd(mask, causalCheck3D)

		if slidingWindow > 0 {
			isSink := LessThan(absK, Const(g, int32(k.SinkPositions)))
			dist := Sub(absQ, absK)
			withinWindow := LessThan(dist, Const(g, int32(slidingWindow)))
			slidingCheck := LogicalOr(isSink, withinWindow)

			slidingCheck3D := ExpandDims(slidingCheck, 0)
			slidingCheck3D = BroadcastToDims(slidingCheck3D, batchSize, qSeqLen, cacheSeqLen)

			mask = LogicalAnd(mask, slidingCheck3D)
		}
	}

	return mask
}
