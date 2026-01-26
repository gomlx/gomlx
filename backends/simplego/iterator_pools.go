// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import "sync"

// Iterator pools for reusing iterator structs during execution.
// Pools are indexed by rank (0-maxPooledRank). Ranks beyond maxPooledRank
// fall back to regular allocation.

const maxPooledRank = 8

// broadcastIteratorPools pools broadcastIterator structs by rank.
var broadcastIteratorPools [maxPooledRank + 1]sync.Pool

// getBroadcastIterator gets a broadcastIterator from the pool or allocates a new one.
// The caller must call putBroadcastIterator when done.
func getBroadcastIterator(rank int) *broadcastIterator {
	if rank > maxPooledRank {
		return &broadcastIterator{
			perAxesIdx:  make([]int, rank),
			targetDims:  make([]int, rank),
			isBroadcast: make([]bool, rank),
			strides:     make([]int, rank),
		}
	}
	if v := broadcastIteratorPools[rank].Get(); v != nil {
		bi := v.(*broadcastIterator)
		bi.flatIdx = 0
		clear(bi.perAxesIdx)
		return bi
	}
	return &broadcastIterator{
		perAxesIdx:  make([]int, rank),
		targetDims:  make([]int, rank),
		isBroadcast: make([]bool, rank),
		strides:     make([]int, rank),
	}
}

// putBroadcastIterator returns a broadcastIterator to the pool.
func putBroadcastIterator(bi *broadcastIterator) {
	rank := len(bi.perAxesIdx)
	if rank <= maxPooledRank {
		broadcastIteratorPools[rank].Put(bi)
	}
}

// transposeIteratorPools pools transposeIterator structs by rank.
var transposeIteratorPools [maxPooledRank + 1]sync.Pool

// transposeWorkspace holds temporary slices used during transpose iterator initialization.
type transposeWorkspace struct {
	stridesOnOutput     []int
	reversePermutations []int
}

// transposeWorkspacePools pools transposeWorkspace structs by rank.
var transposeWorkspacePools [maxPooledRank + 1]sync.Pool

// getTransposeIterator gets a transposeIterator from the pool or allocates a new one.
// The caller must call putTransposeIterator when done.
func getTransposeIterator(rank int) *transposeIterator {
	if rank > maxPooledRank {
		return &transposeIterator{
			perAxisIdx:     make([]int, rank),
			perAxisStrides: make([]int, rank),
			dimensions:     make([]int, rank),
		}
	}
	if v := transposeIteratorPools[rank].Get(); v != nil {
		it := v.(*transposeIterator)
		it.flatIdx = 0
		clear(it.perAxisIdx)
		return it
	}
	return &transposeIterator{
		perAxisIdx:     make([]int, rank),
		perAxisStrides: make([]int, rank),
		dimensions:     make([]int, rank),
	}
}

// putTransposeIterator returns a transposeIterator to the pool.
func putTransposeIterator(it *transposeIterator) {
	rank := len(it.perAxisIdx)
	if rank <= maxPooledRank {
		transposeIteratorPools[rank].Put(it)
	}
}

// getTransposeWorkspace gets temporary slices for transpose initialization.
func getTransposeWorkspace(rank int) *transposeWorkspace {
	if rank > maxPooledRank {
		return &transposeWorkspace{
			stridesOnOutput:     make([]int, rank),
			reversePermutations: make([]int, rank),
		}
	}
	if v := transposeWorkspacePools[rank].Get(); v != nil {
		return v.(*transposeWorkspace)
	}
	return &transposeWorkspace{
		stridesOnOutput:     make([]int, rank),
		reversePermutations: make([]int, rank),
	}
}

// putTransposeWorkspace returns transpose workspace to the pool.
func putTransposeWorkspace(ws *transposeWorkspace) {
	rank := len(ws.stridesOnOutput)
	if rank <= maxPooledRank {
		transposeWorkspacePools[rank].Put(ws)
	}
}

// reduceIteratorPools pools reduceOutputIterator structs by rank.
var reduceIteratorPools [maxPooledRank + 1]sync.Pool

// getReduceIterator gets a reduceOutputIterator from the pool or allocates a new one.
// The caller must call putReduceIterator when done.
func getReduceIterator(rank int) *reduceOutputIterator {
	if rank > maxPooledRank {
		return &reduceOutputIterator{
			perAxisIdx:    make([]int, rank),
			dimensions:    make([]int, rank),
			perAxisStride: make([]int, rank),
		}
	}
	if v := reduceIteratorPools[rank].Get(); v != nil {
		it := v.(*reduceOutputIterator)
		it.flatIdx = 0
		clear(it.perAxisIdx)
		return it
	}
	return &reduceOutputIterator{
		perAxisIdx:    make([]int, rank),
		dimensions:    make([]int, rank),
		perAxisStride: make([]int, rank),
	}
}

// putReduceIterator returns a reduceOutputIterator to the pool.
func putReduceIterator(it *reduceOutputIterator) {
	rank := len(it.perAxisIdx)
	if rank <= maxPooledRank {
		reduceIteratorPools[rank].Put(it)
	}
}
