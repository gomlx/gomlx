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

// whileStateWorkspace holds reusable slices for while loop execution.
type whileStateWorkspace struct {
	state       []*Buffer
	donateState []bool
}

// whileStateWorkspacePools pools whileStateWorkspace structs by state count.
var whileStateWorkspacePools [maxPooledRank + 1]sync.Pool

// getWhileStateWorkspace gets a whileStateWorkspace from the pool or allocates a new one.
func getWhileStateWorkspace(stateCount int) *whileStateWorkspace {
	if stateCount > maxPooledRank {
		return &whileStateWorkspace{
			state:       make([]*Buffer, stateCount),
			donateState: make([]bool, stateCount),
		}
	}
	if v := whileStateWorkspacePools[stateCount].Get(); v != nil {
		return v.(*whileStateWorkspace)
	}
	return &whileStateWorkspace{
		state:       make([]*Buffer, stateCount),
		donateState: make([]bool, stateCount),
	}
}

// putWhileStateWorkspace returns a whileStateWorkspace to the pool.
func putWhileStateWorkspace(ws *whileStateWorkspace) {
	stateCount := len(ws.state)
	if stateCount <= maxPooledRank {
		// Clear pointer slices to avoid holding references that prevent GC
		clear(ws.state)
		clear(ws.donateState)
		whileStateWorkspacePools[stateCount].Put(ws)
	}
}

// sortWorkspace holds reusable slices for sort execution.
type sortWorkspace struct {
	outputs    []*Buffer
	indices    []int
	compInputs []*Buffer
}

// sortWorkspacePools pools sortWorkspace structs by input count.
// Key is inputCount; indices size varies but we size to max seen.
var sortWorkspacePools [maxPooledRank + 1]sync.Pool

// getSortWorkspace gets a sortWorkspace from the pool or allocates a new one.
func getSortWorkspace(inputCount, axisSize int) *sortWorkspace {
	if inputCount > maxPooledRank {
		return &sortWorkspace{
			outputs:    make([]*Buffer, inputCount),
			indices:    make([]int, axisSize),
			compInputs: make([]*Buffer, 2*inputCount),
		}
	}
	if v := sortWorkspacePools[inputCount].Get(); v != nil {
		ws := v.(*sortWorkspace)
		// Resize indices if needed
		if cap(ws.indices) < axisSize {
			ws.indices = make([]int, axisSize)
		} else {
			ws.indices = ws.indices[:axisSize]
		}
		return ws
	}
	return &sortWorkspace{
		outputs:    make([]*Buffer, inputCount),
		indices:    make([]int, axisSize),
		compInputs: make([]*Buffer, 2*inputCount),
	}
}

// putSortWorkspace returns a sortWorkspace to the pool.
func putSortWorkspace(ws *sortWorkspace) {
	inputCount := len(ws.outputs)
	if inputCount <= maxPooledRank {
		// Clear pointer slices to avoid holding references that prevent GC
		clear(ws.outputs)
		clear(ws.compInputs)
		sortWorkspacePools[inputCount].Put(ws)
	}
}

// closureInputsWorkspace holds reusable slices for closure input construction.
// It provides flattened Buffers and Owned slices that can be sliced into for each closure.
type closureInputsWorkspace struct {
	// closureInputs is the slice of ClosureInputs structs (one per closure)
	closureInputs []ClosureInputs
	// buffers is a flat backing slice for all Buffers across closures
	buffers []*Buffer
	// owned is a flat backing slice for all Owned flags across closures
	owned []bool
}

// closureInputsWorkspacePools pools closureInputsWorkspace by number of closures.
var closureInputsWorkspacePools [4]sync.Pool // 0-3 closures (If/While have 2, Sort has 1)

// getClosureInputsWorkspace gets a workspace from the pool or allocates a new one.
// captureCounts is the number of captured inputs for each closure.
func getClosureInputsWorkspace(captureCounts []int) *closureInputsWorkspace {
	numClosures := len(captureCounts)
	totalCaptures := 0
	for _, c := range captureCounts {
		totalCaptures += c
	}

	var ws *closureInputsWorkspace
	if numClosures < len(closureInputsWorkspacePools) {
		if v := closureInputsWorkspacePools[numClosures].Get(); v != nil {
			ws = v.(*closureInputsWorkspace)
			// Resize backing slices if needed
			if cap(ws.buffers) < totalCaptures {
				ws.buffers = make([]*Buffer, totalCaptures)
			} else {
				ws.buffers = ws.buffers[:totalCaptures]
			}
			if cap(ws.owned) < totalCaptures {
				ws.owned = make([]bool, totalCaptures)
			} else {
				ws.owned = ws.owned[:totalCaptures]
				clear(ws.owned)
			}
		}
	}

	if ws == nil {
		ws = &closureInputsWorkspace{
			closureInputs: make([]ClosureInputs, numClosures),
			buffers:       make([]*Buffer, totalCaptures),
			owned:         make([]bool, totalCaptures),
		}
	}

	// Set up closureInputs to point into backing slices
	offset := 0
	for i, count := range captureCounts {
		ws.closureInputs[i] = ClosureInputs{
			Buffers: ws.buffers[offset : offset+count],
			Owned:   ws.owned[offset : offset+count],
		}
		offset += count
	}

	return ws
}

// putClosureInputsWorkspace returns a workspace to the pool.
func putClosureInputsWorkspace(ws *closureInputsWorkspace) {
	numClosures := len(ws.closureInputs)
	if numClosures < len(closureInputsWorkspacePools) {
		// Clear pointer slices to avoid holding references
		clear(ws.buffers)
		closureInputsWorkspacePools[numClosures].Put(ws)
	}
}
