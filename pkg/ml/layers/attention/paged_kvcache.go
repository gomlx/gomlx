package attention

import (
	"errors"
	"sync"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
)

// Paged KV cache errors.
var (
	ErrNoFreeBlocks = errors.New("paged KV cache: no free blocks available")
)

// PagedKVCacheConfig configures the paged KV cache.
type PagedKVCacheConfig struct {
	// NumBlocks is the total number of physical blocks available.
	NumBlocks int

	// BlockSize is the number of tokens stored per block.
	// Common values: 16, 32.
	BlockSize int

	// NumKVHeads is the number of key/value attention heads.
	NumKVHeads int

	// HeadDim is the dimension of each attention head.
	HeadDim int

	// DType is the data type for cached KV entries.
	DType dtypes.DType
}

// BlockManager manages allocation and freeing of physical KV cache blocks.
// It is CPU-side only and operates outside the computation graph.
//
// Physical storage has shape [NumBlocks, NumKVHeads, BlockSize, HeadDim].
// Each request has a page table mapping logical block indices to physical
// block indices. Logical position p maps to:
//
//	block = pageTable[p / BlockSize]
//	offset = p % BlockSize
type BlockManager struct {
	config PagedKVCacheConfig

	mu         sync.Mutex
	freeBlocks []int            // stack of free physical block indices
	pageTable  map[uint64][]int // requestID → ordered physical block indices
}

// NewBlockManager creates a new block manager with the given configuration.
func NewBlockManager(config PagedKVCacheConfig) *BlockManager {
	free := make([]int, config.NumBlocks)
	for i := range config.NumBlocks {
		free[i] = config.NumBlocks - 1 - i // stack order: pop gives lowest first
	}
	return &BlockManager{
		config:     config,
		freeBlocks: free,
		pageTable:  make(map[uint64][]int),
	}
}

// AllocateBlocks allocates n physical blocks for the given request.
// Returns the indices of the allocated blocks.
func (bm *BlockManager) AllocateBlocks(requestID uint64, n int) ([]int, error) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	if len(bm.freeBlocks) < n {
		return nil, ErrNoFreeBlocks
	}

	allocated := make([]int, n)
	for i := range n {
		allocated[i] = bm.freeBlocks[len(bm.freeBlocks)-1]
		bm.freeBlocks = bm.freeBlocks[:len(bm.freeBlocks)-1]
	}

	bm.pageTable[requestID] = append(bm.pageTable[requestID], allocated...)
	return allocated, nil
}

// ReleaseRequest returns all blocks for the given request to the free pool
// and removes the request's page table entry.
func (bm *BlockManager) ReleaseRequest(requestID uint64) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	blocks, ok := bm.pageTable[requestID]
	if !ok {
		return
	}

	bm.freeBlocks = append(bm.freeBlocks, blocks...)
	delete(bm.pageTable, requestID)
}

// GetPageTable returns the page table (list of physical block indices) for a request.
func (bm *BlockManager) GetPageTable(requestID uint64) []int {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	result := make([]int, len(bm.pageTable[requestID]))
	copy(result, bm.pageTable[requestID])
	return result
}

// NumFreeBlocks returns the number of available blocks.
func (bm *BlockManager) NumFreeBlocks() int {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	return len(bm.freeBlocks)
}

// BlocksNeeded returns the number of blocks needed to store seqLen tokens.
func (bm *BlockManager) BlocksNeeded(seqLen int) int {
	return (seqLen + bm.config.BlockSize - 1) / bm.config.BlockSize
}

// EnsureBlocks ensures the request has enough blocks for the given sequence length.
// Allocates additional blocks if needed.
func (bm *BlockManager) EnsureBlocks(requestID uint64, seqLen int) error {
	needed := bm.BlocksNeeded(seqLen)

	bm.mu.Lock()
	defer bm.mu.Unlock()

	current := len(bm.pageTable[requestID])
	if current >= needed {
		return nil
	}

	toAllocate := needed - current
	if len(bm.freeBlocks) < toAllocate {
		return ErrNoFreeBlocks
	}

	allocated := make([]int, toAllocate)
	for i := range toAllocate {
		allocated[i] = bm.freeBlocks[len(bm.freeBlocks)-1]
		bm.freeBlocks = bm.freeBlocks[:len(bm.freeBlocks)-1]
	}
	bm.pageTable[requestID] = append(bm.pageTable[requestID], allocated...)
	return nil
}

// DetachBlocks removes the specified blocks from a request's page table
// without returning them to the free pool. Used for prefix caching where
// blocks are shared across requests.
func (bm *BlockManager) DetachBlocks(requestID uint64, blocks []int) {
	bm.mu.Lock()
	defer bm.mu.Unlock()

	pt := bm.pageTable[requestID]
	if pt == nil {
		return
	}

	detachSet := make(map[int]struct{}, len(blocks))
	for _, b := range blocks {
		detachSet[b] = struct{}{}
	}

	remaining := make([]int, 0, len(pt))
	for _, b := range pt {
		if _, found := detachSet[b]; !found {
			remaining = append(remaining, b)
		}
	}
	bm.pageTable[requestID] = remaining
}

// RecycleBlocks adds blocks directly to the free pool without touching any
// page table. Used when prefix-cached blocks have their reference count
// drop to zero and can be reused.
func (bm *BlockManager) RecycleBlocks(blocks []int) {
	bm.mu.Lock()
	defer bm.mu.Unlock()
	bm.freeBlocks = append(bm.freeBlocks, blocks...)
}

// PhysicalCacheShape returns the shape of the physical KV cache storage.
func (c PagedKVCacheConfig) PhysicalCacheShape() shapes.Shape {
	return shapes.Make(c.DType, c.NumBlocks, c.NumKVHeads, c.BlockSize, c.HeadDim)
}

// PagedKVCacheGetVars returns the physical key and value cache variables.
// Variables are created on first access with zero-initialization.
// Shape: [NumBlocks, NumKVHeads, BlockSize, HeadDim]
func PagedKVCacheGetVars(ctx *context.Context, config PagedKVCacheConfig) (keyVar, valueVar *context.Variable) {
	return KVCacheGetVars(ctx, config.PhysicalCacheShape())
}

// PagedKVCacheReset zeroes out the physical KV cache storage.
func PagedKVCacheReset(ctx *context.Context) {
	// Reuse the standard KVCacheReset — variable naming is the same.
	KVCacheReset(ctx)
}

// pagedCacheWriteElement writes one token's key/value to the paged cache using
// the page table to map a logical position to a physical block and offset.
//
// Parameters:
//   - keyCache, valueCache: [NumBlocks, NumKVHeads, BlockSize, HeadDim]
//   - headsIdx, dimIdx: scalar int32(0) — zero start indices for heads and dim axes
//   - pageTable: [maxBlocks] int32 — physical block indices for this request
//   - pos: scalar int32 — logical position in sequence
//   - newKey, newValue: [1, NumKVHeads, 1, HeadDim] — new projections for one token
//   - blockSize: tokens per block
func pagedCacheWriteElement(
	keyCache, valueCache *Node,
	headsIdx, dimIdx *Node,
	pageTable, pos *Node,
	newKey, newValue *Node,
	blockSize int,
) (*Node, *Node) {
	blockIdx := DivScalar(pos, blockSize)
	offset := ModScalar(pos, blockSize)
	physicalBlock := Reshape(GatherSlices(pageTable, []int{0}, blockIdx, []int{1}, false))

	keyCache = DynamicUpdateSlice(keyCache, newKey, []*Node{physicalBlock, headsIdx, offset, dimIdx})
	valueCache = DynamicUpdateSlice(valueCache, newValue, []*Node{physicalBlock, headsIdx, offset, dimIdx})
	return keyCache, valueCache
}

// PagedKVCacheWrite writes new key/value entries at the specified logical position
// using the page table to map to physical blocks.
//
// Parameters:
//   - ctx: Context for cache variables
//   - g: Graph
//   - config: Paged cache configuration
//   - pageTableNode: [maxBlocks] int32 — physical block indices for this request
//   - logicalPosition: Scalar int32 — logical position in sequence
//   - newKey: [1, NumKVHeads, 1, HeadDim] — new key projection for one token
//   - newValue: [1, NumKVHeads, 1, HeadDim] — new value projection for one token
func PagedKVCacheWrite(ctx *context.Context, g *Graph, config PagedKVCacheConfig, pageTableNode *Node, logicalPosition *Node, newKey, newValue *Node) {
	keyVar, valueVar := PagedKVCacheGetVars(ctx, config)
	keyCache := keyVar.ValueGraph(g)
	valueCache := valueVar.ValueGraph(g)

	pos := ConvertDType(logicalPosition, dtypes.Int32)
	pos = Reshape(pos) // scalar

	headsIdx := Const(g, int32(0))
	dimIdx := Const(g, int32(0))

	keyCache, valueCache = pagedCacheWriteElement(
		keyCache, valueCache,
		headsIdx, dimIdx,
		pageTableNode, pos,
		newKey, newValue,
		config.BlockSize,
	)

	keyVar.SetValueGraph(keyCache)
	valueVar.SetValueGraph(valueCache)
}

// PagedKVCacheRead gathers all cached KV entries for a request using its page table.
// Returns keys and values with shape [1, numKVHeads, seqLen, headDim] where
// seqLen = numBlocks * blockSize.
//
// Parameters:
//   - ctx: Context for cache variables
//   - g: Graph
//   - config: Paged cache configuration
//   - pageTableNode: [numBlocks] int32 — physical block indices
//   - numBlocks: Number of blocks to read (compile-time constant)
func PagedKVCacheRead(ctx *context.Context, g *Graph, config PagedKVCacheConfig, pageTableNode *Node, numBlocks int) (keys, values *Node) {
	keyVar, valueVar := PagedKVCacheGetVars(ctx, config)
	keyCache := keyVar.ValueGraph(g)   // [NumBlocks, NumKVHeads, BlockSize, HeadDim]
	valueCache := valueVar.ValueGraph(g)

	// Gather blocks using page table indices.
	// pageTableNode: [numBlocks] int32 → indices: [numBlocks, 1]
	indices := ExpandDims(pageTableNode, -1)

	// Gather from keyCache: indexed on axis 0 (block dim)
	// Result: [numBlocks, NumKVHeads, BlockSize, HeadDim]
	gatheredKeys := Gather(keyCache, indices)
	gatheredValues := Gather(valueCache, indices)

	// Reshape: [numBlocks, numKVHeads, blockSize, headDim]
	//       → [numKVHeads, numBlocks*blockSize, headDim]
	//       → [1, numKVHeads, seqLen, headDim]
	seqLen := numBlocks * config.BlockSize

	// Transpose: [numBlocks, numKVHeads, blockSize, headDim] → [numKVHeads, numBlocks, blockSize, headDim]
	gatheredKeys = TransposeAllDims(gatheredKeys, 1, 0, 2, 3)
	gatheredValues = TransposeAllDims(gatheredValues, 1, 0, 2, 3)

	// Reshape: [numKVHeads, numBlocks, blockSize, headDim] → [numKVHeads, seqLen, headDim]
	gatheredKeys = Reshape(gatheredKeys, config.NumKVHeads, seqLen, config.HeadDim)
	gatheredValues = Reshape(gatheredValues, config.NumKVHeads, seqLen, config.HeadDim)

	// Add batch dim: [1, numKVHeads, seqLen, headDim]
	keys = ExpandDims(gatheredKeys, 0)
	values = ExpandDims(gatheredValues, 0)
	return
}

// PagedKVCacheWriteBatched writes new key/value entries for a batch of requests,
// each with their own page table and position.
//
// Parameters:
//   - ctx: Context for cache variables
//   - g: Graph
//   - config: Paged cache configuration
//   - pageTables: [batchSize, maxBlocksPerRequest] int32 — per-request page tables
//   - positions: [batchSize] int32 — per-request logical positions
//   - newKeys: [batchSize, NumKVHeads, 1, HeadDim] — new keys
//   - newValues: [batchSize, NumKVHeads, 1, HeadDim] — new values
func PagedKVCacheWriteBatched(ctx *context.Context, g *Graph, config PagedKVCacheConfig, pageTables *Node, positions *Node, newKeys, newValues *Node) {
	keyVar, valueVar := PagedKVCacheGetVars(ctx, config)
	keyCache := keyVar.ValueGraph(g)
	valueCache := valueVar.ValueGraph(g)

	batchSize := newKeys.Shape().Dimensions[0]
	posI32 := ConvertDType(positions, dtypes.Int32)

	headsIdx := Const(g, int32(0))
	dimIdx := Const(g, int32(0))

	for b := range batchSize {
		// Get this batch element's position and page table.
		pos := Reshape(Slice(posI32, AxisElem(b)))

		batchPageTable := Slice(pageTables, AxisElem(b), AxisRange())
		batchPageTable = Reshape(batchPageTable, batchPageTable.Shape().Dimensions[len(batchPageTable.Shape().Dimensions)-1])

		// Extract this batch element's key/value: [1, numKVHeads, 1, headDim]
		batchKey := Slice(newKeys, AxisElem(b), AxisRange(), AxisRange(), AxisRange())
		batchVal := Slice(newValues, AxisElem(b), AxisRange(), AxisRange(), AxisRange())

		keyCache, valueCache = pagedCacheWriteElement(
			keyCache, valueCache,
			headsIdx, dimIdx,
			batchPageTable, pos,
			batchKey, batchVal,
			config.BlockSize,
		)
	}

	keyVar.SetValueGraph(keyCache)
	valueVar.SetValueGraph(valueCache)
}

// BuildPageTableTensor creates a tensor from page table block indices,
// padded to maxBlocksPerRequest. Padding uses block index 0 (safe because
// padded positions are never read).
func BuildPageTableTensor(pageTable []int, maxBlocksPerRequest int) *tensors.Tensor {
	padded := make([]int32, maxBlocksPerRequest)
	for i, block := range pageTable {
		if i >= maxBlocksPerRequest {
			break
		}
		padded[i] = int32(block)
	}
	return tensors.FromValue(padded)
}
