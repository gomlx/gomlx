package simplego

import (
	"sync"
	"unsafe"

	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/xsync"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

// PreBlockedWeightCache caches pre-blocked weight buffers for reuse.
// Keys are the original buffer addresses to enable lookup during matmul.
type PreBlockedWeightCache struct {
	mu     sync.RWMutex
	cache  map[uintptr]*PreBlockedWeight
	lruIdx int64
}

// NewPreBlockedWeightCache creates a new pre-blocked weight cache.
func NewPreBlockedWeightCache() *PreBlockedWeightCache {
	return &PreBlockedWeightCache{
		cache: make(map[uintptr]*PreBlockedWeight),
	}
}

// bufferKey returns a unique key for a buffer based on its data pointer.
func bufferKey(buf *Buffer) uintptr {
	switch flat := buf.flat.(type) {
	case []float32:
		if len(flat) > 0 {
			return uintptr(unsafe.Pointer(&flat[0]))
		}
	case []float64:
		if len(flat) > 0 {
			return uintptr(unsafe.Pointer(&flat[0]))
		}
	case []int8:
		if len(flat) > 0 {
			return uintptr(unsafe.Pointer(&flat[0]))
		}
	case []int16:
		if len(flat) > 0 {
			return uintptr(unsafe.Pointer(&flat[0]))
		}
	case []int32:
		if len(flat) > 0 {
			return uintptr(unsafe.Pointer(&flat[0]))
		}
	case []int64:
		if len(flat) > 0 {
			return uintptr(unsafe.Pointer(&flat[0]))
		}
	case []uint8:
		if len(flat) > 0 {
			return uintptr(unsafe.Pointer(&flat[0]))
		}
	case []uint16:
		if len(flat) > 0 {
			return uintptr(unsafe.Pointer(&flat[0]))
		}
	case []uint32:
		if len(flat) > 0 {
			return uintptr(unsafe.Pointer(&flat[0]))
		}
	case []uint64:
		if len(flat) > 0 {
			return uintptr(unsafe.Pointer(&flat[0]))
		}
	}
	return 0
}

// Get retrieves a pre-blocked weight for the given buffer, if available.
func (c *PreBlockedWeightCache) Get(buf *Buffer) *PreBlockedWeight {
	key := bufferKey(buf)
	if key == 0 {
		return nil
	}
	c.mu.RLock()
	pbw := c.cache[key]
	c.mu.RUnlock()
	return pbw
}

// Set stores a pre-blocked weight for the given buffer.
func (c *PreBlockedWeightCache) Set(buf *Buffer, pbw *PreBlockedWeight) {
	key := bufferKey(buf)
	if key == 0 {
		return
	}
	c.mu.Lock()
	c.cache[key] = pbw
	c.mu.Unlock()
}

// Invalidate removes a pre-blocked weight from the cache.
// Call this when the underlying buffer data has changed or when the buffer is freed.
// This prevents stale cache entries when memory is reused.
func (c *PreBlockedWeightCache) Invalidate(buf *Buffer) {
	key := bufferKey(buf)
	if key == 0 {
		return
	}
	c.mu.Lock()
	delete(c.cache, key)
	c.mu.Unlock()
}

// Clear removes all entries from the cache.
func (c *PreBlockedWeightCache) Clear() {
	c.mu.Lock()
	c.cache = make(map[uintptr]*PreBlockedWeight)
	c.mu.Unlock()
}

// GetOrCreate retrieves a pre-blocked weight or creates one if not cached.
// This is the main entry point for lazily caching pre-blocked weights.
func (c *PreBlockedWeightCache) GetOrCreate(buf *Buffer) *PreBlockedWeight {
	// Check cache first
	if pbw := c.Get(buf); pbw != nil {
		return pbw
	}

	// Create new pre-blocked weight
	pbw := PreBlockWeightForMatMul(buf)
	if pbw != nil {
		c.Set(buf, pbw)
	}
	return pbw
}

// execDotGeneralWithPreBlockedRHS executes DotGeneral using a pre-blocked RHS weight.
// This skips the RHS blocking step entirely, providing significant speedup for inference.
// Supports both unbatched [M, K] × [K, N] and batched [B, M, K] × [K, N] operations.
func execDotGeneralWithPreBlockedRHS(backend *Backend, lhs *Buffer, rhsPreBlocked *PreBlockedWeight, params *dotGeneralNodeData, output *Buffer) error {
	dtype := lhs.shape.DType

	// Get the pre-blocked RHS buffer (shared across all batch elements)
	rhsBlocks := rhsPreBlocked.GetPreBlockedBuffer()

	// We still need to block the LHS (activations)
	blkLog2Dim := rhsPreBlocked.BlockLog2Dim
	blockDim := 1 << blkLog2Dim

	// Get LHS blocked buffer
	lhsBlocks := backend.getBuffer(dtype, params.lhsBlockedShape.Size())
	lhsBlocks.shape = params.lhsBlockedShape
	lhsBlocks.Zeros()

	// Copy LHS to blocked format
	copyFlatToBlock := dotGeneralFlatToBlockDTypeMap.Get(dtype).(func(source, blkOutput *Buffer, contractingAxes, batchAxes []int, batchSize, crossSize, contractingSize, blkLog2Dim int))
	copyFlatToBlock(lhs, lhsBlocks, params.lhsContractingAxes, params.lhsBatchAxes, params.batchSize, params.lhsCrossSize, params.contractingSize, blkLog2Dim)

	// Get output blocked buffer
	// Use params.outputBlockedShape.DType which is the accumulator type (Float32 for FP16/BF16, Int32 for Int8/Uint8)
	accumulatorDType := params.outputBlockedShape.DType
	outputBlocks := backend.getBuffer(accumulatorDType, params.outputBlockedShape.Size())
	outputBlocks.shape = params.outputBlockedShape
	outputBlocks.Zeros()

	// Set up base recursive data for kernel execution
	var recursive dotGeneralRecursiveData
	recursive.backend = backend

	// Get the matrix multiplication kernel for a block.
	kernelBuilder := dotGeneralKernelDTypeMap.Get(dtype).(func(lhs, rhs, output *Buffer, blockDim int) kernelFuncType)
	recursive.kernelFn = kernelBuilder(lhsBlocks, rhsBlocks, outputBlocks, blockDim)

	// Set block counts
	recursive.lhsCrossBlocks = lhsBlocks.shape.Dimensions[1]
	recursive.rhsCrossBlocks = rhsBlocks.shape.Dimensions[1]
	recursive.contractBlocks = lhsBlocks.shape.Dimensions[2]

	// Decide on intra-example parallelism
	maxParallelism := backend.workers.MaxParallelism()
	recursive.maxDepthParallelization = -1 // Disable sub-batch parallelization.
	if backend.workers.IsEnabled() {
		if backend.workers.IsUnlimited() {
			recursive.maxDepthParallelization = 8
		} else {
			recursive.maxDepthParallelization = log2int(maxParallelism) + 1
		}
	}

	// Decide on using parallelism across the batch
	useBatchParallelism := backend.workers.IsEnabled()
	batchSplitSize := 1
	if useBatchParallelism && !backend.workers.IsUnlimited() {
		batchSplitSize = (params.batchSize + maxParallelism - 1) / maxParallelism
	}

	// Loop over examples in the batch
	// RHS is pre-blocked and shared across all batches (rhsBatchOffset = 0)
	wg := xsync.NewDynamicWaitGroup()
	for outerBatchIdx := 0; outerBatchIdx < params.batchSize; outerBatchIdx += batchSplitSize {
		wg.Add(1)
		batchSplitFn := func() {
			for innerBatchIdx := outerBatchIdx; innerBatchIdx < min(outerBatchIdx+batchSplitSize, params.batchSize); innerBatchIdx++ {
				var batchRecursive dotGeneralRecursiveData
				batchRecursive = recursive
				batchRecursive.lhsBatchOffset = innerBatchIdx * recursive.lhsCrossBlocks * recursive.contractBlocks
				batchRecursive.rhsBatchOffset = 0 // RHS is shared - always use batch 0
				batchRecursive.outputBatchOffset = innerBatchIdx * recursive.lhsCrossBlocks * recursive.rhsCrossBlocks
				wg.Add(1)
				batchRecursive.apply(0, recursive.lhsCrossBlocks, 0, recursive.rhsCrossBlocks, 0, recursive.contractBlocks, 0, wg)
			}
			wg.Done()
		}
		if useBatchParallelism {
			backend.workers.WaitToStart(batchSplitFn)
		} else {
			batchSplitFn()
		}
	}
	wg.Wait()

	// Free the LHS block buffer (RHS is pre-blocked and reusable)
	backend.putBuffer(lhsBlocks)

	// Copy output from blocked to flat format
	// Use final output dtype (e.g., Float16) to get correct conversion function
	finalOutputDType := output.shape.DType
	copyOutputBlockToFlat := dotGeneralOutputBlockToFlatDTypeMap.Get(finalOutputDType).(func(blockedSource, output *Buffer))
	copyOutputBlockToFlat(outputBlocks, output)
	backend.putBuffer(outputBlocks)

	return nil
}

// TryExecDotGeneralWithPreBlockedWeights attempts to use pre-blocked weights for DotGeneral.
// Returns true if successful, false if the operation should fall back to standard path.
func TryExecDotGeneralWithPreBlockedWeights(backend *Backend, lhs, rhs *Buffer, params *dotGeneralNodeData, output *Buffer) bool {
	// Check if this is a standard 2D matmul pattern that we can optimize
	if !canUsePreBlockedPath(lhs, rhs, params) {
		return false
	}

	// Check the pre-blocked weight cache
	pbw := backend.preBlockedWeightCache.GetOrCreate(rhs)
	if pbw == nil {
		return false
	}

	// Verify the pre-blocked weight is compatible
	if !CanUsePreBlockedWeight(pbw, rhs.shape, params) {
		return false
	}

	// Execute using pre-blocked RHS
	err := execDotGeneralWithPreBlockedRHS(backend, lhs, pbw, params, output)
	return err == nil
}

// canUsePreBlockedPath checks if the DotGeneral operation can use pre-blocked weights.
// This supports both unbatched matmul [M, K] × [K, N] and batched matmul [B, M, K] × [K, N]
// where the RHS weights are shared across the batch dimension.
func canUsePreBlockedPath(lhs, rhs *Buffer, params *dotGeneralNodeData) bool {
	// RHS (weights) must be 2D: [K, N]
	// LHS can be 2D [M, K] or have batch dimensions that get normalized
	if rhs.shape.Rank() != 2 {
		return false
	}

	// RHS must have no batch dimensions (weights are shared across batch)
	if len(params.rhsBatchAxes) != 0 {
		return false
	}

	// Single contracting axis for both
	if len(params.lhsContractingAxes) != 1 || len(params.rhsContractingAxes) != 1 {
		return false
	}

	// RHS contracts on axis 0 (standard [K, N] weight layout)
	if params.rhsContractingAxes[0] != 0 {
		return false
	}

	// LHS must contract on its last axis (the K dimension)
	// For 2D: [M, K] contracts on axis 1
	// For 3D: [B, M, K] contracts on axis 2
	// In normalized form, this is always the last axis
	lhsRank := lhs.shape.Rank()
	expectedLhsContractingAxis := lhsRank - 1
	if params.lhsContractingAxes[0] != expectedLhsContractingAxis {
		return false
	}

	// Check dtype is supported
	// Float16 and BFloat16 accumulate to Float32 internally and convert back via
	// dgCopyOutputBlockToFlatFloat16/dgCopyOutputBlockToFlatBFloat16.
	// Int8/Uint8 accumulate to Int32 (output dtype is Int32, not converted back).
	dtype := lhs.shape.DType
	switch dtype {
	case dtypes.Float32, dtypes.Float64,
		dtypes.Float16, dtypes.BFloat16,
		dtypes.Int8, dtypes.Uint8,
		dtypes.Int16, dtypes.Int32, dtypes.Int64,
		dtypes.Uint16, dtypes.Uint32, dtypes.Uint64:
		return true
	default:
		return false
	}
}

// PreBlockWeight pre-blocks a weight buffer and caches it for future use.
// Call this at model load time for weight tensors that will be used in matmul.
// Returns the pre-blocked weight or nil if blocking failed.
func (b *Backend) PreBlockWeight(buf *Buffer) *PreBlockedWeight {
	return b.preBlockedWeightCache.GetOrCreate(buf)
}

// PreBlockWeightFromShape pre-blocks a weight for the given shape.
// This is useful when you know the shape but don't have the buffer yet.
func PreBlockedShapeForWeight(originalShape shapes.Shape) shapes.Shape {
	if originalShape.Rank() != 2 {
		return shapes.Shape{}
	}

	dtype := originalShape.DType
	K := originalShape.Dimensions[0]
	N := originalShape.Dimensions[1]

	blockLog2Dim := DotGeneralTargetBlockLog2Dim[dtype]
	return dgCreateBlockedShape(dtype, 1, N, K, blockLog2Dim)
}

// PreBlockedWeightStats returns statistics about the pre-blocked weight cache.
func (b *Backend) PreBlockedWeightStats() (count int, totalBytes int64) {
	b.preBlockedWeightCache.mu.RLock()
	defer b.preBlockedWeightCache.mu.RUnlock()

	count = len(b.preBlockedWeightCache.cache)
	for _, pbw := range b.preBlockedWeightCache.cache {
		totalBytes += int64(pbw.BlockedShape.Size()) * int64(pbw.BlockedShape.DType.Size())
	}
	return
}
