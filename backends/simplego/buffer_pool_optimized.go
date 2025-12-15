package simplego

import (
	"reflect"
	"sync"

	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// Size class boundaries for buffer pooling.
// We use powers of 2 to enable efficient size class lookup.
// This reduces fragmentation and improves cache behavior.
var sizeClasses = []int{
	1 << 8,   // 256 elements
	1 << 10,  // 1K elements
	1 << 12,  // 4K elements
	1 << 14,  // 16K elements
	1 << 16,  // 64K elements
	1 << 18,  // 256K elements
	1 << 20,  // 1M elements
	1 << 22,  // 4M elements
	1 << 24,  // 16M elements
}

// sizeClassPoolKey uses dtype and size class index for more efficient pooling.
type sizeClassPoolKey struct {
	dtype     dtypes.DType
	sizeClass int // index into sizeClasses, or -1 for exact size
}

// getSizeClass returns the index of the appropriate size class for a given size.
// Returns -1 if the size is larger than all size classes (uses exact matching).
func getSizeClass(size int) int {
	for i, classSize := range sizeClasses {
		if size <= classSize {
			return i
		}
	}
	return -1 // Too large, use exact size pooling
}

// getSizeClassCapacity returns the capacity for a given size class.
func getSizeClassCapacity(sizeClass int) int {
	if sizeClass < 0 || sizeClass >= len(sizeClasses) {
		return 0
	}
	return sizeClasses[sizeClass]
}

// OptimizedBufferPool provides size-class based buffer pooling for reduced allocation overhead.
type OptimizedBufferPool struct {
	// pools maps (dtype, sizeClass) to a sync.Pool
	pools sync.Map

	// exactPools for buffers larger than max size class (fallback to exact matching)
	exactPools sync.Map
}

// NewOptimizedBufferPool creates a new optimized buffer pool.
func NewOptimizedBufferPool() *OptimizedBufferPool {
	return &OptimizedBufferPool{}
}

// getPool retrieves or creates a pool for the given dtype and size class.
func (p *OptimizedBufferPool) getPool(dtype dtypes.DType, sizeClass int, exactSize int) *sync.Pool {
	if sizeClass >= 0 {
		key := sizeClassPoolKey{dtype: dtype, sizeClass: sizeClass}
		poolInterface, ok := p.pools.Load(key)
		if !ok {
			capacity := getSizeClassCapacity(sizeClass)
			poolInterface, _ = p.pools.LoadOrStore(key, &sync.Pool{
				New: func() interface{} {
					return &Buffer{
						flat:  reflect.MakeSlice(reflect.SliceOf(dtype.GoType()), capacity, capacity).Interface(),
						shape: shapes.Make(dtype, capacity),
					}
				},
			})
		}
		return poolInterface.(*sync.Pool)
	}

	// Exact size pooling for large buffers
	key := bufferPoolKey{dtype: dtype, length: exactSize}
	poolInterface, ok := p.exactPools.Load(key)
	if !ok {
		poolInterface, _ = p.exactPools.LoadOrStore(key, &sync.Pool{
			New: func() interface{} {
				return &Buffer{
					flat:  reflect.MakeSlice(reflect.SliceOf(dtype.GoType()), exactSize, exactSize).Interface(),
					shape: shapes.Make(dtype, exactSize),
				}
			},
		})
	}
	return poolInterface.(*sync.Pool)
}

// Get retrieves a buffer of at least the requested size.
// The returned buffer may have capacity larger than requested.
func (p *OptimizedBufferPool) Get(dtype dtypes.DType, size int) *Buffer {
	sizeClass := getSizeClass(size)
	pool := p.getPool(dtype, sizeClass, size)
	buf := pool.Get().(*Buffer)
	buf.valid = true
	return buf
}

// Put returns a buffer to the appropriate pool.
func (p *OptimizedBufferPool) Put(buf *Buffer) {
	if buf == nil || !buf.shape.Ok() {
		return
	}
	buf.valid = false

	// Determine which pool to return to based on actual capacity
	capacity := buf.shape.Size()
	sizeClass := getSizeClass(capacity)
	pool := p.getPool(buf.shape.DType, sizeClass, capacity)
	pool.Put(buf)
}

// Clear releases all pooled buffers.
func (p *OptimizedBufferPool) Clear() {
	p.pools.Clear()
	p.exactPools.Clear()
}

// PrewarmPool pre-allocates buffers for common sizes to reduce cold-start latency.
// Call this after model loading to warm up the buffer pool.
func (p *OptimizedBufferPool) PrewarmPool(dtype dtypes.DType, sizes []int, count int) {
	for _, size := range sizes {
		sizeClass := getSizeClass(size)
		pool := p.getPool(dtype, sizeClass, size)

		// Pre-allocate and immediately return to pool
		buffers := make([]*Buffer, count)
		for i := 0; i < count; i++ {
			buffers[i] = pool.Get().(*Buffer)
			buffers[i].valid = true
		}
		for _, buf := range buffers {
			buf.valid = false
			pool.Put(buf)
		}
	}
}
