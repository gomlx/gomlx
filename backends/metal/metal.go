//go:build darwin && cgo

// Package metal implements a GoMLX backend that dispatches compute operations to
// Apple Silicon GPUs via Metal compute shaders.
//
// Buffer allocation uses MTLResourceStorageModeShared (unified memory), so there
// is no host↔device copy on Apple Silicon — the CPU and GPU share the same
// physical RAM.
package metal

/*
#cgo CFLAGS: -x objective-c -fobjc-arc -I${SRCDIR}
#cgo LDFLAGS: -framework Metal -framework Foundation
#include "metal.h"
#include <stdlib.h>
#include <string.h>
*/
import "C"
import (
	_ "embed"
	"fmt"
	"os"
	"sync"
	"sync/atomic"
	"unsafe"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

//go:generate xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -c kernels/elementwise.metal -o kernels/elementwise.air
//go:generate xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -c kernels/reduce.metal -o kernels/reduce.air
//go:generate xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -c kernels/dot_general.metal -o kernels/dot_general.air
//go:generate xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -c kernels/fused_ops.metal -o kernels/fused_ops.air
//go:generate xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -c kernels/attention.metal -o kernels/attention.air
//go:generate xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -c kernels/tensor_ops.metal -o kernels/tensor_ops.air
//go:generate xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -c kernels/rng_pcg.metal -o kernels/rng_pcg.air
//go:generate xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -c kernels/batch_norm.metal -o kernels/batch_norm.air
//go:generate xcrun -sdk macosx metal -std=metal3.1 -mmacosx-version-min=14.0 -c kernels/quantized_dense.metal -o kernels/quantized_dense.air
//go:generate xcrun -sdk macosx metallib kernels/elementwise.air kernels/reduce.air kernels/dot_general.air kernels/fused_ops.air kernels/attention.air kernels/tensor_ops.air kernels/rng_pcg.air kernels/batch_norm.air kernels/quantized_dense.air -o gomlx_metal.metallib

//go:embed gomlx_metal.metallib
var embeddedMetallib []byte

var metalReady atomic.Bool

// BackendName is the name used to register and select this backend.
const BackendName = "metal"

func init() {
	backends.Register(BackendName, New)
}

func initMetal() {
	tmpFile, err := os.CreateTemp("", "gomlx-metal-*.metallib")
	if err != nil {
		reportInitError(err)
		return
	}
	name := tmpFile.Name()
	defer os.Remove(name)

	if _, err := tmpFile.Write(embeddedMetallib); err != nil {
		tmpFile.Close()
		reportInitError(err)
		return
	}
	if err := tmpFile.Close(); err != nil {
		reportInitError(err)
		return
	}

	cPath := C.CString(name)
	defer C.free(unsafe.Pointer(cPath))

	if res := C.metal_init(cPath); res != 0 {
		reportInitError(fmt.Errorf("metal_init returned %d", res))
		return
	}

	metalReady.Store(true)
}

func reportInitError(err error) {
	_, _ = fmt.Fprintf(os.Stderr, "gomlx-metal: init: %v\n", err)
}

// ─── Backend ────────────────────────────────────────────────────────────────

// Backend dispatches GoMLX operations to Apple Metal GPU.
type Backend struct {
	notimplemented.Backend
	isFinalized bool

	// execMu serializes Execute (including nested closure execution on this backend).
	execMu sync.Mutex

	// scratch is the active execScratch for the innermost in-flight Execute on this backend.
	scratch *execScratch
}

var _ backends.Backend = (*Backend)(nil)

// metalBackendUsers counts live *Backend values. The Objective-C runtime is
// torn down only when the last backend is Finalized.
var metalBackendUsers atomic.Int32

// New constructs a new Metal backend. Config is currently ignored.
func New(config string) (backends.Backend, error) {
	if !metalReady.Load() {
		initMetal()
	}
	if !metalReady.Load() {
		return nil, errors.New("metal backend not available")
	}
	metalBackendUsers.Add(1)
	return &Backend{}, nil
}

// GetBackend returns a singleton Metal backend.
var GetBackend = sync.OnceValue(func() backends.Backend {
	b, err := New("")
	if err != nil {
		panic(err)
	}
	return b
})

func (b *Backend) Name() string        { return BackendName }
func (b *Backend) String() string      { return b.Name() }
func (b *Backend) Description() string { return "Metal GPU backend for Apple Silicon" }

// NumDevices reports 1: the Metal layer uses MTLCreateSystemDefaultDevice only.
func (b *Backend) NumDevices() int   { return 1 }
func (b *Backend) IsFinalized() bool { return b.isFinalized }

func (b *Backend) DeviceDescription(deviceNum backends.DeviceNum) string {
	return C.GoString(C.metal_device_name())
}

func (b *Backend) Capabilities() backends.Capabilities {
	return MetalCapabilities
}

func (b *Backend) Finalize() {
	if b.isFinalized {
		return
	}
	b.isFinalized = true
	if metalBackendUsers.Add(-1) == 0 {
		C.metal_finalize()
		metalReady.Store(false)
	}
}

func (b *Backend) Builder(name string) backends.Builder {
	return newBuilder(b, name)
}

// ─── Buffer ─────────────────────────────────────────────────────────────────

// Buffer wraps a Metal GPU buffer (StorageModeShared).
type Buffer struct {
	mtl   C.MetalBuffer
	shape shapes.Shape
}

// mtlRefCount tracks how many *Buffer values may reference an MTL buffer. When it
// reaches zero, the buffer is released to Core Foundation / Metal.
var (
	mtlRefMu sync.Mutex
	mtlRefs  = map[uintptr]int{}
)

func ptrKey(m C.MetalBuffer) uintptr {
	return uintptr(unsafe.Pointer(m))
}

// mtlRegisterAlloc records ownership for a buffer just allocated with metal_buffer_alloc (refcount 1).
func mtlRegisterAlloc(m C.MetalBuffer) {
	if m == nil {
		return
	}
	k := ptrKey(m)
	mtlRefMu.Lock()
	mtlRefs[k] = 1
	mtlRefMu.Unlock()
}

// mtlRetain increments the refcount for a shared MTL handle (alias views).
func mtlRetain(m C.MetalBuffer) {
	if m == nil {
		return
	}
	k := ptrKey(m)
	mtlRefMu.Lock()
	mtlRefs[k]++
	mtlRefMu.Unlock()
}

// mtlRelease decrements the refcount and frees when it drops to zero.
func mtlRelease(m C.MetalBuffer) {
	if m == nil {
		return
	}
	k := ptrKey(m)
	mtlRefMu.Lock()
	n := mtlRefs[k] - 1
	if n <= 0 {
		delete(mtlRefs, k)
		mtlRefMu.Unlock()
		C.metal_buffer_free(m)
		return
	}
	mtlRefs[k] = n
	mtlRefMu.Unlock()
}

// newBuffer allocates a Metal buffer for the given shape.
func newBuffer(shape shapes.Shape) *Buffer {
	bytes := shape.Size() * int(shape.DType.Size())
	if bytes <= 0 {
		bytes = 1 // zero-sized tensors: MTL buffer length must be non-zero for a stable handle
	}
	mtl := C.metal_buffer_alloc(C.size_t(bytes))
	mtlRegisterAlloc(mtl)
	return &Buffer{mtl: mtl, shape: shape}
}

// contents returns an unsafe.Pointer to the shared-memory contents.
func (buf *Buffer) contents() unsafe.Pointer {
	return C.metal_buffer_contents(buf.mtl)
}

// ─── DataInterface ──────────────────────────────────────────────────────────

func (b *Backend) BufferFinalize(buffer backends.Buffer) error {
	buf, ok := buffer.(*Buffer)
	if !ok {
		return errors.New("not a metal buffer")
	}
	mtlRelease(buf.mtl)
	buf.mtl = nil
	return nil
}

func (b *Backend) BufferShape(buffer backends.Buffer) (shapes.Shape, error) {
	buf, ok := buffer.(*Buffer)
	if !ok {
		return shapes.Invalid(), errors.New("not a metal buffer")
	}
	return buf.shape, nil
}

func (b *Backend) BufferDeviceNum(buffer backends.Buffer) (backends.DeviceNum, error) {
	return 0, nil // single device
}

func (b *Backend) BufferToFlatData(buffer backends.Buffer, flat any) error {
	buf, ok := buffer.(*Buffer)
	if !ok {
		return errors.New("not a metal buffer")
	}
	dst := flatToBytes(flat)
	n := buf.shape.Size() * int(buf.shape.DType.Size())
	if len(dst) != n {
		return errors.Errorf("flat destination length %d does not match buffer %s (%d bytes)", len(dst), buf.shape, n)
	}
	if n > 0 {
		src := C.GoBytes(buf.contents(), C.int(n))
		copy(dst, src)
	}
	return nil
}

func (b *Backend) BufferFromFlatData(deviceNum backends.DeviceNum, flat any, shape shapes.Shape) (backends.Buffer, error) {
	_ = deviceNum
	n := shape.Size() * int(shape.DType.Size())
	src := flatToBytes(flat)
	if n == 0 {
		return newBuffer(shape), nil
	}
	if len(src) != n {
		return nil, errors.Errorf("flat data length %d does not match shape %s (%d bytes)", len(src), shape, n)
	}
	buf := newBuffer(shape)
	C.memcpy(buf.contents(), unsafe.Pointer(&src[0]), C.size_t(len(src)))
	return buf, nil
}

func (b *Backend) HasSharedBuffers() bool {
	return true // Apple unified memory
}

func (b *Backend) NewSharedBuffer(deviceNum backends.DeviceNum, shape shapes.Shape) (backends.Buffer, any, error) {
	buf := newBuffer(shape)
	flat := flatFromBuffer(buf)
	return buf, flat, nil
}

func (b *Backend) BufferData(buffer backends.Buffer) (any, error) {
	buf, ok := buffer.(*Buffer)
	if !ok {
		return nil, errors.New("not a metal buffer")
	}
	return flatFromBuffer(buf), nil
}

func (b *Backend) BufferCopyToDevice(source backends.Buffer, deviceNum backends.DeviceNum) (backends.Buffer, error) {
	_, ok := source.(*Buffer)
	if !ok {
		return nil, errors.New("not a metal buffer")
	}
	srcDev, err := b.BufferDeviceNum(source)
	if err != nil {
		return nil, err
	}
	if deviceNum == srcDev {
		return nil, errors.New("metal: BufferCopyToDevice: source and destination are the same device (API forbids same-device copy)")
	}
	return nil, errors.New("metal: multi-device BufferCopyToDevice is not supported (only one GPU is addressable)")
}

// ─── Helpers ────────────────────────────────────────────────────────────────

// dtypeToMetal maps GoMLX dtypes to our C enum (elementwise / matmul suffix).
// 0=f16, 1=f32, 2=f64, 3=u32 (bitwise family in elementwise.metal).
// Int32 is not included here so reduces/transpose paths that lack i32 kernels stay on CPU or error clearly.
func dtypeToMetal(dt dtypes.DType) C.int {
	switch dt {
	case dtypes.Float16:
		return 0
	case dtypes.Float32:
		return 1
	case dtypes.Float64:
		return 2
	case dtypes.Uint32:
		return 3
	default:
		return -1
	}
}

// elementwiseDTypeToMetal is for unary/binary elementwise only (suffix _i32 in elementwise.metal).
func elementwiseDTypeToMetal(dt dtypes.DType) C.int {
	if dt == dtypes.Int32 {
		return 5
	}
	return dtypeToMetal(dt)
}

// wherePredValueKind maps payload dtypes for metal_where_bool_pred (must match metal.m where_pred_value_suffix).
// Float64 is unavailable on Apple GPU kernels (no double buffers).
func wherePredValueKind(dt dtypes.DType) C.int {
	switch dt {
	case dtypes.Float16:
		return 0
	case dtypes.Float32:
		return 1
	case dtypes.Int32:
		return 2
	case dtypes.Int64:
		return 3
	case dtypes.Bool:
		return 4
	case dtypes.Int8:
		return 5
	case dtypes.Int16:
		return 6
	case dtypes.Uint8:
		return 7
	case dtypes.Uint16:
		return 8
	case dtypes.Uint32:
		return 9
	case dtypes.Uint64:
		return 10
	default:
		return -1
	}
}

// scatterElemKind selects scatter_* kernels: 0=f32, 1=i32, 2=u32, 3=i64, 4=u64.
func scatterElemKind(dt dtypes.DType) C.int {
	switch dt {
	case dtypes.Float32:
		return 0
	case dtypes.Int32:
		return 1
	case dtypes.Uint32:
		return 2
	case dtypes.Int64:
		return 3
	case dtypes.Uint64:
		return 4
	default:
		return -1
	}
}

// metalDTypeBoolPred is the dtype code passed to metal_unary_op/metal_binary_op for
// uchar / Go bool buffers (empty kernel name suffix in metal.m dtype_suffix).
func metalDTypeBoolPred() C.int { return 4 }

// dtypeToMetalIota maps dtypes that have iota_* kernels in tensor_ops.metal.
func dtypeToMetalIota(dt dtypes.DType) C.int {
	switch dt {
	case dtypes.Float16:
		return 0
	case dtypes.Float32:
		return 1
	case dtypes.Int32:
		return 3
	case dtypes.Int64:
		return 4
	case dtypes.Uint32:
		return 5
	case dtypes.Uint64:
		return 6
	default:
		return -1
	}
}

// cTransposePerm wraps metal_transpose_perm so callers outside metal.go need not
// reference C symbols that only appear in this file's cgo preamble.
func cTransposePerm(src, dst, cfg *Buffer, total, cfgWords, elemSize int) error {
	if ret := C.gomlx_metal_transpose_perm(src.mtl, dst.mtl, cfg.mtl,
		C.uint32_t(total), C.uint32_t(cfgWords), C.uint32_t(elemSize)); ret != 0 {
		return errors.Errorf("metal_transpose_perm failed: %d", ret)
	}
	return nil
}

// cFusedSDPA wraps metal_fused_sdpa for the same reason as cTransposePerm.
func cFusedSDPA(q, k, v, dst *Buffer, mask *Buffer,
	batch, numHeads, numKVHeads, seqLen, kvLen, headDim uint32,
	scale float64, causal bool, maskType int,
	maskBatchStride, maskHeadStride uint32, qDtype dtypes.DType) error {
	var maskBuf C.MetalBuffer
	if mask != nil {
		maskBuf = mask.mtl
	}
	causalI := C.int(0)
	if causal {
		causalI = 1
	}
	dt := dtypeToMetal(qDtype)
	if ret := C.gomlx_metal_fused_sdpa(q.mtl, k.mtl, v.mtl, maskBuf, dst.mtl,
		C.uint32_t(batch), C.uint32_t(numHeads), C.uint32_t(numKVHeads),
		C.uint32_t(seqLen), C.uint32_t(kvLen), C.uint32_t(headDim),
		C.float(scale), causalI, C.int(maskType),
		C.uint32_t(maskBatchStride), C.uint32_t(maskHeadStride), dt); ret != 0 {
		return errors.Errorf("metal_fused_sdpa failed: %d", ret)
	}
	return nil
}
