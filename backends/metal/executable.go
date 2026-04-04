//go:build darwin && cgo

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
	"reflect"
	"unsafe"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// execScratch tracks owned buffer references created during a single Execute.
// Each entry represents one refcount the current execution is responsible for.
type execScratch struct {
	owned []*Buffer
}

func (s *execScratch) alloc(shape shapes.Shape) *Buffer {
	b := newBuffer(shape)
	s.own(b)
	return b
}

func (s *execScratch) own(buf *Buffer) {
	if s == nil || buf == nil || buf.mtl == nil {
		return
	}
	for _, owned := range s.owned {
		if owned == buf {
			return
		}
	}
	s.owned = append(s.owned, buf)
}

func (s *execScratch) owns(buf *Buffer) bool {
	if s == nil || buf == nil {
		return false
	}
	for _, owned := range s.owned {
		if owned == buf {
			return true
		}
	}
	return false
}

func (s *execScratch) transferOutputsTo(parent *execScratch, outputs []*Buffer) {
	if s == nil || parent == nil {
		return
	}
	for _, out := range outputs {
		if s.owns(out) {
			parent.own(out)
		}
	}
}

// execBackendTL is set for the duration of executeNode (including nested closure
// graphs). Metal Execute paths hold Backend.execMu so this stays per-goroutine-safe.
var execBackendTL *Backend

// releaseExcept frees scratch MTL buffers whose handles are not in keep.
// When a kept MTL handle is only reached via Reshape/Bitcast outputs (not the same
// *Buffer as a scratch row), releaseDeadAliasViews has already dropped retains for
// discarded intermediate views; one release here drops the scratch allocation ref.
// metalHostGPUSync finishes the active batched command buffer (if any) so the CPU
// can safely read StorageModeShared contents produced by the GPU.
func metalHostGPUSync() error {
	if rc := C.metal_encode_barrier_wait(); rc != 0 {
		return errors.Errorf("metal: GPU host sync failed (%d)", rc)
	}
	return nil
}

func (s *execScratch) releaseExcept(keep map[uintptr]struct{}, buffers []*Buffer, outputNodes []*Node) {
	directOut := make(map[*Buffer]struct{})
	for _, out := range outputNodes {
		if b := buffers[out.idx]; b != nil {
			directOut[b] = struct{}{}
		}
	}

	for _, b := range s.owned {
		if b == nil || b.mtl == nil {
			continue
		}
		k := uintptr(unsafe.Pointer(b.mtl))
		if _, kept := keep[k]; kept {
			if _, isDirectOutput := directOut[b]; !isDirectOutput {
				mtlRelease(b.mtl)
				b.mtl = nil
			}
			continue
		}
		mtlRelease(b.mtl)
		b.mtl = nil
	}
}

func allocDuringExec(shape shapes.Shape) *Buffer {
	if execBackendTL != nil && execBackendTL.scratch != nil {
		return execBackendTL.scratch.alloc(shape)
	}
	return newBuffer(shape)
}

// Executable is a compiled computation graph ready to run on Metal.
type Executable struct {
	backend *Backend
	builder *Builder
}

var _ backends.Executable = (*Executable)(nil)

func (e *Executable) Finalize() {
	e.builder = nil
}

func (e *Executable) Inputs() (names []string, inputShapes []shapes.Shape) {
	params := e.builder.mainFn.parameters
	names = make([]string, len(params))
	inputShapes = make([]shapes.Shape, len(params))
	for i, node := range params {
		p := node.data.(*nodeParameter)
		names[i] = p.name
		inputShapes[i] = node.shape
	}
	return
}

func (e *Executable) Outputs() (outputShapes []shapes.Shape) {
	outputs := e.builder.mainFn.outputs
	outputShapes = make([]shapes.Shape, len(outputs))
	for i, node := range outputs {
		outputShapes[i] = node.shape
	}
	return
}

// Execute runs the compiled graph. It walks the DAG in order, dispatching each
// node to the appropriate Metal kernel.
func (e *Executable) Execute(inputs []backends.Buffer, donate []bool, defaultDevice backends.DeviceNum) (res []backends.Buffer, err error) {
	_ = defaultDevice
	b := e.backend
	b.execMu.Lock()
	defer b.execMu.Unlock()

	fn := e.builder.mainFn
	if len(inputs) != len(fn.parameters) {
		return nil, errors.Errorf("Execute: expected %d inputs, got %d", len(fn.parameters), len(inputs))
	}

	if donate != nil && len(donate) != len(inputs) {
		return nil, errors.Errorf("Execute: donate length %d does not match inputs %d", len(donate), len(inputs))
	}

	if fn.compiled == nil {
		return nil, errors.New("Execute: main function not compiled (internal error)")
	}

	C.metal_encode_begin()
	defer func() {
		if rc := C.metal_encode_end_wait(); rc != 0 && err == nil {
			err = errors.Errorf("metal: GPU command buffer completion failed (%d)", rc)
		}
	}()

	paramBufs := make([]*Buffer, len(fn.parameters))

	for i, paramNode := range fn.parameters {
		buf, ok := inputs[i].(*Buffer)
		if !ok {
			return nil, errors.Errorf("Execute: input #%d is not a metal buffer", i)
		}
		if !buf.shape.Equal(paramNode.shape) {
			return nil, errors.Errorf("Execute: input #%d shape %s does not match parameter %q (%s)",
				i, buf.shape, paramNode.data.(*nodeParameter).name, paramNode.shape)
		}
		paramBufs[i] = buf
	}

	out, errRun := fn.compiled.runMain(e.backend, paramBufs, donate)

	if errRun != nil {
		return nil, errRun
	}
	res = make([]backends.Buffer, len(out))
	for i := range out {
		res[i] = out[i]
	}
	return res, err
}

// executeNode dispatches a single node to the appropriate Metal kernel.
func executeNode(backend *Backend, node *Node, inputs []*Buffer) (*Buffer, error) {
	prev := execBackendTL
	execBackendTL = backend
	defer func() { execBackendTL = prev }()

	switch node.opType {
	case backends.OpTypeConstant:
		return executeConstant(node)
	case backends.OpTypeIdentity:
		return inputs[0], nil

	// Unary
	case backends.OpTypeAbs, backends.OpTypeNeg, backends.OpTypeCeil, backends.OpTypeFloor,
		backends.OpTypeRound, backends.OpTypeSign, backends.OpTypeSqrt, backends.OpTypeRsqrt,
		backends.OpTypeExp, backends.OpTypeExpm1, backends.OpTypeLog, backends.OpTypeLog1p,
		backends.OpTypeSin, backends.OpTypeCos, backends.OpTypeTanh, backends.OpTypeErf,
		backends.OpTypeLogistic, backends.OpTypeIsFinite, backends.OpTypeIsNaN,
		backends.OpTypeLogicalNot, backends.OpTypeBitwiseNot, backends.OpTypeClz, backends.OpTypeBitCount:
		return executeUnary(node, inputs[0])

	// Binary
	case backends.OpTypeAdd, backends.OpTypeSub, backends.OpTypeMul, backends.OpTypeDiv,
		backends.OpTypePow, backends.OpTypeRem, backends.OpTypeMax, backends.OpTypeMin,
		backends.OpTypeAtan2,
		backends.OpTypeBitwiseAnd, backends.OpTypeBitwiseOr, backends.OpTypeBitwiseXor,
		backends.OpTypeLogicalAnd, backends.OpTypeLogicalOr, backends.OpTypeLogicalXor,
		backends.OpTypeEqual, backends.OpTypeNotEqual,
		backends.OpTypeLessThan, backends.OpTypeLessOrEqual,
		backends.OpTypeGreaterThan, backends.OpTypeGreaterOrEqual,
		backends.OpTypeEqualTotalOrder, backends.OpTypeNotEqualTotalOrder,
		backends.OpTypeLessThanTotalOrder, backends.OpTypeLessOrEqualTotalOrder,
		backends.OpTypeGreaterThanTotalOrder, backends.OpTypeGreaterOrEqualTotalOrder:
		return executeBinary(node, inputs[0], inputs[1])

	// Reductions
	case backends.OpTypeReduceSum, backends.OpTypeReduceProduct,
		backends.OpTypeReduceMax, backends.OpTypeReduceMin,
		backends.OpTypeReduceBitwiseAnd, backends.OpTypeReduceBitwiseOr, backends.OpTypeReduceBitwiseXor,
		backends.OpTypeReduceLogicalAnd, backends.OpTypeReduceLogicalOr, backends.OpTypeReduceLogicalXor:
		return executeReduce(node, inputs[0])

	// Tensor ops
	case backends.OpTypeReshape:
		return executeReshape(node, inputs[0])
	case backends.OpTypeTranspose:
		return executeTranspose(node, inputs[0])
	case backends.OpTypeBroadcast:
		return executeBroadcast(node, inputs[0])
	case backends.OpTypeBroadcastInDim:
		return executeBroadcastInDim(node, inputs[0])
	case backends.OpTypeWhere:
		return executeWhere(node, inputs[0], inputs[1], inputs[2])
	case backends.OpTypeConcatenate:
		return executeConcatenate(node, inputs)
	case backends.OpTypeSlice:
		return executeSlice(node, inputs[0])
	case backends.OpTypeIota:
		return executeIota(node)
	case backends.OpTypeConvertDType:
		return executeConvertDType(node, inputs[0])
	case backends.OpTypePad:
		return executePad(node, inputs[0], inputs[1])
	case backends.OpTypeReverse:
		return executeReverse(node, inputs[0])
	case backends.OpTypeGather:
		return executeGather(node, inputs[0], inputs[1])
	case backends.OpTypeScatterSum, backends.OpTypeScatterMax, backends.OpTypeScatterMin:
		return executeScatter(node, inputs[0], inputs[1], inputs[2])
	case backends.OpTypeArgMinMax:
		return executeArgMinMax(node, inputs[0])
	case backends.OpTypeConvGeneral:
		return executeConvGeneral(node, inputs[0], inputs[1])
	case backends.OpTypeReduceWindow:
		return executeReduceWindow(node, inputs[0])
	case backends.OpTypeBitcast:
		return executeBitcast(node, inputs[0])

	// Matmul
	case backends.OpTypeDot:
		return executeDot(node, inputs[0], inputs[1])
	case backends.OpTypeDotGeneral:
		return executeDotGeneral(node, inputs[0], inputs[1])

	// Fused
	case backends.OpTypeFusedSoftmax:
		return executeFusedSoftmax(node, inputs[0])
	case backends.OpTypeFusedGelu:
		return executeFusedGelu(node, inputs[0])
	case backends.OpTypeFusedLayerNorm:
		return executeFusedLayerNorm(node, inputs)
	case backends.OpTypeFusedScaledDotProductAttention:
		return executeFusedSDPA(node, inputs)

	default:
		return nil, errors.Wrapf(backends.ErrNotImplemented, "metal: op %s not implemented", node.opType)
	}
}

// ─── Config buffer helper ──────────────────────────────────────────────────

// makeConfigBuffer creates a Metal buffer from a uint32 slice and returns it.
// The caller must free it when done.
func makeConfigBuffer(data []uint32) *Buffer {
	shape := shapes.Make(dtypes.Uint32, len(data))
	buf := newBuffer(shape)

	if len(data) == 0 {
		return buf
	}

	bytes := len(data) * 4
	C.memcpy(buf.contents(), unsafe.Pointer(&data[0]), C.size_t(bytes))
	return buf
}

// freeConfigBuffer releases a config buffer.
func freeConfigBuffer(buf *Buffer) {
	if buf != nil && buf.mtl != nil {
		mtlRelease(buf.mtl)
		buf.mtl = nil
	}
}

// ─── Op executors ───────────────────────────────────────────────────────────

func executeConstant(node *Node) (*Buffer, error) {
	buf := allocDuringExec(node.shape)
	src := flatToBytes(node.data)
	want := node.shape.Size() * int(node.shape.DType.Size())

	if len(src) != want {
		return nil, errors.Errorf("Constant: flat size %d bytes, shape %s expects %d", len(src), node.shape, want)
	}

	if len(src) > 0 {
		C.memcpy(buf.contents(), unsafe.Pointer(&src[0]), C.size_t(len(src)))
	}
	return buf, nil
}

var opTypeToKernelName = map[backends.OpType]string{
	backends.OpTypeAbs:        "op_abs",
	backends.OpTypeNeg:        "op_neg",
	backends.OpTypeCeil:       "op_ceil",
	backends.OpTypeFloor:      "op_floor",
	backends.OpTypeRound:      "op_round",
	backends.OpTypeSign:       "op_sign",
	backends.OpTypeSqrt:       "op_sqrt",
	backends.OpTypeRsqrt:      "op_rsqrt",
	backends.OpTypeExp:        "op_exp",
	backends.OpTypeExpm1:      "op_expm1",
	backends.OpTypeLog:        "op_log",
	backends.OpTypeLog1p:      "op_log1p",
	backends.OpTypeSin:        "op_sin",
	backends.OpTypeCos:        "op_cos",
	backends.OpTypeTanh:       "op_tanh",
	backends.OpTypeErf:        "op_erf",
	backends.OpTypeLogistic:   "op_logistic",
	backends.OpTypeIsFinite:   "op_is_finite_pred",
	backends.OpTypeIsNaN:      "op_is_nan_pred",
	backends.OpTypeLogicalNot: "op_logical_not",
	backends.OpTypeBitwiseNot: "op_bitwise_not",
	backends.OpTypeClz:        "op_clz",
	backends.OpTypeBitCount:   "op_bitcount",
}

func executeUnary(node *Node, input *Buffer) (*Buffer, error) {
	name, ok := opTypeToKernelName[node.opType]
	if !ok {
		return nil, errors.Errorf("no metal kernel for unary op %s", node.opType)
	}

	dst := allocDuringExec(node.shape)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	n := C.uint32_t(node.shape.Size())
	var dt C.int
	if backends.OpTypeLogicalNot == node.opType && input.shape.DType == dtypes.Bool {
		dt = metalDTypeBoolPred()
	} else {
		dt = elementwiseDTypeToMetal(input.shape.DType)
		if dt < 0 {
			return nil, errors.Errorf("metal: unsupported unary input dtype %s", input.shape.DType)
		}
	}

	if ret := C.metal_unary_op(cName, input.mtl, dst.mtl, n, dt); ret != 0 {
		return nil, errors.Errorf("metal_unary_op(%s) failed: %d", name, ret)
	}
	return dst, nil
}

var binaryOpKernelName = map[backends.OpType]string{
	backends.OpTypeAdd:            "op_add",
	backends.OpTypeSub:            "op_sub",
	backends.OpTypeMul:            "op_mul",
	backends.OpTypeDiv:            "op_div",
	backends.OpTypePow:            "op_pow",
	backends.OpTypeRem:            "op_rem",
	backends.OpTypeMax:            "op_max",
	backends.OpTypeMin:            "op_min",
	backends.OpTypeAtan2:          "op_atan2",
	backends.OpTypeBitwiseAnd:     "op_bitwise_and",
	backends.OpTypeBitwiseOr:      "op_bitwise_or",
	backends.OpTypeBitwiseXor:     "op_bitwise_xor",
	backends.OpTypeLogicalAnd:     "op_logical_and",
	backends.OpTypeLogicalOr:      "op_logical_or",
	backends.OpTypeLogicalXor:     "op_logical_xor",
	backends.OpTypeEqual:          "op_equal_pred",
	backends.OpTypeNotEqual:       "op_not_equal_pred",
	backends.OpTypeLessThan:       "op_less_pred",
	backends.OpTypeLessOrEqual:    "op_less_or_equal_pred",
	backends.OpTypeGreaterThan:    "op_greater_pred",
	backends.OpTypeGreaterOrEqual: "op_greater_or_equal_pred",

	backends.OpTypeEqualTotalOrder:          "op_equal_total_order_pred",
	backends.OpTypeNotEqualTotalOrder:       "op_not_equal_total_order_pred",
	backends.OpTypeLessThanTotalOrder:       "op_less_total_order_pred",
	backends.OpTypeLessOrEqualTotalOrder:    "op_less_or_equal_total_order_pred",
	backends.OpTypeGreaterThanTotalOrder:    "op_greater_total_order_pred",
	backends.OpTypeGreaterOrEqualTotalOrder: "op_greater_or_equal_total_order_pred",
}

func executeBinary(node *Node, lhs, rhs *Buffer) (*Buffer, error) {
	name, ok := binaryOpKernelName[node.opType]
	if !ok {
		return nil, errors.Errorf("no metal kernel for binary op %s", node.opType)
	}

	if !lhs.shape.Equal(rhs.shape) {
		return nil, errors.Errorf("metal: binary op %s requires matching lhs/rhs shapes, got %s vs %s",
			node.opType, lhs.shape, rhs.shape)
	}
	if !lhs.shape.EqualDimensions(node.shape) {
		return nil, errors.Errorf("metal: binary op %s output dimensions %s vs operands %s",
			node.opType, node.shape, lhs.shape)
	}
	// Comparisons (and logical ops) may use Bool node.shape with numeric/bool operands.
	if node.shape.DType != dtypes.Bool && !lhs.shape.Equal(node.shape) {
		return nil, errors.Errorf("metal: binary op %s node shape %s does not match operands %s",
			node.opType, node.shape, lhs.shape)
	}

	dst := allocDuringExec(node.shape)
	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))
	n := C.uint32_t(node.shape.Size())
	var dt C.int
	switch node.opType {
	case backends.OpTypeLogicalAnd, backends.OpTypeLogicalOr, backends.OpTypeLogicalXor:
		if lhs.shape.DType != dtypes.Bool || rhs.shape.DType != dtypes.Bool {
			return nil, errors.Errorf("metal: logical op requires bool operands, got %s and %s",
				lhs.shape.DType, rhs.shape.DType)
		}
		dt = metalDTypeBoolPred()
	default:
		elemDT := lhs.shape.DType
		dt = elementwiseDTypeToMetal(elemDT)
		if dt < 0 {
			return nil, errors.Errorf("metal: unsupported binary operand dtype %s", elemDT)
		}
	}

	if ret := C.metal_binary_op(cName, lhs.mtl, rhs.mtl, dst.mtl, n, dt); ret != 0 {
		return nil, errors.Errorf("metal_binary_op(%s) failed: %d", name, ret)
	}
	return dst, nil
}

var reduceOpKernelName = map[backends.OpType]string{
	backends.OpTypeReduceSum:        "reduce_sum",
	backends.OpTypeReduceProduct:    "reduce_product",
	backends.OpTypeReduceMax:        "reduce_max",
	backends.OpTypeReduceMin:        "reduce_min",
	backends.OpTypeReduceBitwiseAnd: "reduce_bitwise_and",
	backends.OpTypeReduceBitwiseOr:  "reduce_bitwise_or",
	backends.OpTypeReduceBitwiseXor: "reduce_bitwise_xor",
	backends.OpTypeReduceLogicalAnd: "reduce_logical_and",
	backends.OpTypeReduceLogicalOr:  "reduce_logical_or",
	backends.OpTypeReduceLogicalXor: "reduce_logical_xor",
}

func executeReduce(node *Node, input *Buffer) (*Buffer, error) {
	name, ok := reduceOpKernelName[node.opType]
	if !ok {
		return nil, errors.Errorf("no metal kernel for reduce op %s", node.opType)
	}

	data := node.data.(*reduceData)
	inShape := input.shape
	var dt C.int

	switch node.opType {
	case backends.OpTypeReduceBitwiseAnd, backends.OpTypeReduceBitwiseOr, backends.OpTypeReduceBitwiseXor:
		if inShape.DType != dtypes.Uint32 {
			return nil, errors.Errorf("metal reduce: bitwise reduction requires Uint32, got %s", inShape.DType)
		}
		dt = 3
	case backends.OpTypeReduceLogicalAnd, backends.OpTypeReduceLogicalOr, backends.OpTypeReduceLogicalXor:
		if inShape.DType != dtypes.Bool {
			return nil, errors.Errorf("metal reduce: logical reduction requires Bool, got %s", inShape.DType)
		}
		dt = metalDTypeBoolPred()
	case backends.OpTypeReduceSum, backends.OpTypeReduceProduct, backends.OpTypeReduceMax, backends.OpTypeReduceMin:
		switch inShape.DType {
		case dtypes.Int32:
			dt = 5
		case dtypes.Uint32:
			dt = 3
		default:
			dt = dtypeToMetal(inShape.DType)
		}
		if dt < 0 {
			return nil, errors.Errorf("metal reduce: unsupported dtype %s", inShape.DType)
		}
	default:
		dt = dtypeToMetal(inShape.DType)
		if dt < 0 {
			return nil, errors.Errorf("metal reduce: unsupported dtype %s", inShape.DType)
		}
	}

	// Multi-axis reduction: reduce axes one at a time from highest to lowest
	// to avoid recomputing strides. Each step reduces one axis.
	axes := make([]int, len(data.axes))
	copy(axes, data.axes)
	// Sort descending so removing higher axes first doesn't shift lower ones
	for i := 0; i < len(axes); i++ {
		for j := i + 1; j < len(axes); j++ {
			if axes[j] > axes[i] {
				axes[i], axes[j] = axes[j], axes[i]
			}
		}
	}

	current := input
	currentDims := make([]int, len(inShape.Dimensions))
	copy(currentDims, inShape.Dimensions)

	cName := C.CString(name)
	defer C.free(unsafe.Pointer(cName))

	for _, axis := range axes {
		// Transpose so that the reduce axis is last
		rank := len(currentDims)
		if axis != rank-1 {
			// Need to transpose axis to last position
			perm := make([]int, rank)
			idx := 0
			for i := 0; i < rank; i++ {
				if i != axis {
					perm[idx] = i
					idx++
				}
			}
			perm[rank-1] = axis

			transposedDims := make([]int, rank)
			for i, p := range perm {
				transposedDims[i] = currentDims[p]
			}

			transposed := allocDuringExec(shapes.Make(inShape.DType, transposedDims...))
			if err := transposeBufferUnified(current, transposed, currentDims, transposedDims, perm, inShape.DType); err != nil {
				return nil, err
			}
			current = transposed
			currentDims = transposedDims
		}

		innerSize := currentDims[len(currentDims)-1]
		outerSize := 1
		for i := 0; i < len(currentDims)-1; i++ {
			outerSize *= currentDims[i]
		}

		newDims := currentDims[:len(currentDims)-1]
		if len(newDims) == 0 {
			newDims = []int{}
		}
		dst := allocDuringExec(shapes.Make(inShape.DType, newDims...))

		if ret := C.metal_reduce_op(cName, current.mtl, dst.mtl,
			C.uint32_t(outerSize), C.uint32_t(innerSize), dt); ret != 0 {
			return nil, errors.Errorf("metal_reduce_op(%s) failed: %d", name, ret)
		}

		current = dst
		currentDims = newDims
	}

	return current, nil
}

// transposeBufferCPU performs a generic transpose on CPU via shared memory.
func transposeBufferCPU(src, dst *Buffer, srcDims, dstDims []int, perm []int, elemSize int) {
	size := 1
	for _, d := range srcDims {
		size *= d
	}
	srcStrides := computeStrides(srcDims)
	dstStrides := computeStrides(dstDims)
	srcPtr := (*[1 << 30]byte)(src.contents())
	dstPtr := (*[1 << 30]byte)(dst.contents())

	for flatIdx := 0; flatIdx < size; flatIdx++ {
		remaining := flatIdx
		srcFlat := 0
		for dim := 0; dim < len(dstDims); dim++ {
			idx := remaining / dstStrides[dim]
			remaining %= dstStrides[dim]
			srcFlat += idx * srcStrides[perm[dim]]
		}
		srcOff := srcFlat * elemSize
		dstOff := flatIdx * elemSize
		copy(dstPtr[dstOff:dstOff+elemSize], srcPtr[srcOff:srcOff+elemSize])
	}

}

const maxTransposeRank = 16

// transposeBufferUnified dispatches a general permutation transpose to GPU and
// falls back to CPU only for oversized rank.
func transposeBufferUnified(src, dst *Buffer, srcDims, dstDims []int, perm []int, dt dtypes.DType) error {
	rank := len(srcDims)
	if rank != len(perm) || rank != len(dstDims) || rank > maxTransposeRank {
		transposeBufferCPU(src, dst, srcDims, dstDims, perm, int(dt.Size()))
		return nil
	}

	srcStrides := computeStrides(srcDims)
	cfg := make([]uint32, 2+3*rank)
	cfg[0] = uint32(rank)
	total := dst.shape.Size()
	if total < 0 {
		return errors.New("transpose: invalid shape")
	}
	cfg[1] = uint32(total)
	for i := range rank {
		cfg[2+i] = uint32(srcDims[i])
	}
	for i := range rank {
		cfg[2+rank+i] = uint32(perm[i])
	}
	for i := range rank {
		cfg[2+2*rank+i] = uint32(srcStrides[i])
	}

	cbuf := makeConfigBuffer(cfg)
	defer freeConfigBuffer(cbuf)
	return cTransposePerm(src, dst, cbuf, total, len(cfg), int(dt.Size()))
}

func executeReshape(node *Node, input *Buffer) (*Buffer, error) {
	// Reshape is zero-copy, same underlying data, different shape interpretation.
	mtlRetain(input.mtl)
	view := &Buffer{mtl: input.mtl, shape: node.shape}
	if execBackendTL != nil && execBackendTL.scratch != nil {
		execBackendTL.scratch.own(view)
	}
	return view, nil
}

func executeTranspose(node *Node, input *Buffer) (*Buffer, error) {
	dst := allocDuringExec(node.shape)
	perm := node.data.(*transposeData).permutation
	inShape := input.shape
	outShape := node.shape
	if err := transposeBufferUnified(input, dst, inShape.Dimensions, outShape.Dimensions, perm, inShape.DType); err != nil {
		return nil, err
	}
	return dst, nil
}

// ─── Broadcast ──────────────────────────────────────────────────────────────

func executeBroadcast(node *Node, input *Buffer) (*Buffer, error) {
	dst := allocDuringExec(node.shape)
	srcSize := C.uint32_t(input.shape.Size())
	dstSize := C.uint32_t(node.shape.Size())
	elemSize := C.uint32_t(node.shape.DType.Size())
	if ret := C.metal_broadcast(input.mtl, dst.mtl, srcSize, dstSize, elemSize); ret != 0 {
		return nil, errors.Errorf("metal_broadcast failed: %d", ret)
	}
	return dst, nil
}

// ─── BroadcastInDim ─────────────────────────────────────────────────────────

func executeBroadcastInDim(node *Node, input *Buffer) (*Buffer, error) {
	broadcastDims := node.data.([]int)
	inShape := input.shape
	outShape := node.shape
	rank := outShape.Rank()

	// Build config: [rank, output_strides[rank], operand_strides[rank]]
	outStrides := computeStrides(outShape.Dimensions)
	inStrides := computeStrides(inShape.Dimensions)

	config := make([]uint32, 1+2*rank)
	config[0] = uint32(rank)
	for i := 0; i < rank; i++ {
		config[1+i] = uint32(outStrides[i])
	}
	// Map operand strides to output dimensions via broadcastDims
	for i := 0; i < rank; i++ {
		config[1+rank+i] = 0 // default: broadcast (stride 0)
	}
	for inAxis, outAxis := range broadcastDims {
		if inShape.Dimensions[inAxis] > 1 {
			config[1+rank+outAxis] = uint32(inStrides[inAxis])
		}
	}

	configBuf := makeConfigBuffer(config)
	defer freeConfigBuffer(configBuf)

	dst := allocDuringExec(outShape)
	total := C.uint32_t(outShape.Size())
	elemSize := C.uint32_t(outShape.DType.Size())

	if ret := C.metal_broadcast_in_dim(input.mtl, dst.mtl, configBuf.mtl,
		total, C.uint32_t(len(config)), elemSize); ret != 0 {
		return nil, errors.Errorf("metal_broadcast_in_dim failed: %d", ret)
	}
	return dst, nil
}

// ─── Where ──────────────────────────────────────────────────────────────────

func executeWhere(node *Node, pred, onTrue, onFalse *Buffer) (*Buffer, error) {
	return whereBuffers(node.shape, pred, onTrue, onFalse)
}

func whereBuffers(outShape shapes.Shape, pred, onTrue, onFalse *Buffer) (*Buffer, error) {
	dst := allocDuringExec(outShape)
	size := outShape.Size()
	if pred.shape.DType == dtypes.Bool {
		n := C.uint32_t(size)
		vk := wherePredValueKind(outShape.DType)
		if vk < 0 {
			return nil, errors.Errorf("metal_where_bool_pred: unsupported value dtype %s", outShape.DType)
		}
		if ret := C.metal_where_bool_pred(pred.mtl, onTrue.mtl, onFalse.mtl, dst.mtl, n, vk); ret != 0 {
			return nil, errors.Errorf("metal_where_bool_pred failed: %d", ret)
		}

		return dst, nil
	}

	n := C.uint32_t(size)
	dt := dtypeToMetal(outShape.DType)
	if ret := C.metal_where(pred.mtl, onTrue.mtl, onFalse.mtl, dst.mtl, n, dt); ret != 0 {
		return nil, errors.Errorf("metal_where failed: %d", ret)
	}
	return dst, nil
}

func metalGatherScatterAxisPermute(dst *Buffer, row *Buffer, indices *Buffer, baseElem, axisStride, axisSize uint32) error {
	es := uint32(dst.shape.DType.Size())
	if ret := C.metal_gather_axis_row_bytes(dst.mtl, row.mtl,
		C.uint32_t(baseElem), C.uint32_t(axisStride), C.uint32_t(axisSize), C.uint32_t(es)); ret != 0 {
		return errors.Errorf("metal_gather_axis_row_bytes failed: %d", ret)
	}
	if ret := C.metal_scatter_axis_row_perm_bytes(row.mtl, dst.mtl, indices.mtl,
		C.uint32_t(baseElem), C.uint32_t(axisStride), C.uint32_t(axisSize), C.uint32_t(es)); ret != 0 {
		return errors.Errorf("metal_scatter_axis_row_perm_bytes failed: %d", ret)
	}
	return nil
}

func metalSortLoadPairBytes(flat, lhs, rhs, idx *Buffer, baseElem, axisStride, es, sortI, sortJ uint32) error {
	if ret := C.metal_sort_load_pair_bytes(flat.mtl, lhs.mtl, rhs.mtl, idx.mtl,
		C.uint32_t(baseElem), C.uint32_t(axisStride), C.uint32_t(es),
		C.uint32_t(sortI), C.uint32_t(sortJ)); ret != 0 {
		return errors.Errorf("metal_sort_load_pair_bytes failed: %d", ret)
	}
	return nil
}

func metalSortBitonicSwapIdx(idx, pred *Buffer, stepK, stepJ, n uint32) error {
	if ret := C.metal_sort_bitonic_swap_idx(idx.mtl, pred.mtl,
		C.uint32_t(stepK), C.uint32_t(stepJ), C.uint32_t(n)); ret != 0 {
		return errors.Errorf("metal_sort_bitonic_swap_idx failed: %d", ret)
	}
	return nil
}

func metalSortAdjacentSwapIdx(idx, pred *Buffer, pairI, n uint32, swapWhenPredNonzero bool) error {
	sw := C.uint32_t(0)
	if swapWhenPredNonzero {
		sw = 1
	}
	if ret := C.metal_sort_adjacent_swap_idx(idx.mtl, pred.mtl,
		C.uint32_t(pairI), C.uint32_t(n), sw); ret != 0 {
		return errors.Errorf("metal_sort_adjacent_swap_idx failed: %d", ret)
	}
	return nil
}

// ─── Concatenate ────────────────────────────────────────────────────────────

func executeConcatenate(node *Node, inputs []*Buffer) (*Buffer, error) {
	data := node.data.(*concatData)
	axis := data.axis
	outShape := node.shape
	numInputs := len(inputs)

	// Compute inner_block_size = product of dims after concat axis
	innerBlockSize := 1
	for i := axis + 1; i < outShape.Rank(); i++ {
		innerBlockSize *= outShape.Dimensions[i]
	}

	// Compute outer_size = product of dims before concat axis
	outerSize := 1
	for i := 0; i < axis; i++ {
		outerSize *= outShape.Dimensions[i]
	}

	// Pack all inputs into a single contiguous mega-buffer
	elemSize := int(outShape.DType.Size())
	totalInputElems := 0
	for _, inp := range inputs {
		totalInputElems += inp.shape.Size()
	}

	megaBuf := allocDuringExec(shapes.Make(outShape.DType, totalInputElems))
	megaPtr := (*[1 << 30]byte)(megaBuf.contents())
	offset := 0
	axisSizes := make([]uint32, numInputs)
	baseOffsets := make([]uint32, numInputs)
	for i, inp := range inputs {
		axisSizes[i] = uint32(inp.shape.Dimensions[axis])
		baseOffsets[i] = uint32(offset)
		sz := inp.shape.Size() * elemSize
		srcPtr := (*[1 << 30]byte)(inp.contents())
		copy(megaPtr[offset*elemSize:(offset+inp.shape.Size())*elemSize], srcPtr[:sz])
		offset += inp.shape.Size()
	}

	// Build config: [num_inputs, inner_block_size, axis_sizes..., base_offsets...]
	config := make([]uint32, 2+2*numInputs)
	config[0] = uint32(numInputs)
	config[1] = uint32(innerBlockSize)
	for i := 0; i < numInputs; i++ {
		config[2+i] = axisSizes[i]
		config[2+numInputs+i] = baseOffsets[i]
	}

	configBuf := makeConfigBuffer(config)
	defer freeConfigBuffer(configBuf)

	dst := allocDuringExec(outShape)
	total := C.uint32_t(outShape.Size())
	elemSizeC := C.uint32_t(outShape.DType.Size())

	if ret := C.metal_concatenate(megaBuf.mtl, dst.mtl, configBuf.mtl,
		total, C.uint32_t(len(config)), elemSizeC); ret != 0 {
		return nil, errors.Errorf("metal_concatenate failed: %d", ret)
	}
	return dst, nil
}

// ─── Slice ──────────────────────────────────────────────────────────────────

func executeSlice(node *Node, input *Buffer) (*Buffer, error) {
	data := node.data.(*sliceData)
	inShape := input.shape
	rank := inShape.Rank()
	inStrides := computeStrides(inShape.Dimensions)
	outDims := node.shape.Dimensions

	// Config: [rank, starts[rank], strides[rank], out_dims[rank], in_strides[rank]]
	config := make([]uint32, 1+4*rank)
	config[0] = uint32(rank)
	for i := 0; i < rank; i++ {
		config[1+i] = uint32(data.starts[i])
		s := 1
		if data.strides != nil && i < len(data.strides) {
			s = data.strides[i]
		}
		config[1+rank+i] = uint32(s)
		config[1+2*rank+i] = uint32(outDims[i])
		config[1+3*rank+i] = uint32(inStrides[i])
	}

	configBuf := makeConfigBuffer(config)
	defer freeConfigBuffer(configBuf)

	dst := allocDuringExec(node.shape)
	total := C.uint32_t(node.shape.Size())
	es := C.uint32_t(node.shape.DType.Size())

	if ret := C.metal_slice(input.mtl, dst.mtl, configBuf.mtl,
		total, C.uint32_t(len(config)), es); ret != 0 {
		return nil, errors.Errorf("metal_slice failed: %d", ret)
	}
	return dst, nil
}

// ─── Iota ───────────────────────────────────────────────────────────────────

func executeIota(node *Node) (*Buffer, error) {
	data := node.data.(*iotaData)
	outShape := node.shape
	iotaDim := data.iotaDimension

	// Decompose into [batch_size, iota_size, repeat_size]
	batchSize := 1
	for i := 0; i < iotaDim; i++ {
		batchSize *= outShape.Dimensions[i]
	}
	iotaSize := outShape.Dimensions[iotaDim]
	repeatSize := 1
	for i := iotaDim + 1; i < outShape.Rank(); i++ {
		repeatSize *= outShape.Dimensions[i]
	}

	dt := dtypeToMetalIota(outShape.DType)
	if dt < 0 {
		return nil, errors.Errorf("metal iota: unsupported dtype %s", outShape.DType)
	}

	dst := allocDuringExec(outShape)

	if ret := C.metal_iota(dst.mtl, C.uint32_t(batchSize), C.uint32_t(iotaSize),
		C.uint32_t(repeatSize), dt); ret != 0 {
		return nil, errors.Errorf("metal_iota failed: %d", ret)
	}
	return dst, nil
}

// ─── ConvertDType ───────────────────────────────────────────────────────────

func executeConvertDType(node *Node, input *Buffer) (*Buffer, error) {
	srcDt := dtypeToMetalExt(input.shape.DType)
	dstDt := dtypeToMetalExt(node.shape.DType)
	if srcDt < 0 {
		return nil, errors.Errorf("metal convert: unsupported source dtype %s", input.shape.DType)
	}
	if dstDt < 0 {
		return nil, errors.Errorf("metal convert: unsupported destination dtype %s", node.shape.DType)
	}

	dst := allocDuringExec(node.shape)
	n := C.uint32_t(node.shape.Size())

	if srcDt == dstDt {
		// Same type, just copy
		bytes := node.shape.Size() * int(node.shape.DType.Size())
		C.memcpy(dst.contents(), input.contents(), C.size_t(bytes))
		return dst, nil
	}

	if ret := C.metal_convert_dtype(input.mtl, dst.mtl, n, srcDt, dstDt); ret != 0 {
		return nil, errors.Errorf("metal_convert_dtype failed: %d", ret)
	}
	return dst, nil
}

// gpuConvertSameShape converts a buffer to toDType, same dimensions as src.
func gpuConvertSameShape(src *Buffer, toDType dtypes.DType) (*Buffer, error) {
	if src.shape.DType == toDType {
		return nil, errors.New("gpuConvertSameShape: src dtype equals toDType (use explicit copy if needed)")
	}
	outShape := shapes.Make(toDType, src.shape.Dimensions...)
	dst := allocDuringExec(outShape)
	n := C.uint32_t(src.shape.Size())
	srcDt := dtypeToMetalExt(src.shape.DType)
	dstDt := dtypeToMetalExt(toDType)
	if srcDt < 0 || dstDt < 0 {
		return nil, errors.Errorf("gpuConvertSameShape: unsupported pair %s -> %s", src.shape.DType, toDType)
	}
	if ret := C.metal_convert_dtype(src.mtl, dst.mtl, n, srcDt, dstDt); ret != 0 {
		return nil, errors.Errorf("metal_convert_dtype failed: %d", ret)
	}
	return dst, nil
}

// ─── Pad ────────────────────────────────────────────────────────────────────

func executePad(node *Node, input, fillValue *Buffer) (*Buffer, error) {
	data := node.data.(*padData)
	inShape := input.shape
	outShape := node.shape
	rank := inShape.Rank()
	inStrides := computeStrides(inShape.Dimensions)

	// Config: [rank, pad_low[rank], pad_interior[rank], in_dims[rank], out_dims[rank], in_strides[rank]]
	config := make([]uint32, 1+5*rank)
	config[0] = uint32(rank)
	for i := 0; i < rank; i++ {
		config[1+i] = uint32(data.axesConfig[i].Start)
		config[1+rank+i] = uint32(data.axesConfig[i].Interior)
		config[1+2*rank+i] = uint32(inShape.Dimensions[i])
		config[1+3*rank+i] = uint32(outShape.Dimensions[i])
		config[1+4*rank+i] = uint32(inStrides[i])
	}

	configBuf := makeConfigBuffer(config)
	defer freeConfigBuffer(configBuf)

	dst := allocDuringExec(outShape)
	total := C.uint32_t(outShape.Size())
	es := C.uint32_t(outShape.DType.Size())

	// Note: pad kernel expects buffers in order: src, pad_value, dst, config, total
	// But our C API wraps it as: src, pad_value, dst, config, total
	if ret := C.metal_pad(input.mtl, fillValue.mtl, dst.mtl, configBuf.mtl,
		total, C.uint32_t(len(config)), es); ret != 0 {
		return nil, errors.Errorf("metal_pad failed: %d", ret)
	}
	return dst, nil
}

// ─── Reverse ────────────────────────────────────────────────────────────────

func executeReverse(node *Node, input *Buffer) (*Buffer, error) {
	data := node.data.(*reverseData)
	inShape := input.shape
	rank := inShape.Rank()
	strides := computeStrides(inShape.Dimensions)

	// Config: [rank, dims[rank], strides[rank], reverse_flags[rank]]
	config := make([]uint32, 1+3*rank)
	config[0] = uint32(rank)

	revSet := make(map[int]bool)
	for _, a := range data.axes {
		revSet[a] = true
	}
	for i := 0; i < rank; i++ {
		config[1+i] = uint32(inShape.Dimensions[i])
		config[1+rank+i] = uint32(strides[i])
		if revSet[i] {
			config[1+2*rank+i] = 1
		}
	}

	configBuf := makeConfigBuffer(config)
	defer freeConfigBuffer(configBuf)

	dst := allocDuringExec(node.shape)
	total := C.uint32_t(node.shape.Size())
	es := C.uint32_t(node.shape.DType.Size())

	if ret := C.metal_reverse(input.mtl, dst.mtl, configBuf.mtl,
		total, C.uint32_t(len(config)), es); ret != 0 {
		return nil, errors.Errorf("metal_reverse failed: %d", ret)
	}
	return dst, nil
}

// ─── Gather ─────────────────────────────────────────────────────────────────

func executeGather(node *Node, operand, indices *Buffer) (*Buffer, error) {
	if indices.shape.DType != dtypes.Int32 {
		return nil, errors.Errorf("gather: indices must be Int32, got %s", indices.shape.DType)
	}

	data := node.data.(*gatherData)
	opShape := operand.shape
	idxShape := indices.shape
	outShape := node.shape

	opStrides := computeStrides(opShape.Dimensions)
	idxStrides := computeStrides(idxShape.Dimensions)
	outStrides := computeStrides(outShape.Dimensions)

	// Pack config buffer per the gather kernel layout:
	// [operand_rank, indices_rank, output_rank, index_vector_axis,
	//  num_offset_axes, num_collapsed, num_start_idx_map,
	//  operand_dims, operand_strides, indices_dims, indices_strides,
	//  output_dims, output_strides, offset_axes, collapsed_axes,
	//  start_index_map, slice_sizes]
	configLen := 7 + 2*opShape.Rank() + 2*idxShape.Rank() + 2*outShape.Rank() +
		len(data.offsetOutputAxes) + len(data.collapsedSliceAxes) +
		len(data.startIndexMap) + len(data.sliceSizes)

	config := make([]uint32, configLen)
	config[0] = uint32(opShape.Rank())
	config[1] = uint32(idxShape.Rank())
	config[2] = uint32(outShape.Rank())
	config[3] = uint32(data.indexVectorAxis)
	config[4] = uint32(len(data.offsetOutputAxes))
	config[5] = uint32(len(data.collapsedSliceAxes))
	config[6] = uint32(len(data.startIndexMap))

	off := 7
	for i := 0; i < opShape.Rank(); i++ {
		config[off+i] = uint32(opShape.Dimensions[i])
	}
	off += opShape.Rank()
	for i := 0; i < opShape.Rank(); i++ {
		config[off+i] = uint32(opStrides[i])
	}
	off += opShape.Rank()
	for i := 0; i < idxShape.Rank(); i++ {
		config[off+i] = uint32(idxShape.Dimensions[i])
	}
	off += idxShape.Rank()
	for i := 0; i < idxShape.Rank(); i++ {
		config[off+i] = uint32(idxStrides[i])
	}
	off += idxShape.Rank()
	for i := 0; i < outShape.Rank(); i++ {
		config[off+i] = uint32(outShape.Dimensions[i])
	}
	off += outShape.Rank()
	for i := 0; i < outShape.Rank(); i++ {
		config[off+i] = uint32(outStrides[i])
	}
	off += outShape.Rank()
	for i, v := range data.offsetOutputAxes {
		config[off+i] = uint32(v)
	}
	off += len(data.offsetOutputAxes)
	for i, v := range data.collapsedSliceAxes {
		config[off+i] = uint32(v)
	}
	off += len(data.collapsedSliceAxes)
	for i, v := range data.startIndexMap {
		config[off+i] = uint32(v)
	}
	off += len(data.startIndexMap)
	for i, v := range data.sliceSizes {
		config[off+i] = uint32(v)
	}

	configBuf := makeConfigBuffer(config)
	defer freeConfigBuffer(configBuf)

	dst := allocDuringExec(outShape)
	total := C.uint32_t(outShape.Size())
	es := C.uint32_t(operand.shape.DType.Size())

	if ret := C.metal_gather(operand.mtl, indices.mtl, dst.mtl, configBuf.mtl,
		total, C.uint32_t(len(config)), es); ret != 0 {
		return nil, errors.Errorf("metal_gather failed: %d", ret)
	}
	return dst, nil
}

// ─── Scatter ────────────────────────────────────────────────────────────────

// scatterEncodeAndRun dispatches scatter_sum/max/min for buffers whose element type
// matches sk (0=f32, 1=i32, 2=u32).
func scatterEncodeAndRun(node *Node, operand, indices, updates, dst *Buffer, sk C.int) error {
	data := node.data.(*scatterData)
	opShape := operand.shape
	idxShape := indices.shape
	updShape := updates.shape

	opStrides := computeStrides(opShape.Dimensions)
	idxStrides := computeStrides(idxShape.Dimensions)
	updStrides := computeStrides(updShape.Dimensions)

	configLen := 7 + 2*opShape.Rank() + 2*idxShape.Rank() + 2*updShape.Rank() +
		len(data.updateWindowAxes) + len(data.insertedWindowAxes) +
		len(data.scatterAxesToOperandAxes)

	config := make([]uint32, configLen)
	config[0] = uint32(opShape.Rank())
	config[1] = uint32(idxShape.Rank())
	config[2] = uint32(updShape.Rank())
	config[3] = uint32(data.indexVectorAxis)
	config[4] = uint32(len(data.updateWindowAxes))
	config[5] = uint32(len(data.insertedWindowAxes))
	config[6] = uint32(len(data.scatterAxesToOperandAxes))

	off := 7

	for i := 0; i < opShape.Rank(); i++ {
		config[off+i] = uint32(opShape.Dimensions[i])
	}
	off += opShape.Rank()
	for i := 0; i < opShape.Rank(); i++ {
		config[off+i] = uint32(opStrides[i])
	}
	off += opShape.Rank()
	for i := 0; i < idxShape.Rank(); i++ {
		config[off+i] = uint32(idxShape.Dimensions[i])
	}
	off += idxShape.Rank()
	for i := 0; i < idxShape.Rank(); i++ {
		config[off+i] = uint32(idxStrides[i])
	}
	off += idxShape.Rank()
	for i := 0; i < updShape.Rank(); i++ {
		config[off+i] = uint32(updShape.Dimensions[i])
	}
	off += updShape.Rank()
	for i := 0; i < updShape.Rank(); i++ {
		config[off+i] = uint32(updStrides[i])
	}
	off += updShape.Rank()
	for i, v := range data.updateWindowAxes {
		config[off+i] = uint32(v)
	}
	off += len(data.updateWindowAxes)
	for i, v := range data.insertedWindowAxes {
		config[off+i] = uint32(v)
	}
	off += len(data.insertedWindowAxes)
	for i, v := range data.scatterAxesToOperandAxes {
		config[off+i] = uint32(v)
	}

	configBuf := makeConfigBuffer(config)
	defer freeConfigBuffer(configBuf)

	total := C.uint32_t(updShape.Size())

	var ret C.int
	switch node.opType {
	case backends.OpTypeScatterSum:
		ret = C.metal_scatter_sum(operand.mtl, indices.mtl, updates.mtl, dst.mtl,
			configBuf.mtl, total, C.uint32_t(len(config)), sk)
	case backends.OpTypeScatterMax:
		ret = C.metal_scatter_max(operand.mtl, indices.mtl, updates.mtl, dst.mtl,
			configBuf.mtl, total, C.uint32_t(len(config)), sk)
	case backends.OpTypeScatterMin:
		ret = C.metal_scatter_min(operand.mtl, indices.mtl, updates.mtl, dst.mtl,
			configBuf.mtl, total, C.uint32_t(len(config)), sk)
	default:
		return errors.Errorf("metal: unsupported scatter op %s", node.opType)
	}
	if ret != 0 {
		return errors.Errorf("metal scatter (%s) failed: %d", node.opType, ret)
	}
	return nil
}

func executeScatter(node *Node, operand, indices, updates *Buffer) (*Buffer, error) {
	if indices.shape.DType != dtypes.Int32 {
		return nil, errors.Errorf("scatter: indices must be Int32, got %s", indices.shape.DType)
	}
	if operand.shape.DType != updates.shape.DType {
		return nil, errors.Errorf("scatter: operand dtype %s != updates dtype %s",
			operand.shape.DType, updates.shape.DType)
	}

	// float16: no 16-bit device atomics in MSL — accumulate in float32, convert back.
	if operand.shape.DType == dtypes.Float16 {
		opF32, err := gpuConvertSameShape(operand, dtypes.Float32)
		if err != nil {
			return nil, err
		}
		updF32, err := gpuConvertSameShape(updates, dtypes.Float32)
		if err != nil {
			return nil, err
		}
		dstF32 := allocDuringExec(shapes.Make(dtypes.Float32, operand.shape.Dimensions...))
		if err := scatterEncodeAndRun(node, opF32, indices, updF32, dstF32, 0); err != nil {
			return nil, err
		}
		return gpuConvertSameShape(dstF32, dtypes.Float16)
	}

	sk := scatterElemKind(operand.shape.DType)
	if sk < 0 {
		return nil, errors.Errorf(
			"scatter: metal scatter supports float16/float32/int32/uint32/int64/uint64 only, got operand %s (updates %s)",
			operand.shape.DType, updates.shape.DType)
	}

	dst := allocDuringExec(operand.shape)
	if err := scatterEncodeAndRun(node, operand, indices, updates, dst, sk); err != nil {
		return nil, err
	}
	return dst, nil
}

// ─── ArgMinMax ──────────────────────────────────────────────────────────────

func executeArgMinMax(node *Node, input *Buffer) (*Buffer, error) {
	if input.shape.DType != dtypes.Float32 && input.shape.DType != dtypes.Float16 {
		return nil, errors.Errorf("argminmax: metal supports float16/f32 input only, got %s", input.shape.DType)
	}
	data := node.data.(*argMinMaxData)
	inShape := input.shape
	axis := data.axis

	prefixSize := 1
	for i := 0; i < axis; i++ {
		prefixSize *= inShape.Dimensions[i]
	}
	reduceSize := inShape.Dimensions[axis]
	suffixSize := 1
	for i := axis + 1; i < inShape.Rank(); i++ {
		suffixSize *= inShape.Dimensions[i]
	}

	dt := dtypeToMetal(inShape.DType)
	isMin := C.int(0)
	if data.isMin {
		isMin = 1
	}

	outDType := data.outputDType
	if outDType == dtypes.Int64 {
		tmpShape := shapes.Make(dtypes.Int32, node.shape.Dimensions...)
		tmp := allocDuringExec(tmpShape)
		if ret := C.metal_argminmax(input.mtl, tmp.mtl,
			C.uint32_t(prefixSize), C.uint32_t(reduceSize), C.uint32_t(suffixSize),
			isMin, dt); ret != 0 {
			return nil, errors.Errorf("metal_argminmax failed: %d", ret)
		}
		dst := allocDuringExec(node.shape)
		n := C.uint32_t(tmpShape.Size())
		if ret := C.metal_cast_i32_to_i64(tmp.mtl, dst.mtl, n); ret != 0 {
			return nil, errors.Errorf("metal_cast_i32_to_i64 failed: %d", ret)
		}
		mtlRelease(tmp.mtl)
		tmp.mtl = nil
		return dst, nil
	}

	dst := allocDuringExec(node.shape)
	if ret := C.metal_argminmax(input.mtl, dst.mtl,
		C.uint32_t(prefixSize), C.uint32_t(reduceSize), C.uint32_t(suffixSize),
		isMin, dt); ret != 0 {
		return nil, errors.Errorf("metal_argminmax failed: %d", ret)
	}
	return dst, nil
}

func qdenseGPUKind(scheme backends.QuantizationScheme, wdt dtypes.DType) (int, error) {
	switch scheme {
	case backends.QuantNF4:
		switch wdt {
		case dtypes.Uint4, dtypes.Int4:
			return 1, nil
		case dtypes.Uint8, dtypes.Int8:
			return 0, nil
		default:
			return -1, errors.Errorf("FusedQuantizedDense NF4: need int8/uint8 or int4/uint4 weights, got %s", wdt)
		}
	case backends.QuantLinear:
		switch wdt {
		case dtypes.Int8:
			return 2, nil
		case dtypes.Uint8:
			return 3, nil
		case dtypes.Int4:
			return 4, nil
		case dtypes.Uint4:
			return 5, nil
		default:
			return -1, errors.Errorf("FusedQuantizedDense Linear: unsupported weight dtype %s", wdt)
		}
	default:
		return -1, errors.Errorf("FusedQuantizedDense: unsupported scheme %v", scheme)
	}
}

func executeFusedQuantizedDense(node *Node, inputs []*Buffer) (*Buffer, error) {
	data := node.data.(*nodeFusedQuantizedDense)
	xBuf := inputs[0]
	wBuf := inputs[1]
	sBuf := inputs[2]

	var zeroPointsBuf, biasBuf *Buffer
	nextIdx := 3
	if data.hasZeroPoint {
		zeroPointsBuf = inputs[nextIdx]
		nextIdx++
	}
	if data.hasBias {
		biasBuf = inputs[nextIdx]
	}

	xEffective := xBuf
	if xBuf.shape.DType == dtypes.Float16 {
		var err error
		xEffective, err = gpuConvertSameShape(xBuf, dtypes.Float32)
		if err != nil {
			return nil, err
		}
	} else if xBuf.shape.DType != dtypes.Float32 {
		return nil, errors.Errorf("FusedQuantizedDense: x must be float16 or float32, got %s", xBuf.shape.DType)
	}

	K := xEffective.shape.Dimensions[xEffective.shape.Rank()-1]
	N := wBuf.shape.Dimensions[1]
	M := xEffective.shape.Size() / K
	numBlocks := (N + data.blockSize - 1) / data.blockSize

	kind, err := qdenseGPUKind(data.scheme, wBuf.shape.DType)
	if err != nil {
		return nil, err
	}

	out := allocDuringExec(node.shape)
	outKernel := out
	if node.shape.DType == dtypes.Float16 {
		outKernel = allocDuringExec(shapes.Make(dtypes.Float32, node.shape.Dimensions...))
	}

	var zpM C.MetalBuffer
	if zeroPointsBuf != nil {
		zpM = zeroPointsBuf.mtl
	}
	var biasM C.MetalBuffer
	if biasBuf != nil {
		biasM = biasBuf.mtl
	}

	cfg := []uint32{
		uint32(M), uint32(K), uint32(N),
		uint32(data.blockSize), uint32(numBlocks),
		0, 0, uint32(data.activation),
	}
	if biasBuf != nil {
		cfg[5] = 1
	}
	if zeroPointsBuf != nil {
		cfg[6] = 1
	}

	if ret := C.metal_quantized_dense(
		xEffective.mtl, wBuf.mtl, sBuf.mtl, zpM, biasM, outKernel.mtl,
		(*C.uint32_t)(unsafe.Pointer(&cfg[0])), C.uint32_t(len(cfg)),
		C.uint32_t(M*N), C.int(kind),
	); ret != 0 {
		return nil, errors.Errorf("metal_quantized_dense failed: %d", ret)
	}
	if node.shape.DType == dtypes.Float16 {
		n := C.uint32_t(node.shape.Size())
		if ret := C.metal_convert_dtype(outKernel.mtl, out.mtl, n, 1, 0); ret != 0 {
			return nil, errors.Errorf("metal_quantized_dense f32->f16: %d", ret)
		}
	}
	return out, nil
}

// ─── ConvGeneral ────────────────────────────────────────────────────────────

func executeConvGeneral(node *Node, input, kernel *Buffer) (*Buffer, error) {
	dt := input.shape.DType
	if dt != dtypes.Float32 && dt != dtypes.Float16 {
		return nil, errors.Errorf("conv_general: metal supports float16/float32, got input %s", dt)
	}
	if kernel.shape.DType != dt {
		return nil, errors.Errorf("conv_general: kernel dtype %s must match input %s", kernel.shape.DType, dt)
	}
	if dtypeToMetal(dt) < 0 {
		return nil, errors.Errorf("conv_general: unsupported dtype %s", dt)
	}

	data := node.data.(*convGeneralData)
	inShape := input.shape
	kShape := kernel.shape
	outShape := node.shape
	axes := data.axes
	spatialRank := len(axes.InputSpatial)

	// Config buffer layout for conv_general kernel:
	// [spatial_rank, batch_size, in_channels, out_channels, channel_group_count,
	//  input_spatial_dims[spatial_rank], kernel_spatial_dims[spatial_rank],
	//  output_spatial_dims[spatial_rank], strides[spatial_rank],
	//  input_dilations[spatial_rank], kernel_dilations[spatial_rank],
	//  paddings_low[spatial_rank], paddings_high[spatial_rank],
	//  input_batch_axis, input_channel_axis, input_spatial_axes[spatial_rank],
	//  kernel_in_channel_axis, kernel_out_channel_axis, kernel_spatial_axes[spatial_rank],
	//  output_batch_axis, output_channel_axis, output_spatial_axes[spatial_rank]]
	configLen := 5 + 8*spatialRank + 6 + 3*spatialRank
	config := make([]uint32, configLen)

	config[0] = uint32(spatialRank)
	config[1] = uint32(inShape.Dimensions[axes.InputBatch])
	config[2] = uint32(inShape.Dimensions[axes.InputChannels])
	config[3] = uint32(kShape.Dimensions[axes.KernelOutputChannels])
	config[4] = uint32(data.channelGroupCount)

	off := 5
	for i := 0; i < spatialRank; i++ {
		config[off+i] = uint32(inShape.Dimensions[axes.InputSpatial[i]])
	}
	off += spatialRank
	for i := 0; i < spatialRank; i++ {
		config[off+i] = uint32(kShape.Dimensions[axes.KernelSpatial[i]])
	}
	off += spatialRank
	for i := 0; i < spatialRank; i++ {
		config[off+i] = uint32(outShape.Dimensions[axes.OutputSpatial[i]])
	}
	off += spatialRank
	for i := 0; i < spatialRank; i++ {
		s := 1
		if data.strides != nil && i < len(data.strides) {
			s = data.strides[i]
		}
		config[off+i] = uint32(s)
	}
	off += spatialRank
	for i := 0; i < spatialRank; i++ {
		d := 1
		if data.inputDilations != nil && i < len(data.inputDilations) {
			d = data.inputDilations[i]
		}
		config[off+i] = uint32(d)
	}
	off += spatialRank
	for i := 0; i < spatialRank; i++ {
		d := 1
		if data.kernelDilations != nil && i < len(data.kernelDilations) {
			d = data.kernelDilations[i]
		}
		config[off+i] = uint32(d)
	}
	off += spatialRank
	for i := 0; i < spatialRank; i++ {
		pl := 0
		if data.paddings != nil && i < len(data.paddings) {
			pl = data.paddings[i][0]
		}
		config[off+i] = uint32(pl)
	}
	off += spatialRank
	for i := 0; i < spatialRank; i++ {
		ph := 0
		if data.paddings != nil && i < len(data.paddings) {
			ph = data.paddings[i][1]
		}
		config[off+i] = uint32(ph)
	}
	off += spatialRank

	// Axis indices for input, kernel, output
	config[off] = uint32(axes.InputBatch)
	config[off+1] = uint32(axes.InputChannels)
	off += 2
	for i := 0; i < spatialRank; i++ {
		config[off+i] = uint32(axes.InputSpatial[i])
	}
	off += spatialRank
	config[off] = uint32(axes.KernelInputChannels)
	config[off+1] = uint32(axes.KernelOutputChannels)
	off += 2
	for i := 0; i < spatialRank; i++ {
		config[off+i] = uint32(axes.KernelSpatial[i])
	}
	off += spatialRank
	config[off] = uint32(axes.OutputBatch)
	config[off+1] = uint32(axes.OutputChannels)
	off += 2
	for i := 0; i < spatialRank; i++ {
		config[off+i] = uint32(axes.OutputSpatial[i])
	}

	configBuf := makeConfigBuffer(config)
	defer freeConfigBuffer(configBuf)

	dst := allocDuringExec(outShape)
	total := C.uint32_t(outShape.Size())
	mdt := dtypeToMetal(dt)

	if ret := C.metal_conv_general(input.mtl, kernel.mtl, dst.mtl, configBuf.mtl,
		total, C.uint32_t(len(config)), mdt); ret != 0 {
		return nil, errors.Errorf("metal_conv_general failed: %d", ret)
	}
	return dst, nil
}

// ─── ReduceWindow ───────────────────────────────────────────────────────────

func executeReduceWindow(node *Node, input *Buffer) (*Buffer, error) {
	idt := input.shape.DType
	if idt != dtypes.Float32 && idt != dtypes.Float16 {
		return nil, errors.Errorf("reduce_window: metal supports float16/float32, got %s", idt)
	}
	if dtypeToMetal(idt) < 0 {
		return nil, errors.Errorf("reduce_window: unsupported dtype %s", idt)
	}

	data := node.data.(*reduceWindowData)
	inShape := input.shape
	outShape := node.shape
	rank := inShape.Rank()

	// Map reduction type to int: 0=sum, 1=max, 2=min, 3=product
	reduceType := 0
	switch data.reductionType {
	case backends.ReduceOpSum:
		reduceType = 0
	case backends.ReduceOpMax:
		reduceType = 1
	case backends.ReduceOpMin:
		reduceType = 2
	case backends.ReduceOpProduct:
		reduceType = 3
	}

	inStrides := computeStrides(inShape.Dimensions)

	// Config: [rank, reduce_type,
	//   input_dims[rank], output_dims[rank], window_dims[rank], strides[rank],
	//   base_dilations[rank], window_dilations[rank],
	//   paddings_low[rank], paddings_high[rank], input_strides[rank]]
	configLen := 2 + 9*rank
	config := make([]uint32, configLen)
	config[0] = uint32(rank)
	config[1] = uint32(reduceType)

	off := 2
	for i := 0; i < rank; i++ {
		config[off+i] = uint32(inShape.Dimensions[i])
	}
	off += rank
	for i := 0; i < rank; i++ {
		config[off+i] = uint32(outShape.Dimensions[i])
	}
	off += rank
	for i := 0; i < rank; i++ {
		config[off+i] = uint32(data.windowDimensions[i])
	}
	off += rank
	for i := 0; i < rank; i++ {
		s := 1
		if data.strides != nil && i < len(data.strides) {
			s = data.strides[i]
		}
		config[off+i] = uint32(s)
	}
	off += rank
	for i := 0; i < rank; i++ {
		d := 1
		if data.baseDilations != nil && i < len(data.baseDilations) {
			d = data.baseDilations[i]
		}
		config[off+i] = uint32(d)
	}
	off += rank
	for i := 0; i < rank; i++ {
		d := 1
		if data.windowDilations != nil && i < len(data.windowDilations) {
			d = data.windowDilations[i]
		}
		config[off+i] = uint32(d)
	}
	off += rank
	for i := 0; i < rank; i++ {
		pl := 0
		if data.paddings != nil && i < len(data.paddings) {
			pl = data.paddings[i][0]
		}
		config[off+i] = uint32(pl)
	}
	off += rank
	for i := 0; i < rank; i++ {
		ph := 0
		if data.paddings != nil && i < len(data.paddings) {
			ph = data.paddings[i][1]
		}
		config[off+i] = uint32(ph)
	}
	off += rank
	for i := 0; i < rank; i++ {
		config[off+i] = uint32(inStrides[i])
	}

	configBuf := makeConfigBuffer(config)
	defer freeConfigBuffer(configBuf)

	dst := allocDuringExec(outShape)
	total := C.uint32_t(outShape.Size())
	dt := dtypeToMetal(outShape.DType)

	if ret := C.metal_reduce_window(input.mtl, dst.mtl, configBuf.mtl,
		total, C.uint32_t(len(config)), dt); ret != 0 {
		return nil, errors.Errorf("metal_reduce_window failed: %d", ret)
	}
	return dst, nil
}

// ─── Bitcast ────────────────────────────────────────────────────────────────

func executeBitcast(node *Node, input *Buffer) (*Buffer, error) {
	// Bitcast is zero-copy: same underlying bytes, different type interpretation.
	mtlRetain(input.mtl)
	view := &Buffer{mtl: input.mtl, shape: node.shape}
	if execBackendTL != nil && execBackendTL.scratch != nil {
		execBackendTL.scratch.own(view)
	}
	return view, nil
}

// ─── Dot / DotGeneral ───────────────────────────────────────────────────────

func executeDot(node *Node, lhs, rhs *Buffer) (*Buffer, error) {
	m := uint32(lhs.shape.Dimensions[0])
	k := uint32(lhs.shape.Dimensions[1])
	n := uint32(rhs.shape.Dimensions[1])

	dst := allocDuringExec(node.shape)
	dt := dtypeToMetal(node.shape.DType)
	if ret := C.metal_dot_general(lhs.mtl, rhs.mtl, dst.mtl,
		1, C.uint32_t(m), C.uint32_t(k), C.uint32_t(n), dt); ret != 0 {
		return nil, errors.Errorf("metal_dot_general failed: %d", ret)
	}
	return dst, nil
}

func executeDotGeneral(node *Node, lhs, rhs *Buffer) (*Buffer, error) {
	data := node.data.(*dotGeneralData)

	// Normalize to [batch, m, k] @ [batch, k, n] by computing the effective
	// dimensions from the axis specification.
	lShape := lhs.shape
	rShape := rhs.shape

	// Calculate batch size
	batch := uint32(1)
	for _, a := range data.lhsBatchAxes {
		batch *= uint32(lShape.Dimensions[a])
	}

	// Calculate contracting size
	k := uint32(1)
	for _, a := range data.lhsContractingAxes {
		k *= uint32(lShape.Dimensions[a])
	}

	// Calculate LHS free size (m)
	contractSetL := make(map[int]bool)
	batchSetL := make(map[int]bool)
	for _, a := range data.lhsContractingAxes {
		contractSetL[a] = true
	}
	for _, a := range data.lhsBatchAxes {
		batchSetL[a] = true
	}
	m := uint32(1)
	for i, d := range lShape.Dimensions {
		if !contractSetL[i] && !batchSetL[i] {
			m *= uint32(d)
		}
	}

	// Calculate RHS free size (n)
	contractSetR := make(map[int]bool)
	batchSetR := make(map[int]bool)
	for _, a := range data.rhsContractingAxes {
		contractSetR[a] = true
	}
	for _, a := range data.rhsBatchAxes {
		batchSetR[a] = true
	}
	n := uint32(1)
	for i, d := range rShape.Dimensions {
		if !contractSetR[i] && !batchSetR[i] {
			n *= uint32(d)
		}
	}

	// For the general case, we need to transpose the inputs into [batch, m, k]
	// and [batch, k, n] layout. For the common case of rank-2 matmul or
	// batch matmul where axes are already in the right order, we can skip.
	if lShape.Rank() == 2 && rShape.Rank() == 2 {
		return executeDot(node, lhs, rhs)
	}

	// General case: build permutations to get [batch, free, contract] for LHS
	// and [batch, contract, free] for RHS, then reshape to 3D.
	lhsPerm := make([]int, 0, lShape.Rank())
	for _, a := range data.lhsBatchAxes {
		lhsPerm = append(lhsPerm, a)
	}
	for i := range lShape.Dimensions {
		if !contractSetL[i] && !batchSetL[i] {
			lhsPerm = append(lhsPerm, i)
		}
	}
	for _, a := range data.lhsContractingAxes {
		lhsPerm = append(lhsPerm, a)
	}

	rhsPerm := make([]int, 0, rShape.Rank())
	for _, a := range data.rhsBatchAxes {
		rhsPerm = append(rhsPerm, a)
	}
	for _, a := range data.rhsContractingAxes {
		rhsPerm = append(rhsPerm, a)
	}
	for i := range rShape.Dimensions {
		if !contractSetR[i] && !batchSetR[i] {
			rhsPerm = append(rhsPerm, i)
		}
	}

	// Transpose LHS
	lhsTransDims := make([]int, len(lhsPerm))
	for i, p := range lhsPerm {
		lhsTransDims[i] = lShape.Dimensions[p]
	}
	lhsTrans := allocDuringExec(shapes.Make(lShape.DType, lhsTransDims...))
	if err := transposeBufferUnified(lhs, lhsTrans, lShape.Dimensions, lhsTransDims, lhsPerm, lShape.DType); err != nil {
		return nil, err
	}

	// Transpose RHS
	rhsTransDims := make([]int, len(rhsPerm))
	for i, p := range rhsPerm {
		rhsTransDims[i] = rShape.Dimensions[p]
	}
	rhsTrans := allocDuringExec(shapes.Make(rShape.DType, rhsTransDims...))
	if err := transposeBufferUnified(rhs, rhsTrans, rShape.Dimensions, rhsTransDims, rhsPerm, rShape.DType); err != nil {
		return nil, err
	}

	// Now dispatch as batched matmul
	dst := allocDuringExec(node.shape)
	dt := dtypeToMetal(node.shape.DType)
	if ret := C.metal_dot_general(lhsTrans.mtl, rhsTrans.mtl, dst.mtl,
		C.uint32_t(batch), C.uint32_t(m), C.uint32_t(k), C.uint32_t(n), dt); ret != 0 {
		return nil, errors.Errorf("metal_dot_general failed: %d", ret)
	}
	return dst, nil
}

// ─── Fused ops ──────────────────────────────────────────────────────────────

func executeFusedSoftmax(node *Node, input *Buffer) (*Buffer, error) {
	data := node.data.(*fusedSoftmaxData)
	inShape := input.shape
	axisSize := inShape.Dimensions[data.axis]
	outerSize := inShape.Size() / axisSize

	dst := allocDuringExec(node.shape)
	dt := dtypeToMetal(inShape.DType)
	if dt < 0 {
		return nil, errors.Errorf("metal_fused_softmax: unsupported dtype %s", inShape.DType)
	}

	if ret := C.metal_fused_softmax(input.mtl, dst.mtl,
		C.uint32_t(outerSize), C.uint32_t(axisSize), dt); ret != 0 {
		return nil, errors.Errorf("metal_fused_softmax failed: %d", ret)
	}
	return dst, nil
}

func executeFusedGelu(node *Node, input *Buffer) (*Buffer, error) {
	data := node.data.(*fusedGeluData)
	dst := allocDuringExec(node.shape)
	exact := C.int(0)
	if data.exact {
		exact = 1
	}

	dt := dtypeToMetal(input.shape.DType)
	if dt < 0 {
		return nil, errors.Errorf("metal_fused_gelu: unsupported dtype %s", input.shape.DType)
	}

	if ret := C.metal_fused_gelu(input.mtl, dst.mtl,
		C.uint32_t(node.shape.Size()), exact, dt); ret != 0 {
		return nil, errors.Errorf("metal_fused_gelu failed: %d", ret)
	}
	return dst, nil
}

func executeFusedLayerNorm(node *Node, inputs []*Buffer) (*Buffer, error) {
	data := node.data.(*fusedLayerNormData)
	x := inputs[0]
	inShape := x.shape

	normSize := 1
	for _, ax := range data.axes {
		normSize *= inShape.Dimensions[ax]
	}

	batchSize := inShape.Size() / normSize

	dst := allocDuringExec(node.shape)
	dt := dtypeToMetal(inShape.DType)
	if dt < 0 {
		return nil, errors.Errorf("metal_fused_layernorm: unsupported dtype %s", inShape.DType)
	}

	var gammaBuf, betaBuf C.MetalBuffer
	hasGamma := C.int(0)
	hasBeta := C.int(0)
	if len(inputs) > 1 {
		gammaBuf = inputs[1].mtl
		hasGamma = 1
	}
	if len(inputs) > 2 {
		betaBuf = inputs[2].mtl
		hasBeta = 1
	}

	if ret := C.metal_fused_layernorm(x.mtl, gammaBuf, betaBuf, dst.mtl,
		C.uint32_t(batchSize), C.uint32_t(normSize),
		C.float(data.epsilon), hasGamma, hasBeta, dt); ret != 0 {
		return nil, errors.Errorf("metal_fused_layernorm failed: %d", ret)
	}
	return dst, nil
}

// sdpaMaskStrides returns element strides for a row-major mask shaped [batch, heads?,
// seq, kv] with size-1 axes broadcast (stride 0), matching backends/simplego.
func sdpaMaskStrides(dims []int) (batchStride, headStride int, err error) {
	switch len(dims) {
	case 2:
		return 0, 0, nil
	case 3:
		if dims[0] <= 1 {
			return 0, 0, nil
		}
		return dims[1] * dims[2], 0, nil
	case 4:
		var bs, hs int
		if dims[0] > 1 {
			bs = dims[1] * dims[2] * dims[3]
		}
		if dims[1] > 1 {
			hs = dims[2] * dims[3]
		}
		return bs, hs, nil
	default:
		return 0, 0, errors.Errorf("metal: SDPA mask rank %d not supported (expected 2..4)", len(dims))
	}
}

func sdpaMaterializeBoolMask(mask *Buffer, qDtype dtypes.DType) (*Buffer, error) {
	dst := allocDuringExec(shapes.Make(qDtype, mask.shape.Dimensions...))
	var md C.int
	switch qDtype {
	case dtypes.Float16:
		md = 0
	case dtypes.Float32:
		md = 1
	default:
		return nil, errors.Errorf("metal: SDPA with bool mask not supported for query dtype %s", qDtype)
	}

	n := C.uint32_t(mask.shape.Size())
	if ret := C.metal_bool_mask_to_float(mask.mtl, dst.mtl, n, md); ret != 0 {
		return nil, errors.Errorf("metal_bool_mask_to_float failed: %d", ret)
	}
	return dst, nil
}

func executeFusedSDPA(node *Node, inputs []*Buffer) (*Buffer, error) {
	data := node.data.(*fusedSDPAData)
	q := inputs[0]
	k := inputs[1]
	v := inputs[2]

	qShape := q.shape
	var batch, numHeads, seqLen, headDim uint32
	if data.axesLayout == backends.AxesLayoutBHSD {
		batch = uint32(qShape.Dimensions[0])
		numHeads = uint32(qShape.Dimensions[1])
		seqLen = uint32(qShape.Dimensions[2])
		headDim = uint32(qShape.Dimensions[3])
	} else {
		batch = uint32(qShape.Dimensions[0])
		seqLen = uint32(qShape.Dimensions[1])
		numHeads = uint32(qShape.Dimensions[2])
		headDim = uint32(qShape.Dimensions[3])
	}

	kShape := k.shape
	var kvLen uint32
	if data.axesLayout == backends.AxesLayoutBHSD {
		kvLen = uint32(kShape.Dimensions[2])
	} else {
		kvLen = uint32(kShape.Dimensions[1])
	}

	dst := allocDuringExec(node.shape)

	var mask *Buffer
	maskType := 0
	var maskBatchStride, maskHeadStride uint32
	if len(inputs) > 3 {
		mask = inputs[3]
		// Same as simplego: BSHD mask [b,s,h,kv] → BHSD [b,h,s,kv] for stride-based indexing.
		if data.axesLayout == backends.AxesLayoutBSHD && mask.shape.Rank() == 4 {
			md := mask.shape.Dimensions
			outDims := []int{md[0], md[2], md[1], md[3]}
			tmp := allocDuringExec(shapes.Make(mask.shape.DType, outDims...))
			if err := transposeBufferUnified(mask, tmp, md, outDims, []int{0, 2, 1, 3}, mask.shape.DType); err != nil {
				return nil, err
			}
			mask = tmp
		}

		if mask.shape.Rank() < 2 || mask.shape.Rank() > 4 {
			return nil, errors.Errorf("metal: SDPA mask rank %d not supported (need 2..4)", mask.shape.Rank())
		}

		if mask.shape.DType == dtypes.Bool {
			mat, err := sdpaMaterializeBoolMask(mask, qShape.DType)
			if err != nil {
				return nil, err
			}
			mask = mat
			maskType = 1
		} else {
			if mask.shape.DType != qShape.DType {
				return nil, errors.Errorf("metal: SDPA additive mask dtype %s must match query dtype %s",
					mask.shape.DType, qShape.DType)
			}
			maskType = 2
		}

		mb, mh, err := sdpaMaskStrides(mask.shape.Dimensions)
		if err != nil {
			return nil, err
		}
		maskBatchStride = uint32(mb)
		maskHeadStride = uint32(mh)
	}

	if err := cFusedSDPA(q, k, v, dst, mask, batch, numHeads, uint32(data.numKVHeads), seqLen, kvLen, headDim,
		data.scale, data.causal, maskType, maskBatchStride, maskHeadStride, qShape.DType); err != nil {
		return nil, err
	}
	return dst, nil
}

// ─── Utility functions ──────────────────────────────────────────────────────

func computeStrides(dims []int) []int {
	strides := make([]int, len(dims))
	stride := 1

	for i := len(dims) - 1; i >= 0; i-- {
		strides[i] = stride
		stride *= dims[i]
	}
	
	return strides
}

// dtypeToMetalExt maps GoMLX dtypes to the ConvertDType kernel kind enum.
// 0=f16, 1=f32, 3=i32, 4=i64, 5=u32, 6=u64, 7=bool, 8=i8, 9=i16, 10=u8, 11=u16.
// Float64/double remains unsupported here.
func dtypeToMetalExt(dt dtypes.DType) C.int {
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
	case dtypes.Bool:
		return 7
	case dtypes.Int8:
		return 8
	case dtypes.Int16:
		return 9
	case dtypes.Uint8:
		return 10
	case dtypes.Uint16:
		return 11
	default:
		return -1
	}
}

// executeRNGBitGeneratorGPU runs PCG byte generation on the GPU (see kernels/rng_pcg.metal).
func executeRNGBitGeneratorGPU(node *Node, state *Buffer) ([]*Buffer, error) {
	if len(node.multiOutputsShapes) != 2 {
		return nil, errors.New("RNGBitGenerator: expected 2 outputs")
	}

	newState := allocDuringExec(node.multiOutputsShapes[0])
	rngData := allocDuringExec(node.multiOutputsShapes[1])
	sh := node.multiOutputsShapes[1]
	nbytes := sh.Size() * int(sh.DType.Size())

	if ret := C.metal_rng_pcg_fill(state.mtl, newState.mtl, rngData.mtl, C.uint32_t(nbytes)); ret != 0 {
		return nil, errors.Errorf("metal_rng_pcg_fill failed: %d", ret)
	}

	return []*Buffer{newState, rngData}, nil
}

func metalBnGeomFrom(xShape, scaleShape shapes.Shape, fa int) (C.MetalBnGeom, error) {
	rank := xShape.Rank()
	if fa < 0 {
		fa += rank
	}

	if fa < 0 || fa >= rank {
		return C.MetalBnGeom{}, errors.Errorf("metal batch norm: feature axis %d invalid for rank %d", fa, rank)
	}

	if xShape.DType != dtypes.Float16 && xShape.DType != dtypes.Float32 {
		return C.MetalBnGeom{}, errors.New("metal batch norm: float16/float32 only")
	}
	
	if scaleShape.DType != xShape.DType {
		return C.MetalBnGeom{}, errors.Errorf(
			"metal batch norm: scale dtype %s must match operand %s",
			scaleShape.DType, xShape.DType)
	}

	inner := 1
	
	for i := fa + 1; i < rank; i++ {
		inner *= xShape.Dimensions[i]
	}
	
	ch := xShape.Dimensions[fa]
	outer := xShape.Size() / (ch * inner)

	if outer*ch*inner != xShape.Size() {
		return C.MetalBnGeom{}, errors.New("metal batch norm: unsupported layout (merge outer*channels*inner != numel)")
	}
	
	if scaleShape.Size() != ch {
		return C.MetalBnGeom{}, errors.Errorf("metal batch norm: scale elements %d != feature size %d", scaleShape.Size(), ch)
	}
	
	return C.MetalBnGeom{
		channels: C.uint32_t(ch),
		inner:    C.uint32_t(inner),
		outer:    C.uint32_t(outer),
		numel:    C.uint32_t(xShape.Size()),
	}, nil
}

// executeBatchNormTrainingGPU implements forward batch norm on GPU (kernels/batch_norm.metal).
func executeBatchNormTrainingGPU(node *Node, inputs []*Buffer) ([]*Buffer, error) {
	data := node.data.(*batchNormTrainingData)
	x, scale, offset := inputs[0], inputs[1], inputs[2]
	geom, err := metalBnGeomFrom(x.shape, scale.shape, data.featureAxis)
	
	if err != nil {
		return nil, err
	}

	mdt := dtypeToMetal(x.shape.DType)
	
	if mdt < 0 || mdt > 1 {
		return nil, errors.New("metal batch norm training: expected float16 or float32 operand")
	}

	outNorm := allocDuringExec(node.multiOutputsShapes[0])
	outMean := allocDuringExec(node.multiOutputsShapes[1])
	outVar := allocDuringExec(node.multiOutputsShapes[2])
	
	if ret := C.metal_bn_training_forward(x.mtl, scale.mtl, offset.mtl,
		outNorm.mtl, outMean.mtl, outVar.mtl, geom, C.float(data.epsilon), mdt); ret != 0 {
		return nil, errors.Errorf("metal_bn_training_forward failed: %d", ret)
	}
	
	return []*Buffer{outNorm, outMean, outVar}, nil
}

// executeBatchNormGradientGPU implements batch norm VJP on GPU.
func executeBatchNormGradientGPU(node *Node, inputs []*Buffer) ([]*Buffer, error) {
	data := node.data.(*batchNormGradientData)
	x, scale, mean, variance, dy := inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]
	geom, err := metalBnGeomFrom(x.shape, scale.shape, data.featureAxis)
	if err != nil {
		return nil, err
	}

	if !mean.shape.Equal(scale.shape) || !variance.shape.Equal(scale.shape) {
		return nil, errors.New("metal batch norm gradient: mean/variance must match scale shape")
	}
	if !dy.shape.Equal(x.shape) {
		return nil, errors.New("metal batch norm gradient: gradOutput must match operand shape")
	}

	dx := allocDuringExec(node.multiOutputsShapes[0])
	dgamma := allocDuringExec(node.multiOutputsShapes[1])
	dbeta := allocDuringExec(node.multiOutputsShapes[2])
	mdt := dtypeToMetal(x.shape.DType)
	if mdt < 0 || mdt > 1 {
		return nil, errors.New("metal batch norm gradient: expected float16 or float32 operand")
	}

	if ret := C.metal_bn_gradient(x.mtl, scale.mtl, mean.mtl, variance.mtl, dy.mtl,
		dx.mtl, dgamma.mtl, dbeta.mtl, geom, C.float(data.epsilon), mdt); ret != 0 {
		return nil, errors.Errorf("metal_bn_gradient failed: %d", ret)
	}
	return []*Buffer{dx, dgamma, dbeta}, nil
}

// flatToBytes converts a flat Go slice to a byte slice (zero-copy).
func flatToBytes(flat any) []byte {
	v := reflect.ValueOf(flat)
	if v.Kind() != reflect.Slice {
		return nil
	}

	elemSize := int(v.Type().Elem().Size())
	length := v.Len() * elemSize
	if length == 0 {
		return nil
	}

	return unsafe.Slice((*byte)(unsafe.Pointer(v.Pointer())), length)
}

// flatFromBuffer creates a Go slice backed by the Metal buffer's shared memory.
func flatFromBuffer(buf *Buffer) any {
	size := buf.shape.Size()
	ptr := buf.contents()

	switch buf.shape.DType {
	case dtypes.Float32:
		return unsafe.Slice((*float32)(ptr), size)
	case dtypes.Float64:
		return unsafe.Slice((*float64)(ptr), size)
	case dtypes.Float16:
		return unsafe.Slice((*uint16)(ptr), size)
	case dtypes.Int32:
		return unsafe.Slice((*int32)(ptr), size)
	case dtypes.Int64:
		return unsafe.Slice((*int64)(ptr), size)
	case dtypes.Int8:
		return unsafe.Slice((*int8)(ptr), size)
	case dtypes.Uint8:
		return unsafe.Slice((*uint8)(ptr), size)
	case dtypes.Uint64:
		return unsafe.Slice((*uint64)(ptr), size)
	case dtypes.Bool:
		return unsafe.Slice((*bool)(ptr), size)
	default:
		return unsafe.Slice((*byte)(ptr), size*int(buf.shape.DType.Size()))
	}
}

// checkFlat validates that flat is a typed slice.
func checkFlat(flat any) (dtypes.DType, int, error) {
	v := reflect.ValueOf(flat)
	
	if v.Kind() != reflect.Slice {
		return dtypes.InvalidDType, 0, errors.Errorf("flat should be a slice, got %T", flat)
	}

	dtype := dtypes.FromGoType(v.Type().Elem())
	
	if dtype == dtypes.InvalidDType {
		return dtype, 0, errors.Errorf("unsupported element type %s", v.Type().Elem())
	}
	
	return dtype, v.Len(), nil
}
