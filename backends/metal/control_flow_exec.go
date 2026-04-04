//go:build darwin && cgo

package metal

import (
	"unsafe"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/pkg/errors"
)

// Unstable sort: bitonic when axis length is a power of two (>=2), else odd-even.
// Stable sort: odd-even with strict adjacent swaps (O(n²) compare rounds per fiber; fully GPU, no host fallback).
func sortAxisIsPow2(n int) bool {
	return n > 0 && (n&(n-1)) == 0
}

func metalExecClosureOp(b *Backend, node *Node, results []*Buffer, inputBufs []*Buffer) error {
	switch node.opType {
	case backends.OpTypeIf:
		return metalExecIf(b, node, results, inputBufs)
	case backends.OpTypeWhile:
		return metalExecWhile(b, node, results, inputBufs)
	case backends.OpTypeSort:
		return metalExecSort(b, node, results, inputBufs)
	default:
		return errors.Errorf("internal: unknown closure op %s", node.opType)
	}
}

func closureCaptureBufs(node *Node, results []*Buffer, closureIdx int) ([]*Buffer, error) {
	if closureIdx >= len(node.capturedInputs) {
		return nil, errors.Errorf("missing captured inputs group %d for %s", closureIdx, node.opType)
	}
	caps := node.capturedInputs[closureIdx]
	out := make([]*Buffer, len(caps))
	for i, cn := range caps {
		out[i] = results[cn.idx]
		if out[i] == nil {
			return nil, errors.Errorf("captured node idx %d not computed for %s", cn.idx, node.opType)
		}
	}
	return out, nil
}

func metalExecIf(b *Backend, node *Node, results []*Buffer, inputBufs []*Buffer) error {
	pred := inputBufs[0]
	data := node.data.(*ifNode)

	if data.trueBranch.compiled == nil || data.falseBranch.compiled == nil {
		return errors.New("If: branch not compiled")
	}

	takeTrue, err := scalarBoolFromBuffer(pred, "If")

	if err != nil {
		return err
	}

	var capChosen []*Buffer
	var compiled *functionExecutable

	if takeTrue {
		capChosen, err = closureCaptureBufs(node, results, 0)
		compiled = data.trueBranch.compiled
	} else {
		capChosen, err = closureCaptureBufs(node, results, 1)
		compiled = data.falseBranch.compiled
	}

	if err != nil {
		return err
	}

	branchOuts, err := compiled.run(b, nil, capChosen)

	if err != nil {
		if takeTrue {
			return errors.WithMessage(err, "If: true branch")
		}

		return errors.WithMessage(err, "If: false branch")
	}

	for i, on := range node.multiOutputsNodes {
		results[on.idx] = branchOuts[i]
	}

	return nil
}

func metalExecWhile(b *Backend, node *Node, results []*Buffer, inputBufs []*Buffer) error {
	data := node.data.(*whileNode)
	stateCount := data.stateCount

	if len(inputBufs) < stateCount {
		return errors.Errorf("While: need %d state inputs", stateCount)
	}

	state := make([]*Buffer, stateCount)
	copy(state, inputBufs[:stateCount])
	initialState := append([]*Buffer(nil), inputBufs[:stateCount]...)

	capCond, err := closureCaptureBufs(node, results, 0)

	if err != nil {
		return err
	}

	capBody, err := closureCaptureBufs(node, results, 1)

	if err != nil {
		return err
	}

	if data.cond.compiled == nil || data.body.compiled == nil {
		return errors.New("While: cond/body not compiled")
	}

	for iter := 0; ; iter++ {
		condOut, err := data.cond.compiled.run(b, state, capCond)

		if err != nil {
			return errors.WithMessagef(err, "While: cond at iter %d", iter)
		}

		condBuf := condOut[0]
		condTrue, err := scalarBoolFromBuffer(condBuf, "While")

		if err != nil {
			return errors.WithMessagef(err, "While: cond at iter %d", iter)
		}

		condProtect := append(append(append(make([]*Buffer, 0, len(state)+len(capCond)+len(capBody)+len(initialState)),
			state...), capCond...), capBody...)
		condProtect = append(condProtect, initialState...)
		releaseTmpBufferUnlessAliased(condBuf, condProtect)

		if !condTrue {
			break
		}

		newState, err := data.body.compiled.run(b, state, capBody)

		if err != nil {
			return errors.WithMessagef(err, "While: body at iter %d", iter)
		}

		nextProtect := append(append(append(
			make([]*Buffer, 0, len(newState)+len(capCond)+len(capBody)+len(initialState)),
			newState...), capCond...), capBody...)
		nextProtect = append(nextProtect, initialState...)

		for i := 0; i < stateCount; i++ {
			oldState := state[i]
			state[i] = newState[i]
			if oldState != state[i] {
				releaseTmpBufferUnlessAliased(oldState, nextProtect)
			}
		}
	}

	for i := range stateCount {
		results[node.multiOutputsNodes[i].idx] = state[i]
	}
	
	return nil
}

func scalarBoolFromBuffer(buf *Buffer, opName string) (bool, error) {
	if err := metalHostGPUSync(); err != nil {
		return false, errors.WithMessagef(err, "%s: sync predicate", opName)
	}
	
	predAny := flatFromBuffer(buf)
	predBools, ok := predAny.([]bool)
	
	if !ok {
		return false, errors.Errorf("%s: need scalar bool buffer, got %T", opName, predAny)
	}
	
	if len(predBools) != 1 {
		return false, errors.Errorf("%s: need scalar bool, got %d elements", opName, len(predBools))
	}
	
	return predBools[0], nil
}

func tensorBufferBytes(buf *Buffer) []byte {
	n := buf.shape.Size() * int(buf.shape.DType.Size())
	
	if n <= 0 {
		return nil
	}
	
	return unsafe.Slice((*byte)(buf.contents()), n)
}

func releaseTmpBuffer(buf *Buffer) {
	if buf != nil && buf.mtl != nil {
		mtlRelease(buf.mtl)
		buf.mtl = nil
	}
}

func bufferAliasedIn(buf *Buffer, protected []*Buffer) bool {
	if buf == nil {
		return false
	}
	
	for _, p := range protected {
		if p == nil {
			continue
		}
	
		if p == buf || (p.mtl != nil && buf.mtl != nil && p.mtl == buf.mtl) {
			return true
		}
	}
	
	return false
}

// releaseTmpBufferUnlessAliased drops a refcount only when buf is not the same
// allocation (or MTL handle) as any protected buffer — cond/comparator results
// may alias state or captures.
func releaseTmpBufferUnlessAliased(buf *Buffer, protected []*Buffer) {
	if buf == nil || buf.mtl == nil {
		return
	}
	
	if bufferAliasedIn(buf, protected) {
		return
	}
	
	mtlRelease(buf.mtl)
	buf.mtl = nil
}

func metalSortRowGPUBitonic(
	b *Backend, data *sortNode, tensors, outputs []*Buffer, compParams, capComp []*Buffer,
	idxBuf *Buffer, baseOffset, axisStride, axisSize, inputCount int,
) error {
	indexFlat := flatFromBuffer(idxBuf).([]int32)
	
	for i := range indexFlat {
		indexFlat[i] = int32(i)
	}
	
	sortBaseProtect := append(append(append(make([]*Buffer, 0,
		len(tensors)+len(outputs)+len(compParams)+len(capComp)+1), tensors...), outputs...), compParams...)
	sortBaseProtect = append(sortBaseProtect, capComp...)
	sortBaseProtect = append(sortBaseProtect, idxBuf)

	var cmpTrash []*Buffer
	
	for k := uint32(2); k <= uint32(axisSize); k <<= 1 {
		for j := k >> 1; j > 0; j >>= 1 {
			for gid := uint32(0); gid < uint32(axisSize); gid++ {
				ix := gid ^ j
	
				if ix <= gid {
					continue
				}
	
				for t := 0; t < inputCount; t++ {
					es := uint32(outputs[t].shape.DType.Size())
					if err := metalSortLoadPairBytes(outputs[t], compParams[2*t], compParams[2*t+1], idxBuf,
						uint32(baseOffset), uint32(axisStride), es, gid, ix); err != nil {
						return errors.WithMessage(err, "Sort: load pair")
					}
				}
	
				compOut, err := data.comparator.compiled.run(b, compParams, capComp)
	
				if err != nil {
					return errors.WithMessage(err, "Sort: comparator")
				}
	
				bb := compOut[0]
	
				if err := metalSortBitonicSwapIdx(idxBuf, bb, k, j, uint32(axisSize)); err != nil {
					return errors.WithMessage(err, "Sort: bitonic swap")
				}
	
				cmpTrash = append(cmpTrash, bb)
			}
		}
	}
	
	for _, bb := range cmpTrash {
		releaseTmpBufferUnlessAliased(bb, sortBaseProtect)
	}
	
	return nil
}

func metalSortRowGPUOddEven(
	b *Backend, data *sortNode, tensors, outputs []*Buffer, compParams, capComp []*Buffer,
	idxBuf *Buffer, baseOffset, axisStride, axisSize, inputCount int,
	stable bool,
) error {
	indexFlat := flatFromBuffer(idxBuf).([]int32)
	
	for i := range indexFlat {
		indexFlat[i] = int32(i)
	}
	
	sortBaseProtect := append(append(append(make([]*Buffer, 0,
		len(tensors)+len(outputs)+len(compParams)+len(capComp)+1), tensors...), outputs...), compParams...)
	sortBaseProtect = append(sortBaseProtect, capComp...)
	sortBaseProtect = append(sortBaseProtect, idxBuf)

	var cmpTrash []*Buffer
	
	for range axisSize {
		for parity := range 2 {
			for i := parity; i+1 < axisSize; i += 2 {
				for t := range inputCount {
					es := uint32(outputs[t].shape.DType.Size())
					si, sj := uint32(i), uint32(i+1)
					if stable {
						// lhs = keys at i+1, rhs = keys at i => pred means strict inversion (stable).
						si, sj = uint32(i+1), uint32(i)
					}
	
					if err := metalSortLoadPairBytes(outputs[t], compParams[2*t], compParams[2*t+1], idxBuf,
						uint32(baseOffset), uint32(axisStride), es, si, sj); err != nil {
						return errors.WithMessage(err, "Sort: load pair")
					}
				}
	
				compOut, err := data.comparator.compiled.run(b, compParams, capComp)
	
				if err != nil {
					return errors.WithMessage(err, "Sort: comparator")
				}
	
				bb := compOut[0]
	
				if err := metalSortAdjacentSwapIdx(idxBuf, bb, uint32(i), uint32(axisSize), stable); err != nil {
					return errors.WithMessage(err, "Sort: adjacent swap")
				}
	
				cmpTrash = append(cmpTrash, bb)
			}
		}
	}

	for _, bb := range cmpTrash {
		releaseTmpBufferUnlessAliased(bb, sortBaseProtect)
	}
	
	return nil
}

func metalExecSort(b *Backend, node *Node, results []*Buffer, inputBufs []*Buffer) error {
	data := node.data.(*sortNode)
	axis := data.axis
	isStable := data.isStable
	inputCount := data.inputCount
	
	if len(inputBufs) < inputCount {
		return errors.Errorf("Sort: expected %d tensor inputs", inputCount)
	}
	
	tensors := inputBufs[:inputCount]

	capComp, err := closureCaptureBufs(node, results, 0)
	
	if err != nil {
		return err
	}
	
	if data.comparator.compiled == nil {
		return errors.New("Sort: comparator not compiled")
	}

	shape := tensors[0].shape
	rank := shape.Rank()
	axisSize := shape.Dimensions[axis]

	outerSize := 1
	
	for i := range axis {
		outerSize *= shape.Dimensions[i]
	}
	
	innerSize := 1
	
	for i := axis + 1; i < rank; i++ {
		innerSize *= shape.Dimensions[i]
	}
	
	axisStride := innerSize

	useGPUBitonic := !isStable && axisSize >= 2 && sortAxisIsPow2(axisSize)
	useGPUOddEven := !isStable && axisSize >= 2 && !sortAxisIsPow2(axisSize)
	useGPUStableOddEven := isStable && axisSize >= 2

	outputs := make([]*Buffer, inputCount)
	rowTemps := make([]*Buffer, inputCount)
	
	for i, t := range tensors {
		outputs[i] = allocDuringExec(t.shape)
		es := int(t.shape.DType.Size())
		n := t.shape.Size() * es
		if n > 0 {
			copy(tensorBufferBytes(outputs[i]), tensorBufferBytes(t))
		}
		rowTemps[i] = allocDuringExec(shapes.Make(dtypes.Uint8, axisSize*es))
	}
	
	idxBuf := allocDuringExec(shapes.Make(dtypes.Int32, axisSize))
	
	defer func() {
		releaseTmpBuffer(idxBuf)
		for _, rt := range rowTemps {
			releaseTmpBuffer(rt)
		}
	}()

	compParams := make([]*Buffer, 2*inputCount)
	
	for i := range inputCount {
		sc := shapesScalar(tensors[i].shape.DType)
		compParams[2*i] = allocDuringExec(sc)
		compParams[2*i+1] = allocDuringExec(sc)
	}
	
	defer func() {
		for _, p := range compParams {
			releaseTmpBuffer(p)
		}
	}()

	for outer := 0; outer < outerSize; outer++ {
		for inner := 0; inner < innerSize; inner++ {
			baseOffset := outer*axisSize*innerSize + inner
	
			if axisSize < 2 {
				continue
			}
	
			if useGPUBitonic {
				if err := metalSortRowGPUBitonic(b, data, tensors, outputs, compParams, capComp, idxBuf,
					baseOffset, axisStride, axisSize, inputCount); err != nil {
					return err
				}
	
				be := uint32(baseOffset)
				ast := uint32(axisStride)
				asz := uint32(axisSize)
	
				for k, ob := range outputs {
					if err := metalGatherScatterAxisPermute(ob, rowTemps[k], idxBuf, be, ast, asz); err != nil {
						return errors.WithMessage(err, "Sort: axis permute")
					}
				}
	
				continue
			}
			if useGPUOddEven {
				if err := metalSortRowGPUOddEven(b, data, tensors, outputs, compParams, capComp, idxBuf,
					baseOffset, axisStride, axisSize, inputCount, false); err != nil {
					return err
				}
	
				be := uint32(baseOffset)
				ast := uint32(axisStride)
				asz := uint32(axisSize)
	
				for k, ob := range outputs {
					if err := metalGatherScatterAxisPermute(ob, rowTemps[k], idxBuf, be, ast, asz); err != nil {
						return errors.WithMessage(err, "Sort: axis permute")
					}
				}
	
				continue
			}
			if useGPUStableOddEven {
				if err := metalSortRowGPUOddEven(b, data, tensors, outputs, compParams, capComp, idxBuf,
					baseOffset, axisStride, axisSize, inputCount, true); err != nil {
					return err
				}
	
				be := uint32(baseOffset)
				ast := uint32(axisStride)
				asz := uint32(axisSize)
	
				for k, ob := range outputs {
					if err := metalGatherScatterAxisPermute(ob, rowTemps[k], idxBuf, be, ast, asz); err != nil {
						return errors.WithMessage(err, "Sort: axis permute")
					}
				}
	
				continue
			}

			return errors.Errorf("Sort: internal error, no GPU sort path (axisSize=%d isStable=%v)", axisSize, isStable)
		}
	}

	for i, on := range node.multiOutputsNodes {
		results[on.idx] = outputs[i]
	}

	return nil
}

func shapesScalar(dt dtypes.DType) shapes.Shape {
	return shapes.Make(dt)
}
