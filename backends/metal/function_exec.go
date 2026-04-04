//go:build darwin && cgo

package metal

import (
	"unsafe"

	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// functionExecutable is compiled execution metadata for a Function (main, named, or closure).
type functionExecutable struct {
	function          *Function
	numNodesToProcess int
	numUses           []int
	dependents        [][]int
	outputNodes       []*Node
}

func newFunctionExecutable(f *Function) (*functionExecutable, error) {
	if !f.returned {
		return nil, errors.New("function must have Return() called before compilation")
	}

	var numNodesToProcess int
	
	for _, output := range f.outputs {
		numNodesToProcess = max(numNodesToProcess, output.idx+1)
	}
	
	fe := &functionExecutable{
		function:          f,
		outputNodes:       f.outputs,
		numNodesToProcess: numNodesToProcess,
		numUses:           make([]int, numNodesToProcess),
		dependents:        make([][]int, numNodesToProcess),
	}
	
	for _, output := range f.outputs {
		fe.countNodeUsesAndDependents(output)
	}
	
	return fe, nil
}

func (fe *functionExecutable) countNodeUsesAndDependents(node *Node) {
	nodeIdx := node.idx
	fe.numUses[nodeIdx]++
	
	if fe.numUses[nodeIdx] == 1 {
		for _, input := range node.inputs {
			fe.dependents[input.idx] = append(fe.dependents[input.idx], nodeIdx)
			fe.countNodeUsesAndDependents(input)
		}
	
		for _, closureCaptures := range node.capturedInputs {
			for _, capturedInput := range closureCaptures {
				fe.dependents[capturedInput.idx] = append(fe.dependents[capturedInput.idx], nodeIdx)
				fe.countNodeUsesAndDependents(capturedInput)
			}
		}
	}
}

// liveBuffers is the set of *Buffer values that remain referenced after this
// invocation: parameters, captured locals, and declared outputs.
func (fe *functionExecutable) liveBuffers(results []*Buffer) map[*Buffer]struct{} {
	live := make(map[*Buffer]struct{})
	
	for _, p := range fe.function.parameters {
		if b := results[p.idx]; b != nil {
			live[b] = struct{}{}
		}
	}
	
	for _, c := range fe.function.capturedLocalNodes {
		if b := results[c.idx]; b != nil {
			live[b] = struct{}{}
		}
	}
	
	for _, o := range fe.outputNodes {
		if b := results[o.idx]; b != nil {
			live[b] = struct{}{}
		}
	}
	
	return live
}

// releaseDeadAliasViews releases one MTL refcount per Reshape/Bitcast node whose
// result buffer is not live. Each such op paired mtlRetain with a view that only
// this *Buffer represented; if nothing references that view after the run, we must
// balance here (intermediates, including chains from parameters or nested calls).
func (fe *functionExecutable) releaseDeadAliasViews(results []*Buffer) {
	live := fe.liveBuffers(results)
	
	for idx := 0; idx < fe.numNodesToProcess; idx++ {
		buf := results[idx]
	
		if buf == nil || buf.mtl == nil {
			continue
		}
	
		if _, ok := live[buf]; ok {
			continue
		}
	
		switch fe.function.nodes[idx].opType {
		case backends.OpTypeReshape, backends.OpTypeBitcast:
			mtlRelease(buf.mtl)
			buf.mtl = nil
		default:
		}
	}
}

func (fe *functionExecutable) keepSet(results []*Buffer) map[uintptr]struct{} {
	keep := make(map[uintptr]struct{})
	
	for _, p := range fe.function.parameters {
		if buf := results[p.idx]; buf != nil && buf.mtl != nil {
			keep[uintptr(unsafe.Pointer(buf.mtl))] = struct{}{}
		}
	}
	
	for _, c := range fe.function.capturedLocalNodes {
		if buf := results[c.idx]; buf != nil && buf.mtl != nil {
			keep[uintptr(unsafe.Pointer(buf.mtl))] = struct{}{}
		}
	}

	for _, o := range fe.outputNodes {
		if buf := results[o.idx]; buf != nil && buf.mtl != nil {
			keep[uintptr(unsafe.Pointer(buf.mtl))] = struct{}{}
		}
	}
	return keep
}

// liveNodeIndices marks nodes backward-reachable from declared outputs (inputs and
// captured dependencies). Needed so scratch.releaseExcept does not free buffers
// for live intermediates (e.g. values fed into Call or tuple Sort).
func (fe *functionExecutable) liveNodeIndices() []bool {
	live := make([]bool, fe.numNodesToProcess)
	
	var q []int
	enqueue := func(j int) {
		if j < 0 || j >= fe.numNodesToProcess || live[j] {
			return
		}
	
		live[j] = true
		q = append(q, j)
	}
	
	for _, on := range fe.outputNodes {
		enqueue(on.idx)
	}
	
	for head := 0; head < len(q); head++ {
		nidx := q[head]
		n := fe.function.nodes[nidx]
	
		for _, in := range n.inputs {
			if in != nil {
				enqueue(in.idx)
			}
		}
	
		for _, group := range n.capturedInputs {
			for _, cn := range group {
				if cn != nil {
					enqueue(cn.idx)
				}
			}
		}
	}
	
	return live
}

// scratchKeepSet extends keepSet with MTL handles for every live node's result
// buffer so intermediates on paths to outputs survive scratch cleanup.
func (fe *functionExecutable) scratchKeepSet(results []*Buffer) map[uintptr]struct{} {
	keep := fe.keepSet(results)
	
	for i, ok := range fe.liveNodeIndices() {
		if !ok {
			continue
		}
	
		b := results[i]
	
		if b != nil && b.mtl != nil {
			keep[uintptr(unsafe.Pointer(b.mtl))] = struct{}{}
		}
	}
	
	return keep
}

func (fe *functionExecutable) runMain(b *Backend, paramBufs []*Buffer, donate []bool) ([]*Buffer, error) {
	for i, d := range donate {
		if d {
			return nil, errors.Errorf("metal: input buffer donation is not supported (donate[%d]=true)", i)
		}
	}
	
	scratch := &execScratch{}
	saved := b.scratch
	b.scratch = scratch
	
	defer func() { b.scratch = saved }()

	results := make([]*Buffer, fe.numNodesToProcess)
	
	for i, p := range fe.function.parameters {
		results[p.idx] = paramBufs[i]
	}
	
	if err := fe.executeSequential(b, results); err != nil {
		return nil, err
	}
	
	out := make([]*Buffer, len(fe.outputNodes))
	
	for i, on := range fe.outputNodes {
		out[i] = results[on.idx]
	
		if out[i] == nil {
			return nil, errors.Errorf("output %d not computed", i)
		}
	}
	
	scratch.transferOutputsTo(saved, out)
	fe.releaseDeadAliasViews(results)
	scratch.releaseExcept(fe.scratchKeepSet(results), results, fe.outputNodes)
	return out, nil
}

// run executes a closure or named function inside an already-locked backend Execute.
func (fe *functionExecutable) run(b *Backend, inputs []*Buffer, captured []*Buffer) ([]*Buffer, error) {
	if len(inputs) != len(fe.function.parameters) {
		return nil, errors.Errorf("function expects %d inputs, got %d", len(fe.function.parameters), len(inputs))
	}
	
	if len(captured) != len(fe.function.capturedLocalNodes) {
		return nil, errors.Errorf("function expects %d captured values, got %d",
			len(fe.function.capturedLocalNodes), len(captured))
	}
	
	scratch := &execScratch{}
	saved := b.scratch
	b.scratch = scratch
	defer func() { b.scratch = saved }()

	results := make([]*Buffer, fe.numNodesToProcess)
	
	for i, p := range fe.function.parameters {
		results[p.idx] = inputs[i]
	}
	
	for i, cn := range fe.function.capturedLocalNodes {
		results[cn.idx] = captured[i]
	}
	
	if err := fe.executeSequential(b, results); err != nil {
		return nil, err
	}
	
	out := make([]*Buffer, len(fe.outputNodes))
	
	for i, on := range fe.outputNodes {
		out[i] = results[on.idx]
		if out[i] == nil {
			return nil, errors.Errorf("output %d not computed", i)
		}
	}
	
	scratch.transferOutputsTo(saved, out)
	fe.releaseDeadAliasViews(results)
	scratch.releaseExcept(fe.scratchKeepSet(results), results, fe.outputNodes)
	return out, nil
}

func (fe *functionExecutable) executeSequential(b *Backend, results []*Buffer) error {
	prevTL := execBackendTL
	execBackendTL = b
	defer func() { execBackendTL = prevTL }()

	for nodeIdx := 0; nodeIdx < fe.numNodesToProcess; nodeIdx++ {
		if results[nodeIdx] != nil {
			continue
		}

		if fe.numUses[nodeIdx] == 0 {
			continue
		}
		
		node := fe.function.nodes[nodeIdx]
		
		if node.isNodeSelectOutput {
			continue
		}
		
		switch node.opType {
		case backends.OpTypeConstant:
			buf, err := executeConstant(node)
			if err != nil {
				return err
			}
			results[nodeIdx] = buf
			continue
		case backends.OpTypeParameter, backends.OpTypeCapturedValue:
			continue
		}

		inputBufs := make([]*Buffer, len(node.inputs))
		
		for i, inp := range node.inputs {
			inputBufs[i] = results[inp.idx]
			if inputBufs[i] == nil {
				return errors.Errorf("input %d for node %s (idx %d) not computed", i, node.opType, nodeIdx)
			}
		}

		switch node.opType {
		case backends.OpTypeCall:
			if err := metalExecCall(b, node, results, inputBufs); err != nil {
				return err
			}
		case backends.OpTypeIf, backends.OpTypeWhile, backends.OpTypeSort:
			if err := metalExecClosureOp(b, node, results, inputBufs); err != nil {
				return err
			}
		case backends.OpTypeRNGBitGenerator:
			outs, err := executeRNGBitGeneratorGPU(node, inputBufs[0])
			if err != nil {
				return err
			}
			for i, on := range node.multiOutputsNodes {
				results[on.idx] = outs[i]
			}
		case backends.OpTypeBatchNormForTraining:
			outs, err := executeBatchNormTrainingGPU(node, inputBufs)
			if err != nil {
				return err
			}
			for i, on := range node.multiOutputsNodes {
				results[on.idx] = outs[i]
			}
		case backends.OpTypeBatchNormGradient:
			outs, err := executeBatchNormGradientGPU(node, inputBufs)
			if err != nil {
				return err
			}
			for i, on := range node.multiOutputsNodes {
				results[on.idx] = outs[i]
			}
		case backends.OpTypeFusedQuantizedDense:
			res, err := executeFusedQuantizedDense(node, inputBufs)
			if err != nil {
				return err
			}
			results[nodeIdx] = res
		default:
			res, err := executeNode(b, node, inputBufs)
			if err != nil {
				return err
			}
			results[nodeIdx] = res
		}
	}
	return nil
}

func metalExecCall(b *Backend, node *Node, results []*Buffer, inputBufs []*Buffer) error {
	data := node.data.(*callNode)
	
	if data.target.compiled == nil {
		return errors.Errorf("Call: target %q not compiled", data.target.name)
	}

	var capBufs []*Buffer
	
	if len(data.target.capturedLocalNodes) > 0 {
		var err error
		capBufs, err = closureCaptureBufs(node, results, 0)
		if err != nil {
			return errors.WithMessagef(err, "Call: %q captures", data.target.name)
		}
	}
	
	outs, err := data.target.compiled.run(b, inputBufs, capBufs)
	
	if err != nil {
		return errors.WithMessagef(err, "Call: executing %q", data.target.name)
	}
	
	for i, on := range node.multiOutputsNodes {
		results[on.idx] = outs[i]
	}
	
	return nil
}
