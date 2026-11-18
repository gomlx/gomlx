// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package autodiff implements reverse-mode automatic differentiation (symbolic backpropagation)
// for GoMLX computation graphs ([graph.Graph]).
//
// Its main entry point is [Gradient], which appends graph nodes computing the partial derivatives
// of a scalar output with respect to selected target nodes using Vector-Jacobian Products (VJP).
// It also honors [graph.StopGradient], custom gradients via [graph.IdentityWithCustomGradient],
// and activation checkpointing via [graph.Node.Checkpoint] / [graph.Node.StopCheckpoint].
package autodiff

import (
	"fmt"
	"math"
	"os"
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/internal/graphbridge"
	"github.com/gomlx/gomlx/internal/graphimpl"
	. "github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
)

// This file implements reverse-mode automatic differentiation, using AccumulatedVJP (Vector Jacobian Product).
// There are many sources discussing this topic, some below:
//
// Jax Autodiff Cookbook: https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html
// What is Automatic Differentiation ? (YouTube video), https://www.youtube.com/watch?v=wG_nF1awSSY&t=864s
//
// Overall in this file we assume the following conventions:
//
// * root node: the final output of the graph. The objective it to generate the gradient of this value with
//      respect to a list of selected gradient nodes (or selected graph inputNodes).
// * selected gradient nodes: the nodes with respect to which we want to calculate the gradient of the output.
//      Typically, in a machine learning set up these  will be the variables (aka. weights).
// * VJP / Adjoint: VJP stands for "Vector Jacobian Product", and it's the  accumulated reverse gradient of the root
//      node with respect to the current node being processed. The final gradients will the adjoints/VJP on the selected
//      gradient nodes. Notice the "V" is not necessarily a vector, it could be of any dimension. And the "V" of one
//      node is the "VJP" of its output node. They are generated in reverse order, from the Graph outputs back to
//      its inputNodes.
// * "new nodes": new nodes that being created on the fly to calculate the adjoints. They are not included in the
//      reverse graph.

// reverseGraph stores information of the Graph in reverse order, in order to use
type reverseGraph struct {
	Graph *graph.Graph
	Root  *graph.Node // This is the output node

	ReverseNodes []*reverseNode
	NumConsumers []int
}

type reverseNode struct {
	Node *graph.Node

	// Consumers is the list of nodes that utilize the output of this node. That is, the nodes whose inputNodes
	// include this node.
	Consumers []*reverseNode

	// Selected indicates whether this is one of the nodes for which we want the gradient.
	Selected bool

	// Included is true for nodes to which the root node has a dependency. Nodes not included are irrelevant for root.
	Included bool

	// Useful is true when this node is in the path to one of the nodes we are calculating the gradient with
	// respect to.
	// For nodes not marked as useful, we don't need to generate the VJP values (aka adjoints).
	Useful bool

	// AccumulatedVJP is the gradient of the root node with respect to the output of this node. In the end it will be the sum
	// of the VJPs back-propagated by all its consumers. Once all of them are included, this node is ready to push
	// its VJP to its inputNodes.
	AccumulatedVJP *graph.Node

	// VJPsForMultiOutputs holds the individual VJPs for a multi-outputs nodes: they are only collapsed to a VJP when
	// all the VJPs for the outputs have been calculated.
	VJPsForMultiOutputs []*graph.Node
}

// accumulateInputVJPs validates and accumulates VJPs for each of node's inputs
// into the reverse graph. It checks that each VJP has the expected shape and
// either initializes or adds to the existing accumulated VJP.
func (rg *reverseGraph) accumulateInputVJPs(node *graph.Node, inputsVJPs []*graph.Node, outputShape shapes.Shape) {
	for ii, input := range node.Inputs() {
		vjp := inputsVJPs[ii]
		if vjp == nil {
			continue
		}
		combinedShape := combineOutputShape(outputShape, input.Shape())
		if !vjp.Shape().Equal(combinedShape) {
			if node.Trace() != nil {
				_, _ = fmt.Fprintf(os.Stderr, "Trace for node in error: %s\n%+v\n\n", node, node.Trace())
			}
			Panicf("invalid Gradient calculation for node %q: invalid shape (or DType) for calculated AccumulatedVJP for "+
				"input #%d (out of %d): input shape=%s, calculated AccumulatedVJP shape=%s (wanted %s)"+
				" -- this probably indicates a bug in the code, please report the issue.",
				node, ii, len(node.Inputs()), input.Shape(), vjp.Shape(), combinedShape)
		}
		rInput := rg.ReverseNodes[input.Id()]
		if rInput.AccumulatedVJP == nil {
			rInput.AccumulatedVJP = vjp
		} else {
			rInput.AccumulatedVJP = graph.Add(rInput.AccumulatedVJP, vjp)
		}
	}
}

// The jacobian
func combineOutputShape(outputShape, inputShape shapes.Shape) shapes.Shape {
	outputShape.DType = inputShape.DType // Make sure they are the same.
	return shapes.ConcatenateDimensions(outputShape, inputShape)
}

// Gradient builds new graph nodes representing the reverse-mode partial derivatives of scalar
// `output` with respect to each node in `gradientNodes`, returning one gradient node per target
// in the same order and with the same shape and DType as the corresponding target.
//
// How it works:
//
//   - Reverse-mode backpropagation (VJP): Starting from adjoint `1` at `output`, Gradient walks
//     backwards in topological order along the subgraph connecting `gradientNodes` to `output`,
//     accumulating Vector-Jacobian Products at each node into new nodes on the same [graph.Graph].
//   - Disconnected or stopped targets: If no differentiable path reaches a node in `gradientNodes`
//     (for example because it was unused or blocked by [graph.StopGradient]), its returned
//     gradient is [graph.ZerosLike](node).
//   - Custom gradients & checkpointing: Per-node custom gradients set with
//     [graph.IdentityWithCustomGradient] take precedence over [VJPRegistration]. Nodes inside a
//     checkpointed region ([graph.Node.Checkpoint] / [graph.Node.StopCheckpoint]) are lazily
//     rematerialized on the backward pass using scheduling barriers so activations don't have to
//     be kept alive across the forward pass.
//   - Higher-order gradients: Because Gradient appends standard [graph.Node] operations to the
//     graph, higher-order derivatives can be computed by calling Gradient again on a scalar
//     derived from its outputs (provided the backward ops involved have registered VJPs).
//
// Constraints and panics:
//
//   - `output` must be a non-complex scalar (`output.Rank() == 0 && !output.DType().IsComplex()`);
//     if your loss is non-scalar, reduce it first (e.g. with [graph.ReduceAllSum] or
//     [graph.ReduceMean]).
//   - `output` and all `gradientNodes` must belong to the same [graph.Graph] while it is still
//     being built (before compilation).
//   - Panics if an operation on a differentiable path from `gradientNodes` to `output` has no
//     custom gradient and no entry in [VJPRegistration].
//
// Example:
//
//	loss := graph.ReduceAllSum(graph.Add(graph.Square(x), graph.Mul(x, y)))
//	grads := autodiff.Gradient(loss, x, y)
//	gradX, gradY := grads[0], grads[1]
func Gradient(output *graph.Node, gradientNodes ...*graph.Node) []*graph.Node {
	allInputNodes := make([]*graph.Node, 0, len(gradientNodes)+1)
	allInputNodes = append(allInputNodes, output)
	allInputNodes = append(allInputNodes, gradientNodes...)
	g := validateBuildingGraphFromInputs(allInputNodes...)

	outputShape := output.Shape()
	if outputShape.Rank() > 0 || outputShape.DType.IsComplex() {
		Panicf("only gradients of a non-complex scalar with respect to tensors are accepted, not jacobians, "+
			"that is, output must be float scalar, got %s", output.Shape())
	}

	rg := newReverseGraph(g, output, gradientNodes)
	rOutput := rg.ReverseNodes[output.Id()]
	// Initialize gradient of the output with respect to itself to 1. When outputShape.Rank() != 0
	// we will need to find something akin to a matrix identity for possibly higher dimensional tensors.
	rOutput.AccumulatedVJP = graph.Ones(output.Graph(), shapes.Make(outputShape.DType))

	rematSweep := newRematerializationSweep()

	// Whether we need the gradient for the node.
	needGradientForNode := func(node *graph.Node) bool {
		if node.StopGradient() {
			return false
		}
		rNode := rg.ReverseNodes[node.Id()]
		return rNode.Included && rNode.Useful
	}

	// Loop from final node backwards, back propagating the gradients. Notice that the nodes are ordered according to
	// the DAG, meaning that by the time g.nodes[ii] is reached, all nodes consuming its outputs will already have been
	// accounted for, and their VJPs summed up.
	for nodeIdx := output.Id(); nodeIdx >= 0; nodeIdx -= 1 {
		node := g.Nodes()[nodeIdx]
		rNode := rg.ReverseNodes[nodeIdx]

		// No need to propagate VJP if either node is not of interest, if there is a stop gradient, or
		// if its inputNodes are not of interest.
		if !needGradientForNode(node) {
			continue
		}
		needInputs := slices.ContainsFunc(node.Inputs(), needGradientForNode)
		if !needInputs {
			continue
		}

		// Special case for multiple-outputs: their vjp need to be aggregated differently.
		if node.NumOutputs() > 1 {
			var hasVJP bool
			for ii := range node.NumOutputs() {
				if rNode.VJPsForMultiOutputs[ii] != nil {
					hasVJP = true
					break
				}
			}
			if !hasVJP {
				// Skip back-propagation for this node, since there are no arriving VJPs.
				continue
			}

			// Fill missing VJPs with zeros (no VJP was incident on them), so the backward pass is calculated properly.
			// * Except if using vjpAlternateOutputs, in which case we don't need the zeros, they
			//   will be handled later in the when using the alternate path for VJP.
			nodeImpl := graphbridge.NodeImpl(node)
			if len(nodeImpl.VJPAlternateOutputs()) == 0 {
				var splitOutputs []*graph.Node
				for ii, shape := range nodeImpl.OutputShapes() {
					if rNode.VJPsForMultiOutputs[ii] == nil {
						if shape.IsDynamic() {
							if splitOutputs == nil {
								splitOutputs = splitNode(node)
							}
							rNode.VJPsForMultiOutputs[ii] = graph.ZerosLike(splitOutputs[ii])
						} else {
							rNode.VJPsForMultiOutputs[ii] = graph.Zeros(node.Graph(), shape)
						}
					}
				}
			}
		}

		// For the usual single output nodes, if there are no gradients arriving to rNode -- e.g.:
		// there was a `stopGradient`, we can't propagate the gradient either.
		if node.NumOutputs() == 1 && rNode.AccumulatedVJP == nil {
			continue
		}

		// Fused ops with decomposed alternatives: transfer VJPs to the alternate output
		// nodes and let the normal gradient loop process the decomposed subgraph.
		// The alternates have lower nodeIdx (built first by InternalFusedOpCaller/Multi),
		// so the loop will reach them later.
		// Works for both single-output (1-element slice) and multi-output fused ops.
		vjpAltOutputs := graphbridge.NodeImpl(node).VJPAlternateOutputs()
		if len(vjpAltOutputs) > 0 {
			// Collect the arriving VJPs: for single-output nodes use AccumulatedVJP,
			// for multi-output nodes use VJPsForMultiOutputs.
			vjps := rNode.VJPsForMultiOutputs
			if node.NumOutputs() == 1 {
				vjps = []*graph.Node{rNode.AccumulatedVJP}
			}
			for i, altImpl := range vjpAltOutputs {
				alt := graphbridge.Node(altImpl)
				if alt == nil || vjps[i] == nil {
					continue
				}
				rAlt := rg.ReverseNodes[alt.Id()]
				if rAlt.AccumulatedVJP == nil {
					rAlt.AccumulatedVJP = vjps[i]
				} else {
					rAlt.AccumulatedVJP = graph.Add(rAlt.AccumulatedVJP, vjps[i])
				}
				markSubgraphUseful(rg, alt)
			}
			continue
		}

		nodeImpl := graphbridge.NodeImpl(node)
		if nodeImpl.Type() == graphimpl.NodeTypeSplitNode {
			// SplitNode just pushes v to the specific element of the tuple.
			// This is a special case because it doesn't return a single VJP to accumulate, but needs to be set
			// in a special way.
			params := nodeImpl.NodeInputs().(*graphimpl.NodeInputsSplitNode)
			rInput := rg.ReverseNodes[params.MultiOutputNode.Id()]
			if rInput.VJPsForMultiOutputs[params.Index] == nil {
				rInput.VJPsForMultiOutputs[params.Index] = rNode.AccumulatedVJP
			} else {
				// This usually shouldn't happen, since a Node is only split once ... but just in case.
				rInput.VJPsForMultiOutputs[params.Index] = graph.Add(rInput.VJPsForMultiOutputs[params.Index], rNode.AccumulatedVJP)
			}
			continue
		}

		// Find vjpFn that calculates backpropagation for this node.
		vjpFn := node.CustomGradient()
		if vjpFn == nil {
			var ok bool
			vjpFn, ok = VJPRegistration[node.Type()]
			if !ok {
				Panicf("graph has node %s with type %q, for which no gradient is defined yet, cannot generate graph gradient", node, node.Type())
			}
		}

		vjpsForOutputs := rNode.VJPsForMultiOutputs
		if node.NumOutputs() == 1 {
			vjpsForOutputs = []*graph.Node{rNode.AccumulatedVJP}
		}

		nodeToUse := node
		if nodeImpl.NeedsRematerialization() {
			var trigger *graph.Node
			if node.NumOutputs() == 1 {
				trigger = rNode.AccumulatedVJP
			} else {
				for _, vjp := range rNode.VJPsForMultiOutputs {
					if vjp != nil {
						trigger = vjp
						break
					}
				}
			}
			rematInputs := rematSweep.forwardInputs(node, trigger)
			nodeToUse = graphbridge.Node(nodeImpl.CloneWithInputs(graphbridge.NodeImpls(rematInputs)...))
		}

		inputsVJPs := vjpFn(nodeToUse, vjpsForOutputs, outputShape)
		if len(inputsVJPs) != len(node.Inputs()) {
			Panicf("AccumulatedVJP(%s) returned %d VJPs, but it has %d inputNodes, implementation of auto-differentiation for node failed",
				node, len(inputsVJPs), len(node.Inputs()))
		}
		rg.accumulateInputVJPs(node, inputsVJPs, outputShape)
	}

	gradients := make([]*graph.Node, len(gradientNodes))
	for ii, node := range gradientNodes {
		rNode := rg.ReverseNodes[node.Id()]
		if rNode.AccumulatedVJP == nil {
			// If there is no path from the output to the gradient node (possibly because of
			// a StopGradient) return zero.
			// TODO: fix the shape if the output wrt which we are calculating the gradient is not a scalar.
			gradients[ii] = graph.ZerosLike(node)

		} else {
			gradients[ii] = rNode.AccumulatedVJP
		}
	}
	return gradients
}

func newReverseGraph(g *graph.Graph, root *graph.Node, gradientNodes []*graph.Node) *reverseGraph {
	nodes := g.Nodes()
	numNodes := len(nodes)
	if root == nil {
		root = nodes[numNodes-1]
	}
	rg := &reverseGraph{
		Graph:        g,
		Root:         root,
		ReverseNodes: make([]*reverseNode, numNodes),
		NumConsumers: make([]int, numNodes),
	}

	// Stitch reverse "consumer" links to graph.
	for ii, node := range nodes {
		rNode := &reverseNode{Node: node}
		rg.ReverseNodes[ii] = rNode
		if node.NumOutputs() > 1 {
			rNode.VJPsForMultiOutputs = make([]*graph.Node, node.NumOutputs())
		}
		for _, input := range node.Inputs() {
			rg.NumConsumers[input.Id()] += 1
		}
	}
	for ii, node := range nodes {
		rNode := rg.ReverseNodes[ii]
		rNode.Consumers = make([]*reverseNode, 0, rg.NumConsumers[ii])
		for _, input := range node.Inputs() {
			rInput := rg.ReverseNodes[input.Id()]
			rInput.Consumers = append(rInput.Consumers, rNode)
		}
	}

	// Mark nodes with a path from root as Included.
	recursivePathFromRoot(rg, root)

	// Mark gradient nodes as selected, and recursively mark all the nodes
	// in a path from root to the selected gradient nodes as Useful.
	for _, selected := range gradientNodes {
		rNode := rg.ReverseNodes[selected.Id()]
		rNode.Selected = true
		recursiveMarkAsUseful(rg, rNode)
	}

	return rg
}

// recursivePathFromRoot mark nodes and its inputNodes recursively as Included.
func recursivePathFromRoot(rg *reverseGraph, node *graph.Node) {
	rNode := rg.ReverseNodes[node.Id()]
	if rNode.Included {
		// Already visited.
		return
	}
	rNode.Included = true
	for _, input := range node.Inputs() {
		recursivePathFromRoot(rg, input)
	}
}

func recursiveMarkAsUseful(rg *reverseGraph, rNode *reverseNode) {
	if !rNode.Included || rNode.Useful {
		// Not relevant or already marked as useful.
		return
	}
	rNode.Useful = true
	for _, consumer := range rNode.Consumers {
		recursiveMarkAsUseful(rg, consumer)
	}
}

// markSubgraphUseful walks the decomposed subgraph rooted at node and marks
// every reachable node as Included and Useful, so the gradient loop will
// process them. This is called when a fused node's VJP is transferred to its
// vjpAlternateOutput — without this, the decomposed nodes may have been
// marked as dead code and would be skipped.
func markSubgraphUseful(rg *reverseGraph, node *graph.Node) {
	rNode := rg.ReverseNodes[node.Id()]
	if rNode.Included && rNode.Useful {
		return // Already marked.
	}
	rNode.Included = true
	rNode.Useful = true
	for _, input := range node.Inputs() {
		markSubgraphUseful(rg, input)
	}
}

// VJP returns the $v \dot Jacobian$ of the given `node`, with respect to each of its inputNodes (given
// by `node.Inputs()`).
//
// outputShape is the shape of the value for which we are calculating the gradient for.
// For now this is only used for Gradient, so one can expect outputShape to be scalar, and `v.Shape()`
// to be the same as `output.Shape()`. But this won't be true once Jacobian functionality (like a
// Gradient where output is a non-scalar tensor),
// is defined.
//
// Args:
//
//	node: node for which we are calculating the backward gradient. An important part of `node` are its inputNodes
//	   given by `node.Inputs()`. The VJP function must one gradient per input.
//	v: gradient (or jacobian) of what we care about with the respect to the output of `node`, also known as the
//	   adjoint. VJP needs to calculate the next adjoint (`v`) but with respect to the inputNodes of `node` instead.
//	outputShape: for now fixed as scalar, and can be ignored. When we add support for Jacobian this will hold
//	   the outputShape of the thing we are calculating the Jacobian for.
//
// Returns:
//
//	The adjoint (the updated `vjps`) to each of `node` inputNodes. That is, the gradient of the loss (typically, but of
//	whatever we are calculating the gradient of) with respect to each of the `node` inputNodes.
//
// VJP is an alias to graph.VJP.
type VJP = graph.VJP

// SingleOutputVJP for VJP of ops that have a single output (most of them).
type SingleOutputVJP func(node, v *graph.Node, outputShape shapes.Shape) []*graph.Node

// vjpForSingleOutput is simple converter from SingleOutputVJP to generic VJP.
func vjpForSingleOutput(vjpFn SingleOutputVJP) VJP {
	return func(node *graph.Node, vjpOutputs []*graph.Node, outputShape shapes.Shape) []*graph.Node {
		return vjpFn(node, vjpOutputs[0], outputShape)
	}
}

// VJPRegistration maps each node type to its implementation of VJP. If implementing a new outputOps, or
// for experimentation, one can dynamically change this.
//
// Notice xla.GetTupleElementNode is specialized inside the main reverse autodiff code, and is not
// in the table here.
var VJPRegistration = map[graphimpl.NodeType]VJP{
	graphimpl.NodeTypeInvalid:                        vjpForSingleOutput(noOpVJP),
	graphimpl.NodeTypeConstant:                       vjpForSingleOutput(nilVJP),
	graphimpl.NodeTypeParameter:                      vjpForSingleOutput(nilVJP),
	graphimpl.NodeTypeConvertDType:                   vjpForSingleOutput(convertDTypeVJP),
	graphimpl.NodeTypeWhere:                          vjpForSingleOutput(whereVJP),
	graphimpl.NodeTypeNeg:                            vjpForSingleOutput(negVJP),
	graphimpl.NodeTypeAbs:                            vjpForSingleOutput(absVJP),
	graphimpl.NodeTypeExp:                            vjpForSingleOutput(expVJP),
	graphimpl.NodeTypeExpm1:                          vjpForSingleOutput(expm1VJP),
	graphimpl.NodeTypeLog:                            vjpForSingleOutput(logVJP),
	graphimpl.NodeTypeLog1p:                          vjpForSingleOutput(log1pVJP),
	graphimpl.NodeTypeCos:                            vjpForSingleOutput(cosVJP),
	graphimpl.NodeTypeSin:                            vjpForSingleOutput(sinVJP),
	graphimpl.NodeTypeTanh:                           vjpForSingleOutput(tanhVJP),
	graphimpl.NodeTypeAdd:                            vjpForSingleOutput(addVJP),
	graphimpl.NodeTypeSub:                            vjpForSingleOutput(subVJP),
	graphimpl.NodeTypeMul:                            vjpForSingleOutput(mulVJP),
	graphimpl.NodeTypeDiv:                            vjpForSingleOutput(divVJP),
	graphimpl.NodeTypePow:                            vjpForSingleOutput(powVJP),
	graphimpl.NodeTypeSqrt:                           vjpForSingleOutput(sqrtVJP),
	graphimpl.NodeTypeRsqrt:                          vjpForSingleOutput(rsqrtVJP),
	graphimpl.NodeTypeErf:                            vjpForSingleOutput(erfVJP),
	graphimpl.NodeTypeBatchNormForTraining:           batchNormForTrainingVJP,
	graphimpl.NodeTypeFusedDense:                     vjpForSingleOutput(fusedDenseVJP),
	graphimpl.NodeTypeFusedQuantizedDense:            vjpForSingleOutput(fusedQuantizedDenseVJP),
	graphimpl.NodeTypeFusedLayerNorm:                 vjpForSingleOutput(fusedLayerNormVJP),
	graphimpl.NodeTypeFusedActivation:                vjpForSingleOutput(fusedActivationVJP),
	graphimpl.NodeTypeFusedScaledDotProductAttention: fusedScaledDotProductAttentionVJP,
	graphimpl.NodeTypeFusedTopK:                      fusedTopKVJP,

	// Bitwise/integer operations: we define a zero gradient to their inputs.
	graphimpl.NodeTypeLogicalAnd: vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeLogicalOr:  vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeLogicalXor: vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeLogicalNot: vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeBitwiseAnd: vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeBitwiseOr:  vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeBitwiseXor: vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeBitwiseNot: vjpForSingleOutput(zeroVJP),

	// Complex numbers.
	graphimpl.NodeTypeReal:    vjpForSingleOutput(realVJP),
	graphimpl.NodeTypeImag:    vjpForSingleOutput(imagVJP),
	graphimpl.NodeTypeConj:    vjpForSingleOutput(conjVJP),
	graphimpl.NodeTypeComplex: vjpForSingleOutput(complexVJP),

	graphimpl.NodeTypeMax:                   vjpForSingleOutput(minMaxVJP),
	graphimpl.NodeTypeMin:                   vjpForSingleOutput(minMaxVJP),
	graphimpl.NodeTypeReshape:               vjpForSingleOutput(reshapeVJP),
	graphimpl.NodeTypeReduceSum:             vjpForSingleOutput(reduceSumVJP),
	graphimpl.NodeTypeReduceMax:             vjpForSingleOutput(reduceMaxVJP),
	graphimpl.NodeTypeReduceMin:             vjpForSingleOutput(reduceMinVJP),
	graphimpl.NodeTypeLogistic:              vjpForSingleOutput(logisticVJP),
	graphimpl.NodeTypeDotGeneral:            vjpForSingleOutput(dotGeneralVJP),
	graphimpl.NodeTypeSlice:                 vjpForSingleOutput(sliceVJP),
	graphimpl.NodeTypeGather:                vjpForSingleOutput(gatherVJP),
	graphimpl.NodeTypeScatterSum:            vjpForSingleOutput(scatterSumVJP),
	graphimpl.NodeTypeScatterMax:            vjpForSingleOutput(scatterMaxOrMinVJP),
	graphimpl.NodeTypeScatterMin:            vjpForSingleOutput(scatterMaxOrMinVJP),
	graphimpl.NodeTypeConcatenate:           vjpForSingleOutput(concatenateVJP),
	graphimpl.NodeTypeConvGeneral:           vjpForSingleOutput(convGeneralVJP),
	graphimpl.NodeTypeReduceWindow:          vjpForSingleOutput(reduceWindowVJP),
	graphimpl.NodeTypeTranspose:             vjpForSingleOutput(transposeVJP),
	graphimpl.NodeTypeBroadcastInDim:        vjpForSingleOutput(broadcastInDimVJP),
	graphimpl.NodeTypeDynamicBroadcastInDim: vjpForSingleOutput(dynamicBroadcastInDimVJP),
	graphimpl.NodeTypeFFT:                   vjpForSingleOutput(fftVJP),
	graphimpl.NodeTypeDynamicSlice:          vjpForSingleOutput(dynamicSliceVJP),
	graphimpl.NodeTypeDynamicUpdateSlice:    vjpForSingleOutput(dynamicUpdateSliceVJP),
	graphimpl.NodeTypeDynamicDimensionSize:  vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeDynamicReshape:        vjpForSingleOutput(dynamicReshapeVJP),
	graphimpl.NodeTypeDynamicShape:          vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeCumMax:                vjpForSingleOutput(cumMaxVJP),
	graphimpl.NodeTypeCumSum:                vjpForSingleOutput(cumSumVJP),
	graphimpl.NodeTypePad:                   vjpForSingleOutput(padVJP),
	graphimpl.NodeTypeDynamicPad:            vjpForSingleOutput(dynamicPadVJP),
	graphimpl.NodeTypeIota:                  vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeDynamicIota:           vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeNonZero:               vjpForSingleOutput(zeroVJP),
	graphimpl.NodeTypeReverse:               vjpForSingleOutput(reverseVJP),
	graphimpl.NodeTypeOptimizationBarrier:   vjpOptimizationBarrier,
	graphimpl.NodeTypeSchedulingBarrier:     vjpForSingleOutput(vjpSchedulingBarrier),
}

// nilVJP returns no gradient, for functions without any inputNodes.
func nilVJP(_, _ *graph.Node, _ shapes.Shape) []*graph.Node {
	return nil
}

// noOpVJP works for anything that has not impact on the gradient, like a No-Op.
func noOpVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	return []*graph.Node{v}
}

// zeroVJP can be used for ops that don't back-propagate any gradient -- like logical operations.
// It's equivalent to doing a StopGradient on the operation.
func zeroVJP(node, v *graph.Node, outputShape shapes.Shape) []*graph.Node {
	return make([]*graph.Node, len(node.Inputs()))
}

// vjpForDefaultBroadcast returns the VJP of the default broadcasting on operations like Add, Mul, Sub, etc.
// It is a reduce-sum of the broadcast dimensions.
func vjpForDefaultBroadcast(node, input, v *graph.Node) *graph.Node {
	_ = validateBuildingGraphFromInputs(node, input, v)
	if input.Shape().Equal(node.Shape()) {
		// If there was no broadcast involved, VJP is the identity.
		return v
	} else if input.IsScalar() {
		// If there was a broadcast from a scalar, the VJP is the full reduction of the V tensor.
		return graph.ReduceAllSum(v)
	}

	// Reduce-sum on the dimensions it was broadcast during the op. Search for all dimensions that
	// are 1 in the input and not 1 in the output (including dynamic dimensions, where dim == -1).
	var reduceDims []int
	for ii, dim := range input.Shape().Dimensions {
		if dim == 1 && v.Shape().Dimensions[ii] != 1 {
			reduceDims = append(reduceDims, ii)
		}
	}
	reduced := v
	if len(reduceDims) > 0 {
		reduced = graph.ReduceSum(v, reduceDims...)
	}
	// Since ReduceSum collapsed those reduced dimensions of now size 1, we need to reshape it back to the
	// original input format.
	var vjp *graph.Node
	err := TryCatch[error](func() {
		vjp = graph.DynamicReshapeLike(reduced, input)
	})
	if err != nil {
		err = errors.WithMessagef(err, "AutoGrad: calculating the AccumulatedVJP of a broadcast: v.Shape()=%s, input.Shape()=%s", v.Shape(), input.Shape())
		panic(err)
	}
	return vjp
}

// Gradient of the type conversion, just converts back the adjoint dtype to
// its input dtype.
//
// Except if the conversion is from float->complex: it's a valid conversion, but for the back-propagation
// we propagate back only the real part.
func convertDTypeVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	_ = node
	inputDType := node.Inputs()[0].DType()
	if node.DType().IsComplex() && inputDType.IsFloat() {
		// Just take the real part of the adjoint, since the input has no effect on the imaginary output.
		return []*graph.Node{graph.ConvertDType(graph.Real(v), inputDType)}
	}
	return []*graph.Node{graph.ConvertDType(v, inputDType)}
}

func negVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	_ = node
	return []*graph.Node{graph.Neg(v)}
}

func absVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	x := node.Inputs()[0]
	if x.DType().IsComplex() {
		// For complex numbers, Abs() is defined as the Euclidean distance in the complex
		// plane from 0.
		//
		// Abs(x_c) = Sqrt(x_re^2 + x_im^2)
		// dAbs(x_c)/dx_re = 1/2 * (x_re^2+x_im^2)^(-1/2) * (2*x_re) = x_re / Abs(x_c)
		// dAbs(x_c)/dx_im = 1/2 * (x_re^2+x_im^2)^(-1/2) * (2*x_im) = x_im / Abs(x_c)
		//
		dxReal := graph.Div(graph.Real(x), node)
		dxImag := graph.Div(graph.Imag(x), node)
		// Multiply by the output adjoint to get the input adjoint.
		dxReal = graph.Mul(dxReal, v)
		dxImag = graph.Mul(dxImag, v)
		// Now there is a sleight of hand here: we are using
		//   dAbs(x_c)/d(x_c) = dAbs(x_c)/dx_re + i.dAbs(x_c)/dx_im ... which is not really true.
		return []*graph.Node{graph.Complex(dxReal, dxImag)}
	}

	// Notice that d(abs(x))/dx at 0 is actually undefined. This will take 1, so it assumes the positive side.
	// This usually has little impact on training, but makes some optimizations where the limits of operations
	// have to behave the same consistent (e.g: layers.BinaryCrossentropyLogitsLoss).
	return []*graph.Node{graph.Mul(v, graph.SignPlusOrMinus(node.Inputs()[0]))}
}

func expVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	return []*graph.Node{graph.Mul(v, node)}
}

// expm1VJP uses d(e^x - 1)/dx = e^x = Expm1(x) + 1.
func expm1VJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	return []*graph.Node{graph.Mul(v, graph.OnePlus(node))}
}

func logVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	return []*graph.Node{graph.Mul(v, graph.Reciprocal(node.Inputs()[0]))}
}

func log1pVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	one := graph.ScalarOne(node.Graph(), node.Inputs()[0].DType())
	return []*graph.Node{graph.Mul(v, graph.Reciprocal(graph.Add(one, node.Inputs()[0])))}
}

func sinVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	x := node.Inputs()[0]
	return []*graph.Node{graph.Mul(v, graph.Cos(x))}
}

func cosVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	x := node.Inputs()[0]
	return []*graph.Node{graph.Mul(v, graph.Neg(graph.Sin(x)))}
}

func tanhVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	tanhX := node // node holds the output of tanh(x)
	return []*graph.Node{graph.Mul(v, graph.OneMinus(graph.Square(tanhX)))}
}

func erfVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	x := node.Inputs()[0]
	// d/dx(erf(x)) = 2⋅e^(-x²)/√π
	c := 2.0 / math.Sqrt(math.Pi)
	dErf := graph.MulScalar(graph.Exp(graph.Neg(graph.Square(x))), c)
	return []*graph.Node{graph.Mul(v, dErf)}
}

func sqrtVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	// d(x^0.5)/dx = 0.5 * x^(-0.5) = 0.5/sqrt(x)
	return []*graph.Node{graph.Mul(v, graph.MulScalar(graph.Reciprocal(node), 0.5))}
}

func rsqrtVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	// node = x^(-0.5) => d(x^(-0.5))/dx = -0.5 * x^(-1.5) = -0.5 * node^3
	return []*graph.Node{graph.Mul(v, graph.MulScalar(graph.Mul(node, graph.Square(node)), -0.5))}
}

// realVJP projects v back to a complex number -- the gradient on the imaginary
// is zero.
func realVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	g := node.Graph()
	return []*graph.Node{graph.Complex(v, graph.ScalarZero(g, v.DType()))}
}

// imagVJP projects v back to a complex number -- the gradient on the real part
// is zero.
func imagVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	g := node.Graph()
	return []*graph.Node{graph.Complex(graph.ScalarZero(g, v.DType()), v)}
}

// conjVJP takes the conjugate of the output's adjoint.
func conjVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	return []*graph.Node{graph.Conj(v)}
}

// complexVJP splits the output's adjoint into its components.
func complexVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	return []*graph.Node{graph.Real(v), graph.Imag(v)}
}

func addVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputsVJPs := make([]*graph.Node, len(node.Inputs()))
	for ii, input := range node.Inputs() {
		inputsVJPs[ii] = vjpForDefaultBroadcast(node, input, v)
	}
	return inputsVJPs
}

func subVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	return []*graph.Node{
		vjpForDefaultBroadcast(node, node.Inputs()[0], v),
		graph.Neg(vjpForDefaultBroadcast(node, node.Inputs()[1], v)),
	}
}

func whereVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	condition := node.Inputs()[0]
	zeros := graph.ZerosLike(v)
	ifTrueVJP := graph.Where(condition, v, zeros)
	ifFalseVJP := graph.Where(condition, zeros, v)
	return []*graph.Node{
		nil, // No gradient wrt condition.
		vjpForDefaultBroadcast(node, node.Inputs()[1], ifTrueVJP),
		vjpForDefaultBroadcast(node, node.Inputs()[2], ifFalseVJP),
	}
}

// VJP formulation for Mul (without consideration to broadcasting):
// F(a,b) = a*b ->  v*dF/da = v*b ; v*dG/db = v*a
func mulVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputsVJPs := make([]*graph.Node, 2)
	broadcastInputs := make([]*graph.Node, 2)
	for ii := range 2 {
		broadcastInputs[ii] = node.Inputs()[ii]
		if !broadcastInputs[ii].Shape().Equal(node.Shape()) {
			broadcastInputs[ii] = graph.BroadcastLike(broadcastInputs[ii], node)
		}
	}
	for ii := range 2 {
		vMul := graph.Mul(v, broadcastInputs[1-ii])
		inputsVJPs[ii] = vjpForDefaultBroadcast(node, node.Inputs()[ii], vMul)
	}
	return inputsVJPs
}

// VJP formulation for Div (without consideration to broadcasting):
// F(a,b) = a/b ->  v*dF/da = v/b ; v*dF/db = -v*a/b^2
func divVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputsVJPs := make([]*graph.Node, 2)
	broadcastInputs := make([]*graph.Node, 2)
	for ii := range 2 {
		broadcastInputs[ii] = node.Inputs()[ii]
		if !broadcastInputs[ii].Shape().Equal(node.Shape()) {
			broadcastInputs[ii] = graph.BroadcastLike(broadcastInputs[ii], node)
		}
	}
	a := broadcastInputs[0]
	b := broadcastInputs[1]
	inputsVJPs[0] = vjpForDefaultBroadcast(node, node.Inputs()[0], graph.Div(v, b))
	inputsVJPs[1] = vjpForDefaultBroadcast(node, node.Inputs()[1],
		graph.Neg(graph.Mul(v, graph.Div(a, graph.Mul(b, b))))) // -v*a/b^2
	return inputsVJPs
}

// VJP formulation for Pow (without consideration to broadcasting):
// F(a,b) = pow(a,b) = a^b ->  v*dF/da = v*b*a^(b-1) ; v*dF/db = v*log(a)*a^b
//
// Notice this will break (NaN) if "a" is negative.
// TODO: handle the negative case for complex numbers, if one wants that.
func powVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	broadcastInputs := make([]*graph.Node, 2)
	for ii := range 2 {
		broadcastInputs[ii] = node.Inputs()[ii]
		if !broadcastInputs[ii].Shape().Equal(node.Shape()) {
			broadcastInputs[ii] = graph.BroadcastLike(broadcastInputs[ii], node)
		}
	}
	a, b := broadcastInputs[0], broadcastInputs[1]
	powAB := node
	// v*dF/da = v*b*a^(b-1)
	dA := graph.Mul(v, graph.Mul(b, graph.Pow(a, graph.AddScalar(b, -1))))
	// v*dF/db = v*log(a)*a^b
	dB := graph.Mul(v, graph.Mul(graph.Log(a), powAB))

	return []*graph.Node{
		vjpForDefaultBroadcast(node, node.Inputs()[0], dA),
		vjpForDefaultBroadcast(node, node.Inputs()[1], dB),
	}
}

func minMaxVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	// We push the adjoint gradient to one side or the other, depending on which is the max.
	// Notice because NonNegativeIndicator(0) == 1, the gradient of Max(x, y) w.r.t. x, where x == y will be 1.0
	// (as opposed to 0).
	side0Indicator := graph.NonNegativeIndicator(graph.Sub(node.Inputs()[0], node.Inputs()[1]))
	side1Indicator := graph.OneMinus(side0Indicator)
	if node.Type() == graphimpl.NodeTypeMin {
		// If min, swap directions.
		side0Indicator, side1Indicator = side1Indicator, side0Indicator
	}
	return []*graph.Node{
		vjpForDefaultBroadcast(node, node.Inputs()[0], graph.Mul(v, side0Indicator)),
		vjpForDefaultBroadcast(node, node.Inputs()[1], graph.Mul(v, side1Indicator)),
	}
}

func reduceSumVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	// Reconstruct exactly the reduced dimensions.
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsReduceSum)
	x := graphbridge.Node(params.X)
	reducedDims := params.Axes
	if len(reducedDims) == 0 {
		// Reduced all dims, reconstruct those.
		reducedDims = xslices.Iota(0, x.Rank())
	}

	// Expand rank of v to match the input, by re-creating
	// the reduced dimensions with size 1.
	newShape := x.Shape().Clone()
	for _, dim := range reducedDims {
		newShape.Dimensions[dim] = 1
	}
	expandedV := graph.ReshapeWithShape(v, newShape)

	// Now all we need it to broadcast the v on the reduced dimensions.
	vjp := graph.BroadcastLike(expandedV, x)
	return []*graph.Node{vjp}
}

func cumMaxVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsCumMax)
	x := graphbridge.Node(inputs.Operand)
	axis := inputs.Axis
	options := inputs.Options
	g := x.Graph()

	dim := x.Shape().Dimensions[axis]
	if dim == 0 || (dim == 1 && options.Exclusive) {
		return []*graph.Node{graph.ZerosLike(x)}
	}

	cur := x
	grad := v
	if options.Reverse {
		cur = graph.Reverse(cur, axis)
		grad = graph.Reverse(grad, axis)
	}

	if options.Exclusive {
		padAxes := make([]compute.PadAxis, cur.Rank())
		padAxes[axis] = compute.PadAxis{End: 1}
		grad = graph.Pad(graph.SliceAxis(grad, axis, graph.AxisRange(1, dim)), graph.ScalarZero(g, grad.DType()), padAxes...)
	}

	windowSizes := xslices.SliceWithValue(cur.Rank(), 1)
	windowSizes[axis] = dim
	strides := xslices.SliceWithValue(cur.Rank(), 1)
	paddings := make([][2]int, cur.Rank())
	paddings[axis][0] = dim - 1

	gradX := checkedSelectAndScatter(cur, grad, compute.ReduceOpMax, windowSizes, strides, paddings)
	if options.Reverse {
		gradX = graph.Reverse(gradX, axis)
	}
	return []*graph.Node{gradX}
}

func cumSumVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsCumSum)
	return []*graph.Node{graph.CumSum(v, inputs.Axis, &graph.CumSumOptions{
		Exclusive: inputs.Options.Exclusive,
		Reverse:   !inputs.Options.Reverse,
	})}
}

func padVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsPad)
	x := graphbridge.Node(inputs.X)
	fillValue := graphbridge.Node(inputs.FillValue)
	g := node.Graph()
	rank := x.Rank()

	hasNegativeEdge := false
	hasInterior := false
	for i := range rank {
		var padAxis compute.PadAxis
		if i < len(inputs.AxesConfig) {
			padAxis = inputs.AxesConfig[i]
		}
		if padAxis.Start < 0 || padAxis.End < 0 {
			hasNegativeEdge = true
		}
		if padAxis.Interior > 0 && x.Shape().Dimensions[i] > 1 {
			hasInterior = true
		}
	}

	var gradX *graph.Node
	if !hasNegativeEdge {
		starts := make([]int, rank)
		limits := make([]int, rank)
		strides := make([]int, rank)
		for i := range rank {
			var padAxis compute.PadAxis
			if i < len(inputs.AxesConfig) {
				padAxis = inputs.AxesConfig[i]
			}
			starts[i] = padAxis.Start
			limits[i] = padAxis.Start + (x.Shape().Dimensions[i]-1)*(1+padAxis.Interior) + 1
			strides[i] = 1 + padAxis.Interior
		}
		gradX = graphbridge.Node(graphimpl.BackendSlice(graphbridge.NodeImpl(v), starts, limits, strides))
	} else {
		// Undo Start/End padding/cropping by padding with negated Start/End and zero fill.
		invEdgeConfig := make([]compute.PadAxis, rank)
		for i := range rank {
			if i < len(inputs.AxesConfig) {
				invEdgeConfig[i] = compute.PadAxis{
					Start: -inputs.AxesConfig[i].Start,
					End:   -inputs.AxesConfig[i].End,
				}
			}
		}
		gradX = graph.Pad(v, graph.ScalarZero(g, x.DType()), invEdgeConfig...)
		if hasInterior {
			starts := make([]int, rank)
			limits := make([]int, rank)
			strides := make([]int, rank)
			for i := range rank {
				stride := 1
				if i < len(inputs.AxesConfig) && x.Shape().Dimensions[i] > 1 {
					stride += inputs.AxesConfig[i].Interior
				}
				limits[i] = gradX.Shape().Dimensions[i]
				strides[i] = stride
			}
			gradX = graphbridge.Node(graphimpl.BackendSlice(graphbridge.NodeImpl(gradX), starts, limits, strides))
		}
	}
	gradFill := graph.ZerosLike(fillValue)
	return []*graph.Node{gradX, gradFill}
}

func dynamicPadVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsDynamicPad)
	x := graphbridge.Node(inputs.X)
	fillValue := graphbridge.Node(inputs.FillValue)
	g := node.Graph()
	invSpecs := make([]graph.DynamicPadAxisSpec, len(inputs.AxesConfig))
	valIdx := 2
	for i, padAxis := range inputs.AxesConfig {
		if padAxis.InteriorValue != nil || padAxis.Interior != 0 {
			Panicf("dynamicPadVJP: gradient for interior padding not supported")
		}
		if padAxis.StartValue != nil {
			invSpecs[i].StartNode = graph.Neg(node.Inputs()[valIdx])
			valIdx++
		} else {
			invSpecs[i].Start = -padAxis.Start
		}
		if padAxis.EndValue != nil {
			invSpecs[i].EndNode = graph.Neg(node.Inputs()[valIdx])
			valIdx++
		} else {
			invSpecs[i].End = -padAxis.End
		}
	}
	gradX := graph.DynamicPad(v, graph.ScalarZero(g, x.DType()), invSpecs...)
	if !gradX.Shape().Equal(x.Shape()) {
		gradX = graph.DynamicReshapeLike(gradX, x)
	}
	gradFill := graph.ZerosLike(fillValue)
	vjps := make([]*graph.Node, len(node.Inputs()))
	vjps[0] = gradX
	vjps[1] = gradFill
	return vjps
}

func reverseVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	inputs := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsReverse)
	return []*graph.Node{graph.Reverse(v, inputs.Axes...)}
}

func reduceMaxVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	// Reconstruct exactly the reduced dimensions, and build a newShape
	// (same shape as if we had done ReduceAndKeep(input[0]))
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsReduceMax)
	x := graphbridge.Node(params.X)
	reducedDims := params.Axes
	if len(reducedDims) == 0 {
		// Reduced all dims, reconstruct those.
		reducedDims = xslices.Iota(0, x.Rank())
	}
	newShape := x.Shape().Clone()
	for _, dim := range reducedDims {
		newShape.Dimensions[dim] = 1
	}

	// Expand the node output (with max) to match the input. And then creates
	// an indicator to which positions are at the max values.
	maxAtOriginalRank := graph.ReshapeWithShape(node, newShape)
	maxIndicatorAtInput := graph.NonNegativeIndicator(graph.Sub(x, maxAtOriginalRank))

	// Expand rank of v to match the input, by re-creating
	// the reduced dimensions with size 1 and then broadcasting.
	expandedV := graph.ReshapeWithShape(v, newShape)
	expandedV = graph.BroadcastLike(expandedV, x)

	// vjp is only propagated to the elements at the max value.
	vjp := graph.Mul(expandedV, maxIndicatorAtInput)
	return []*graph.Node{vjp}
}

func reduceMinVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	// Reconstruct exactly the reduced dimensions, and build a newShape
	// (same shape as if we had done ReduceAndKeep(input[0]))
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsReduceMin)
	x := graphbridge.Node(params.X)
	reducedDims := params.Axes
	if len(reducedDims) == 0 {
		// Reduced all dims, reconstruct those.
		reducedDims = xslices.Iota(0, x.Rank())
	}
	newShape := x.Shape().Clone()
	for _, dim := range reducedDims {
		newShape.Dimensions[dim] = 1
	}

	// Expand the node output (with min) to match the input. And then creates
	// an indicator to which positions are at the min values.
	minAtOriginalRank := graph.ReshapeWithShape(node, newShape)
	minIndicatorAtInput := graph.NonPositiveIndicator(graph.Sub(x, minAtOriginalRank))

	// Expand rank of v to match the input, by re-creating
	// the reduced dimensions with size 1 and then broadcasting.
	expandedV := graph.ReshapeWithShape(v, newShape)
	expandedV = graph.BroadcastLike(expandedV, x)

	// vjp is only propagated to the elements at the min value.
	vjp := graph.Mul(expandedV, minIndicatorAtInput)
	return []*graph.Node{vjp}
}

func reshapeVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	// ReshapeWithShape back to its inputNodes shape.
	return []*graph.Node{graph.ReshapeWithShape(v, node.Inputs()[0].Shape())}
}

func logisticVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	g := node.Graph()
	// d\sigma(x)/dx = sigma(x) * (1 - sigma(x)
	// See https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
	grad := graph.Mul(node, graph.Sub(graph.ScalarOne(g, node.Shape().DType), node))
	vjp := graph.Mul(v, grad)
	return []*graph.Node{vjp}
}

// sliceVJP generates the adjoint gradient term for SliceNode, encoded by the
// SliceWithStridesXLA function.
func sliceVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	g := node.Graph()
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsSlice)
	x := graphbridge.Node(params.X)
	rank := x.Rank()
	dimensions := x.Shape().Dimensions

	// The incoming adjoint v must be applied only where the slice came from, and
	// the new adjoint will have zero, filled with padding (`Pad()`) elsewhere.
	//
	// Here we build the PadAxis configuration for each axis.
	padding := make([]graph.PadAxis, rank)
	for ii := range padding {
		padding[ii].Start = params.Starts[ii]
		if params.Strides[ii] <= 1 {
			padding[ii].End = dimensions[ii] - params.Limits[ii]
		} else {
			// Padding at the end may be affected by the strides.
			padding[ii].Interior = params.Strides[ii] - 1
			dimToPad := dimensions[ii] - params.Starts[ii]
			dimToPad -= (v.Shape().Dimensions[ii]-1)*params.Strides[ii] + 1
			padding[ii].End = dimToPad // What is missing to make v the same shape as x at axis ii.
		}
	}
	return []*graph.Node{
		graph.Pad(v, graph.ScalarZero(g, x.DType()), padding...),
	}
}

// gatherVJP generates the adjoint gradient term for  GatherXL node. It works only for simple gather operations,
// in particular it works with graph.Gather.
func gatherVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	// TODO: check that values are compatible with Gather() and return an error if not.
	input := node.Inputs()[0]
	indices := node.Inputs()[1]
	inputShape := input.Shape()
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsGather)

	// Gather() case: params.SliceSizes is 1 for the first dimensions, and full in the last. Plus the initial
	// dimensions are all collapsed.
	if len(params.SliceSizes) != inputShape.Rank() {
		Panicf("gradient from Gather with len(params.SliceSizes) != input.Rank() not defined -- params.SliceSizes=%v, input.Shape()=%s",
			params.SliceSizes, inputShape)
	}
	isSimpleGather := true // Whether all indexed axes are at the start, and slices are the full dimension.
	for axis, inputSize := range inputShape.Dimensions {
		if axis < len(params.CollapsedSliceAxes) {
			if params.CollapsedSliceAxes[axis] != axis {
				isSimpleGather = false
				break
			}
			if params.SliceSizes[axis] != 1 {
				isSimpleGather = false
				break
			}
		} else {
			// For non-collapsed axes, we expect the slice to be the full dimension.
			if params.SliceSizes[axis] != inputSize {
				isSimpleGather = false
				break
			}
		}
	}
	if isSimpleGather {
		return []*graph.Node{
			graph.ScatterSum(graph.Zeros(node.Graph(), inputShape), indices, v, params.IndicesAreSorted, false),
			nil, // No gradients for indices.
		}
	}

	// GatherSlices(): params.SliceSizes are variable, but there are no paramsCollapsedSliceDims.
	//fmt.Printf("\tgatherVJP: operand=%s, start=%s, paramsIndexVectorAxis=%d, offsetDims=%v, paramsCollapsedSliceDims=%v, params.StartIndexMap=%v, params.SliceSizes=%v\n",
	//	input.Shape(), indices.Shape(), paramsIndexVectorAxis, offsetDims, paramsCollapsedSliceDims, params.StartIndexMap, params.SliceSizes)
	isGatherSlices := len(params.CollapsedSliceAxes) == 0
	if !isGatherSlices {
		Panicf("xlaGather operation for which no gradient was defined. Please use only Gather() or GatherSlices().")
	}

	// Find scatterXLA parameters to reverse the GatherSlice:
	operand := graph.ZerosLike(input)
	startIndices := indices
	outputPrefixRank := startIndices.Rank() - 1 // Prefix dimensions of the output of the GatherSlice.
	updates := v
	//fmt.Printf("\tgatherVJP: updates=%s\n", updates.Shape())

	// updateWindowsDims: one per every dimension of the input, offset by the initial outputPrefixRank.
	updateWindowsDims := make([]int, 0, inputShape.Rank())
	for ii := 0; ii < inputShape.Rank(); ii++ {
		updateWindowsDims = append(updateWindowsDims, ii+outputPrefixRank)
	}
	var insertedWindowDims []int                     // Empty, since the original GatherSlice don's have any paramsCollapsedSliceDims.
	scatterDimsToOperandDims := params.StartIndexMap // Same map used in GatherSlice.
	// We don't make any assumptions on uniqueness or sortedness of the indices. Likely the slices will overlap.
	return []*graph.Node{
		graphbridge.Node(graphimpl.BackendScatterSum(graphbridge.NodeImpl(operand), graphbridge.NodeImpl(startIndices), graphbridge.NodeImpl(updates), params.IndexVectorAxis, updateWindowsDims, insertedWindowDims, scatterDimsToOperandDims,
			params.IndicesAreSorted, false)),
		nil, // No gradients for indices.
	}
}

// batchNormForTrainingVJP generates the gradient with respect to the operand and the scale and offset
// parameters. It uses the XLA implemented gradient.
func batchNormForTrainingVJP(node *graph.Node, vjps []*graph.Node, _ shapes.Shape) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsBatchNormForTraining)
	splitOutputs := splitNode(node)
	mean, variance := splitOutputs[1], splitOutputs[2]
	gradOutput := vjps[0]
	//fmt.Printf("operand: %s\n", graphbridge.Node(params.Operand))
	//fmt.Printf("scale: %s\n", graphbridge.Node(params.Scale))
	//graphbridge.Node(params.Operand).SetLogged("operand")
	//graphbridge.Node(params.Scale).SetLogged("scale")
	//mean.SetLogged("mean")
	//variance.SetLogged("variance")
	//fmt.Printf("batchNormForTrainingVJP: epsilon=%g\n", params.Epsilon)
	gradOperand, gradScale, gradOffset := graphbridge.BatchNormGradient(
		graphbridge.Node(params.Operand), graphbridge.Node(params.Scale), mean, variance, gradOutput, params.Epsilon, params.FeatureAxis)
	return []*graph.Node{gradOperand, gradScale, gradOffset}
}

// transposeVJP generates the "vector dot jacobian" w.r.t. the input of transpose. It's
// simply the transpose of the incoming vector.
func transposeVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsTranspose)
	reversePermutation := make([]int, len(params.Permutation))
	for to, from := range params.Permutation {
		reversePermutation[from] = to
	}
	vjp := graph.TransposeAllAxes(v, reversePermutation...)
	return []*graph.Node{vjp}
}

// broadcastInDimVJP generates the "vector dot jacobian" w.r.t. the input of BroadcastInDim.
// One just needs to reduce the broadcast dimensions.
func broadcastInDimVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsBroadcastInDim)
	x := graphbridge.Node(params.Operand)
	shape := params.OutputShape

	if x.Rank() != len(params.BroadcastAxes) {
		Panicf("there must be a broadcastAxes for each axis in x, instead got x.Shape()=%s and broadcastAxes=%v",
			x.Shape(), params.BroadcastAxes)
	}

	axesPreserved := make([]bool, shape.Rank())
	for inputAxis, outputAxis := range params.BroadcastAxes {
		if x.Shape().Dimensions[inputAxis] == shape.Dimensions[outputAxis] {
			axesPreserved[outputAxis] = true
		} else {
			if x.Shape().Dimensions[inputAxis] != 1 {
				Panicf("unexpected broadcast from shape %s to shape %s at axis %d -- don't know how to calculate gradient",
					x.Shape(), shape, inputAxis)
			}
		}
	}
	axesToReduce := make([]int, 0, shape.Rank())
	for axis, preserved := range axesPreserved {
		if !preserved {
			axesToReduce = append(axesToReduce, axis)
		}
	}
	gradWrtX := v
	if len(axesToReduce) > 0 {
		gradWrtX = graph.ReduceSum(v, axesToReduce...)
	}
	if gradWrtX.Rank() != x.Rank() {
		// X had some axes of dimension 1 that were reduced, we simply reshape it here.
		gradWrtX = graph.Reshape(gradWrtX, x.Shape().Dimensions...)
	}
	return []*graph.Node{gradWrtX}
}

// dynamicSliceVJP generates the "vector dot jacobian" w.r.t. the input of DynamicSlice.
func dynamicSliceVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsDynamicSlice)
	vjpOperand := graph.ZerosLike(graphbridge.Node(params.Operand))
	vjpOperand = graph.DynamicUpdateSlice(vjpOperand, v, graphbridge.Nodes(params.StartIndices))

	vjps := make([]*graph.Node, len(node.Inputs()))
	vjps[0] = vjpOperand // No gradients wrt. the startIndices, only wrt. the operand.
	return vjps
}

// dynamicUpdateSliceVJP generates the "vector dot jacobian" w.r.t. the input of DynamicUpdateSlice.
func dynamicUpdateSliceVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsDynamicUpdateSlice)
	vjpOperand := v
	vjpOperand = graph.DynamicUpdateSlice(vjpOperand, graph.ZerosLike(graphbridge.Node(params.Update)), graphbridge.Nodes(params.StartIndices))
	vjpUpdate := graph.DynamicSlice(v, graphbridge.Nodes(params.StartIndices), graphbridge.Node(params.Update).Shape().Dimensions)

	vjps := make([]*graph.Node, len(node.Inputs()))
	// No gradients wrt. the startIndices, only wrt. the operand and the update.
	vjps[0] = vjpOperand
	vjps[1] = vjpUpdate
	return vjps
}

// vjpOptimizationBarrier generates the "vector dot jacobian" w.r.t. the inputs of OptimizationBarrier.
// Since OptimizationBarrier is mathematically an identity function for each of its inputs,
// its gradient is just the identity function of the output gradients.
func vjpOptimizationBarrier(node *graph.Node, vjpOutputs []*graph.Node, _ shapes.Shape) []*graph.Node {
	return vjpOutputs
}

// vjpSchedulingBarrier generates the "vector dot jacobian" w.r.t. the inputs of SchedulingBarrier.
// Since SchedulingBarrier is mathematically an identity function of the first operand,
// its gradient is just the incoming gradient for the operand, and nil (zero) for the dependencies.
func vjpSchedulingBarrier(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	vjps := make([]*graph.Node, len(node.Inputs()))
	vjps[0] = v
	return vjps
}

// dynamicBroadcastInDimVJP generates the "vector dot jacobian" w.r.t. the input of DynamicBroadcastInDim.
func dynamicBroadcastInDimVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	params := graphbridge.NodeImpl(node).NodeInputs().(*graphimpl.NodeInputsDynamicBroadcastInDim)
	operand := graphbridge.Node(params.Operand)
	outShape := node.Shape()

	if operand.Rank() != len(params.BroadcastAxes) {
		Panicf("there must be a broadcastAxes for each axis in operand, instead got operand.Shape()=%s and broadcastAxes=%v",
			operand.Shape(), params.BroadcastAxes)
	}

	// Classify output axes:
	//  - Unmapped output axes (newly inserted) and static-1 operand axes (inDim == 1 && outDim != 1) are always reduced.
	//  - Dynamic operand axes mapped to a static target axis or a different dynamic axis name may either broadcast (when
	//    concrete inDim == 1) or match (when concrete inDim == outDim) at runtime, so their reduction is conditional.
	mappedInputAxis := make([]int, outShape.Rank())
	for i := range mappedInputAxis {
		mappedInputAxis[i] = -1
	}
	for inputAxis, outputAxis := range params.BroadcastAxes {
		mappedInputAxis[outputAxis] = inputAxis
	}

	alwaysReduceOutputAxes := make([]int, 0, outShape.Rank())
	var reinsertStatic1InputAxes []int
	var runtimeMaybeBcastInputAxes []int

	for outputAxis := range outShape.Rank() {
		inputAxis := mappedInputAxis[outputAxis]
		if inputAxis < 0 {
			alwaysReduceOutputAxes = append(alwaysReduceOutputAxes, outputAxis)
			continue
		}
		inDim := operand.Shape().Dimensions[inputAxis]
		outDim := outShape.Dimensions[outputAxis]
		switch {
		case inDim == 1 && outDim != 1:
			alwaysReduceOutputAxes = append(alwaysReduceOutputAxes, outputAxis)
			reinsertStatic1InputAxes = append(reinsertStatic1InputAxes, inputAxis)
		case inDim == shapes.DynamicDim && (outDim != shapes.DynamicDim || operand.Shape().AxisName(inputAxis) != outShape.AxisName(outputAxis)):
			runtimeMaybeBcastInputAxes = append(runtimeMaybeBcastInputAxes, inputAxis)
		}
	}

	gradWrtOperand := v
	if len(alwaysReduceOutputAxes) > 0 {
		gradWrtOperand = graph.ReduceSum(v, alwaysReduceOutputAxes...)
	}

	if len(runtimeMaybeBcastInputAxes) == 0 {
		if !gradWrtOperand.Shape().Equal(operand.Shape()) {
			gradWrtOperand = graph.DynamicReshapeLike(gradWrtOperand, operand)
		}
		vjps := make([]*graph.Node, len(node.Inputs()))
		vjps[0] = gradWrtOperand
		return vjps
	}

	// Restore rank to operand.Rank() by re-inserting size-1 axes at static-1 broadcast positions.
	if operand.Rank() == 1 && gradWrtOperand.IsScalar() {
		gradWrtOperand = graph.Reshape(gradWrtOperand, 1)
	} else if len(reinsertStatic1InputAxes) > 0 {
		gradWrtOperand = graph.ExpandAxes(gradWrtOperand, reinsertStatic1InputAxes...)
	}

	g := operand.Graph()
	caps := g.Backend().Capabilities()
	supportsIf := caps.Functions && caps.Operations[compute.OpTypeIf]

	for _, ax := range runtimeMaybeBcastInputAxes {
		curGrad := gradWrtOperand
		inSize := graph.DimensionSize(operand, ax)
		one := graph.ScalarOne(g, inSize.DType())
		targetSpecs := graph.DimensionSpecsFor(curGrad)
		targetSpecs[ax] = graph.DimensionSpecFor(operand, ax)

		if supportsIf {
			pred := graph.Equal(inSize, one)
			trueFn := graph.NewClosure(g, func(g *graph.Graph) []*graph.Node {
				red := graph.ReduceSum(curGrad, ax)
				if curGrad.Rank() > 1 {
					red = graph.ExpandAxes(red, ax)
				} else {
					red = graph.Reshape(red, 1)
				}
				return []*graph.Node{graph.DynamicReshape(red, targetSpecs...)}
			})
			falseFn := graph.NewClosure(g, func(g *graph.Graph) []*graph.Node {
				return []*graph.Node{graph.DynamicReshape(curGrad, targetSpecs...)}
			})
			gradWrtOperand = graph.If(pred, trueFn, falseFn)[0]
		} else {
			// Closure-free fallback (e.g. ONNX backend): scatter-sum along axis ax with indices
			// clamped to max(inSize - 1, 0), which sums all slices into index 0 when inSize == 1
			// and maps index j -> j when inSize == outDim (and leaves zeros when outDim == 0).
			work := curGrad
			permSpecs := slices.Clone(targetSpecs)
			if ax != 0 {
				work = graph.Transpose(work, ax, 0)
				permSpecs[0], permSpecs[ax] = permSpecs[ax], permSpecs[0]
			}
			zeros := graph.DynamicBroadcastInDim(graph.ScalarZero(g, work.DType()), nil, permSpecs...)
			iotaIdx := graph.DynamicIota(g, inSize.DType(), 0, graph.DimensionSpecFor(work, 0))
			maxIdx := graph.Max(graph.Sub(inSize, one), graph.ScalarZero(g, inSize.DType()))
			clampedIdx := graph.ExpandAxes(graph.Min(iotaIdx, maxIdx), -1)
			scattered := graph.ScatterSum(zeros, clampedIdx, work, false, false)
			if ax != 0 {
				scattered = graph.Transpose(scattered, ax, 0)
			}
			gradWrtOperand = scattered
		}
	}

	if !gradWrtOperand.Shape().Equal(operand.Shape()) {
		gradWrtOperand = graph.DynamicReshapeLike(gradWrtOperand, operand)
	}
	vjps := make([]*graph.Node, len(node.Inputs()))
	vjps[0] = gradWrtOperand
	return vjps
}

// dynamicReshapeVJP generates the "vector dot jacobian" w.r.t. the input of DynamicReshape.
func dynamicReshapeVJP(node, v *graph.Node, _ shapes.Shape) []*graph.Node {
	operand := node.Inputs()[0]
	vjp := graph.DynamicReshapeLike(v, operand)
	vjps := make([]*graph.Node, len(node.Inputs()))
	vjps[0] = vjp
	return vjps
}
