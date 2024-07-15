/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package graph

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/xla"
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
	Graph *Graph
	Root  *Node // This is the output node

	ReverseNodes []*reverseNode
	NumConsumers []int
}

type reverseNode struct {
	Node *Node

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
	AccumulatedVJP *Node

	// VJPsForTuple holds the individual VJPs for a tuple: they are only collapsed to a VJP at the end.
	VJPsForTuple []*Node
}

// The jacobian
func combineOutputShape(outputShape, inputShape shapes.Shape) shapes.Shape {
	outputShape.DType = inputShape.DType // Make sure they are the same.
	return shapes.ConcatenateDimensions(outputShape, inputShape)
}

// Gradient creates new nodes for the gradients of the output with respect to each node in gradientNodes.
// The output must be a scalar -- otherwise this would be called Jacobian.
// TODO: Define a Jacobian.
func Gradient(output *Node, gradientNodes ...*Node) []*Node {
	allInputNodes := make([]*Node, 0, len(gradientNodes)+1)
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
	rOutput.AccumulatedVJP = Ones(output.graph, shapes.Make(outputShape.DType))

	// Whether we need the gradient for the node.
	needGradientForNode := func(node *Node) bool {
		if node.stopGradient {
			return false
		}
		rNode := rg.ReverseNodes[node.Id()]
		return rNode.Included && rNode.Useful
	}

	// Loop from final node backwards, back propagating the gradients. Notice that the nodes are ordered according to
	// the DAG, meaning that by the time g.nodes[ii] is reached, all nodes consuming its outputs will already have been
	// accounted for, and their VJPs summed up.
	for nodeIdx := output.Id(); nodeIdx >= 0; nodeIdx -= 1 {
		node := g.nodes[nodeIdx]
		rNode := rg.ReverseNodes[nodeIdx]

		// No need to propagate VJP if either node is not of interest, if there is a stop gradient, or
		// if its inputNodes are not of interest.
		if !needGradientForNode(node) {
			continue
		}
		needInputs := false
		for _, input := range node.Inputs() {
			if needGradientForNode(input) {
				needInputs = true
				break
			}
		}
		if !needInputs {
			continue
		}

		// Special case for tuples: their vjp need to be aggregated differently.
		if node.shape.IsTuple() {
			for ii, shape := range node.shape.TupleShapes {
				if rNode.VJPsForTuple[ii] == nil {
					rNode.VJPsForTuple[ii] = Zeros(node.Graph(), shape)
				}
			}
			rNode.AccumulatedVJP = Tuple(rNode.VJPsForTuple...)
		}
		if node.Type() == xla.GetTupleElementNode {
			// GetTupleElement just pushes v to the specific element of the tuple.
			elementIdx := node.serializedNode.Int
			input := node.inputNodes[0]
			rInput := rg.ReverseNodes[input.Id()]
			if rInput.VJPsForTuple[elementIdx] == nil {
				rInput.VJPsForTuple[elementIdx] = rNode.AccumulatedVJP
			} else {
				rInput.VJPsForTuple[elementIdx] = Add(rInput.VJPsForTuple[elementIdx], rNode.AccumulatedVJP)
			}
			continue
		}

		if rNode.AccumulatedVJP == nil {
			// No gradients arriving to rNode -- e.g.: there was a `stopGradient`. So we can't propagate the gradient
			// either.
			continue
		}

		vjpFn := node.customVJP
		if vjpFn == nil {
			var ok bool
			vjpFn, ok = VJPRegistration[node.Type()]
			if !ok {
				Panicf("graph has node %s, for which no gradient is defined yet, cannot generate graph gradient", node)
			}
		}
		inputsVJPs := vjpFn(node, rNode.AccumulatedVJP, outputShape)
		if len(inputsVJPs) != len(node.Inputs()) {
			Panicf("AccumulatedVJP(%s) returned %d VJPs, but it has %d inputNodes, implementation of auto-differentiation for node failed",
				node, len(inputsVJPs), len(node.Inputs()))
		}
		//fmt.Printf("\tFrom node %s\n", node)
		for ii, input := range node.Inputs() {
			vjp := inputsVJPs[ii]
			if vjp == nil {
				// Skip this vjp, input is assumed to be static.
				continue
			}
			//fmt.Printf("\t\tSetting vjp for %s: %s\n", input, vjp)
			combinedShape := combineOutputShape(outputShape, input.shape)
			if !vjp.shape.Eq(combinedShape) {
				Panicf("invalid Gradient calculation for node %q: invalid shape for calculated AccumulatedVJP for "+
					"input #%d (out of %d): input shape=%s, calculated AccumulatedVJP shape=%s (wanted %s)"+
					" -- this probably indicates a bug in the code, please report the issue.",
					node, ii, len(node.Inputs()), input.shape, vjp.shape, combinedShape)
			}
			rInput := rg.ReverseNodes[input.Id()]
			if rInput.AccumulatedVJP == nil {
				rInput.AccumulatedVJP = vjp
			} else {
				rInput.AccumulatedVJP = Add(rInput.AccumulatedVJP, vjp)
			}
		}
	}

	gradients := make([]*Node, len(gradientNodes))
	for ii, node := range gradientNodes {
		rNode := rg.ReverseNodes[node.Id()]
		if rNode.AccumulatedVJP == nil {
			// If there is no path from the output to the gradient node (possibly because of
			// a StopGradient) return zero.
			// TODO: fix the shape if the output wrt which we are calculating the gradient is not a scalar.
			gradients[ii] = ZerosLike(node)

		} else {
			gradients[ii] = rNode.AccumulatedVJP
		}
	}
	return gradients
}

func newReverseGraph(g *Graph, root *Node, gradientNodes []*Node) *reverseGraph {
	numNodes := len(g.nodes)
	if root == nil {
		root = g.nodes[numNodes-1]
	}
	rg := &reverseGraph{
		Graph:        g,
		Root:         root,
		ReverseNodes: make([]*reverseNode, numNodes),
		NumConsumers: make([]int, numNodes),
	}

	// Stitch reverse "consumer" links to graph.
	for ii, node := range g.nodes {
		rNode := &reverseNode{Node: node}
		rg.ReverseNodes[ii] = rNode
		if node.shape.IsTuple() {
			rNode.VJPsForTuple = make([]*Node, node.shape.TupleSize())
		}
		for _, input := range node.inputNodes {
			rg.NumConsumers[input.Id()] += 1
		}
	}
	for ii, node := range g.nodes {
		rNode := rg.ReverseNodes[ii]
		rNode.Consumers = make([]*reverseNode, 0, rg.NumConsumers[ii])
		for _, input := range node.inputNodes {
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
func recursivePathFromRoot(rg *reverseGraph, node *Node) {
	rNode := rg.ReverseNodes[node.Id()]
	if rNode.Included {
		// Already visited.
		return
	}
	rNode.Included = true
	for _, input := range node.inputNodes {
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
//	`node`: node for which we are calculating the backward gradient. An important part of `node` are its inputNodes
//	   given by `node.Inputs()`. The VJP function must one gradient per input.
//	`v`: gradient (or jacobian) of what we care about with the respect to the output of `node`, also known as the
//	   adjoint. VJP needs to calculate the next adjoint (`v`) but with respect to the inputNodes of `node` instead.
//	`outputShape`: for now fixed as scalar, and can be ignored. When we add support for Jacobian this will hold
//	   the shape of the thing we are calculating the Jacobian for.
//
// Returns:
//
//	The adjoint (the updated `v`) to each of `node` inputNodes. That is, the gradient of the loss (typically, but of
//	whatever we are calculating the gradient of) with respect to each of the `node` inputNodes.
type VJP func(node, v *Node, outputShape shapes.Shape) []*Node

// VJPRegistration maps each node type to its implementation of VJP. If implementing a new op, or
// for experimentation, one can dynamically change this.
//
// Notice xla.GetTupleElementNode is specialized inside the main reverse autodiff code, and is not
// in the table here.
var VJPRegistration = map[xla.NodeType]VJP{
	xla.InvalidNode:     noOpVJP, // NoOp
	xla.ConstantNode:    nilVJP,
	xla.ParameterNode:   nilVJP,
	xla.ConvertTypeNode: convertTypeVJP,
	xla.WhereNode:       whereVJP,
	xla.NegNode:         negVJP,
	xla.AbsNode:         absVJP,
	xla.ExpNode:         expVJP,
	xla.LogNode:         logVJP,
	xla.Log1pNode:       log1pVJP,
	xla.TanhNode:        tanhVJP,
	xla.AddNode:         addVJP,
	xla.SubNode:         subVJP,
	xla.MulNode:         mulVJP,
	xla.DivNode:         divVJP,
	xla.SqrtNode:        sqrtVJP,

	// Complex numbers.
	xla.RealNode:    realVJP,
	xla.ImagNode:    imagVJP,
	xla.ConjNode:    conjVJP,
	xla.ComplexNode: complexVJP,

	xla.MaxNode:                minMaxVJP,
	xla.MinNode:                minMaxVJP,
	xla.ReshapeNode:            reshapeVJP,
	xla.ReduceSumNode:          reduceSumVJP,
	xla.ReduceMaxNode:          reduceMaxVJP,
	xla.LogisticNode:           logisticVJP,
	xla.DotNode:                dotVJP,
	xla.DotGeneralNode:         dotGeneralVJP,
	xla.SliceNode:              sliceVJP,
	xla.GatherNode:             gatherVJP,
	xla.ConcatenateNode:        concatenateVJP,
	xla.ConvGeneralDilatedNode: convGeneralDilatedVJP,
	xla.ReduceWindowNode:       reduceWindowVJP,
	xla.BatchNormTrainingNode:  batchNormTrainingVJP,
	xla.TransposeNode:          transposeVJP,
	xla.BroadcastInDimNode:     broadcastInDimVJP,
	xla.FftNode:                fftVJP,
}

// nilVJP returns no gradient, for functions without any inputNodes.
func nilVJP(_, _ *Node, _ shapes.Shape) []*Node {
	return nil
}

// noOpVJP works for anything that has not impact on the gradient, like a No-Op.
func noOpVJP(node, v *Node, _ shapes.Shape) []*Node {
	return []*Node{v}
}

// vjpForDefaultBroadcast returns the VJP of the default broadcasting on operations like Add, Mul, Sub, etc.
// It is a reduce-sum of the broadcast dimensions.
func vjpForDefaultBroadcast(node, input, v *Node) *Node {
	_ = validateBuildingGraphFromInputs(node, input, v)
	if input.shape.Eq(node.shape) {
		// If there was no broadcast involved, VJP is the identity.
		return v
	} else if input.shape.IsScalar() {
		// If there was a broadcast from a scalar, the VJP is the full reduction of the V tensor.
		return ReduceAllSum(v)
	}

	// Reduce-sum on the dimensions it was broadcast during the sum. Search for all dimensions that
	// are 1 in the input, and > 1 in the output.
	var reduceDims []int
	for ii, dim := range input.shape.Dimensions {
		if dim == 1 && v.shape.Dimensions[ii] > 1 {
			reduceDims = append(reduceDims, ii)
		}
	}
	reduced := ReduceSum(v, reduceDims...)
	// Since ReduceSum collapsed those reduced dimensions of now size 1, we need to reshape it back to the
	// original input format.
	var vjp *Node
	err := TryCatch[error](func() {
		vjp = ReshapeWithShape(reduced, input.shape)
	})
	if err != nil {
		err = errors.WithMessagef(err, "AutoGrad: calculating the AccumulatedVJP of a broadcast: v.shape=%s, input.shape=%s", v.Shape(), input.Shape())
		panic(err)
	}
	return vjp
}

// Gradient of the type conversion, just converts back the adjoint dtype to
// its input dtype.
//
// Except if the conversion is from float->complex: it's a valid conversion, but for the back-propagation
// we propagate back only the real part.
func convertTypeVJP(node, v *Node, _ shapes.Shape) []*Node {
	_ = node
	inputDType := node.inputNodes[0].DType()
	if node.DType().IsComplex() && inputDType.IsFloat() {
		// Just take the real part of the adjoint, since the input has no effect on the imaginary output.
		return []*Node{ConvertDType(Real(v), inputDType)}
	}
	return []*Node{ConvertDType(v, inputDType)}
}

func negVJP(node, v *Node, _ shapes.Shape) []*Node {
	_ = node
	return []*Node{Neg(v)}
}

func absVJP(node, v *Node, _ shapes.Shape) []*Node {
	x := node.inputNodes[0]
	if x.DType().IsComplex() {
		// For complex numbers, Abs() is defined as the Euclidean distance in the complex
		// plane from 0.
		//
		// Abs(x_c) = Sqrt(x_re^2 + x_im^2)
		// dAbs(x_c)/dx_re = 1/2 * (x_re^2+x_im^2)^(-1/2) * (2*x_re) = x_re / Abs(x_c)
		// dAbs(x_c)/dx_im = 1/2 * (x_re^2+x_im^2)^(-1/2) * (2*x_im) = x_im / Abs(x_c)
		//
		dxReal := Div(Real(x), node)
		dxImag := Div(Imag(x), node)
		// Multiply by the output adjoint to get the input adjoint.
		dxReal = Mul(dxReal, v)
		dxImag = Mul(dxImag, v)
		// Now there is a sleight of hand here: we are using
		//   dAbs(x_c)/d(x_c) = dAbs(x_c)/dx_re + i.dAbs(x_c)/dx_im ... which is not really true.
		return []*Node{Complex(dxReal, dxImag)}
	}

	// Notice that d(abs(x))/dx at 0 is actually undefined. This will take 1, so it assumes the positive side.
	// This usually has little impact on training, but makes some optimizations where the limits of operations
	// have to behave the same consistent (e.g: layers.BinaryCrossentropyLogitsLoss).
	return []*Node{Mul(v, SignPlusOrMinus(node.inputNodes[0]))}
}

func expVJP(node, v *Node, _ shapes.Shape) []*Node {
	return []*Node{Mul(v, node)}
}

func logVJP(node, v *Node, _ shapes.Shape) []*Node {
	return []*Node{Mul(v, Inverse(node.inputNodes[0]))}
}

func log1pVJP(node, v *Node, _ shapes.Shape) []*Node {
	one := ScalarOne(node.Graph(), node.inputNodes[0].DType())
	return []*Node{Mul(v, Inverse(Add(one, node.inputNodes[0])))}
}

func tanhVJP(node, v *Node, _ shapes.Shape) []*Node {
	tanhX := node // node holds the output of tanh(x)
	return []*Node{Mul(v, OneMinus(Square(tanhX)))}
}

func sqrtVJP(node, v *Node, _ shapes.Shape) []*Node {
	// d(x^0.5)/dx = 0.5 * x^(-0.5) = 0.5/sqrt(x)
	return []*Node{Mul(v, MulScalar(Inverse(node), 0.5))}
}

// realVJP projects v back to a complex number -- the gradient on the imaginary
// is zero.
func realVJP(node, v *Node, _ shapes.Shape) []*Node {
	g := node.Graph()
	return []*Node{Complex(v, ScalarZero(g, v.DType()))}
}

// imagVJP projects v back to a complex number -- the gradient on the real part
// is zero.
func imagVJP(node, v *Node, _ shapes.Shape) []*Node {
	g := node.Graph()
	return []*Node{Complex(ScalarZero(g, v.DType()), v)}
}

// conjVJP takes the conjugate of the output's adjoint.
func conjVJP(node, v *Node, _ shapes.Shape) []*Node {
	return []*Node{Conj(v)}
}

// complexVJP splits the output's adjoint into its components.
func complexVJP(node, v *Node, _ shapes.Shape) []*Node {
	return []*Node{Real(v), Imag(v)}
}

func addVJP(node, v *Node, _ shapes.Shape) []*Node {
	inputsVJPs := make([]*Node, len(node.inputNodes))
	for ii, input := range node.inputNodes {
		inputsVJPs[ii] = vjpForDefaultBroadcast(node, input, v)
	}
	return inputsVJPs
}

func subVJP(node, v *Node, _ shapes.Shape) []*Node {
	return []*Node{
		vjpForDefaultBroadcast(node, node.inputNodes[0], v),
		Neg(vjpForDefaultBroadcast(node, node.inputNodes[1], v)),
	}
}

func whereVJP(node, v *Node, _ shapes.Shape) []*Node {
	condition := node.inputNodes[0]
	zeros := ZerosLike(v)
	return []*Node{
		nil, // No gradient wrt condition.
		Where(condition, v, zeros),
		Where(condition, zeros, v),
	}
}

// VJP formulation for Mul (without consideration to broadcasting):
// F(a,b) = a*b ->  v*dF/da = v*b ; v*dG/db = v*a
func mulVJP(node, v *Node, _ shapes.Shape) []*Node {
	inputsVJPs := make([]*Node, 2)
	broadcastInputs := make([]*Node, 2)
	for ii := 0; ii < 2; ii++ {
		broadcastInputs[ii] = node.inputNodes[ii]
		if !broadcastInputs[ii].shape.Eq(node.shape) {
			broadcastInputs[ii] = BroadcastToShape(broadcastInputs[ii], node.shape)
		}
	}
	for ii := 0; ii < 2; ii++ {
		vMul := Mul(v, broadcastInputs[1-ii])
		inputsVJPs[ii] = vjpForDefaultBroadcast(node, node.inputNodes[ii], vMul)
	}
	return inputsVJPs
}

// VJP formulation for Div (without consideration to broadcasting):
// F(a,b) = a/b ->  v*dF/da = v/b ; v*dF/db = -v*a/b^2
func divVJP(node, v *Node, _ shapes.Shape) []*Node {
	inputsVJPs := make([]*Node, 2)
	broadcastInputs := make([]*Node, 2)
	for ii := 0; ii < 2; ii++ {
		broadcastInputs[ii] = node.inputNodes[ii]
		if !broadcastInputs[ii].shape.Eq(node.shape) {
			broadcastInputs[ii] = BroadcastToShape(broadcastInputs[ii], node.shape)
		}
	}
	a := broadcastInputs[0]
	b := broadcastInputs[1]
	inputsVJPs[0] = vjpForDefaultBroadcast(node, node.inputNodes[0], Div(v, b))
	inputsVJPs[1] = vjpForDefaultBroadcast(node, node.inputNodes[1],
		Neg(Mul(v, Div(a, Mul(b, b))))) // -v*a/b^2
	return inputsVJPs
}

func minMaxVJP(node, v *Node, _ shapes.Shape) []*Node {
	// We push the adjoint gradient to one side or the other, depending on which is the max.
	// Notice because PositiveIndicator(0) == 1, the gradient of Max(x, y) w.r.t. x, where x == y will be 1.0
	// (as opposed to 0).
	side0Indicator := PositiveIndicator(Sub(node.inputNodes[0], node.inputNodes[1]))
	side1Indicator := OneMinus(side0Indicator)
	if node.Type() == xla.MinNode {
		// If min, swap directions.
		side0Indicator, side1Indicator = side1Indicator, side0Indicator
	}
	return []*Node{
		vjpForDefaultBroadcast(node, node.inputNodes[0], Mul(v, side0Indicator)),
		vjpForDefaultBroadcast(node, node.inputNodes[1], Mul(v, side1Indicator)),
	}
}

func reduceSumVJP(node, v *Node, _ shapes.Shape) []*Node {
	// Reconstruct exactly the reduced dimensions.
	rankInput := node.inputNodes[0].shape.Rank()
	reducedDims := node.serializedNode.Ints
	if len(reducedDims) == 0 {
		// Reduced all dims, reconstruct those.
		reducedDims = make([]int, rankInput)
		for ii := 0; ii < rankInput; ii++ {
			reducedDims[ii] = ii
		}
	}

	// Expand rank of v to match the input, by re-creating
	// the reduced dimensions with size 1.
	newShape := node.inputNodes[0].shape.Clone()
	for _, dim := range reducedDims {
		newShape.Dimensions[dim] = 1
	}
	expandedV := ReshapeWithShape(v, newShape)

	// Now all we need it to broadcast the v on the reduced dimensions.
	// Notice the second input to a reduction is its initial value, a constant. There
	// is no need to push a gradient to that.
	vjp := BroadcastToShape(expandedV, node.inputNodes[0].shape)
	return []*Node{vjp, nil}
}

func reduceMaxVJP(node, v *Node, _ shapes.Shape) []*Node {
	// Reconstruct exactly the reduced dimensions, and build a newShape
	// (same shape as if we had done ReduceAndKeep(input[0]))
	rankInput := node.inputNodes[0].shape.Rank()
	reducedDims := node.serializedNode.Ints
	if len(reducedDims) == 0 {
		// Reduced all dims, reconstruct those.
		reducedDims = make([]int, rankInput)
		for ii := 0; ii < rankInput; ii++ {
			reducedDims[ii] = ii
		}
	}
	newShape := node.inputNodes[0].shape.Clone()
	for _, dim := range reducedDims {
		newShape.Dimensions[dim] = 1
	}

	// Expand the node output (with max) to match the input. And then creates
	// an indicator to which positions are at the max values.
	maxAtOriginalRank := ReshapeWithShape(node, newShape)
	maxIndicatorAtInput := PositiveIndicator(Sub(node.inputNodes[0], maxAtOriginalRank))

	// Expand rank of v to match the input, by re-creating
	// the reduced dimensions with size 1 and then broadcasting.
	expandedV := ReshapeWithShape(v, newShape)
	expandedV = BroadcastToShape(expandedV, node.inputNodes[0].shape)

	// vjp is only propagated to the elements at the max value.
	vjp := Mul(expandedV, maxIndicatorAtInput)
	return []*Node{vjp, nil}
}

func reshapeVJP(node, v *Node, _ shapes.Shape) []*Node {
	// ReshapeWithShape back to its inputNodes shape.
	return []*Node{ReshapeWithShape(v, node.inputNodes[0].shape)}
}

func logisticVJP(node, v *Node, _ shapes.Shape) []*Node {
	g := node.Graph()
	// d\sigma(x)/dx = sigma(x) * (1 - sigma(x)
	// See https://towardsdatascience.com/derivative-of-the-sigmoid-function-536880cf918e
	grad := Mul(node, Sub(ScalarOne(g, node.shape.DType), node))
	vjp := Mul(v, grad)
	return []*Node{vjp}
}

func dotVJP(node, v *Node, _ shapes.Shape) []*Node {
	// Case 1: dot product of two vectors.
	if node.inputNodes[0].shape.Rank() == 1 && node.inputNodes[1].shape.Rank() == 1 {
		return []*Node{
			Mul(v, node.inputNodes[1]),
			Mul(v, node.inputNodes[0]),
		}
	}

	// Case 2: matrix[i, j] x vector[j] -> vector[i]
	if node.inputNodes[0].shape.Rank() == 2 && node.inputNodes[1].shape.Rank() == 1 {
		v = ExpandDims(v, -1)
		v = BroadcastToShape(v, node.inputNodes[0].shape)
		return []*Node{
			Mul(v, ExpandDims(node.inputNodes[1], 0)),
			ReduceSum(Mul(v, node.inputNodes[0]), 0),
		}
	}

	// Case 3: matrix[i, j] x matrix[j, k] -> matrix[i, k]
	if node.inputNodes[0].shape.Rank() != 2 || node.inputNodes[1].shape.Rank() != 2 {
		Panicf("Dot node with combination of ranks not accepted: %s, %s", node.inputNodes[0].shape, node.inputNodes[1].shape)
	}
	dimI, dimJ, dimK := node.shape.Dimensions[0], node.inputNodes[0].shape.Dimensions[1], node.shape.Dimensions[1]
	v = ExpandDims(v, 1) // Shape [i, 1, k]
	v = BroadcastToShape(v, shapes.Make(node.shape.DType, dimI, dimJ, dimK))
	return []*Node{
		ReduceSum(Mul(v, ExpandDims(node.inputNodes[1], 0)), 2),
		ReduceSum(Mul(v, ExpandDims(node.inputNodes[0], -1)), 0),
	}
}

// sliceVJP generates the adjoint gradient term for SliceNode, encoded by the
// SliceWithStridesXLA function.
func sliceVJP(node, v *Node, _ shapes.Shape) []*Node {
	g := node.Graph()
	x := node.Inputs()[0]
	serialized := node.serializedNode

	// The incoming adjoint v must be applied only where the slice came from, and
	// the new adjoint will have zero, filled with padding (`Pad()`) elsewhere.
	//
	// Here we build the PadAxis configuration for each axis.
	rank := x.Rank()
	dimensions := x.Shape().Dimensions
	starts := serialized.Ints[0:rank]
	limits := serialized.Ints[rank : 2*rank]
	strides := serialized.Ints[2*rank:]
	padding := make([]PadAxis, rank)
	for ii := range padding {
		padding[ii].Start = starts[ii]
		if strides[ii] <= 1 {
			padding[ii].End = dimensions[ii] - limits[ii]
		} else {
			// Padding at the end may be affected by the strides.
			padding[ii].Interior = strides[ii] - 1
			dimToPad := dimensions[ii] - starts[ii]
			dimToPad -= (v.Shape().Dimensions[ii]-1)*strides[ii] + 1
			padding[ii].End = dimToPad // What is missing to make v the same shape as x at axis ii.
		}
	}
	return []*Node{
		Pad(v, ScalarZero(g, x.DType()), padding...),
	}
}

// gatherVJP generates the adjoint gradient term for  GatherXL node. It works only for simple gather operations,
// in particular it works with graph.Gather.
func gatherVJP(node, v *Node, _ shapes.Shape) []*Node {
	// TODO: check that values are compatible with Gather() and return an error if not.
	input := node.inputNodes[0]
	indices := node.inputNodes[1]
	inputShape := input.Shape()
	params := node.inputs.(*nodeInputsGather)

	// Gather() case: params.sliceSizes is 1 for the first dimensions, and full in the last. Plus the initial
	// dimensions are all collapsed.
	if len(params.sliceSizes) != inputShape.Rank() {
		Panicf("gradient from Gather with len(params.sliceSizes) != input.Rank() not defined -- params.sliceSizes=%v, input.Shape()=%s",
			params.sliceSizes, inputShape)
	}
	isSimpleGather := true // Whether all indexed axes are at the start, and slices are the full dimension.
	for axis, inputSize := range inputShape.Dimensions {
		if axis < len(params.collapsedSliceAxes) {
			if params.collapsedSliceAxes[axis] != axis {
				isSimpleGather = false
				break
			}
			if params.sliceSizes[axis] != 1 {
				isSimpleGather = false
				break
			}
		} else {
			// For non-collapsed axes, we expect the slice to be the full dimension.
			if params.sliceSizes[axis] != inputSize {
				isSimpleGather = false
				break
			}
		}
	}
	if isSimpleGather {
		return []*Node{
			ScatterAdd(Zeros(node.graph, inputShape), indices, v, params.indicesAreSorted, false),
			nil, // No gradients for indices.
		}
	}

	// GatherSlices(): params.sliceSizes are variable, but there are no paramsCollapsedSliceDims.
	//fmt.Printf("\tgatherVJP: operand=%s, start=%s, paramsIndexVectorAxis=%d, offsetDims=%v, paramsCollapsedSliceDims=%v, params.startIndexMap=%v, params.sliceSizes=%v\n",
	//	input.shape, indices.shape, paramsIndexVectorAxis, offsetDims, paramsCollapsedSliceDims, params.startIndexMap, params.sliceSizes)
	isGatherSlices := len(params.collapsedSliceAxes) == 0
	if !isGatherSlices {
		Panicf("xlaGather operation for which no gradient was defined. Please use only Gather() or GatherSlices().")
	}

	// Find scatterXLA parameters to reverse the GatherSlice:
	operand := ZerosLike(input)
	startIndices := indices
	outputPrefixRank := startIndices.Rank() - 1 // Prefix dimensions of the output of the GatherSlice.
	updates := v
	//fmt.Printf("\tgatherVJP: updates=%s\n", updates.shape)

	// updateWindowsDims: one per every dimension of the input, offset by the initial outputPrefixRank.
	updateWindowsDims := make([]int, 0, inputShape.Rank())
	for ii := 0; ii < inputShape.Rank(); ii++ {
		updateWindowsDims = append(updateWindowsDims, ii+outputPrefixRank)
	}
	var insertedWindowDims []int                     // Empty, since the original GatherSlice don's have any paramsCollapsedSliceDims.
	scatterDimsToOperandDims := params.startIndexMap // Same map used in GatherSlice.
	// We don't make any assumptions on uniqueness or sortedness of the indices. Likely the slices will overlap.
	return []*Node{
		backendScatterAdd(operand, startIndices, updates, params.indexVectorAxis, updateWindowsDims, insertedWindowDims, scatterDimsToOperandDims,
			params.indicesAreSorted, false),
		nil, // No gradients for indices.
	}
}

// batchNormTrainingVJP generates the gradient with respect to the operand and the scale and offset
// parameters. It uses the XLA implemented gradient.
func batchNormTrainingVJP(node, v *Node, _ shapes.Shape) []*Node {
	operand, scale := node.inputNodes[0], node.inputNodes[1]
	mean := GetTupleElement(node, 1)     // Output 1 of batchNormTraining, the batchMean.
	variance := GetTupleElement(node, 2) // Output 2 of batchNormTraining, the batchVariance.
	epsilon := node.serializedNode.Float
	featureAxis := node.serializedNode.Int
	gradOutput := GetTupleElement(v, 0)
	gradOperand, gradScale, gradOffset := batchNormGradXLA(operand, scale, mean, variance, gradOutput, epsilon, featureAxis)
	return []*Node{gradOperand, gradScale, gradOffset}
}

// transposeVJP generates the "vector dot jacobian" w.r.t. the input of transpose. It's
// simply the transpose of the incoming vector.
func transposeVJP(node, v *Node, _ shapes.Shape) []*Node {
	permutations := node.serializedNode.Ints
	reversePermutations := make([]int, len(permutations))
	for to, from := range permutations {
		reversePermutations[from] = to
	}
	vjp := TransposeAllDims(v, reversePermutations...)
	return []*Node{vjp}
}

// broadcastInDimVJP generates the "vector dot jacobian" w.r.t. the input of broadcast.
// One just needs to reduce the broadcast dimensions.
func broadcastInDimVJP(node, v *Node, _ shapes.Shape) []*Node {
	x := node.inputNodes[0]
	outputShape := v.Shape()
	shape := node.serializedNode.Shape
	broadcastDims := node.serializedNode.Ints

	if x.Rank() != len(broadcastDims) {
		Panicf("there must be a broadcastDim for each axis in x, instead got x.shape=%s and broadcastDims=%v",
			x.shape, broadcastDims)
	}

	axesPreserved := make([]bool, outputShape.Rank())
	for inputAxis, outputAxis := range broadcastDims {
		if x.Shape().Dimensions[inputAxis] == shape.Dimensions[outputAxis] {
			axesPreserved[outputAxis] = true
		} else {
			if x.Shape().Dimensions[inputAxis] != 1 {
				Panicf("unexpected broadcast from shape %s to shape %s at axis %d -- don't know how to calculate gradient",
					x.Shape(), shape, inputAxis)
			}
		}
	}
	dimsToReduce := make([]int, 0, outputShape.Rank())
	for axis, preserved := range axesPreserved {
		if !preserved {
			dimsToReduce = append(dimsToReduce, axis)
		}
	}
	gradWrtX := v
	if len(dimsToReduce) > 0 {
		gradWrtX = ReduceSum(v, dimsToReduce...)
	}
	if gradWrtX.Rank() != x.Rank() {
		// X had some axes of dimension 1 that were reduced, we simply reshape it here.
		gradWrtX = Reshape(gradWrtX, x.Shape().Dimensions...)
	}
	return []*Node{gradWrtX}
}
