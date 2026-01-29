// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"math"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

func init() {
	// Register VJPs for fused ops so they can be used during training.
	VJPRegistration[NodeTypeSoftmax] = vjpForSingleOutput(softmaxVJP)
	VJPRegistration[NodeTypeGelu] = vjpForSingleOutput(geluVJP)
	VJPRegistration[NodeTypeLayerNorm] = vjpForSingleOutput(layerNormVJP)
	VJPRegistration[NodeTypeLinear] = vjpForSingleOutput(linearVJP)
	VJPRegistration[NodeTypeLinearActivation] = vjpForSingleOutput(linearActivationVJP)
}

// softmaxVJP computes the VJP for fused Softmax.
//
// Given s = softmax(x, axis):
//
//	ds/dx · v = s * (v - ReduceAndKeep(v * s, axis))
func softmaxVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsSoftmax)
	s := node // node is the softmax output
	vs := Mul(v, s)
	sumVS := ReduceAndKeep(vs, ReduceSum, params.axis)
	return []*Node{Mul(s, Sub(v, sumVS))}
}

// geluVJP computes the VJP for fused Gelu (exact mode).
//
// Given gelu(x) = x * Φ(x), where Φ(x) = 0.5 * (1 + erf(x / √2)):
//
//	dgelu/dx = Φ(x) + x * φ(x)
//
// where φ(x) = (1/√(2π)) * exp(-x²/2) is the standard normal PDF.
//
//	VJP = v * (Φ(x) + x * φ(x))
func geluVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsGelu)
	x := params.x

	// Φ(x) = 0.5 * (1 + erf(x / √2))
	cdf := MulScalar(AddScalar(Erf(DivScalar(x, math.Sqrt2)), 1), 0.5)

	// φ(x) = (1/√(2π)) * exp(-x²/2)
	pdf := MulScalar(Exp(MulScalar(Mul(x, x), -0.5)), 1.0/math.Sqrt(2.0*math.Pi))

	// dgelu/dx = Φ(x) + x * φ(x)
	grad := Add(cdf, Mul(x, pdf))
	return []*Node{Mul(v, grad)}
}

// layerNormVJP computes the VJP for fused LayerNorm.
//
// LayerNorm: y = gamma * (x - mean) / sqrt(var + eps) + beta
//
// Gradients:
//
//	dy/dx: via chain rule through normalization
//	dy/dgamma: sum(v * xhat) over batch dims
//	dy/dbeta: sum(v) over batch dims
//
// where xhat = (x - mean) / sqrt(var + eps).
func layerNormVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsLayerNorm)
	x := params.x
	axes := params.axes
	epsilon := params.epsilon

	// Recompute forward pass intermediates.
	mean := ReduceAndKeep(x, ReduceMean, axes...)
	xCentered := Sub(x, mean)
	variance := ReduceAndKeep(Mul(xCentered, xCentered), ReduceMean, axes...)
	invStd := Rsqrt(AddScalar(variance, epsilon))
	xhat := Mul(xCentered, invStd)

	// Apply gamma scaling to upstream gradient if present.
	var vScaled *Node
	if params.gamma != nil {
		vScaled = Mul(v, params.gamma)
	} else {
		vScaled = v
	}

	// Gradient w.r.t. x:
	// dx = invStd * (vScaled - mean(vScaled) - xhat * mean(vScaled * xhat))
	meanVScaled := ReduceAndKeep(vScaled, ReduceMean, axes...)
	meanVScaledXhat := ReduceAndKeep(Mul(vScaled, xhat), ReduceMean, axes...)
	dx := Mul(invStd, Sub(Sub(vScaled, meanVScaled), Mul(xhat, meanVScaledXhat)))

	results := []*Node{dx}

	if params.gamma != nil {
		// dy/dgamma = sum(v * xhat) over non-normalizing (batch) dimensions.
		dgamma := reduceToBroadcastShape(Mul(v, xhat), params.gamma, axes, x)
		results = append(results, dgamma)
	}

	if params.beta != nil {
		// dy/dbeta = sum(v) over non-normalizing (batch) dimensions.
		dbeta := reduceToBroadcastShape(v, params.beta, axes, x)
		results = append(results, dbeta)
	}

	return results
}

// reduceToBroadcastShape reduces a gradient to match a parameter's shape.
// The parameter was broadcast from normAxes dimensions to x's full shape,
// so we sum over all non-normalizing (batch) dimensions.
func reduceToBroadcastShape(grad, param *Node, normAxes []int, x *Node) *Node {
	normSet := make(map[int]bool, len(normAxes))
	for _, a := range normAxes {
		normSet[a] = true
	}
	var batchAxes []int
	for i := 0; i < x.Rank(); i++ {
		if !normSet[i] {
			batchAxes = append(batchAxes, i)
		}
	}
	if len(batchAxes) > 0 {
		grad = ReduceSum(grad, batchAxes...)
	}
	return ReshapeWithShape(grad, param.Shape())
}

// linearVJP computes the VJP for fused Linear.
//
// Linear: y = x @ weight^T + bias
// where x: [..., in_features], weight: [out_features, in_features]
//
//	dy/dx = v @ weight                          (v: [..., out], weight: [out, in] → [..., in])
//	dy/dweight = sum_batch(v^T @ x)             (→ [out, in])
//	dy/dbias = sum_batch(v)                     (→ [out])
func linearVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsLinear)
	x := params.x
	weight := params.weight
	xRank := x.Rank()

	// dx = v @ weight
	// v: [..., out], weight: [out, in] → contract on out axis
	// v's last axis (out) contracts with weight's first axis (out).
	lastAxis := xRank - 1 // position of "out" in v (same rank as x but last dim is out)
	dx := DotGeneral(v, []int{lastAxis}, []int{}, weight, []int{0}, []int{})

	// dweight: contract batch dims of v and x, keep out from v and in from x.
	// v: [..., out], x: [..., in]
	// Batch dims: 0..rank-2, contracting those.
	// Result: [out, in]
	var batchDimsV, batchDimsX []int
	for i := 0; i < xRank-1; i++ {
		batchDimsV = append(batchDimsV, i)
		batchDimsX = append(batchDimsX, i)
	}
	// lhs=v contracting=batch, batch=none; rhs=x contracting=batch, batch=none
	// This gives outer product of the non-batch dims: [out] x [in] = [out, in]
	dweight := DotGeneral(v, batchDimsV, []int{}, x, batchDimsX, []int{})

	results := []*Node{dx, dweight}

	if params.bias != nil {
		// dbias = sum(v) over batch dims → [out]
		if len(batchDimsV) > 0 {
			dbias := ReduceSum(v, batchDimsV...)
			results = append(results, dbias)
		} else {
			results = append(results, v)
		}
	}

	return results
}

// linearActivationVJP computes the VJP for fused LinearActivation.
//
// LinearActivation: y = activation(x @ weight^T + bias)
//
// Chain rule: backprop through activation first, then through linear.
func linearActivationVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsLinearActivation)
	x := params.x
	weight := params.weight
	xRank := x.Rank()

	// Recompute pre-activation: z = x @ weight^T + bias.
	lastAxis := xRank - 1
	z := DotGeneral(x, []int{lastAxis}, []int{}, weight, []int{1}, []int{})
	if params.bias != nil {
		z = Add(z, params.bias)
	}

	// Compute v * activation'(z).
	var vz *Node
	switch params.activation {
	case backends.ActivationNone:
		vz = v
	case backends.ActivationRelu:
		zero := ScalarZero(z.Graph(), z.DType())
		vz = Where(GreaterThan(z, zero), v, zero)
	case backends.ActivationGelu:
		cdf := MulScalar(AddScalar(Erf(DivScalar(z, math.Sqrt2)), 1), 0.5)
		pdf := MulScalar(Exp(MulScalar(Mul(z, z), -0.5)), 1.0/math.Sqrt(2.0*math.Pi))
		vz = Mul(v, Add(cdf, Mul(z, pdf)))
	case backends.ActivationSilu:
		sig := Logistic(z)
		one := ScalarOne(z.Graph(), z.DType())
		vz = Mul(v, Mul(sig, Add(one, Mul(z, Sub(one, sig)))))
	case backends.ActivationTanh:
		t := Tanh(z)
		one := ScalarOne(z.Graph(), z.DType())
		vz = Mul(v, Sub(one, Mul(t, t)))
	default:
		Panicf("linearActivationVJP: unsupported activation type %s", params.activation)
	}

	// Now compute linear VJP with vz as the upstream gradient.
	// dx = vz @ weight
	dx := DotGeneral(vz, []int{lastAxis}, []int{}, weight, []int{0}, []int{})

	// dweight = sum_batch(vz^T @ x) → [out, in]
	var batchDims []int
	for i := 0; i < xRank-1; i++ {
		batchDims = append(batchDims, i)
	}
	dweight := DotGeneral(vz, batchDims, []int{}, x, batchDims, []int{})

	results := []*Node{dx, dweight}

	if params.bias != nil {
		if len(batchDims) > 0 {
			dbias := ReduceSum(vz, batchDims...)
			results = append(results, dbias)
		} else {
			results = append(results, vz)
		}
	}

	return results
}
