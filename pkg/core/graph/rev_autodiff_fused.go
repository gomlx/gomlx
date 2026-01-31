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
	VJPRegistration[NodeTypeFusedSoftmax] = vjpForSingleOutput(softmaxVJP)
	VJPRegistration[NodeTypeFusedGelu] = vjpForSingleOutput(geluVJP)
	VJPRegistration[NodeTypeFusedLayerNorm] = vjpForSingleOutput(layerNormVJP)
	VJPRegistration[NodeTypeFusedDense] = vjpForSingleOutput(denseVJP)
}

// softmaxVJP computes the VJP for fused Softmax.
//
// Given s = softmax(x, axis):
//
//	ds/dx · v = s * (v - ReduceAndKeep(v * s, axis))
func softmaxVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsFusedSoftmax)
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
	params := node.inputs.(*nodeInputsFusedGelu)
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
	params := node.inputs.(*nodeInputsFusedLayerNorm)
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

// denseVJP computes the VJP for fused Dense.
//
// Dense: y = activation(x @ W + bias)
// where x: [..., in_features], W: [in_features, out_features]
//
// Chain rule: backprop through activation first (if any), then through dense.
func denseVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsFusedDense)
	x := params.x
	weight := params.weight
	xRank := x.Rank()
	lastAxis := xRank - 1

	// If activation is present, backprop through it first.
	if params.activation != backends.ActivationNone {
		// Recompute pre-activation: z = x @ W + bias.
		z := DotGeneral(x, []int{lastAxis}, []int{}, weight, []int{0}, []int{})
		if params.bias != nil {
			z = Add(z, params.bias)
		}

		// Compute v * activation'(z).
		switch params.activation {
		case backends.ActivationRelu:
			zero := ScalarZero(z.Graph(), z.DType())
			v = Where(GreaterThan(z, zero), v, zero)
		case backends.ActivationGelu:
			cdf := MulScalar(AddScalar(Erf(DivScalar(z, math.Sqrt2)), 1), 0.5)
			pdf := MulScalar(Exp(MulScalar(Mul(z, z), -0.5)), 1.0/math.Sqrt(2.0*math.Pi))
			v = Mul(v, Add(cdf, Mul(z, pdf)))
		case backends.ActivationSilu:
			sig := Logistic(z)
			one := ScalarOne(z.Graph(), z.DType())
			v = Mul(v, Mul(sig, Add(one, Mul(z, Sub(one, sig)))))
		case backends.ActivationTanh:
			t := Tanh(z)
			one := ScalarOne(z.Graph(), z.DType())
			v = Mul(v, Sub(one, Mul(t, t)))
		default:
			Panicf("denseVJP: unsupported activation type %s", params.activation)
		}
	}

	// dx = v @ W^T
	dx := DotGeneral(v, []int{lastAxis}, []int{}, weight, []int{1}, []int{})

	// dW: contract batch dims of v and x, keep in from x and out from v.
	var batchDimsV, batchDimsX []int
	for i := 0; i < xRank-1; i++ {
		batchDimsV = append(batchDimsV, i)
		batchDimsX = append(batchDimsX, i)
	}
	dweight := DotGeneral(x, batchDimsX, []int{}, v, batchDimsV, []int{})

	results := []*Node{dx, dweight}

	if params.bias != nil {
		if len(batchDimsV) > 0 {
			dbias := ReduceSum(v, batchDimsV...)
			results = append(results, dbias)
		} else {
			results = append(results, v)
		}
	}

	return results
}
