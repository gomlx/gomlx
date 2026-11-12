// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package nn

import (
	"slices"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapeinference"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/envutil"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/ggml"
	"github.com/gomlx/gomlx/ml/layers/activation"
	. "github.com/gomlx/gomlx/support/exceptions"
)

// QuantizedDense performs a quantized dense (linear) projection with optional bias,
// activation, pre-quantized input activations, and optional output re-quantization:
//
//   - Weight-only quantization (cfg.XQuantization == nil):
//     y = activation(x @ dequant(weight, cfg.WeightQuantization) + bias)
//     (transposing dequant(weight) when cfg.WeightLayout == compute.DenseLayoutOutputsInput).
//   - Pre-quantized x + weight integer matmul (cfg.XQuantization != nil):
//     supports QuantLinear (exact Int32 accumulation + xScale * wScale) and
//     QuantGGML (GGMLQ8_0 / GGMLQ8_K paired block dot products).
//   - Bias conventions (when bias != nil, shape [N]):
//   - Floating-point bias: added in floating-point AFTER dequantization / (xScale * wScale)
//     rescaling and BEFORE activation / OutputQuantization.
//   - Int32 bias (only for QuantLinear XQuantization when neither xScale nor wScale is sub-blocked
//     along K): added directly to the Int32 accumulator BEFORE converting to Float32 and multiplying
//     by xScale * wScale, i.e. z[m, n] = float32(acc_int32[m, n] + bias[n]) * (xScale[m] * wScale[n]).
//   - Optional output quantization (cfg.OutputQuantization != nil):
//     re-quantizes the post-activation float output to Int8 or Uint8.
//
// An optional activation.Type may be passed as optionalActivation (convenience shorthand
// when cfg.Activation is left at its zero value ActivationNone).
//
// Migration note: bias is the 3rd parameter (matching nn.Dense), QuantGGML weights require
// cfg.WeightLayout = compute.DenseLayoutOutputsInput, scalar [] and 1D [N] scales no longer
// need [K, NumBlocks] broadcasts, and QuantLinear ZeroPoint uses the standard pre-scale
// convention real = (float(q) - float(zp)) * scale (legacy additive offsets convert via
// ZeroPoint = Neg(Div(oldOffset, scale))).
//
// When the backend supports fused FusedQuantizedDense (and FusedQuantizedDenseVJP for
// weight-only fine-tuning such as QLoRA), the native implementation is used without
// materializing dequantized [K, N] float weights in forward or backward passes;
// otherwise QuantizedDense falls back automatically to primitive graph operations.
func QuantizedDense(
	x, weight, bias *Node,
	cfg QuantizedDenseConfig,
	optionalActivation ...activation.Type,
) *Node {
	if cfg.WeightQuantization == nil {
		Panicf("nn.QuantizedDense: cfg.WeightQuantization must not be nil")
	}
	if len(optionalActivation) > 0 {
		if len(optionalActivation) > 1 {
			Panicf("nn.QuantizedDense() can only take one optional activation, got %v", optionalActivation)
		}
		optActCfg := optionalActivation[0].ToBackend()
		if cfg.Activation.Type != compute.ActivationNone && optActCfg.Type != cfg.Activation.Type {
			Panicf("nn.QuantizedDense(): conflicting activations in cfg.Activation (%v) and optionalActivation (%v)",
				cfg.Activation.Type, optionalActivation[0])
		}
		if optActCfg.Type != compute.ActivationNone {
			cfg.Activation = optActCfg
		}
	}

	// SwiGLU halves the output feature dimension and is never fused inside FusedQuantizedDense
	// (mirroring FusedDense); emit the linear projection with ActivationNone followed by SwiGLU.
	if cfg.Activation.Type == compute.ActivationSwiGLU {
		if cfg.OutputQuantization != nil {
			Panicf("nn.QuantizedDense: ActivationSwiGLU cannot be combined with OutputQuantization")
		}
		noActCfg := cfg
		noActCfg.Activation = compute.ActivationConfig{Type: compute.ActivationNone}
		denseOut := QuantizedDense(x, weight, bias, noActCfg)
		return activation.SwiGLU(denseOut)
	}

	decomposed := func() *Node {
		return quantizedDenseDecomposed(x, weight, bias, cfg)
	}

	if enabled, err := envutil.ReadBool(FusionEnv, true); err != nil {
		panic(err)
	} else if !enabled {
		return decomposed()
	}

	g := x.Graph()
	backend := g.Backend()
	if backend != nil && !backend.Capabilities().Operations[compute.OpTypeFusedQuantizedDense] {
		return decomposed()
	}

	isWeightOnlyFloat := cfg.XQuantization == nil && cfg.OutputQuantization == nil
	hasFusedVJP := isWeightOnlyFloat && backend != nil && backend.Capabilities().Operations[compute.OpTypeFusedQuantizedDenseVJP]

	if hasFusedVJP && cfg.Activation.Type.VJPRequiresInput() {
		noActCfg := cfg
		noActCfg.Activation = compute.ActivationConfig{Type: compute.ActivationNone}
		denseOut, _ := InternalFusedOpCaller(
			func() *Node { return BackendFusedQuantizedDense(x, weight, bias, noActCfg) },
			func() *Node { return quantizedDenseDecomposed(x, weight, bias, noActCfg) },
			false,
		)
		return applyBackendActivationConfig(denseOut, cfg.Activation)
	}

	// Only attach vjpAlternateOutputs when the operation is differentiable (weight-only float x/y),
	// the backend lacks FusedQuantizedDenseVJP, and a decomposed fallback exists for the scheme.
	useAlternateVJP := isWeightOnlyFloat && !hasFusedVJP && canDecomposeConfig(cfg)
	res, _ := InternalFusedOpCaller(
		func() *Node { return BackendFusedQuantizedDense(x, weight, bias, cfg) },
		decomposed,
		useAlternateVJP,
	)
	return res
}

func canDecomposeConfig(cfg QuantizedDenseConfig) bool {
	if cfg.WeightQuantization == nil {
		return false
	}
	if cfg.WeightQuantization.Scheme == compute.QuantGGML && !ggml.CanDecompose(cfg.WeightQuantization.GGMLType) {
		return false
	}
	if cfg.XQuantization != nil && cfg.XQuantization.Scheme == compute.QuantGGML && !ggml.CanDecompose(cfg.XQuantization.GGMLType) {
		return false
	}
	return true
}

func quantizationToShapes(q *Quantization) *shapeinference.QuantizationShapes {
	if q == nil {
		return nil
	}
	qs := &shapeinference.QuantizationShapes{
		Scheme:         q.Scheme,
		Scale:          shapes.Invalid(),
		ZeroPoint:      shapes.Invalid(),
		BlockAxis:      q.BlockAxis,
		BlockSize:      q.BlockSize,
		GGMLType:       q.GGMLType,
		QuantizedDType: q.QuantizedDType,
	}
	if q.Scale != nil {
		qs.Scale = q.Scale.Shape()
	}
	if q.ZeroPoint != nil {
		qs.ZeroPoint = q.ZeroPoint.Shape()
	}
	return qs
}

// quantizedDenseDecomposed implements QuantizedDense using primitive graph ops.
func quantizedDenseDecomposed(x, weight, bias *Node, cfg QuantizedDenseConfig) *Node {
	var biasShape shapes.Shape
	if bias != nil {
		biasShape = bias.Shape()
	}
	info, err := shapeinference.FusedQuantizedDenseWithInfo(
		x.Shape(),
		weight.Shape(),
		biasShape,
		shapeinference.QuantizedDenseConfigShapes{
			WeightQuantization: quantizationToShapes(cfg.WeightQuantization),
			XQuantization:      quantizationToShapes(cfg.XQuantization),
			OutputQuantization: quantizationToShapes(cfg.OutputQuantization),
			QuantizedMatmul:    cfg.QuantizedMatmul,
			Activation:         cfg.Activation,
			WeightLayout:       cfg.WeightLayout,
		},
	)
	if err != nil {
		panic(err)
	}

	wq := cfg.WeightQuantization
	var y *Node

	switch {
	case wq.Scheme == compute.QuantGGML:
		y = quantizedDenseDecomposedGGML(x, weight, bias, cfg, info)
	case cfg.XQuantization != nil:
		y = quantizedDenseDecomposedPreQuantizedLinear(x, weight, bias, cfg, info)
	default:
		y = quantizedDenseDecomposedWeightOnly(x, weight, bias, cfg, info)
	}

	if cfg.Activation.Type != compute.ActivationNone {
		y = applyBackendActivationConfig(y, cfg.Activation)
	}
	if cfg.OutputQuantization != nil {
		y = quantizeOutputDecomposed(y, cfg.OutputQuantization, info)
	}
	return y
}

func quantizedDenseDecomposedGGML(
	x, weight, bias *Node,
	cfg QuantizedDenseConfig,
	info shapeinference.QuantizedDenseInfo,
) *Node {
	wq := cfg.WeightQuantization
	if !ggml.CanDecompose(wq.GGMLType) {
		Panicf("nn.QuantizedDense: GGML weight format %s does not have a decomposed fallback (requires fused backend support)", wq.GGMLType)
	}

	dequantW := ggml.Dequantize(weight, wq.GGMLType, info.N) // [N, K] Float32
	xForDot := x
	if cfg.XQuantization != nil {
		xq := cfg.XQuantization
		if !ggml.CanDecompose(xq.GGMLType) {
			Panicf("nn.QuantizedDense: GGML XQuantization format %s does not have a decomposed fallback", xq.GGMLType)
		}
		xShape := x.Shape()
		if xShape.IsDynamic() {
			Panicf("nn.QuantizedDense: decomposed GGML XQuantization does not support dynamic batch dimensions (%s)", xShape)
		}
		bytesPerRowX := xShape.Dimensions[xShape.Rank()-1]
		M := xShape.Size() / bytesPerRowX
		x2d := Reshape(x, M, bytesPerRowX)
		xDeq2D := ggml.Dequantize(x2d, xq.GGMLType, M) // [M, K] Float32
		xDims := slices.Clone(xShape.Dimensions)
		xDims[len(xDims)-1] = info.K
		xForDot = Reshape(xDeq2D, xDims...)
	} else if info.FloatComputeDType != dtypes.Float32 {
		dequantW = ConvertDType(dequantW, info.FloatComputeDType)
	}

	y := DotGeneral(xForDot, []int{xForDot.Rank() - 1}, nil, dequantW, []int{1}, nil)
	if bias != nil {
		b := bias
		if b.DType() != y.DType() {
			b = ConvertDType(b, y.DType())
		}
		y = Add(y, ExpandLeftToRank(b, y.Rank()))
	}
	return y
}

func quantizedDenseDecomposedWeightOnly(
	x, weight, bias *Node,
	cfg QuantizedDenseConfig,
	info shapeinference.QuantizedDenseInfo,
) *Node {
	g := x.Graph()
	wq := cfg.WeightQuantization
	wShape := weight.Shape()
	d0, d1 := wShape.Dimensions[0], wShape.Dimensions[1]

	var dequant *Node
	switch wq.Scheme {
	case compute.QuantNF4:
		dequant = dequantNF4FromTyped(g, weight, d0, d1)
	case compute.QuantLinear:
		dequant = ConvertDType(weight, dtypes.Float32)
		if wq.ZeroPoint != nil {
			zpExpanded := expandWeightParam(wq.ZeroPoint, info.WeightZeroPointLayout, d0, d1)
			dequant = Sub(dequant, zpExpanded)
		}
	default:
		Panicf("nn.QuantizedDense: unsupported quantization scheme %v", wq.Scheme)
	}

	scaleExpanded := expandWeightParam(wq.Scale, info.WeightScaleLayout, d0, d1)
	dequant = Mul(dequant, scaleExpanded)
	if info.FloatComputeDType != dtypes.Float32 {
		dequant = ConvertDType(dequant, info.FloatComputeDType)
	}

	rhsContractAxis := 0
	if cfg.WeightLayout == compute.DenseLayoutOutputsInput {
		rhsContractAxis = 1
	}
	y := DotGeneral(x, []int{x.Rank() - 1}, nil, dequant, []int{rhsContractAxis}, nil)
	if bias != nil {
		b := bias
		if b.DType() != y.DType() {
			b = ConvertDType(b, y.DType())
		}
		y = Add(y, ExpandLeftToRank(b, y.Rank()))
	}
	return y
}

func quantizedDenseDecomposedPreQuantizedLinear(
	x, weight, bias *Node,
	cfg QuantizedDenseConfig,
	info shapeinference.QuantizedDenseInfo,
) *Node {
	xq := cfg.XQuantization
	wq := cfg.WeightQuantization
	wShape := weight.Shape()
	d0, d1 := wShape.Dimensions[0], wShape.Dimensions[1]
	K, N := info.K, info.N

	x2d, unflatten := flattenLastAxisTo2D(x, K)
	rhsContractAxis := 0
	wKBlockSize := info.WeightScaleLayout.RowBlockSize
	if cfg.WeightLayout == compute.DenseLayoutOutputsInput {
		rhsContractAxis = 1
		wKBlockSize = info.WeightScaleLayout.ColBlockSize
	}

	// Exact Int32 accumulation when neither x nor weight partitions K into multiple scale blocks.
	if info.XScaleLayout.ColBlockSize == K && wKBlockSize == K {
		xI32 := ConvertDType(x2d, dtypes.Int32)
		if xq.ZeroPoint != nil {
			xZPI32 := zeroPointToInt32(expandActivationParam2D(xq.ZeroPoint, info.XZeroPointLayout, x2d, K))
			xI32 = Sub(xI32, xZPI32)
		}

		wI32 := ConvertDType(weight, dtypes.Int32)
		if wq.ZeroPoint != nil {
			wZPI32 := zeroPointToInt32(expandWeightParam(wq.ZeroPoint, info.WeightZeroPointLayout, d0, d1))
			wI32 = Sub(wI32, wZPI32)
		}

		accI32 := DotGeneral(xI32, []int{1}, nil, wI32, []int{rhsContractAxis}, nil) // [M, N] Int32
		if bias != nil && bias.DType() == dtypes.Int32 {
			accI32 = Add(accI32, ExpandLeftToRank(bias, 2))
		}

		xSc := expandActivationScalePerRowOrScalar(xq.Scale, info.XScaleLayout, x2d)
		wScFull := expandWeightParam(wq.Scale, info.WeightScaleLayout, d0, d1)
		var wScRow *Node // [1, N]
		if cfg.WeightLayout == compute.DenseLayoutInputOutputs {
			wScRow = Slice(wScFull, AxisRange(0, 1), AxisRange())
		} else {
			wScCol := Slice(wScFull, AxisRange(), AxisRange(0, 1)) // [N, 1]
			wScRow = Transpose(wScCol, 0, 1)                       // [1, N]
		}

		y2d := Mul(ConvertDType(accI32, dtypes.Float32), Mul(xSc, wScRow))
		if bias != nil && bias.DType() != dtypes.Int32 {
			y2d = Add(y2d, ExpandLeftToRank(ConvertDType(bias, dtypes.Float32), 2))
		}
		return unflatten(y2d, N)
	}

	// K-blocked scales: dequantize each block to Float32 and contract.
	xDeq := ConvertDType(x2d, dtypes.Float32)
	if xq.ZeroPoint != nil {
		xZP := roundIfFloatZeroPoint(xq.ZeroPoint, expandActivationParam2D(xq.ZeroPoint, info.XZeroPointLayout, x2d, K))
		xDeq = Sub(xDeq, xZP)
	}
	xScaleExp := expandActivationParam2D(xq.Scale, info.XScaleLayout, x2d, K)
	xDeq = Mul(xDeq, xScaleExp)

	wDeq := ConvertDType(weight, dtypes.Float32)
	if wq.ZeroPoint != nil {
		wZP := roundIfFloatZeroPoint(wq.ZeroPoint, expandWeightParam(wq.ZeroPoint, info.WeightZeroPointLayout, d0, d1))
		wDeq = Sub(wDeq, wZP)
	}
	wScaleExp := expandWeightParam(wq.Scale, info.WeightScaleLayout, d0, d1)
	wDeq = Mul(wDeq, wScaleExp)

	y2d := DotGeneral(xDeq, []int{1}, nil, wDeq, []int{rhsContractAxis}, nil)
	if bias != nil {
		y2d = Add(y2d, ExpandLeftToRank(ConvertDType(bias, dtypes.Float32), 2))
	}
	return unflatten(y2d, N)
}

func quantizeOutputDecomposed(y *Node, oq *Quantization, info shapeinference.QuantizedDenseInfo) *Node {
	g := y.Graph()
	N := info.N
	y2d, unflatten := flattenLastAxisTo2D(ConvertDType(y, dtypes.Float32), N)

	outScale := expandActivationParam2D(oq.Scale, info.OutputScaleLayout, y2d, N)
	q := roundToEvenNode(Div(y2d, outScale))
	if oq.ZeroPoint != nil {
		outZP := expandActivationParam2D(oq.ZeroPoint, info.OutputZeroPointLayout, y2d, N)
		q = Add(q, outZP)
	}

	targetDType := info.OutputShape.DType
	switch targetDType {
	case dtypes.Int8:
		q = Clamp(Scalar(g, dtypes.Float32, float32(-128.0)), q, Scalar(g, dtypes.Float32, float32(127.0)))
	case dtypes.Uint8:
		q = Clamp(Scalar(g, dtypes.Float32, float32(0.0)), q, Scalar(g, dtypes.Float32, float32(255.0)))
	default:
		Panicf("nn.QuantizedDense: unsupported OutputQuantization DType %s", targetDType)
	}
	qInt := ConvertDType(roundToEvenNode(q), targetDType)
	return unflatten(qInt, N)
}

func flattenLastAxisTo2D(t *Node, lastDim int) (t2d *Node, unflatten func(res2d *Node, outLastDim int) *Node) {
	tShape := t.Shape()
	rank := tShape.Rank()
	if rank == 2 {
		return t, func(res2d *Node, _ int) *Node { return res2d }
	}
	if rank == 1 {
		return Reshape(t, 1, lastDim), func(res2d *Node, outLastDim int) *Node {
			return Reshape(res2d, outLastDim)
		}
	}
	if tShape.IsDynamic() {
		g := t.Graph()
		t2d = DynamicReshape(t, NamedInferredDim(g.UniqueName("qd_batch")), StaticDim(lastDim))
		return t2d, func(res2d *Node, outLastDim int) *Node {
			specs := DimensionSpecsFor(t)
			specs[rank-1] = StaticDim(outLastDim)
			return DynamicReshape(res2d, specs...)
		}
	}
	M := tShape.Size() / lastDim
	return Reshape(t, M, lastDim), func(res2d *Node, outLastDim int) *Node {
		outDims := slices.Clone(tShape.Dimensions)
		outDims[rank-1] = outLastDim
		return Reshape(res2d, outDims...)
	}
}

func expandWeightParam(param *Node, layout shapeinference.BlockParamLayout, d0, d1 int) *Node {
	p := ConvertDType(param, dtypes.Float32)
	pShape := p.Shape()
	if pShape.Rank() == 2 && pShape.Dimensions[0] == d0 && pShape.Dimensions[1] == d1 {
		return p
	}
	numRowBlocks := (d0 + layout.RowBlockSize - 1) / layout.RowBlockSize
	numColBlocks := layout.NumColBlocks
	p4 := Reshape(p, numRowBlocks, 1, numColBlocks, 1)
	b4 := BroadcastToDims(p4, numRowBlocks, layout.RowBlockSize, numColBlocks, layout.ColBlockSize)
	fullRows := numRowBlocks * layout.RowBlockSize
	fullCols := numColBlocks * layout.ColBlockSize
	expanded := Reshape(b4, fullRows, fullCols)
	if fullRows != d0 || fullCols != d1 {
		expanded = Slice(expanded, AxisRange(0, d0), AxisRange(0, d1))
	}
	return expanded
}

func expandActivationScalePerRowOrScalar(param *Node, layout shapeinference.ActivationParamLayout, ref2d *Node) *Node {
	p := ConvertDType(param, dtypes.Float32)
	switch layout.Kind {
	case shapeinference.ActParamPerTensor, shapeinference.ActParamPerColBlock:
		return Reshape(p, 1, 1)
	case shapeinference.ActParamPerRow, shapeinference.ActParamPerRowColBlock:
		if ref2d.Shape().IsDynamic() {
			return DynamicReshape(p, NamedInferredDim(ref2d.Graph().UniqueName("qd_act_row")), StaticDim(1))
		}
		return Reshape(p, ref2d.Shape().Dimensions[0], 1)
	default:
		return Reshape(p, 1, 1)
	}
}

func expandActivationParam2D(param *Node, layout shapeinference.ActivationParamLayout, ref2d *Node, features int) *Node {
	p := ConvertDType(param, dtypes.Float32)
	switch layout.Kind {
	case shapeinference.ActParamPerTensor:
		return Reshape(p, 1, 1)
	case shapeinference.ActParamPerRow:
		if ref2d.Shape().IsDynamic() {
			return DynamicReshape(p, NamedInferredDim(ref2d.Graph().UniqueName("qd_act_p_row")), StaticDim(1))
		}
		return Reshape(p, ref2d.Shape().Dimensions[0], 1)
	case shapeinference.ActParamPerColBlock:
		p3 := Reshape(p, 1, layout.NumColBlocks, 1)
		b3 := BroadcastToDims(p3, 1, layout.NumColBlocks, layout.ColBlockSize)
		fullCols := layout.NumColBlocks * layout.ColBlockSize
		exp := Reshape(b3, 1, fullCols)
		if fullCols != features {
			exp = Slice(exp, AxisRange(), AxisRange(0, features))
		}
		return exp
	case shapeinference.ActParamPerRowColBlock:
		if ref2d.Shape().IsDynamic() {
			Panicf("nn.QuantizedDense: decomposed 2D per-row-block activation parameter does not support dynamic batch (%s)", ref2d.Shape())
		}
		M := ref2d.Shape().Dimensions[0]
		p3 := Reshape(p, M, layout.NumColBlocks, 1)
		b3 := BroadcastToDims(p3, M, layout.NumColBlocks, layout.ColBlockSize)
		fullCols := layout.NumColBlocks * layout.ColBlockSize
		exp := Reshape(b3, M, fullCols)
		if fullCols != features {
			exp = Slice(exp, AxisRange(), AxisRange(0, features))
		}
		return exp
	default:
		return Reshape(p, 1, 1)
	}
}

func zeroPointToInt32(zpFloat *Node) *Node {
	return ConvertDType(roundToEvenNode(zpFloat), dtypes.Int32)
}

func roundIfFloatZeroPoint(origZP, expandedFloat *Node) *Node {
	if origZP.DType().IsFloat() {
		return roundToEvenNode(expandedFloat)
	}
	return expandedFloat
}

func roundToEvenNode(x *Node) *Node {
	g := x.Graph()
	f := Floor(x)
	frac := Sub(x, f)
	half := Scalar(g, dtypes.Float32, float32(0.5))
	two := Scalar(g, dtypes.Float32, float32(2.0))
	one := Scalar(g, dtypes.Float32, float32(1.0))
	gtHalf := GreaterThan(frac, half)
	eqHalf := Equal(frac, half)
	isOdd := Equal(Abs(Rem(f, two)), one)
	return Where(LogicalOr(gtHalf, LogicalAnd(eqHalf, isOdd)), Add(f, one), f)
}

func applyBackendActivationConfig(x *Node, cfg compute.ActivationConfig) *Node {
	switch cfg.Type {
	case compute.ActivationNone:
		return x
	case compute.ActivationRelu:
		return activation.Relu(x)
	case compute.ActivationSigmoid:
		return activation.Apply(activation.TypeSigmoid, x)
	case compute.ActivationHardSigmoid:
		return activation.HardSigmoid(x)
	case compute.ActivationLeakyRelu:
		return activation.LeakyRelu(x)
	case compute.ActivationSelu:
		return activation.Selu(x)
	case compute.ActivationSilu:
		return activation.Swish(x)
	case compute.ActivationHardSwish:
		return activation.HardSwish(x)
	case compute.ActivationTanh:
		return activation.Apply(activation.TypeTanh, x)
	case compute.ActivationGelu:
		return activation.Gelu(x)
	case compute.ActivationGeluApproximate:
		return activation.GeluApproximate(x)
	case compute.ActivationSwiGLU:
		return activation.SwiGLU(x)
	default:
		Panicf("nn.QuantizedDense: unsupported activation %v", cfg.Type)
		return nil
	}
}

// dequantNF4FromTyped dequantizes NF4 weights (Uint4/Int4/Uint8/Int8) to [d0, d1] Float32
// using the QLoRA lookup table. Nibble indices are masked to [0..15].
func dequantNF4FromTyped(g *Graph, weights *Node, d0, d1 int) *Node {
	wDType := weights.Shape().DType
	var indices *Node
	switch wDType {
	case dtypes.Uint4:
		indices = ConvertDType(weights, dtypes.Int32)
	case dtypes.Int4, dtypes.Uint8, dtypes.Int8:
		indices = BitwiseAnd(ConvertDType(weights, dtypes.Int32), Scalar(g, dtypes.Int32, int32(0x0F)))
	default:
		Panicf("dequantNF4FromTyped: expected Int4/Uint4/Int8/Uint8 weights, got %s", wDType)
	}
	nf4Table := Const(g, compute.NF4LookupTable[:]) // [16] Float32
	indicesForGather := Reshape(indices, d0, d1, 1) // [d0, d1, 1]
	return Gather(nf4Table, indicesForGather)       // [d0, d1] Float32
}
