// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package linear

import (
	"math"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/support/exceptions"
)

const (
	// DefaultChunkSize is the default chunk size for scalar-decay and non-decay recurrences
	// (Simple Linear Attention, Mamba-2/SSD, DeltaNet, Gated DeltaNet).
	DefaultChunkSize = 64

	// DefaultChannelDecayChunkSize is the default chunk size for channel-wise decay recurrences
	// (GLA, KDA), keeping the exact overflow-free [C, C, Dk] intra-chunk factor compact.
	DefaultChannelDecayChunkSize = 16
)

// CoreOptions configures the stateless [Core] linear attention / gated delta rule computation.
type CoreOptions struct {
	// LogDecay (g) is the optional log-space decay factor (values must be <= 0; state is scaled by exp(g)).
	// Supported shapes:
	//   - nil: no decay (Simple Linear Attention or un-gated DeltaNet).
	//   - [B, S, H]: scalar decay per head (Mamba-2 / SSD or Gated DeltaNet).
	//   - [B, S, H, Dk]: channel-wise decay per key dimension (GLA or KDA).
	LogDecay *Node

	// Beta is the optional delta-rule update gate shaped [B, S, H] (typically in (0, 1)).
	//   - nil: disables the delta erasure rule (Simple Linear Attention, Mamba-2/SSD, GLA),
	//     skipping the WY/UT triangular matrix solve for ~2x lower FLOPs.
	//   - non-nil: enables the surgical delta rule S_t = decay(S_{t-1})(I - beta_t k_t k_t^T) + beta_t k_t v_t^T
	//     (DeltaNet, Gated DeltaNet, KDA).
	Beta *Node

	// InitialState is the optional initial recurrent state shaped [B, H, Dk, Dv].
	// If nil, starts from a zero state.
	InitialState *Node

	// UseQKL2Norm applies L2 normalization along head_k_dim (with eps=1e-6 in Float32) to query and key.
	UseQKL2Norm bool

	// QueryScale scales queries by this factor. If 0, defaults to 1 / sqrt(Dk).
	// Set to a negative value (e.g. -1) to disable query scaling (scale = 1.0).
	QueryScale float64

	// ChunkSize sets the chunk length along the sequence axis for chunkwise execution.
	// Must be a power of 2 >= 2 (e.g., 16, 32, 64).
	// If <= 0, defaults to DefaultChunkSize (64) for scalar/no decay and DefaultChannelDecayChunkSize (16)
	// for channel-wise decay.
	ChunkSize int

	// UnrollChunks, when true (and sequence length is static), unrolls the inter-chunk scan at
	// graph-build time instead of emitting a graph.While loop. This enables reverse-mode autodiff
	// (core/autodiff.Gradient) through Core during training.
	// Ignored when sequence length is dynamic (which always uses graph.While).
	UnrollChunks bool
}

// Core computes linear attention or the (gated) delta rule over pre-projected tensors.
//
// Inputs (BSHD layout):
//   - query: [B, S, numKHeads, Dk]
//   - key:   [B, S, numKHeads, Dk]
//   - value: [B, S, numVHeads, Dv] (numVHeads must be a positive multiple of numKHeads)
//
// All recurrence math runs in Float32 for numerical stability; the output is cast back to
// query's original dtype while finalState is returned in Float32 with shape [B, numVHeads, Dk, Dv].
func Core(query, key, value *Node, opts CoreOptions) (output, finalState *Node) {
	if query.Rank() != 4 || key.Rank() != 4 || value.Rank() != 4 {
		exceptions.Panicf("linear.Core expects rank-4 BSHD tensors for query (%s), key (%s), and value (%s)",
			query.Shape(), key.Shape(), value.Shape())
	}
	numKHeads := key.Shape().Dimensions[2]
	headKDim := key.Shape().Dimensions[3]
	numVHeads := value.Shape().Dimensions[2]
	if numKHeads <= 0 || numVHeads <= 0 || numVHeads%numKHeads != 0 {
		exceptions.Panicf("linear.Core requires numVHeads (%d) to be a positive multiple of numKHeads (%d)",
			numVHeads, numKHeads)
	}
	if opts.LogDecay != nil && opts.LogDecay.Rank() != 3 && opts.LogDecay.Rank() != 4 {
		exceptions.Panicf("linear.Core: LogDecay must be rank 3 [B,S,H] or rank 4 [B,S,H,Dk], got %s", opts.LogDecay.Shape())
	}
	if opts.Beta != nil && opts.Beta.Rank() != 3 {
		exceptions.Panicf("linear.Core: Beta must be rank 3 [B,S,H], got %s", opts.Beta.Shape())
	}

	origDType := query.DType()

	// Grouped value attention: expand Q and K heads to match numVHeads.
	if numKHeads != numVHeads {
		rep := numVHeads / numKHeads
		query = RepeatInterleaveHeads(query, rep)
		key = RepeatInterleaveHeads(key, rep)
	}

	// Promote Q, K, V, LogDecay, Beta to Float32 for numerical stability.
	qF32 := ConvertDType(query, dtypes.Float32)
	kF32 := ConvertDType(key, dtypes.Float32)
	vF32 := ConvertDType(value, dtypes.Float32)
	if opts.UseQKL2Norm {
		qF32 = L2NormLastAxis(qF32, 1e-6)
		kF32 = L2NormLastAxis(kF32, 1e-6)
	}
	scale := opts.QueryScale
	if scale == 0 {
		scale = 1.0 / math.Sqrt(float64(headKDim))
	}
	if scale > 0 && scale != 1.0 {
		qF32 = MulScalar(qF32, scale)
	}

	var gDecay, beta *Node
	if opts.LogDecay != nil {
		gDecay = ConvertDType(opts.LogDecay, dtypes.Float32)
	}
	if opts.Beta != nil {
		beta = ConvertDType(opts.Beta, dtypes.Float32)
	}

	var outF32 *Node
	if query.Shape().Dimensions[1] == 1 {
		outF32, finalState = coreSingleStep(qF32, kF32, vF32, gDecay, beta, opts.InitialState)
	} else {
		chunkSize := opts.ChunkSize
		if chunkSize <= 0 {
			if gDecay != nil && gDecay.Rank() == 4 {
				chunkSize = DefaultChannelDecayChunkSize
			} else {
				chunkSize = DefaultChunkSize
			}
		}
		if chunkSize < 2 || (chunkSize&(chunkSize-1)) != 0 {
			exceptions.Panicf("linear.Core: ChunkSize must be a power of 2 >= 2, got %d", chunkSize)
		}
		outF32, finalState = coreChunked(qF32, kF32, vF32, gDecay, beta, opts.InitialState, chunkSize, opts.UnrollChunks)
	}
	return ConvertDType(outF32, origDType), finalState
}

// coreSingleStep executes a single recurrent step for static seqLen == 1.
// All inputs are Float32:
//   - q, k: [B, 1, H, Dk], v: [B, 1, H, Dv]
//   - gDecay: nil, [B, 1, H], or [B, 1, H, Dk]
//   - beta: nil or [B, 1, H]
//   - initialState: nil or [B, H, Dk, Dv]
func coreSingleStep(q, k, v, gDecay, beta, initialState *Node) (out, newState *Node) {
	g := q.Graph()
	batchSpec := DimensionSpecFor(q, 0)
	numHeads := q.Shape().Dimensions[2]
	headKDim := q.Shape().Dimensions[3]
	headVDim := v.Shape().Dimensions[3]

	var state *Node
	if initialState == nil {
		state = DynamicBroadcastInDim(ScalarZero(g, dtypes.Float32), nil,
			batchSpec, StaticDim(numHeads), StaticDim(headKDim), StaticDim(headVDim))
	} else {
		state = ConvertDType(initialState, dtypes.Float32)
	}

	q0 := Squeeze(q, 1) // [B, H, Dk]
	k0 := Squeeze(k, 1) // [B, H, Dk]
	v0 := Squeeze(v, 1) // [B, H, Dv]

	// 1. Apply decay if configured.
	if gDecay != nil {
		var decay0 *Node
		if gDecay.Rank() == 3 {
			decay0 = ExpandAxes(Exp(Squeeze(gDecay, 1)), -2, -1) // [B, H, 1, 1]
		} else {
			decay0 = ExpandDims(Exp(Squeeze(gDecay, 1)), -1) // [B, H, Dk, 1]
		}
		state = Mul(state, decay0)
	}

	// 2. Apply delta update (if beta != nil) or pure associative write (if beta == nil).
	if beta != nil {
		beta0 := ExpandDims(Squeeze(beta, 1), -1)                                    // [B, H, 1]
		kvMem := DotGeneral(k0, []int{2}, []int{0, 1}, state, []int{2}, []int{0, 1}) // [B, H, Dv]
		delta := Mul(Sub(v0, kvMem), beta0)                                          // [B, H, Dv]
		state = Add(state, Mul(ExpandDims(k0, -1), ExpandDims(delta, -2)))           // [B, H, Dk, Dv]
	} else {
		state = Add(state, Mul(ExpandDims(k0, -1), ExpandDims(v0, -2))) // [B, H, Dk, Dv]
	}

	// 3. Read output along q.
	out0 := DotGeneral(q0, []int{2}, []int{0, 1}, state, []int{2}, []int{0, 1}) // [B, H, Dv]
	return ExpandDims(out0, 1), state
}

// coreChunked implements the unified chunkwise parallel scan for all 6 linear-attention variants:
//   - gDecay == nil / [B,S,H] (scalar) / [B,S,H,Dk] (channel-wise)
//   - beta == nil (no delta erasure, skips triangular solve) / [B,S,H] (delta rule via WY/UT transform)
func coreChunked(query, key, value, gDecay, beta, initialState *Node, chunkSize int, unrollChunks bool) (out, finalState *Node) {
	g := query.Graph()
	batchSpec := DimensionSpecFor(query, 0)
	seqDim := query.Shape().Dimensions[1]
	seqAxisName := query.Shape().AxisName(1)
	numHeads := query.Shape().Dimensions[2]
	headKDim := query.Shape().Dimensions[3]
	headVDim := value.Shape().Dimensions[3]

	hasDecay := gDecay != nil
	channelDecay := hasDecay && gDecay.Rank() == 4
	useDelta := beta != nil

	// Transpose to [B, H, S, ...] layout.
	q := TransposeAllDims(query, 0, 2, 1, 3)
	k := TransposeAllDims(key, 0, 2, 1, 3)
	v := TransposeAllDims(value, 0, 2, 1, 3)
	var bT, dT *Node
	if useDelta {
		bT = TransposeAllDims(beta, 0, 2, 1) // [B, H, S]
	}
	if hasDecay {
		if channelDecay {
			dT = TransposeAllDims(gDecay, 0, 2, 1, 3) // [B, H, S, Dk]
		} else {
			dT = TransposeAllDims(gDecay, 0, 2, 1) // [B, H, S]
		}
	}

	// Pad sequence axis (axis 2) to a multiple of chunkSize with zeros.
	// Zero-padded tokens have k=0, v=0, beta=0, decay=0 (exp(0)=1), leaving finalState unchanged.
	zeroF32 := ScalarZero(g, dtypes.Float32)
	var padSizeNode *Node
	var staticPadSize int
	if seqDim != shapes.DynamicDim {
		staticPadSize = (chunkSize - seqDim%chunkSize) % chunkSize
		if staticPadSize > 0 {
			pad4 := []PadAxis{{}, {}, {End: staticPadSize}, {}}
			pad3 := []PadAxis{{}, {}, {End: staticPadSize}}
			q = Pad(q, zeroF32, pad4...)
			k = Pad(k, zeroF32, pad4...)
			v = Pad(v, zeroF32, pad4...)
			if useDelta {
				bT = Pad(bT, zeroF32, pad3...)
			}
			if hasDecay {
				if channelDecay {
					dT = Pad(dT, zeroF32, pad4...)
				} else {
					dT = Pad(dT, zeroF32, pad3...)
				}
			}
		}
	} else {
		seqLenDyn := DimensionSize(q, 2)
		cNode := Scalar(g, seqLenDyn.DType(), chunkSize)
		padSizeNode = Rem(Sub(cNode, Rem(seqLenDyn, cNode)), cNode)
		paddedAxisName := shapes.UniqueAxisName("linattn_padded_seq")
		pad4D := []DynamicPadAxisSpec{
			DynamicPadAxis(0, 0, 0),
			DynamicPadAxis(0, 0, 0),
			{EndNode: padSizeNode, TargetAxisName: paddedAxisName},
			DynamicPadAxis(0, 0, 0),
		}
		pad3D := []DynamicPadAxisSpec{
			DynamicPadAxis(0, 0, 0),
			DynamicPadAxis(0, 0, 0),
			{EndNode: padSizeNode, TargetAxisName: paddedAxisName},
		}
		q = DynamicPad(q, zeroF32, pad4D...)
		k = DynamicPad(k, zeroF32, pad4D...)
		v = DynamicPad(v, zeroF32, pad4D...)
		if useDelta {
			bT = DynamicPad(bT, zeroF32, pad3D...)
		}
		if hasDecay {
			if channelDecay {
				dT = DynamicPad(dT, zeroF32, pad4D...)
			} else {
				dT = DynamicPad(dT, zeroF32, pad3D...)
			}
		}
	}

	// Reshape into chunks: [B, H, numChunks, C, ...].
	// Share one numChunksSpec across all tensors so dynamic axis names unify identically.
	q = DynamicReshape(q, batchSpec, StaticDim(numHeads), InferredDim(), StaticDim(chunkSize), StaticDim(headKDim))
	numChunksSpec := DimensionSpecFor(q, 2)
	k = DynamicReshape(k, batchSpec, StaticDim(numHeads), numChunksSpec, StaticDim(chunkSize), StaticDim(headKDim))
	v = DynamicReshape(v, batchSpec, StaticDim(numHeads), numChunksSpec, StaticDim(chunkSize), StaticDim(headVDim))

	var kBeta, vBeta *Node
	if useDelta {
		bT = DynamicReshape(bT, batchSpec, StaticDim(numHeads), numChunksSpec, StaticDim(chunkSize))
		bExp := ExpandDims(bT, -1) // [B, H, N, C, 1]
		kBeta = Mul(k, bExp)
		vBeta = Mul(v, bExp)
	} else {
		vBeta = v
	}

	// Cumulative decay and intra-chunk attention / UT matrices.
	causalMask := makeLowerTriangularMask(g, chunkSize, 0)         // j <= i, [1, 1, 1, C, C]
	strictlyUpperMask := makeUpperTriangularMask(g, chunkSize, 1)  // j > i,  [1, 1, 1, C, C]
	strictlyLowerMask := makeLowerTriangularMask(g, chunkSize, -1) // j < i,  [1, 1, 1, C, C]
	negInf := Infinity(g, dtypes.Float32, -1)

	var (
		intraChunkAttn *Node // [B, H, N, C, C]
		newValues      *Node // [B, H, N, C, Dv]
		kCumDecay      *Node // [B, H, N, C, Dk] (only non-nil when useDelta)
		qDecayed       *Node // [B, H, N, C, Dk]
		kDecayed       *Node // [B, H, N, C, Dk]
		chunkDecay     *Node // nil (no decay), [B, H, N, 1, 1] (scalar), or [B, H, N, Dk, 1] (channel-wise)
	)

	if !hasDecay {
		// No decay (Simple Linear Attention or un-gated DeltaNet).
		rawQK := DotGeneral(q, []int{4}, []int{0, 1, 2}, k, []int{4}, []int{0, 1, 2})
		intraChunkAttn = Where(BroadcastLike(causalMask, rawQK), rawQK, ZerosLike(rawQK))
		qDecayed = q
		kDecayed = k
		if useDelta {
			rawKK := DotGeneral(kBeta, []int{4}, []int{0, 1, 2}, k, []int{4}, []int{0, 1, 2})
			utLower := Where(BroadcastLike(strictlyLowerMask, rawKK), rawKK, ZerosLike(rawKK))
			utInv := invertUnitLowerTriangular(utLower)
			newValues = DotGeneral(utInv, []int{4}, []int{0, 1, 2}, vBeta, []int{3}, []int{0, 1, 2})
			kCumDecay = DotGeneral(utInv, []int{4}, []int{0, 1, 2}, kBeta, []int{3}, []int{0, 1, 2})
		} else {
			newValues = vBeta
		}
	} else if !channelDecay {
		// Scalar per-head log-decay (Mamba-2 / SSD or Gated DeltaNet).
		dT = DynamicReshape(dT, batchSpec, StaticDim(numHeads), numChunksSpec, StaticDim(chunkSize))
		cumDecay := CumSum(dT, 3) // [B, H, N, C]

		pairwiseDiff := Sub(ExpandDims(cumDecay, 4), ExpandDims(cumDecay, 3)) // [B, H, N, C, C]
		pairwiseDiff = Where(BroadcastLike(strictlyUpperMask, pairwiseDiff), BroadcastLike(negInf, pairwiseDiff), pairwiseDiff)
		pairwiseDecay := Exp(pairwiseDiff)

		intraChunkAttn = Mul(DotGeneral(q, []int{4}, []int{0, 1, 2}, k, []int{4}, []int{0, 1, 2}), pairwiseDecay)

		expCumDecay := Exp(cumDecay)                                                             // [B, H, N, C]
		cumDecayLast := Slice(cumDecay, AxisRange().Spacer(), AxisRange(chunkSize-1, chunkSize)) // [B, H, N, 1]
		qDecayed = Mul(q, ExpandDims(expCumDecay, -1))
		kDecayed = Mul(k, ExpandDims(Exp(Sub(cumDecayLast, cumDecay)), -1))
		chunkDecay = ExpandDims(Exp(cumDecayLast), -1) // [B, H, N, 1, 1]

		if useDelta {
			utSystem := Mul(DotGeneral(kBeta, []int{4}, []int{0, 1, 2}, k, []int{4}, []int{0, 1, 2}), pairwiseDecay)
			utLower := Where(BroadcastLike(strictlyLowerMask, utSystem), utSystem, ZerosLike(utSystem))
			utInv := invertUnitLowerTriangular(utLower)
			decayedKBeta := Mul(kBeta, ExpandDims(expCumDecay, -1))
			newValues = DotGeneral(utInv, []int{4}, []int{0, 1, 2}, vBeta, []int{3}, []int{0, 1, 2})
			kCumDecay = DotGeneral(utInv, []int{4}, []int{0, 1, 2}, decayedKBeta, []int{3}, []int{0, 1, 2})
		} else {
			newValues = vBeta
		}
	} else {
		// Channel-wise log-decay per key dim (GLA or KDA): dT is [B, H, N, C, Dk].
		// Compute exact overflow-free pairwise decay exp(G_i - G_j) for j <= i (guaranteed <= 0).
		dT = DynamicReshape(dT, batchSpec, StaticDim(numHeads), numChunksSpec, StaticDim(chunkSize), StaticDim(headKDim))
		cumDecay := CumSum(dT, 3) // [B, H, N, C, Dk]

		// pairwiseDiff: [B, H, N, C_i, C_j, Dk]
		pairwiseDiff := Sub(ExpandDims(cumDecay, 4), ExpandDims(cumDecay, 3))
		upper6D := ExpandDims(strictlyUpperMask, -1) // [1, 1, 1, C, C, 1]
		pairwiseDiff = Where(BroadcastLike(upper6D, pairwiseDiff), BroadcastLike(negInf, pairwiseDiff), pairwiseDiff)
		pairwiseDecay := Exp(pairwiseDiff) // [B, H, N, C, C, Dk], values in [0, 1]

		// intraChunkAttn[i, j] = sum_d q[i, d] * k[j, d] * exp(G[i, d] - G[j, d])
		qkPair := Mul(ExpandDims(q, 4), ExpandDims(k, 3)) // [B, H, N, C, C, Dk]
		intraChunkAttn = ReduceSum(Mul(qkPair, pairwiseDecay), 5)

		expCumDecay := Exp(cumDecay)                                                                           // [B, H, N, C, Dk]
		cumDecayLast := Slice(cumDecay, AxisRange(), AxisRange(), AxisRange(), AxisRange(chunkSize-1, chunkSize), AxisRange()) // [B, H, N, 1, Dk]
		qDecayed = Mul(q, expCumDecay)
		kDecayed = Mul(k, Exp(Sub(cumDecayLast, cumDecay)))
		// chunkDecay: [B, H, N, Dk, 1] to broadcast against state [B, H, Dk, Dv].
		chunkDecay = TransposeAllDims(Exp(cumDecayLast), 0, 1, 2, 4, 3)

		if useDelta {
			kkPair := Mul(ExpandDims(kBeta, 4), ExpandDims(k, 3)) // [B, H, N, C, C, Dk]
			utSystem := ReduceSum(Mul(kkPair, pairwiseDecay), 5)  // [B, H, N, C, C]
			utLower := Where(BroadcastLike(strictlyLowerMask, utSystem), utSystem, ZerosLike(utSystem))
			utInv := invertUnitLowerTriangular(utLower)
			decayedKBeta := Mul(kBeta, expCumDecay)
			newValues = DotGeneral(utInv, []int{4}, []int{0, 1, 2}, vBeta, []int{3}, []int{0, 1, 2})
			kCumDecay = DotGeneral(utInv, []int{4}, []int{0, 1, 2}, decayedKBeta, []int{3}, []int{0, 1, 2})
		} else {
			newValues = vBeta
		}
	}

	// Initial recurrent state: [B, H, Dk, Dv].
	var initState *Node
	if initialState == nil {
		initState = DynamicBroadcastInDim(zeroF32, nil,
			batchSpec, StaticDim(numHeads), StaticDim(headKDim), StaticDim(headVDim))
	} else {
		initState = ConvertDType(initialState, dtypes.Float32)
	}

	// chunkStep advances one chunk given per-chunk tensors [B, H, ...] and current state [B, H, Dk, Dv].
	chunkStep := func(st, nv, kc, qc, kd, ia, cd *Node) (chunkOut, newSt *Node) {
		vNew := nv
		if kc != nil {
			vNew = Sub(nv, DotGeneral(kc, []int{3}, []int{0, 1}, st, []int{2}, []int{0, 1}))
		}
		interAttn := DotGeneral(qc, []int{3}, []int{0, 1}, st, []int{2}, []int{0, 1})
		chunkOut = Add(interAttn, DotGeneral(ia, []int{3}, []int{0, 1}, vNew, []int{2}, []int{0, 1}))
		decayedSt := st
		if cd != nil {
			decayedSt = Mul(st, cd)
		}
		newSt = Add(decayedSt, DotGeneral(kd, []int{2}, []int{0, 1}, vNew, []int{2}, []int{0, 1}))
		return chunkOut, newSt
	}

	// Inter-chunk sequential scan: unroll when static & (numChunks == 1 or unrollChunks), else While loop.
	var coreOutChunks *Node // [B, H, numChunks, C, Dv]
	numChunksStatic := q.Shape().Dimensions[2]
	if numChunksStatic != shapes.DynamicDim && (numChunksStatic == 1 || unrollChunks) {
		st := initState
		outChunks := make([]*Node, numChunksStatic)
		for i := range numChunksStatic {
			nv := Squeeze(Slice(newValues, AxisRange(), AxisRange(), AxisElem(i), AxisRange(), AxisRange()), 2)
			var kc *Node
			if kCumDecay != nil {
				kc = Squeeze(Slice(kCumDecay, AxisRange(), AxisRange(), AxisElem(i), AxisRange(), AxisRange()), 2)
			}
			qc := Squeeze(Slice(qDecayed, AxisRange(), AxisRange(), AxisElem(i), AxisRange(), AxisRange()), 2)
			kd := Squeeze(Slice(kDecayed, AxisRange(), AxisRange(), AxisElem(i), AxisRange(), AxisRange()), 2)
			ia := Squeeze(Slice(intraChunkAttn, AxisRange(), AxisRange(), AxisElem(i), AxisRange(), AxisRange()), 2)
			var cd *Node
			if chunkDecay != nil {
				cd = Squeeze(Slice(chunkDecay, AxisRange(), AxisRange(), AxisElem(i), AxisRange(), AxisRange()), 2)
			}
			outChunks[i], st = chunkStep(st, nv, kc, qc, kd, ia, cd)
		}
		finalState = st
		coreOutChunks = Stack(outChunks, 2) // [B, H, numChunks, C, Dv]
	} else {
		// Move numChunks to axis 0 so each While iteration performs contiguous Gather & DynamicUpdateSlice.
		nv0 := TransposeAllDims(newValues, 2, 0, 1, 3, 4)
		var kc0 *Node
		if kCumDecay != nil {
			kc0 = TransposeAllDims(kCumDecay, 2, 0, 1, 3, 4)
		}
		qc0 := TransposeAllDims(qDecayed, 2, 0, 1, 3, 4)
		kd0 := TransposeAllDims(kDecayed, 2, 0, 1, 3, 4)
		ia0 := TransposeAllDims(intraChunkAttn, 2, 0, 1, 3, 4)
		var cd0 *Node
		if chunkDecay != nil {
			cd0 = TransposeAllDims(chunkDecay, 2, 0, 1, 3, 4)
		}

		outInit := ZerosLike(nv0) // [numChunks, B, H, C, Dv]
		numChunksNode := ConvertDType(DimensionSize(nv0, 0), dtypes.Int32)

		condFn := NewClosure(g, func(g *Graph) []*Node {
			idx := Parameter(g, "idx", shapes.Scalar[int32]())
			_ = Parameter(g, "state", initState.Shape())
			_ = Parameter(g, "out", outInit.Shape())
			return []*Node{LessThan(idx, numChunksNode)}
		})

		bodyFn := NewClosure(g, func(g *Graph) []*Node {
			idx := Parameter(g, "idx", shapes.Scalar[int32]())
			st := Parameter(g, "state", initState.Shape())
			outBuf := Parameter(g, "out", outInit.Shape())

			nv := Gather(nv0, idx)
			var kc *Node
			if kc0 != nil {
				kc = Gather(kc0, idx)
			}
			qc := Gather(qc0, idx)
			kd := Gather(kd0, idx)
			ia := Gather(ia0, idx)
			var cd *Node
			if cd0 != nil {
				cd = Gather(cd0, idx)
			}
			chunkOut, newSt := chunkStep(st, nv, kc, qc, kd, ia, cd)

			zero := Const(g, int32(0))
			newOut := DynamicUpdateSlice(outBuf, ExpandDims(chunkOut, 0), []*Node{idx, zero, zero, zero, zero})
			return []*Node{Add(idx, Const(g, int32(1))), newSt, newOut}
		})

		results := While(condFn, bodyFn, Const(g, int32(0)), initState, outInit)
		finalState = results[1]
		coreOutChunks = TransposeAllDims(results[2], 1, 2, 0, 3, 4) // [B, H, numChunks, C, Dv]
	}

	// Flatten chunk dimension back to [B, H, totalSeq, Dv], drop padding to [B, H, S, Dv], and transpose to [B, S, H, Dv].
	coreOutFlat := DynamicReshape(coreOutChunks, batchSpec, StaticDim(numHeads), InferredDim(), StaticDim(headVDim))
	if seqDim != shapes.DynamicDim {
		if staticPadSize > 0 {
			coreOutFlat = Slice(coreOutFlat, AxisRange(), AxisRange(), AxisRange(0, seqDim), AxisRange())
		}
	} else {
		coreOutFlat = DynamicPad(coreOutFlat, zeroF32,
			DynamicPadAxis(0, 0, 0),
			DynamicPadAxis(0, 0, 0),
			DynamicPadAxisSpec{EndNode: Neg(padSizeNode), TargetAxisName: seqAxisName},
			DynamicPadAxis(0, 0, 0),
		)
	}
	return TransposeAllDims(coreOutFlat, 0, 2, 1, 3), finalState
}

// invertUnitLowerTriangular computes the exact inverse of (I + L) where L is a strictly
// lower-triangular matrix of shape [..., C, C] (with C a power of 2 >= 2) using a numerically
// stable block-recursive Schur complement decomposition:
//   - For base blocks up to size 16, applies dyadic block-doubling D_{k+1} = D_k - D_k @ L_k @ D_k.
//   - For C > 16 (e.g., 32, 64, 128), inverts all C/16 diagonal 16x16 blocks in parallel and merges
//     adjacent blocks bottom-up via LL = -B_inv @ C_blk @ A_inv.
func invertUnitLowerTriangular(L *Node) *Node {
	g := L.Graph()
	rank := L.Rank()
	c := L.Shape().Dimensions[rank-1]
	if c < 2 || (c&(c-1)) != 0 || L.Shape().Dimensions[rank-2] != c {
		exceptions.Panicf("invertUnitLowerTriangular expects square power-of-2 last two dims (>= 2), got %s", L.Shape())
	}

	// Helper: invert all blocks of shape [..., base, base] via dyadic block-doubling (1 -> 2 -> ... -> base).
	invertDyadic := func(LBlk *Node, base int) *Node {
		r := LBlk.Rank()
		eye := makeIdentityMatrix(g, base, r)
		D := Sub(BroadcastLike(eye, LBlk), Mul(LBlk, makeDyadicLowerBlockMask(g, base, 1, r)))
		batchAxes := make([]int, r-2)
		for i := range batchAxes {
			batchAxes[i] = i
		}
		for m := 2; m < base; m *= 2 {
			Lm := Mul(LBlk, makeDyadicLowerBlockMask(g, base, m, r))
			DL := DotGeneral(D, []int{r - 1}, batchAxes, Lm, []int{r - 2}, batchAxes)
			DLD := DotGeneral(DL, []int{r - 1}, batchAxes, D, []int{r - 2}, batchAxes)
			D = Sub(D, DLD)
		}
		return D
	}

	if c <= 16 {
		return invertDyadic(L, c)
	}

	// Step 1: Extract numBlocks = c/16 diagonal 16x16 blocks and invert them in parallel.
	const base = 16
	numBlocks := c / base
	diagBlocks := make([]*Node, numBlocks)
	for b := range numBlocks {
		s, e := b*base, (b+1)*base
		diagBlocks[b] = Slice(L, AxisRange().Spacer(), AxisRange(s, e), AxisRange(s, e))
	}
	invBlocks := invertDyadic(Stack(diagBlocks, rank-2), base) // [..., numBlocks, base, base]

	// Step 2: Bottom-up Schur complement merge: size m -> 2*m until a single c x c block remains.
	blockAxis := rank - 2
	for m := base; m < c; m *= 2 {
		pairs := c / (2 * m)
		aList := make([]*Node, pairs)
		bList := make([]*Node, pairs)
		cList := make([]*Node, pairs)
		for p := range pairs {
			aList[p] = Slice(invBlocks, AxisRange().Spacer(), AxisElem(2*p), AxisRange(), AxisRange())
			bList[p] = Slice(invBlocks, AxisRange().Spacer(), AxisElem(2*p+1), AxisRange(), AxisRange())
			rStart := (2*p + 1) * m
			cStart := (2 * p) * m
			cList[p] = Slice(L, AxisRange().Spacer(), AxisRange(rStart, rStart+m), AxisRange(cStart, cStart+m))
		}
		if pairs == 1 {
			A := Squeeze(aList[0], blockAxis)
			B := Squeeze(bList[0], blockAxis)
			C := cList[0]
			batchAxes := make([]int, rank-2)
			for i := range batchAxes {
				batchAxes[i] = i
			}
			BC := DotGeneral(B, []int{rank - 1}, batchAxes, C, []int{rank - 2}, batchAxes)
			LL := Neg(DotGeneral(BC, []int{rank - 1}, batchAxes, A, []int{rank - 2}, batchAxes))
			top := Concatenate([]*Node{A, ZerosLike(A)}, rank-1)
			bot := Concatenate([]*Node{LL, B}, rank-1)
			return Concatenate([]*Node{top, bot}, rank-2)
		}
		A := Concatenate(aList, blockAxis) // [..., pairs, m, m]
		B := Concatenate(bList, blockAxis) // [..., pairs, m, m]
		C := Stack(cList, blockAxis)       // [..., pairs, m, m]
		batchAxes := make([]int, rank-1)
		for i := range batchAxes {
			batchAxes[i] = i
		}
		BC := DotGeneral(B, []int{rank}, batchAxes, C, []int{rank - 1}, batchAxes)
		LL := Neg(DotGeneral(BC, []int{rank}, batchAxes, A, []int{rank - 1}, batchAxes))
		top := Concatenate([]*Node{A, ZerosLike(A)}, rank)
		bot := Concatenate([]*Node{LL, B}, rank)
		invBlocks = Concatenate([]*Node{top, bot}, rank-1) // [..., pairs, 2*m, 2*m]
	}
	return Squeeze(invBlocks, blockAxis)
}

func makeUpperTriangularMask(g *Graph, n, diagonalOffset int) *Node {
	data := make([]bool, n*n)
	for i := range n {
		for j := range n {
			if j >= i+diagonalOffset {
				data[i*n+j] = true
			}
		}
	}
	return Reshape(Const(g, data), 1, 1, 1, n, n)
}

func makeLowerTriangularMask(g *Graph, n, diagonalOffset int) *Node {
	data := make([]bool, n*n)
	for i := range n {
		for j := range n {
			if j <= i+diagonalOffset {
				data[i*n+j] = true
			}
		}
	}
	return Reshape(Const(g, data), 1, 1, 1, n, n)
}

func makeIdentityMatrix(g *Graph, n, targetRank int) *Node {
	data := make([]float32, n*n)
	for i := range n {
		data[i*n+i] = 1.0
	}
	dims := make([]int, targetRank)
	for i := range targetRank - 2 {
		dims[i] = 1
	}
	dims[targetRank-2] = n
	dims[targetRank-1] = n
	return Reshape(Const(g, data), dims...)
}

// makeDyadicLowerBlockMask creates a [1,...,1, n, n] float32 mask with 1.0 at the lower-left
// m x m sub-blocks of each (2m) x (2m) diagonal block, and 0.0 elsewhere.
func makeDyadicLowerBlockMask(g *Graph, n, m, targetRank int) *Node {
	data := make([]float32, n*n)
	blockSize := 2 * m
	for b := 0; b < n; b += blockSize {
		for r := b + m; r < b+blockSize; r++ {
			for c := b; c < b+m; c++ {
				data[r*n+c] = 1.0
			}
		}
	}
	dims := make([]int, targetRank)
	for i := range targetRank - 2 {
		dims[i] = 1
	}
	dims[targetRank-2] = n
	dims[targetRank-1] = n
	return Reshape(Const(g, data), dims...)
}
