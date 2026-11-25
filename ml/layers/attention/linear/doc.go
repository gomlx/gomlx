// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package linear implements linear attention, state-space duality (Mamba-2 / SSD),
// Gated Linear Attention (GLA), DeltaNet, Gated DeltaNet, and Kimi Delta Attention (KDA)
// as both stateless core primitives ([Core]) and configurable stateful layer builders ([New]).
//
// # Mathematical Unification
//
// Unlike standard softmax self-attention (whose KV cache grows as O(S) with sequence length S
// and whose prefill cost scales as O(S²)), linear attention and modern recurrent sequence mixers
// maintain a fixed-size associative memory matrix S_t ∈ R^{d_k × d_v} per head, giving O(S)
// prefill complexity and O(1) per-token decode cost and memory.
//
// Across the modern literature, these architectures share a single unified recurrent state update:
//
//	S_t = Decay_t(S_{t-1}) · (I - β_t k_t k_tᵀ) + β_t k_t v_tᵀ
//	y_t = q_tᵀ S_t
//
// where:
//   - Decay_t(S) = diag(exp(g_t)) · S applies multiplicative forgetting (with log-decay g_t ≤ 0),
//     allowing rapid bulk erasure of stale context.
//   - (I - β_t k_t k_tᵀ) is the delta-rule surgical erasure (a generalized Householder reflection
//     for β_t ∈ (0, 1) and L2-normalized k_t), clearing only the memory subspace along k_t before
//     writing the new value v_t so associative memories do not collide or saturate.
//
// Different published architectures correspond to concrete choices on two orthogonal axes
// (DecayMode and whether the Delta Rule is enabled):
//
//	| Architecture          | Decay (g_t)                   | Delta Rule (β_t) | Triangular Solve |
//	|-----------------------|-------------------------------|------------------|------------------|
//	| Simple Linear Attn    | None (g_t = 0)                | Disabled         | Skipped          |
//	| Mamba-2 / SSD         | Scalar per head [B, S, H]     | Disabled         | Skipped          |
//	| GLA                   | Channel-wise [B, S, H, d_k]   | Disabled         | Skipped          |
//	| DeltaNet              | None (g_t = 0)                | Enabled [B,S,H]  | Required (WY/UT) |
//	| Gated DeltaNet        | Scalar per head [B, S, H]     | Enabled [B,S,H]  | Required (WY/UT) |
//	| Kimi Delta Attn (KDA) | Channel-wise [B, S, H, d_k]   | Enabled [B,S,H]  | Required (WY/UT) |
//
// # Execution Algorithms
//
// [Core] automatically dispatches between three mathematically equivalent execution paths:
//
//  1. Single-step direct recurrence (static seqLen == 1): executes in a handful of tensor ops
//     with zero padding, zero chunking, and no loop — ideal for autoregressive token-by-token decode.
//  2. Chunkwise parallel scan with graph.While (default for inference and dynamic seqLen):
//     splits the sequence into chunks of size C (default 64 for scalar/no decay, 16 for channel-wise
//     decay), computes intra-chunk interactions in parallel via GEMMs (solving the WY/UT unit
//     lower-triangular system via an exact, numerically stable block-recursive Schur complement
//     when Delta Rule is active), and scans sequentially across only ceil(S/C) chunks. Works with
//     both static and dynamic sequence lengths across all backends (Go, XLA CPU/CUDA, ONNX, MPS/MLX).
//  3. Unrolled chunkwise scan (UnrollChunks = true, enabled automatically during training for
//     static sequence lengths): unrolls the inter-chunk scan at graph-build time so GoMLX's
//     reverse-mode autodiff (core/autodiff) can differentiate end-to-end without requiring a
//     While-loop VJP.
//
// # Two-Tier API
//
// Mirroring package [github.com/gomlx/gomlx/ml/layers/attention]:
//   - Tier 1 (Stateless): [Core], [CausalDepthwiseConv1D], [RMSNormGated], [L2NormLastAxis] operate
//     directly on pre-projected tensors with no trainable variables or scope-naming assumptions.
//   - Tier 2 (Stateful Builder): [New] (defaulting to Gated DeltaNet) and presets [GatedDeltaNet],
//     [DeltaNet], [GLA], [Simple] manage input projections, optional causal depthwise 1D convolution,
//     decay/beta parameters, QK L2-normalization, gated RMSNorm, and output projection inside a
//     [github.com/gomlx/gomlx/ml/model.Scope].
//
// # Example Usage
//
// Training or inference with Gated DeltaNet (default):
//
//	y := linear.GatedDeltaNet(scope.In("linear_attn"), x, numHeads, headDim).
//		WithShortConv(4).
//		Done()
//
// Incremental cached decoding (e.g., inside a hybrid transformer such as Qwen3.5):
//
//	y, nextConvState, nextRecState := linear.GatedDeltaNet(scope.In("linear_attn"), x, numVHeads, headDim).
//		WithNumKeyHeads(numKHeads).
//		WithState(convState, recState).
//		UseTransposedWeights(true).
//		DoneWithState()
//
// # References & Further Reading
//
//   - Explainer article: "Gated DeltaNet: The Surgical Eraser Solving Linear Attention's Memory Problem",
//     Towards AI:
//     https://pub.towardsai.net/gated-deltanet-the-surgical-eraser-solving-linear-attentions-memory-problem-1e50ca3e42ab
//   - Gated DeltaNet: Songlin Yang, Jan Kautz, Ali Hatamizadeh,
//     "Gated Delta Networks: Improving Mamba2 with Delta Rule" (ICLR 2025):
//     https://arxiv.org/abs/2412.06464
//   - Parallelizing DeltaNet: Songlin Yang, Bailin Wang, Yu Zhang, Yikang Shen, Yoon Kim,
//     "Parallelizing Linear Transformers with the Delta Rule over Sequence Length" (NeurIPS 2024):
//     https://arxiv.org/abs/2406.06484
//   - Original DeltaNet: Imanol Schlag, Kazuki Irie, Jürgen Schmidhuber,
//     "Linear Transformers Are Secretly Fast Weight Programmers" (ICML 2021):
//     https://arxiv.org/abs/2102.11174
//   - Mamba-2 (State Space Duality): Tri Dao, Albert Gu,
//     "Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality" (ICML 2024):
//     https://arxiv.org/abs/2405.21060
//   - Gated Linear Attention (GLA): Songlin Yang, Bailin Wang, Yikang Shen, Rameswar Panda, Yoon Kim,
//     "Gated Linear Attention Transformers with Hardware-Efficient Training" (ICML 2024):
//     https://arxiv.org/abs/2312.06635
//   - Linear Attention: Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, François Fleuret,
//     "Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention" (ICML 2020):
//     https://arxiv.org/abs/2006.16236
//   - RetNet: Yutao Sun et al.,
//     "Retentive Network: A Successor to Transformer for Large Language Models" (2023):
//     https://arxiv.org/abs/2307.08621
//   - Kimi Linear / Kimi Delta Attention (KDA): Moonshot AI,
//     "Kimi Linear: An Expressive, Efficient Attention Architecture" (2025):
//     https://arxiv.org/abs/2510.26692
package linear
