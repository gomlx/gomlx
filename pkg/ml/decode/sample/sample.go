// Package sample provides various sampling strategies for autoregressive generation.
//
// It also includes a BeamSearch implementation in "graph", used by decode.Decoder
// to implement beam search decoding.
package sample

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/graph"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

// RNG is an interface for random number generation used by the sampling strategies.
//
// The context.Context implements this interface.
type RNG interface {
	RandomUniform(g *Graph, shape shapes.Shape) *Node
}

// Strategy represents the different types of sampling available.
//
//go:generate go tool enumer -type=Strategy -trimprefix=Strategy -transform=snake -json -yaml -text -output=gen_strategy_enumer.go
type Strategy int

const (
	StrategyGreedy Strategy = iota
	StrategyTemperature
	StrategyTopK
	StrategyTopP
	StrategyBeamSearch // Requires specialized implementation, see sample.NewBeamSearch or simply use the decode.Decoder with StrategyBeamSearch strategy.
)

// Greedy: select max-logit token.
func Greedy(logits *Node) *Node {
	return ArgMax(logits, -1, dtypes.Int32)
}

// Temperature: scale by temperature and sample via Gumbel-ArgMax.
//
// The rng is used for random number generation, a context.Context implements this interface.
func Temperature(rng RNG, logits *Node, temperature float64) *Node {
	g := logits.Graph()

	if temperature != 1.0 {
		logits = DivScalar(logits, temperature)
	}

	uniform := rng.RandomUniform(g, logits.Shape())
	epsilon := ConstAs(uniform, 1e-10)
	uniform = Max(uniform, epsilon)
	gumbel := Neg(Log(Neg(Log(uniform))))

	noisyLogits := Add(logits, gumbel)
	return ArgMax(noisyLogits, -1, dtypes.Int32)
}

// TopP: nucleus sampling by masking tokens covering ~p mass.
//
// The rng is used for random number generation, a context.Context implements this interface.
func TopP(rng RNG, logits *Node, p float64, temperature float64) *Node {
	g := logits.Graph()

	// Probabilities
	probs := Softmax(logits)

	// Candidate ks: powers of two up to vocabSize + vocabSize
	vocabSize := logits.Shape().Dimensions[logits.Rank()-1]
	ks := []int{1}
	for ks[len(ks)-1] < vocabSize {
		next := ks[len(ks)-1] * 2
		if next > vocabSize {
			next = vocabSize
		}
		if next == ks[len(ks)-1] {
			break
		}
		ks = append(ks, next)
		if next == vocabSize {
			break
		}
	}

	// Default mask: full vocab (last candidate)
	defaultMask := TopKMask(probs, ks[len(ks)-1], -1)
	selectedMask := defaultMask

	// Threshold p as tensor
	pNode := ConstAs(ReduceSum(probs, -1), p) // shape [batch]

	// Select smallest k where cumulative top-k prob >= p
	for i := 0; i < len(ks)-1; i++ {
		k := ks[i]
		topVals, _ := graph.TopK(probs, k, -1)
		cum := ReduceSum(topVals, -1)
		cond := GreaterOrEqual(cum, pNode)
		cond = ExpandDims(cond, -1)
		condShaped := BroadcastToShape(cond, probs.Shape())
		maskK := TopKMask(probs, k, -1)
		selectedMask = Where(condShaped, maskK, selectedMask)
	}

	// Mask outside nucleus
	maskedLogits := Where(
		selectedMask,
		logits,
		Infinity(g, logits.DType(), -1),
	)

	// Temperature sampling
	return Temperature(rng, maskedLogits, temperature)
}

// TopKWithTemperature: mask non-top-k to -inf, then temperature.
//
// The rng is used for random number generation, a context.Context implements this interface.
func TopKWithTemperature(rng RNG, logits *graph.Node, k int, temperature float64) *graph.Node {
	g := logits.Graph()

	// Get mask of top-k elements
	topKMask := graph.TopKMask(logits, k, -1)

	// Set non-top-k logits to -inf so they have zero probability
	maskedLogits := graph.Where(
		topKMask,
		logits,
		graph.Infinity(g, logits.DType(), -1),
	)

	// Apply temperature sampling on the masked logits
	return Temperature(rng, maskedLogits, temperature)
}

// SampleWithStrategy dispatches to greedy|temperature|top_k|top_p.
//
// BeamSearch rquires specialized handling, and will lead to an error if used here.
func SampleWithStrategy(
	ctx RNG,
	logits *graph.Node,
	strategy Strategy,
	temperature float64,
	topK int,
	topP float64,
) *graph.Node {
	switch strategy {
	case StrategyGreedy:
		return Greedy(logits)
	case StrategyTemperature:
		return Temperature(ctx, logits, temperature)
	case StrategyTopK:
		return TopKWithTemperature(ctx, logits, topK, temperature)
	case StrategyTopP:
		return TopP(ctx, logits, topP, temperature)
	case StrategyBeamSearch:
		panic("BeamSearch requires specialized handling, see sample.NewBeamSearch or simply use the decode.Decoder with StrategyBeamSearch strategy.")
	default:
		// Default to greedy
		return Greedy(logits)
	}
}
