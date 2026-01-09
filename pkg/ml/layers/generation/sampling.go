package generation

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gopjrt/dtypes"
)

// GreedySample: select max-logit token.
func GreedySample(ctx *context.Context, logits *Node) *Node {
	return ArgMax(logits, -1, dtypes.Int32)
}

// TemperatureSample: scale by temperature and sample via Gumbel-ArgMax.
func TemperatureSample(ctx *context.Context, logits *Node, temperature float64) *Node {
	g := logits.Graph()

	if temperature != 1.0 {
		logits = DivScalar(logits, temperature)
	}

	uniform := ctx.RandomUniform(g, logits.Shape())
	epsilon := ConstAs(uniform, 1e-10)
	uniform = Max(uniform, epsilon)
	gumbel := Neg(Log(Neg(Log(uniform))))

	noisyLogits := Add(logits, gumbel)
	return ArgMax(noisyLogits, -1, dtypes.Int32)
}

// TopKSample: mask non-top-k to -inf, then temperature.
func TopKSample(ctx *context.Context, logits *Node, k int, temperature float64) *Node {
	g := logits.Graph()

	// Get mask of top-k elements
	topKMask := TopKMask(logits, k)

	// Set non-top-k logits to -inf so they have zero probability
	maskedLogits := Where(
		topKMask,
		logits,
		Infinity(g, logits.DType(), -1),
	)

	// Apply temperature sampling on the masked logits
	return TemperatureSample(ctx, maskedLogits, temperature)
}

// TopPSample: nucleus sampling by masking tokens covering ~p mass.
func TopPSample(ctx *context.Context, logits *Node, p float64, temperature float64) *Node {
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
	defaultMask := TopKMask(probs, ks[len(ks)-1])
	selectedMask := defaultMask

	// Threshold p as tensor
	pNode := ConstAs(ReduceSum(probs, -1), p) // shape [batch]

	// Select smallest k where cumulative top-k prob >= p
	for i := 0; i < len(ks)-1; i++ {
		k := ks[i]
		topVals, _ := TopK(probs, k)
		cum := ReduceSum(topVals, -1)
		cond := GreaterOrEqual(cum, pNode)
		cond = ExpandDims(cond, -1)
		maskK := TopKMask(probs, k)
		selectedMask = Where(cond, maskK, selectedMask)
	}

	// Mask outside nucleus
	maskedLogits := Where(
		selectedMask,
		logits,
		Infinity(g, logits.DType(), -1),
	)

	// Temperature sampling
	return TemperatureSample(ctx, maskedLogits, temperature)
}

// SampleWithStrategy dispatches to greedy|temperature|top_k|top_p.
func SampleWithStrategy(
	ctx *context.Context,
	logits *Node,
	strategy string,
	temperature float64,
	topK int,
	topP float64,
) *Node {
	switch strategy {
	case "greedy":
		return GreedySample(ctx, logits)
	case "temperature":
		return TemperatureSample(ctx, logits, temperature)
	case "top_k":
		return TopKSample(ctx, logits, topK, temperature)
	case "top_p":
		return TopPSample(ctx, logits, topP, temperature)
	default:
		// Default to greedy
		return GreedySample(ctx, logits)
	}
}
