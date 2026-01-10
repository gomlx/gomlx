package main

import (
	"math"
	"math/rand"
	"sort"
)

// sampleToken implements temperature and top-p sampling
func (m *GPT2Model) sampleToken(logits []float32, temperature, topP float32) int {
	if temperature < 0.01 {
		// Greedy sampling
		maxIdx := 0
		maxVal := logits[0]
		for i, v := range logits {
			if v > maxVal {
				maxVal = v
				maxIdx = i
			}
		}
		return maxIdx
	}

	// Apply temperature scaling and find max for numerical stability
	maxLogit := logits[0] / temperature
	for _, v := range logits[1:] {
		scaled := v / temperature
		if scaled > maxLogit {
			maxLogit = scaled
		}
	}

	// Compute exp and sum for softmax (with temperature scaling)
	expSum := float32(0.0)
	probs := make([]float32, len(logits))
	for i, v := range logits {
		probs[i] = float32(math.Exp(float64(v/temperature - maxLogit)))
		expSum += probs[i]
	}

	// Normalize probabilities
	invSum := 1.0 / expSum
	for i := range probs {
		probs[i] *= invSum
	}

	// OPTIMIZATION: Instead of sorting all 50K tokens, use partial selection
	// Most top-p=0.9 only needs top ~100-500 tokens
	type probPair struct {
		idx  int
		prob float32
	}

	// Build array of non-zero probabilities only (skip zeros)
	candidates := make([]probPair, 0, len(probs))
	for i, p := range probs {
		if p > 1e-10 { // Skip negligible probabilities
			candidates = append(candidates, probPair{i, p})
		}
	}

	// Sort only the candidates (typically much fewer than full vocab)
	sort.Slice(candidates, func(i, j int) bool {
		return candidates[i].prob > candidates[j].prob
	})

	// Accumulate probabilities until we reach topP
	cumProb := float32(0.0)
	cutoff := 0
	for i, item := range candidates {
		cumProb += item.prob
		if cumProb >= topP {
			cutoff = i + 1
			break
		}
	}
	if cutoff == 0 {
		cutoff = len(candidates)
	}

	// Renormalize the selected candidates
	renormSum := float32(0.0)
	for i := 0; i < cutoff; i++ {
		renormSum += candidates[i].prob
	}

	// Sample from the renormalized distribution
	r := rand.Float32() * renormSum
	cumProb = 0.0
	for i := 0; i < cutoff; i++ {
		cumProb += candidates[i].prob
		if cumProb >= r {
			return candidates[i].idx
		}
	}

	return candidates[0].idx
}
