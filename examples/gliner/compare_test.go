//go:build gliner_model

package gliner

import (
	"math"
	"testing"

	_ "github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/examples/gliner/safetensors"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/nn"
)

// Expected values from Python GLiNER model
var pythonEmbeddings = map[int][]float32{
	// Token ID 1 (CLS) embedding first 10 values
	1: {0.01062013, 0.01688149, -0.01215875, 0.00884032, -0.0076318, -0.00634906, 0.007761, -0.01576149, 0.03193525, -0.01111121},
	// Token ID 128002 (<<ENT>>) embedding first 10 values
	128002: {-4.5895656e-03, 2.4678833e-05, -3.7752404e-03, -7.0154974e-03, 1.2293180e-02, 5.8466988e-03, -8.4632775e-03, -3.2942323e-03, -2.2009993e-03, -2.2977882e-03},
	// Token ID 483 (company) embedding first 10 values
	483: {0.02775387, 0.03695673, -0.05219839, -0.0003175, -0.061168, 0.21475752, 0.09007239, -0.07032759, 0.02966276, 0.0325502},
}

func TestCompareEmbeddings(t *testing.T) {
	// Load safetensors file
	sf, err := safetensors.Open("model/model.safetensors")
	if err != nil {
		t.Fatalf("Failed to open safetensors: %v", err)
	}

	// Get embedding weights
	embTensor, err := sf.ToTensor("token_rep_layer.bert_layer.model.embeddings.word_embeddings.weight")
	if err != nil {
		t.Fatalf("Failed to get embedding tensor: %v", err)
	}

	t.Logf("Embedding tensor shape: %v", embTensor.Shape())
	hiddenSize := embTensor.Shape().Dimensions[1]

	// Compare embeddings for each token
	tensors.MustConstFlatData(embTensor, func(data []float32) {
		for tokenID, expected := range pythonEmbeddings {
			// Get embedding row for this token
			startIdx := tokenID * hiddenSize
			embedding := data[startIdx : startIdx+10]

			t.Logf("\nToken ID %d:", tokenID)
			t.Logf("  Expected: %v", expected)
			t.Logf("  Got:      %v", embedding)

			// Compare values
			maxDiff := float32(0)
			for i := 0; i < 10; i++ {
				diff := float32(math.Abs(float64(embedding[i] - expected[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("  Max diff: %v", maxDiff)

			if maxDiff > 1e-5 {
				t.Errorf("Token %d embedding mismatch: max diff = %v", tokenID, maxDiff)
			}
		}
	})
}

// Expected LayerNorm output from Python for first 3 tokens
var pythonLayerNormOutput = [][]float32{
	// Token 0 (CLS) first 10 values
	{0.15650524, 0.30060756, -0.13636512, 0.22869597, -0.22486907, -0.26287806, -0.03088734, -0.18700671, 0.33966297, -0.2765567},
	// Token 1 (<<ENT>>) first 10 values
	{-0.18802431, 0.08896533, -0.03260744, -0.23355876, 0.69027966, 0.24202327, -0.52374655, 0.00760016, -0.25436938, -0.10259766},
	// Token 2 (company) first 10 values
	{0.2852087, 0.4144218, -0.34321564, 0.12503932, -0.589309, 1.7212859, 0.69645214, -0.49889567, 0.10481863, 0.28863928},
}

// Expected Q, K, V projections from Python encoder layer 0
var pythonQuery = [][]float32{
	// Token 0 first 10 values
	{2.304645, -0.30320022, 0.25491986, -2.914699, 1.1428206, -1.0579894, -0.6735296, 4.5935593, -0.14897165, 0.6760465},
	// Token 1 first 10 values
	{0.546238, -0.05516491, -0.87824416, -0.8472095, -0.2668614, -0.24001212, -0.97733563, -0.22748654, 0.4627497, 0.2884218},
	// Token 2 first 10 values
	{-0.5770636, 0.54433566, -0.28764668, 0.27122283, 0.5297075, -0.8811788, -0.5615193, -0.5626583, 0.41406262, 0.48786855},
}

var pythonKey = [][]float32{
	{0.1291973, -0.74927914, -0.68894, -0.6117102, -0.826663, -0.82927245, -0.64950794, 1.7371997, -1.165193, -1.1412069},
	{0.2786864, -0.8418843, 0.8469123, -1.320505, -0.00900161, 0.00203325, -1.036603, -0.34268397, -0.04753186, 1.6400472},
	{-7.8986638e-04, -4.7648364e-01, -5.5952583e-02, -5.9768593e-01, 4.1284817e-01, -1.2764841, 9.7471935e-01, -7.5694114e-01, 1.8470110e-01, 1.2434667},
}

var pythonValue = [][]float32{
	{-0.28433904, -0.05205758, -0.43100867, -0.19767651, 1.7283869, -0.25066155, -0.0187161, 0.1834836, -0.18181695, -0.15242408},
	{-1.0139493, -0.05572736, 0.34542596, -0.2705547, 0.65294707, -1.8688242, -0.17428118, -0.14975387, 0.13727742, -0.40490133},
	{1.3351884, -0.6894491, 0.645087, 1.0543563, 0.47976136, 0.09184454, -0.08838114, -1.0633748, 0.3211327, 0.59279025},
}

// Expected c2c and c2p attention scores from Python (head 0, first 3x3)
var pythonC2C = [][]float32{
	{43.97111, 29.607767, 23.26186},
	{22.883812, 13.133279, -1.2066299},
	{5.2045417, -17.901741, -13.577909},
}

var pythonC2P = [][]float32{
	{38.28766, 33.15444, 32.81541},
	{5.4332156, 4.6298027, 2.6263983},
	{-8.8686075, -9.101149, -11.186572},
}

var pythonP2C = [][]float32{
	{11.015256, 5.2241287, 3.86872},
	{6.368396, -3.1240764, 3.9558887},
	{6.158946, -9.047855, -5.934821},
}

var pythonCombined = [][]float32{
	{93.27403, 67.986336, 59.94599},
	{34.685425, 14.639004, 5.375657},
	{2.4948802, -36.050743, -30.699303},
}

var pythonScaled = [][]float32{
	{6.7314734, 4.9064913, 4.3262296},
	{2.503205, 1.0564791, 0.38795465},
	{0.18005247, -2.6017382, -2.2155313},
}

var pythonAttnWeights = [][]float32{
	{0.28532687, 0.04600055, 0.02574889},
	{0.13470097, 0.03170047, 0.01624535},
	{0.17044613, 0.01055531, 0.01553096},
}

var pythonAttnOutput = [][]float32{
	// Token 0, first 10 dims
	{-1.5766129e-01, -7.8504287e-02, -3.8274087e-02, -3.9411131e-02, 1.2203256, -5.2810967e-01, 4.2811487e-02, -1.1489322e-01, -4.5906197e-02, -4.0620729e-02},
}

func TestCompareAttentionScores(t *testing.T) {
	// Load model
	model, err := LoadModel("model")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	backend := graphtest.BuildTestBackend()

	// Input: CLS, <<ENT>>, company (same as Python)
	inputIDs := tensors.FromFlatDataAndDimensions([]int32{1, 128002, 483}, 1, 3)

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, inputIDs *Node) []*Node {
		ctx = ctx.In("gliner").In("token_rep").Checked(false)
		g := inputIDs.Graph()

		// Embeddings + LayerNorm
		embCtx := ctx.In("embeddings")
		embWeights := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings").ValueGraph(g)
		embeddings := Gather(embWeights, InsertAxes(inputIDs, -1))

		lnCtx := embCtx.In("layer_norm")
		gain := lnCtx.GetVariableByScopeAndName(lnCtx.Scope(), "gain").ValueGraph(g)
		offset := lnCtx.GetVariableByScopeAndName(lnCtx.Scope(), "offset").ValueGraph(g)

		mean := ReduceMean(embeddings, -1)
		mean = InsertAxes(mean, -1)
		centered := Sub(embeddings, mean)
		variance := ReduceMean(Mul(centered, centered), -1)
		variance = InsertAxes(variance, -1)
		normalized := Div(centered, Sqrt(Add(variance, ConstAs(embeddings, 1e-7))))
		gain = Reshape(gain, 1, 1, gain.Shape().Dimensions[0])
		offset = Reshape(offset, 1, 1, offset.Shape().Dimensions[0])
		hidden := Add(Mul(normalized, gain), offset)

		// Q, K projections
		layerCtx := ctx.In("encoder").In("layer").In("0")
		attnCtx := layerCtx.In("attention")

		queryW := attnCtx.In("query").GetVariableByScopeAndName(attnCtx.In("query").Scope(), "weights").ValueGraph(g)
		queryB := attnCtx.In("query").GetVariableByScopeAndName(attnCtx.In("query").Scope(), "biases").ValueGraph(g)
		keyW := attnCtx.In("key").GetVariableByScopeAndName(attnCtx.In("key").Scope(), "weights").ValueGraph(g)
		keyB := attnCtx.In("key").GetVariableByScopeAndName(attnCtx.In("key").Scope(), "biases").ValueGraph(g)

		query := Add(Einsum("bsi,oi->bso", hidden, queryW), Reshape(queryB, 1, 1, queryB.Shape().Dimensions[0]))
		key := Add(Einsum("bsi,oi->bso", hidden, keyW), Reshape(keyB, 1, 1, keyB.Shape().Dimensions[0]))

		// Reshape for multi-head attention
		seqLen := 3
		numHeads := 12
		headDim := 64
		query = Reshape(query, 1, seqLen, numHeads, headDim)
		key = Reshape(key, 1, seqLen, numHeads, headDim)
		query = Transpose(query, 1, 2) // [1, 12, 3, 64]
		key = Transpose(key, 1, 2)     // [1, 12, 3, 64]

		// c2c: query @ key.T
		c2c := Einsum("bhqd,bhkd->bhqk", query, key)

		// Get relative position embeddings
		relEmbeddings := ctx.In("rel_embeddings").GetVariableByScopeAndName(ctx.In("rel_embeddings").Scope(), "embeddings").ValueGraph(g)

		// Build position indices for seq_len=3
		// rel_pos = i - j + 256 for (i, j) positions
		indices := make([]int32, seqLen*seqLen)
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				relPos := i - j + 256 // 256 is the offset
				indices[i*seqLen+j] = int32(relPos)
			}
		}
		indicesNode := Const(g, indices)
		indicesNode = Reshape(indicesNode, seqLen, seqLen, 1)

		// Gather position embeddings: [512, 768] -> [3, 3, 768]
		relPosEmb := Gather(relEmbeddings, indicesNode)

		// Project position embeddings through key projection (share_att_key=True)
		// pos_key = rel_emb @ key_proj (no bias for position)
		relPosKey := Einsum("qkh,oh->qko", relPosEmb, keyW)
		// Add bias
		relPosKey = Add(relPosKey, Reshape(keyB, 1, 1, keyB.Shape().Dimensions[0]))
		relPosKey = Reshape(relPosKey, seqLen, seqLen, numHeads, headDim)

		// c2p: query @ pos_key
		c2p := Einsum("bhqd,qkhd->bhqk", query, relPosKey)

		// p2c: pos_query @ key
		relPosQuery := Einsum("qkh,oh->qko", relPosEmb, queryW)
		relPosQuery = Add(relPosQuery, Reshape(queryB, 1, 1, queryB.Shape().Dimensions[0]))
		relPosQuery = Reshape(relPosQuery, seqLen, seqLen, numHeads, headDim)
		p2c := Einsum("qkhd,bhkd->bhqk", relPosQuery, key)

		// Combined
		combined := Add(Add(c2c, c2p), p2c)

		// Scaled
		scaleFactor := 1.0 / (float64(headDim) * 3.0)
		scale := ConstAs(combined, scaleFactor)
		scaled := Mul(combined, Sqrt(scale))

		// Softmax
		attnWeights := nn.Softmax(scaled, -1)

		// Value projection
		valueW := attnCtx.In("value").GetVariableByScopeAndName(attnCtx.In("value").Scope(), "weights").ValueGraph(g)
		valueB := attnCtx.In("value").GetVariableByScopeAndName(attnCtx.In("value").Scope(), "biases").ValueGraph(g)
		value := Add(Einsum("bsi,oi->bso", hidden, valueW), Reshape(valueB, 1, 1, valueB.Shape().Dimensions[0]))
		value = Reshape(value, 1, seqLen, numHeads, headDim)
		value = Transpose(value, 1, 2) // [1, 12, 3, 64]

		// Attention output: [1, 12, 3, 3] @ [1, 12, 3, 64] -> [1, 12, 3, 64]
		attnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, value)

		return []*Node{c2c, c2p, p2c, combined, scaled, attnWeights, attnOutput}
	})

	results := exec.MustExec(inputIDs)
	c2c := results[0]
	c2p := results[1]
	p2c := results[2]
	combined := results[3]
	scaled := results[4]
	attnWeights := results[5]
	attnOutput := results[6]

	t.Logf("c2c shape: %v", c2c.Shape())

	// Helper to compare and print 3x3 matrix
	compare3x3 := func(name string, tensor *tensors.Tensor, expected [][]float32, tolerance float32) {
		t.Logf("\n=== %s Comparison ===", name)
		tensors.MustConstFlatData(tensor, func(data []float32) {
			seqLen := 3
			t.Logf("%s[0, 0, :3, :3]:", name)
			for q := 0; q < 3; q++ {
				row := make([]float32, 3)
				for k := 0; k < 3; k++ {
					idx := q*seqLen + k
					row[k] = data[idx]
				}
				t.Logf("  %v", row)
			}
			for q := 0; q < 3; q++ {
				for k := 0; k < 3; k++ {
					idx := q*seqLen + k
					exp := expected[q][k]
					got := data[idx]
					diff := float32(math.Abs(float64(got - exp)))
					if diff > tolerance {
						t.Errorf("%s[0,0,%d,%d]: expected %v, got %v (diff %v)", name, q, k, exp, got, diff)
					}
				}
			}
		})
	}

	compare3x3("c2c", c2c, pythonC2C, 0.01)
	compare3x3("c2p", c2p, pythonC2P, 0.01)
	compare3x3("p2c", p2c, pythonP2C, 0.01)
	compare3x3("combined", combined, pythonCombined, 0.1)
	compare3x3("scaled", scaled, pythonScaled, 0.01)
	compare3x3("attnWeights", attnWeights, pythonAttnWeights, 0.01)

	// Compare attention output (token 0, head 0, first 10 dims)
	t.Logf("\n=== attnOutput Comparison ===")
	tensors.MustConstFlatData(attnOutput, func(data []float32) {
		// attnOutput shape: [1, 12, 3, 64]
		// head 0, token 0 starts at index 0
		t.Logf("attnOutput[0, 0, 0, :10]: %v", data[:10])
		t.Logf("Python expected:          %v", pythonAttnOutput[0])
		maxDiff := float32(0)
		for i := 0; i < 10; i++ {
			diff := float32(math.Abs(float64(data[i] - pythonAttnOutput[0][i])))
			if diff > maxDiff {
				maxDiff = diff
			}
		}
		t.Logf("Max diff: %v", maxDiff)
		if maxDiff > 0.01 {
			t.Errorf("attnOutput mismatch: max diff = %v", maxDiff)
		}
	})
}

func TestCompareQKV(t *testing.T) {
	// Load model to get weights into context
	model, err := LoadModel("model")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	// Get backend
	backend := graphtest.BuildTestBackend()

	// Input IDs from Python: [1, 128002, 483] (CLS, <<ENT>>, company)
	inputIDs := tensors.FromFlatDataAndDimensions([]int32{1, 128002, 483}, 1, 3)

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, inputIDs *Node) []*Node {
		ctx = ctx.In("gliner").In("token_rep").Checked(false)
		g := inputIDs.Graph()

		// Get embeddings
		embCtx := ctx.In("embeddings")
		embVar := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings")
		if embVar == nil {
			panic("missing embeddings variable")
		}
		embWeights := embVar.ValueGraph(g)

		// Lookup embeddings
		embeddings := Gather(embWeights, InsertAxes(inputIDs, -1))

		// Apply LayerNorm
		lnCtx := embCtx.In("layer_norm")
		gainVar := lnCtx.GetVariableByScopeAndName(lnCtx.Scope(), "gain")
		offsetVar := lnCtx.GetVariableByScopeAndName(lnCtx.Scope(), "offset")
		gain := gainVar.ValueGraph(g)
		offset := offsetVar.ValueGraph(g)

		epsilon := ConstAs(embeddings, 1e-7)
		mean := ReduceMean(embeddings, -1)
		mean = InsertAxes(mean, -1)
		centered := Sub(embeddings, mean)
		variance := ReduceMean(Mul(centered, centered), -1)
		variance = InsertAxes(variance, -1)
		normalized := Div(centered, Sqrt(Add(variance, epsilon)))

		gain = Reshape(gain, 1, 1, gain.Shape().Dimensions[0])
		offset = Reshape(offset, 1, 1, offset.Shape().Dimensions[0])
		hidden := Add(Mul(normalized, gain), offset)

		// Q, K, V projections from encoder layer 0
		layerCtx := ctx.In("encoder").In("layer").In("0")
		attnCtx := layerCtx.In("attention")

		queryW := attnCtx.In("query").GetVariableByScopeAndName(attnCtx.In("query").Scope(), "weights").ValueGraph(g)
		queryB := attnCtx.In("query").GetVariableByScopeAndName(attnCtx.In("query").Scope(), "biases").ValueGraph(g)
		keyW := attnCtx.In("key").GetVariableByScopeAndName(attnCtx.In("key").Scope(), "weights").ValueGraph(g)
		keyB := attnCtx.In("key").GetVariableByScopeAndName(attnCtx.In("key").Scope(), "biases").ValueGraph(g)
		valueW := attnCtx.In("value").GetVariableByScopeAndName(attnCtx.In("value").Scope(), "weights").ValueGraph(g)
		valueB := attnCtx.In("value").GetVariableByScopeAndName(attnCtx.In("value").Scope(), "biases").ValueGraph(g)

		query := Add(Einsum("bsi,oi->bso", hidden, queryW), Reshape(queryB, 1, 1, queryB.Shape().Dimensions[0]))
		key := Add(Einsum("bsi,oi->bso", hidden, keyW), Reshape(keyB, 1, 1, keyB.Shape().Dimensions[0]))
		value := Add(Einsum("bsi,oi->bso", hidden, valueW), Reshape(valueB, 1, 1, valueB.Shape().Dimensions[0]))

		return []*Node{query, key, value}
	})

	results := exec.MustExec(inputIDs)
	query := results[0]
	key := results[1]
	value := results[2]

	t.Logf("Query shape: %v", query.Shape())

	// Compare Query
	t.Log("\n=== Query Comparison ===")
	tensors.MustConstFlatData(query, func(data []float32) {
		hiddenSize := query.Shape().Dimensions[2]
		for tokenIdx, expected := range pythonQuery {
			startIdx := tokenIdx * hiddenSize
			got := data[startIdx : startIdx+10]
			t.Logf("Token %d:", tokenIdx)
			t.Logf("  Expected: %v", expected)
			t.Logf("  Got:      %v", got)

			maxDiff := float32(0)
			for i := 0; i < 10; i++ {
				diff := float32(math.Abs(float64(got[i] - expected[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("  Max diff: %v", maxDiff)

			if maxDiff > 1e-4 {
				t.Errorf("Query token %d mismatch: max diff = %v", tokenIdx, maxDiff)
			}
		}
	})

	// Compare Key
	t.Log("\n=== Key Comparison ===")
	tensors.MustConstFlatData(key, func(data []float32) {
		hiddenSize := key.Shape().Dimensions[2]
		for tokenIdx, expected := range pythonKey {
			startIdx := tokenIdx * hiddenSize
			got := data[startIdx : startIdx+10]
			t.Logf("Token %d:", tokenIdx)
			t.Logf("  Expected: %v", expected)
			t.Logf("  Got:      %v", got)

			maxDiff := float32(0)
			for i := 0; i < 10; i++ {
				diff := float32(math.Abs(float64(got[i] - expected[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("  Max diff: %v", maxDiff)

			if maxDiff > 1e-4 {
				t.Errorf("Key token %d mismatch: max diff = %v", tokenIdx, maxDiff)
			}
		}
	})

	// Compare Value
	t.Log("\n=== Value Comparison ===")
	tensors.MustConstFlatData(value, func(data []float32) {
		hiddenSize := value.Shape().Dimensions[2]
		for tokenIdx, expected := range pythonValue {
			startIdx := tokenIdx * hiddenSize
			got := data[startIdx : startIdx+10]
			t.Logf("Token %d:", tokenIdx)
			t.Logf("  Expected: %v", expected)
			t.Logf("  Got:      %v", got)

			maxDiff := float32(0)
			for i := 0; i < 10; i++ {
				diff := float32(math.Abs(float64(got[i] - expected[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("  Max diff: %v", maxDiff)

			if maxDiff > 1e-4 {
				t.Errorf("Value token %d mismatch: max diff = %v", tokenIdx, maxDiff)
			}
		}
	})
}

// Expected encoder layer 0 output from Python
var pythonEncoderLayer0Output = [][]float32{
	// Token 0 (CLS) first 10 values
	{0.08449504, -0.03506679, -0.01757421, 0.01713596, -0.09546749, -0.43953535, -0.02477938, -0.01732887, -0.02133834, -0.00270626},
	// Token 1 (<<ENT>>) first 10 values
	{-0.4942793, -0.1567398, -0.16105905, 0.32343632, 0.69881266, -0.40586093, -0.8359051, -0.11632655, 0.3747192, -0.04977745},
	// Token 2 (company) first 10 values
	{0.5082177, 0.5892172, -0.33880225, 0.49290422, -0.36382782, 2.3561664, 0.5218573, 0.05740282, -0.21394758, 0.5396901},
}

func TestCompareEncoderLayer0(t *testing.T) {
	// Load model to get weights into context
	model, err := LoadModel("model")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	// Get backend
	backend := graphtest.BuildTestBackend()

	// Full 22-token sequence from Python
	inputIDsData := []int32{1, 128002, 483, 128002, 604, 128002, 1250, 128003, 2013, 1326, 260, 284, 3679, 293, 3299, 9167, 267, 58326, 261, 1482, 260, 2}
	inputIDs := tensors.FromFlatDataAndDimensions(inputIDsData, 1, 22)
	attentionMaskData := make([]int32, 22)
	for i := range attentionMaskData {
		attentionMaskData[i] = 1
	}
	attentionMask := tensors.FromFlatDataAndDimensions(attentionMaskData, 1, 22)

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, inputIDs, attentionMask *Node) []*Node {
		ctx = ctx.In("gliner").In("token_rep").Checked(false)
		g := inputIDs.Graph()

		// Get embeddings
		embCtx := ctx.In("embeddings")
		embVar := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings")
		if embVar == nil {
			panic("missing embeddings variable")
		}
		embWeights := embVar.ValueGraph(g)

		// Lookup embeddings: inputIDs [1, 3] -> embeddings [1, 3, 768]
		embeddings := Gather(embWeights, InsertAxes(inputIDs, -1))

		// Apply LayerNorm
		lnCtx := embCtx.In("layer_norm")
		gainVar := lnCtx.GetVariableByScopeAndName(lnCtx.Scope(), "gain")
		offsetVar := lnCtx.GetVariableByScopeAndName(lnCtx.Scope(), "offset")
		if gainVar == nil || offsetVar == nil {
			panic("missing layer_norm variables")
		}
		gain := gainVar.ValueGraph(g)
		offset := offsetVar.ValueGraph(g)

		// LayerNorm: (x - mean) / sqrt(var + eps) * gain + offset
		epsilon := ConstAs(embeddings, 1e-7)
		mean := ReduceMean(embeddings, -1)
		mean = InsertAxes(mean, -1)
		centered := Sub(embeddings, mean)
		variance := ReduceMean(Mul(centered, centered), -1)
		variance = InsertAxes(variance, -1)
		normalized := Div(centered, Sqrt(Add(variance, epsilon)))

		// Apply gain and offset
		gain = Reshape(gain, 1, 1, gain.Shape().Dimensions[0])
		offset = Reshape(offset, 1, 1, offset.Shape().Dimensions[0])
		hidden := Add(Mul(normalized, gain), offset)

		// Build encoder layer 0 (similar to buildEncoderLayer but for layer 0 only)
		layerCtx := ctx.In("encoder").In("layer").In("0")
		attnCtx := layerCtx.In("attention")

		hiddenSize := 768
		numHeads := 12
		batchSize := 1
		seqLen := 22
		headDim := hiddenSize / numHeads

		residual := hidden

		// Q, K, V projections
		queryW := attnCtx.In("query").GetVariableByScopeAndName(attnCtx.In("query").Scope(), "weights").ValueGraph(g)
		queryB := attnCtx.In("query").GetVariableByScopeAndName(attnCtx.In("query").Scope(), "biases").ValueGraph(g)
		keyW := attnCtx.In("key").GetVariableByScopeAndName(attnCtx.In("key").Scope(), "weights").ValueGraph(g)
		keyB := attnCtx.In("key").GetVariableByScopeAndName(attnCtx.In("key").Scope(), "biases").ValueGraph(g)
		valueW := attnCtx.In("value").GetVariableByScopeAndName(attnCtx.In("value").Scope(), "weights").ValueGraph(g)
		valueB := attnCtx.In("value").GetVariableByScopeAndName(attnCtx.In("value").Scope(), "biases").ValueGraph(g)

		query := Add(Einsum("bsi,oi->bso", hidden, queryW), Reshape(queryB, 1, 1, queryB.Shape().Dimensions[0]))
		key := Add(Einsum("bsi,oi->bso", hidden, keyW), Reshape(keyB, 1, 1, keyB.Shape().Dimensions[0]))
		value := Add(Einsum("bsi,oi->bso", hidden, valueW), Reshape(valueB, 1, 1, valueB.Shape().Dimensions[0]))

		// Reshape for multi-head attention
		query = Reshape(query, batchSize, seqLen, numHeads, headDim)
		key = Reshape(key, batchSize, seqLen, numHeads, headDim)
		value = Reshape(value, batchSize, seqLen, numHeads, headDim)

		// Transpose to [batch, heads, seq, head_dim]
		query = Transpose(query, 1, 2)
		key = Transpose(key, 1, 2)
		value = Transpose(value, 1, 2)

		// Content-to-content attention (c2c)
		c2cScores := Einsum("bhqd,bhkd->bhqk", query, key)

		// Disentangled attention: c2p and p2c
		// Get relative position embeddings
		relEmbeddings := ctx.In("rel_embeddings").GetVariableByScopeAndName(ctx.In("rel_embeddings").Scope(), "embeddings").ValueGraph(g)

		// Apply encoder LayerNorm to relative embeddings (required by DeBERTa v2)
		encLnCtx := ctx.In("encoder").In("layer_norm")
		encLnGain := encLnCtx.GetVariableByScopeAndName(encLnCtx.Scope(), "gain").ValueGraph(g)
		encLnOffset := encLnCtx.GetVariableByScopeAndName(encLnCtx.Scope(), "offset").ValueGraph(g)
		relMean := ReduceMean(relEmbeddings, -1)
		relMean = InsertAxes(relMean, -1)
		relCentered := Sub(relEmbeddings, relMean)
		relVariance := ReduceMean(Mul(relCentered, relCentered), -1)
		relVariance = InsertAxes(relVariance, -1)
		relNormalized := Div(relCentered, Sqrt(Add(relVariance, ConstAs(relEmbeddings, 1e-7))))
		encLnGain = Reshape(encLnGain, 1, encLnGain.Shape().Dimensions[0])
		encLnOffset = Reshape(encLnOffset, 1, encLnOffset.Shape().Dimensions[0])
		relEmbeddings = Add(Mul(relNormalized, encLnGain), encLnOffset)

		// Build position indices for seq_len=22
		// rel_pos = i - j + 256 for (i, j) positions
		indices := make([]int32, seqLen*seqLen)
		for i := 0; i < seqLen; i++ {
			for j := 0; j < seqLen; j++ {
				relPos := i - j + 256 // 256 is the offset (position_buckets)
				indices[i*seqLen+j] = int32(relPos)
			}
		}
		indicesNode := Const(g, indices)
		indicesNode = Reshape(indicesNode, seqLen, seqLen, 1)

		// Gather position embeddings: [512, 768] -> [22, 22, 768]
		relPosEmb := Gather(relEmbeddings, indicesNode)

		// Project position embeddings through key projection (share_att_key=True)
		relPosKey := Einsum("qkh,oh->qko", relPosEmb, keyW)
		relPosKey = Add(relPosKey, Reshape(keyB, 1, 1, keyB.Shape().Dimensions[0]))
		relPosKey = Reshape(relPosKey, seqLen, seqLen, numHeads, headDim)

		// c2p: query @ pos_key
		c2pScores := Einsum("bhqd,qkhd->bhqk", query, relPosKey)

		// p2c: pos_query @ key
		relPosQuery := Einsum("qkh,oh->qko", relPosEmb, queryW)
		relPosQuery = Add(relPosQuery, Reshape(queryB, 1, 1, queryB.Shape().Dimensions[0]))
		relPosQuery = Reshape(relPosQuery, seqLen, seqLen, numHeads, headDim)
		p2cScores := Einsum("qkhd,bhkd->bhqk", relPosQuery, key)

		// Combined: c2c + c2p + p2c
		scores := Add(Add(c2cScores, c2pScores), p2cScores)

		// Scale by 1/sqrt(3*d) for disentangled attention
		scaleFactor := 1.0 / (float64(headDim) * 3.0)
		scale := ConstAs(scores, scaleFactor)
		scores = Mul(scores, Sqrt(scale))

		// Apply attention mask
		mask := InsertAxes(attentionMask, 1, 1)
		mask = BroadcastToDims(mask, scores.Shape().Dimensions...)
		negInf := ConstAs(scores, -1e9)
		zeroMask := Equal(mask, ScalarZero(g, mask.DType()))
		scores = Where(zeroMask, negInf, scores)

		attnWeights := nn.Softmax(scores, -1)
		rawAttnOutput := Einsum("bhqk,bhkd->bhqd", attnWeights, value)

		// Transpose back
		attnOutput := Transpose(rawAttnOutput, 1, 2)
		attnOutput = Reshape(attnOutput, batchSize, seqLen, hiddenSize)
		rawAttnOutputReshaped := attnOutput // Save for comparison

		// Output projection and residual
		denseW := attnCtx.In("output").In("dense").GetVariableByScopeAndName(attnCtx.In("output").In("dense").Scope(), "weights").ValueGraph(g)
		denseB := attnCtx.In("output").In("dense").GetVariableByScopeAndName(attnCtx.In("output").In("dense").Scope(), "biases").ValueGraph(g)
		projected := Add(Einsum("bsi,oi->bso", attnOutput, denseW), Reshape(denseB, 1, 1, denseB.Shape().Dimensions[0]))

		hidden = Add(residual, projected)

		// Attention output LayerNorm
		lnGain := attnCtx.In("output").In("layer_norm").GetVariableByScopeAndName(attnCtx.In("output").In("layer_norm").Scope(), "gain").ValueGraph(g)
		lnOffset := attnCtx.In("output").In("layer_norm").GetVariableByScopeAndName(attnCtx.In("output").In("layer_norm").Scope(), "offset").ValueGraph(g)

		mean = ReduceMean(hidden, -1)
		mean = InsertAxes(mean, -1)
		centered = Sub(hidden, mean)
		variance = ReduceMean(Mul(centered, centered), -1)
		variance = InsertAxes(variance, -1)
		normalized = Div(centered, Sqrt(Add(variance, ConstAs(hidden, 1e-7))))
		lnGain = Reshape(lnGain, 1, 1, lnGain.Shape().Dimensions[0])
		lnOffset = Reshape(lnOffset, 1, 1, lnOffset.Shape().Dimensions[0])
		hidden = Add(Mul(normalized, lnGain), lnOffset)
		afterAttnLN := hidden // Save for comparison

		// Feed-forward network
		ffCtx := layerCtx.In("ff")
		residual = hidden

		// Intermediate
		intW := ffCtx.In("intermediate").In("dense").GetVariableByScopeAndName(ffCtx.In("intermediate").In("dense").Scope(), "weights").ValueGraph(g)
		intB := ffCtx.In("intermediate").In("dense").GetVariableByScopeAndName(ffCtx.In("intermediate").In("dense").Scope(), "biases").ValueGraph(g)
		hidden = Add(Einsum("bsi,oi->bso", hidden, intW), Reshape(intB, 1, 1, intB.Shape().Dimensions[0]))

		// GELU activation
		hidden = Mul(hidden, Mul(Add(Erf(Mul(hidden, ConstAs(hidden, 1.0/math.Sqrt(2.0)))), ConstAs(hidden, 1.0)), ConstAs(hidden, 0.5)))

		// Output
		outW := ffCtx.In("output").In("dense").GetVariableByScopeAndName(ffCtx.In("output").In("dense").Scope(), "weights").ValueGraph(g)
		outB := ffCtx.In("output").In("dense").GetVariableByScopeAndName(ffCtx.In("output").In("dense").Scope(), "biases").ValueGraph(g)
		hidden = Add(Einsum("bsi,oi->bso", hidden, outW), Reshape(outB, 1, 1, outB.Shape().Dimensions[0]))

		hidden = Add(residual, hidden)

		// Output LayerNorm
		ffLnGain := ffCtx.In("output").In("layer_norm").GetVariableByScopeAndName(ffCtx.In("output").In("layer_norm").Scope(), "gain").ValueGraph(g)
		ffLnOffset := ffCtx.In("output").In("layer_norm").GetVariableByScopeAndName(ffCtx.In("output").In("layer_norm").Scope(), "offset").ValueGraph(g)

		mean = ReduceMean(hidden, -1)
		mean = InsertAxes(mean, -1)
		centered = Sub(hidden, mean)
		variance = ReduceMean(Mul(centered, centered), -1)
		variance = InsertAxes(variance, -1)
		normalized = Div(centered, Sqrt(Add(variance, ConstAs(hidden, 1e-7))))
		ffLnGain = Reshape(ffLnGain, 1, 1, ffLnGain.Shape().Dimensions[0])
		ffLnOffset = Reshape(ffLnOffset, 1, 1, ffLnOffset.Shape().Dimensions[0])
		hidden = Add(Mul(normalized, ffLnGain), ffLnOffset)

		return []*Node{hidden, rawAttnOutputReshaped, projected, afterAttnLN, attnWeights}
	})

	results := exec.MustExec(inputIDs, attentionMask)
	output := results[0]
	rawAttnOut := results[1]
	projectedOut := results[2]
	afterAttnLN := results[3]
	attnWeights := results[4]

	t.Logf("Encoder layer 0 output shape: %v", output.Shape())

	// Compare intermediate values with Python
	pythonRawAttnOutput := [][]float32{
		{-0.15766129, -0.07850429, -0.03827409, -0.03941113, 1.2203256, -0.52810967, 0.04281149, -0.11489322, -0.0459062, -0.04062073},
		{-0.17062764, -0.07134039, -0.09115491, -0.02737973, 1.2359278, -0.61449724, 0.04103219, -0.08353028, -0.07121632, -0.05875022},
		{-0.08129565, -0.091498971, -0.047810949, 0.023982638, 1.2765139, -0.39736360, -0.041081034, -0.16317126, 0.0013455893, -0.0011874777},
	}
	t.Log("\n=== Raw attention output comparison ===")
	tensors.MustConstFlatData(rawAttnOut, func(data []float32) {
		hiddenSize := rawAttnOut.Shape().Dimensions[2]
		for i, expected := range pythonRawAttnOutput {
			startIdx := i * hiddenSize
			got := data[startIdx : startIdx+10]
			t.Logf("Token %d - Python: %v", i, expected)
			t.Logf("Token %d - Go:     %v", i, got)
			maxDiff := float32(0)
			for j := 0; j < 10; j++ {
				diff := float32(math.Abs(float64(got[j] - expected[j])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("Token %d max diff: %v", i, maxDiff)
		}
	})

	pythonAfterProjection := [][]float32{
		{0.28168565, -0.3111158, 0.00926845, 0.19431849, 0.28387657, -0.094, -0.31495997, 0.09220595, -0.27428448, 0.20371847},
		{-0.32894257, -0.36176416, 0.00719469, 0.09244777, -0.06917311, -0.1934573, 0.18932983, 0.16917147, -0.4548601, 0.17229097},
		{0.26091233, -0.01073605, 0.24717778, -0.27639714, 0.05800378, 1.2020506, -0.14400327, 0.36196333, -1.0459008, 0.2843488},
	}
	t.Log("\n=== After projection comparison ===")
	tensors.MustConstFlatData(projectedOut, func(data []float32) {
		hiddenSize := projectedOut.Shape().Dimensions[2]
		for i, expected := range pythonAfterProjection {
			startIdx := i * hiddenSize
			got := data[startIdx : startIdx+10]
			t.Logf("Token %d - Python: %v", i, expected)
			t.Logf("Token %d - Go:     %v", i, got)
		}
	})

	pythonAfterAttnLN := [][]float32{
		{0.23385382, -0.11638412, -0.20906517, 0.21312854, -0.08070093, -0.8054729, -0.4016056, -0.1740603, -0.03028945, -0.16091903},
		{-0.61302626, -0.2544766, -0.1170913, -0.20290628, 0.39276397, -0.18110476, -0.37305224, 0.05685458, -0.56115615, -0.0268863},
		{0.22965759, 0.0756548, -0.16319335, -0.19471726, -0.44526237, 2.9951239, 0.28040412, -0.17438309, -0.57844466, 0.25581592},
	}
	t.Log("\n=== After attention residual+LN comparison ===")
	tensors.MustConstFlatData(afterAttnLN, func(data []float32) {
		hiddenSize := afterAttnLN.Shape().Dimensions[2]
		for i, expected := range pythonAfterAttnLN {
			startIdx := i * hiddenSize
			got := data[startIdx : startIdx+10]
			t.Logf("Token %d - Python: %v", i, expected)
			t.Logf("Token %d - Go:     %v", i, got)
		}
	})

	// Print attention weights for first few positions
	t.Log("\n=== Attention weights comparison ===")
	tensors.MustConstFlatData(attnWeights, func(data []float32) {
		// attnWeights shape: [1, 12, 22, 22], head 0
		seqLen := 22
		t.Logf("Go attn_weights[0,0,0,:5]: %v", data[0:5])
		t.Logf("Go attn_weights[0,0,1,:5]: %v", data[seqLen:seqLen+5])
		t.Logf("Go attn_weights[0,0,2,:5]: %v", data[2*seqLen:2*seqLen+5])
		t.Logf("Python [0,0,0,:5]: [0.2853, 0.0460, 0.0257, 0.0282, 0.0105]")
		t.Logf("Python [0,0,1,:5]: [0.1347, 0.0317, 0.0162, 0.0427, 0.0074]")
		t.Logf("Python [0,0,2,:5]: [0.1704, 0.0106, 0.0155, 0.0280, 0.0563]")
	})

	// Compare with Python values
	tensors.MustConstFlatData(output, func(data []float32) {
		hiddenSize := output.Shape().Dimensions[2]
		for tokenIdx, expected := range pythonEncoderLayer0Output {
			startIdx := tokenIdx * hiddenSize
			got := data[startIdx : startIdx+10]

			t.Logf("\nToken %d:", tokenIdx)
			t.Logf("  Expected: %v", expected)
			t.Logf("  Got:      %v", got)

			maxDiff := float32(0)
			for i := 0; i < 10; i++ {
				diff := float32(math.Abs(float64(got[i] - expected[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("  Max diff: %v", maxDiff)

			if maxDiff > 1e-3 {
				t.Errorf("Token %d encoder layer 0 mismatch: max diff = %v", tokenIdx, maxDiff)
			}
		}
	})
}

func TestCompareLayerNorm(t *testing.T) {
	// Load model to get weights into context
	model, err := LoadModel("model")
	if err != nil {
		t.Fatalf("Failed to load model: %v", err)
	}

	ctx := context.New()
	if err := model.LoadWeightsIntoContext(ctx); err != nil {
		t.Fatalf("Failed to load weights: %v", err)
	}

	// Get backend
	backend := graphtest.BuildTestBackend()

	// Build a graph that does just embedding + LayerNorm
	// Input IDs: [CLS, <<ENT>>, company] = [1, 128002, 483]
	inputIDs := tensors.FromFlatDataAndDimensions([]int32{1, 128002, 483}, 1, 3)

	exec := context.MustNewExec(backend, ctx, func(ctx *context.Context, inputIDs *Node) *Node {
		ctx = ctx.In("gliner").In("token_rep").Checked(false)
		g := inputIDs.Graph()

		// Get embeddings
		embCtx := ctx.In("embeddings")
		embVar := embCtx.GetVariableByScopeAndName(embCtx.Scope(), "embeddings")
		if embVar == nil {
			panic("missing embeddings variable")
		}
		embWeights := embVar.ValueGraph(g)

		// Lookup embeddings: inputIDs [1, 3] -> embeddings [1, 3, 768]
		embeddings := Gather(embWeights, InsertAxes(inputIDs, -1))

		// Apply LayerNorm
		lnCtx := embCtx.In("layer_norm")
		gainVar := lnCtx.GetVariableByScopeAndName(lnCtx.Scope(), "gain")
		offsetVar := lnCtx.GetVariableByScopeAndName(lnCtx.Scope(), "offset")
		if gainVar == nil || offsetVar == nil {
			panic("missing layer_norm variables")
		}
		gain := gainVar.ValueGraph(g)
		offset := offsetVar.ValueGraph(g)

		// LayerNorm: (x - mean) / sqrt(var + eps) * gain + offset
		epsilon := ConstAs(embeddings, 1e-7)
		mean := ReduceMean(embeddings, -1)
		mean = InsertAxes(mean, -1)
		centered := Sub(embeddings, mean)
		variance := ReduceMean(Mul(centered, centered), -1)
		variance = InsertAxes(variance, -1)
		normalized := Div(centered, Sqrt(Add(variance, epsilon)))

		// Apply gain and offset
		gain = Reshape(gain, 1, 1, gain.Shape().Dimensions[0])
		offset = Reshape(offset, 1, 1, offset.Shape().Dimensions[0])
		return Add(Mul(normalized, gain), offset)
	})

	results := exec.MustExec(inputIDs)
	output := results[0]

	t.Logf("LayerNorm output shape: %v", output.Shape())

	// Compare with Python values
	tensors.MustConstFlatData(output, func(data []float32) {
		hiddenSize := output.Shape().Dimensions[2]
		for tokenIdx, expected := range pythonLayerNormOutput {
			startIdx := tokenIdx * hiddenSize
			got := data[startIdx : startIdx+10]

			t.Logf("\nToken %d:", tokenIdx)
			t.Logf("  Expected: %v", expected)
			t.Logf("  Got:      %v", got)

			maxDiff := float32(0)
			for i := 0; i < 10; i++ {
				diff := float32(math.Abs(float64(got[i] - expected[i])))
				if diff > maxDiff {
					maxDiff = diff
				}
			}
			t.Logf("  Max diff: %v", maxDiff)

			if maxDiff > 1e-4 {
				t.Errorf("Token %d LayerNorm mismatch: max diff = %v", tokenIdx, maxDiff)
			}
		}
	})
}
