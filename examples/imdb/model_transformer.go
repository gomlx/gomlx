package imdb

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/fnn"
)

// TransformerModelGraph is the part of the model that takes the word/token embeddings to a transformed
// embedding through attention ready to be pooled and read out.
func TransformerModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec
	tokens := inputs[0]
	maskWordTaskWeight := context.GetParamOr(ctx, "imdb_mask_word_task_weight", 0.0)
	useMaskWordTask := maskWordTaskWeight > 0
	if useMaskWordTask {
		// Select word to mask and replace with <masked> token.
	}

	embed, mask := EmbedTokensGraph(ctx, tokens)

	//g := embed.Graph()
	//dtype := embed.DType()
	embed.AssertRank(3)
	batchSize := embed.Shape().Dimensions[0]
	contentLen := embed.Shape().Dimensions[1]
	//embedSize := embed.Shape().Dimensions[2]

	maxAttentionLen := context.GetParamOr(ctx, "transformer_max_att_len", 200)
	var newEmbed *Node
	if maxAttentionLen >= contentLen {
		// Full attention, the normal way.
		newEmbed = TransformerLayers(ctx.In("transformer"), embed, mask)
		embed = Add(embed, newEmbed)
	} else {
		// Split embedding in multiple split embeddings and apply transformer in each of them.
		attLen := maxAttentionLen
		sequenceFrom := 0
		for {
			// x.shape = [batchSize, sequence, embedding]
			sequenceTo := sequenceFrom + attLen
			if sequenceTo > contentLen {
				sequenceTo = contentLen
				sequenceFrom = sequenceTo - attLen
			}
			// part = x[:, sequenceFrom:sequenceTo, :]
			part := Slice(embed, AxisRange(), AxisRange(sequenceFrom, sequenceTo), AxisRange())
			partMask := Slice(mask, AxisRange(), AxisRange(sequenceFrom, sequenceTo))
			// Checked(false) -> to reuse "transformer" scope (same weights on every slice).
			part = TransformerLayers(ctx.In("transformer").Checked(false), part, partMask)
			if newEmbed == nil {
				newEmbed = part
			} else {
				newEmbed = Max(newEmbed, part)
			}
			if sequenceTo == contentLen {
				// Reached end of parts.
				break
			}
			sequenceFrom += max(attLen-10, 10) // Attention window overlap 10 tokens among themselves.
		}
		embed = newEmbed // Notice shape changed to `[batchSize, maxAttLen, embedDim]`
	}

	if useMaskWordTask {
		// Add "masked word" task loss.
		/*
			MaskedWordTaskGraph(ctx.In("masked_word_task"), tokens, embed, mask,
				func(input, mask *Node) *Node {
					return TransformerLayers(ctx.In("transformer").Reuse(), input, mask)
				})
		*/
	}

	// Take the max over the content length, and put an FNN on top.
	// Shape transformation: [batch_size, content_len, embed_size] -> [batch_size, embed_size]
	logits := ReduceMax(embed, 1)
	logits = fnn.New(ctx, logits, 1).Done()
	logits.AssertDims(batchSize, 1)
	return []*Node{logits}
}

// TransformerLayers builds the stacked transformer layers for the model.
func TransformerLayers(ctx *context.Context, embed, mask *Node) *Node {
	g := embed.Graph()
	shape := embed.Shape()
	dtype := embed.DType()
	embedSize := shape.Dimensions[2]

	// Dropout.
	dropoutRate := context.GetParamOr(ctx, "transformer_dropout_rate", -1.0)
	if dropoutRate < 0 {
		dropoutRate = context.GetParamOr(ctx, layers.ParamDropoutRate, 0.0)
	}
	var dropoutNode *Node
	if dropoutRate > 0.0 {
		dropoutNode = Scalar(g, dtype, dropoutRate)
	}

	// Create positional embedding variable: it is 1 in every axis, but for the
	// sequence dimension -- there will be one embedding per position.
	// Shape: [1, maxLen, embedSize]
	posEmbedShape := shape.Clone()
	posEmbedShape.Dimensions[0] = 1
	posEmbedVar := ctx.VariableWithShape("positional", posEmbedShape)
	posEmbed := posEmbedVar.ValueGraph(g)
	embed = Add(embed, posEmbed) // Just add the embeddings, seems to work well.

	// Add the requested number of attention layers.
	numAttLayers := context.GetParamOr(ctx, "transformer_num_att_layers", 1)
	numAttHeads := context.GetParamOr(ctx, "transformer_num_att_heads", 2)
	attKeySize := context.GetParamOr(ctx, "transformer_att_key_size", 8)
	for layerNum := range numAttLayers {
		// Each layer in its own scope.
		ctx := ctx.Inf("%03d_attention_layer", layerNum)
		residual := embed
		embed = layers.MultiHeadAttention(ctx.In("000_attention"), embed, embed, embed, numAttHeads, attKeySize).
			SetKeyMask(mask).SetQueryMask(mask).
			SetOutputDim(embedSize).
			SetValueHeadDim(embedSize).Done()
		if dropoutNode != nil {
			embed = layers.Dropout(ctx.In("001_dropout"), embed, dropoutNode)
		}
		embed = NormalizeSequence(ctx.In("002_normalization"), embed)
		attentionOutput := embed

		// Transformers recipe: 2 dense layers after attention.
		embed = fnn.New(ctx.In("003_fnn"), embed, embedSize).NumHiddenLayers(1, embedSize).Done()
		if dropoutNode != nil {
			embed = layers.Dropout(ctx.In("004_dropout"), embed, dropoutNode)
		}
		embed = Add(embed, attentionOutput)
		embed = NormalizeSequence(ctx.In("005_normalization"), embed)

		// Residual connection:
		if layerNum > 0 {
			embed = Add(residual, embed)
		}
	}
	return embed
}

/*
// MaskedWordTaskGraph builds the computation graph for the predicting a hidden word unsupervised task.
func MaskedWordTaskGraph(ctx *context.Context, tokens, embed, mask *Node,
	transformerFn func(input, mask *Node) *Node) {
	g := embed.Graph()
	batchSize := embed.Shape().Dimensions[0]
	//maxSeqSize := embed.Shape().Dimensions[1]
	embedDim := embed.Shape().Dimensions[2]
	vocabSize := len(LoadedVocab.ListEntries)

	// sequence sizer per example seqSize.Shape -> [batch_size]
	seqSize := ConvertDType(mask, DType)
	seqSize = ReduceSum(seqSize, -1)
	//seqSize.SetLogged("0. seqSize")

	// choice: shape=[batch_size]
	choice := ctx.RandomUniform(g, seqSize.Shape())
	choice = Mul(seqSize, choice)
	choice = ConvertDType(choice, dtypes.Int64)
	//choice.SetLogged("1. choice")

	// Find indices to gather the word token and later the word embedding.
	expandedChoice := InsertAxes(choice, -1)
	batchIndices := Iota(g, shapes.Make(dtypes.Int64, batchSize, 1), 0)
	indices := Concatenate([]*Node{batchIndices, expandedChoice}, -1) // shape=[batchSize, 2]
	wordToken := InsertAxes(Gather(tokens, indices), -1)              // [batchSize, 1]

	// wordMask: shape=[batch_size, seq_size]
	broadcastChoice := BroadcastToDims(expandedChoice, mask.Shape().Dimensions...)
	wordMask := Iota(g, shapes.Make(dtypes.Int64, mask.Shape().Dimensions...), 1)
	wordMask = Equal(wordMask, broadcastChoice)

	// wordMaskedEmbed: shape=[batch_size, seq_size, embedding_dim]
	// It will differ from embed only in chosen masked word position, for which it
	// takes some learned embedding to represent a masked word.
	maskedWordEmbeddingVar := ctx.VariableWithShape("masked_embedding", shapes.Make(DType, 1, 1, embedDim))
	maskedWordEmbedding := maskedWordEmbeddingVar.ValueGraph(g)
	maskedWordEmbedding = BroadcastToShape(maskedWordEmbedding, embed.Shape())
	embedWithMaskedWord := Where(wordMask, embed, maskedWordEmbedding)

	// Use shared transformer layer: embedWithMaskedWord will contain the transformer
	// processed embeddings for all words.
	embedWithMaskedWord = transformerFn(embedWithMaskedWord, mask)

	// Gather the embedding for the one word we are trying to predict.
	maskedWordEmbedding = Gather(embedWithMaskedWord, indices) // Shape=[batchSize, embedDim]

	// Use the embedding of the masked word to predict itself.
	logits := maskedWordEmbedding
	var dropoutRate *Node
	if *flagDropoutRate > 0 {
		dropoutRate = ConstAsDType(g, DType, *flagDropoutRate)
	}

	{
		ctx := ctx.In("output_dense_0")
		if *flagDropoutRate > 0 {
			logits = layers.Dropout(ctx.In("dropout"), logits, dropoutRate)
		}
		logits = layers.DenseWithBias(ctx, logits, *flagNumNodes)
		logits = Normalize(ctx, logits)
	}
	for ii := 1; ii < *flagNumHiddenLayers; ii++ {
		ctx := ctx.In(fmt.Sprintf("output_dense_%d", ii))
		residual := logits
		// Add layer with residual connection.
		if *flagDropoutRate > 0 {
			logits = layers.Dropout(ctx.In("dropout"), logits, dropoutRate)
		}
		logits = Tanh(logits)
		logits = layers.DenseWithBias(ctx, logits, *flagNumNodes)
		logits = Normalize(ctx, logits)
		logits = Add(logits, residual)
	}

	// Final logits layer with dimension `[batch, vocabulary_size]`
	{
		ctx := ctx.In("readout")
		if *flagDropoutRate > 0 {
			logits = layers.Dropout(ctx.In("dropout"), logits, dropoutRate)
		}
		logits = Tanh(logits)

		logits = layers.DenseWithBias(ctx, logits, vocabSize)
	}

	// Calculate loss associated to prediction of masked word.
	loss := losses.SparseCategoricalCrossEntropyLogits([]*Node{wordToken}, []*Node{logits})
	loss = ReduceAllMean(loss)
	loss = MulScalar(loss, *flagMaskWordTask)
	train.AddLoss(ctx, loss)
}
*/
