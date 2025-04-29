package imdb

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/layers/fnn"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
)

// EmbedTokensGraph creates embeddings for tokens and returns them along with the mask of used tokens --
// set to false where padding was used.
//
// - tokens: padded (at the start) tokens shaped (int32)[batch_size, content_len].
//
// Outputs:
//
// - embed: shaped [batch_size, content_len, <imdb_token_embedding_size>].
// - mask: shaped (bool)[batch_size, content_len], indicates where tokens were padded.
func EmbedTokensGraph(ctx *context.Context, tokens *Node) (embed, mask *Node) {
	g := tokens.Graph()
	mask = NotEqual(tokens, ZerosLike(tokens)) // Mask of tokens actually used.

	// The token ids are indexed by frequency. Truncate to the vocabulary size considered, replacing
	// ids higher than that by 0 .
	maxVocab := len(LoadedVocab.ListEntries)
	maxVocab = min(maxVocab, context.GetParamOr(ctx, "imdb_max_vocab", 20_000))

	// Limits tokens to the maxVocab.

	tokens = Where(GreaterOrEqual(tokens, Scalar(g, dtypes.Int32, float64(maxVocab))),
		MulScalar(OnesLike(tokens), float64(maxVocab-1)),
		tokens)

	// Embed tokens: shape=[batchSize, maxLen, embedDim]
	tokensEmbedSize := context.GetParamOr(ctx, "imdb_token_embedding_size", 32)
	embed = layers.Embedding(ctx.In("tokens"), tokens, DType, maxVocab, tokensEmbedSize, false)
	embed = Where(BroadcastToShape(mask, embed.Shape()), embed, ZerosLike(embed))

	wordDropoutRate := context.GetParamOr(ctx, "imdb_word_dropout_rate", 0.0)
	if wordDropoutRate > 0 {
		shape := embed.Shape().Dimensions[:len(embed.Shape().Dimensions)-1]
		dropoutMask := Ones(g, shapes.Make(DType, shape...))
		dropoutMask = layers.Dropout(ctx, dropoutMask, ConstAsDType(g, DType, wordDropoutRate))
		dropoutMask = InsertAxes(dropoutMask, -1)
		embed = Mul(embed, dropoutMask)
	}
	return
}

// BagOfWordsModelGraph builds the computation graph for the "bag of words" model: simply the sum of the embeddings
// for each token included.
func BagOfWordsModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	embed, _ := EmbedTokensGraph(ctx, inputs[0])

	// Take the max over the content length, and put an FNN on top.
	// Shape transformation: [batch_size, content_len, embed_size] -> [batch_size, embed_size]
	embed = ReduceMax(embed, 1)
	logits := fnn.New(ctx, embed, 1).Done()
	return []*Node{logits}
}
