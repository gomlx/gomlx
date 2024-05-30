/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

// IMDB Movie Review library (imdb) demo: you can run this program in 4 different ways:
package main

import (
	"flag"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/gomlx/gomlx/examples/imdb"
	"github.com/gomlx/gomlx/examples/notebook/gonb/margaid"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/ml/train/losses"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
)

// DType used for the demo.
const DType = shapes.Float32

func AssertNoError(err error) {
	if err != nil {
		log.Fatalf("Failed: %+v", err)
	}
}

var (
	flagEval    = flag.Bool("eval", true, "Evaluate model at the end.")
	flagTrain   = flag.Bool("train", true, "Train model. Set to false to evaluate only.")
	flagDataDir = flag.String("data", "~/tmp/imdb", "Directory to cache downloaded and generated dataset files.")

	// Extra options.
	flagMaskWordTask    = flag.Float64("masked_word_task", 0.0, "Include \"masked word\" self-supervised task with given weight.")
	flagUseUnsupervised = flag.Bool("use_unsupervised", false, "Use unsupervised dataset: it's only used to pretrain --masked_word_task. The model trained needs to be further fine-tuned with normal training data later.")

	// Checkpointing model.
	flagCheckpoint     = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
	flagCheckpointKeep = flag.Int("checkpoint_keep", 10, "Number of checkpoints to keep, if --checkpoint is set.")

	// ML Manager creation:
	flagNumThreads  = flag.Int("num_threads", -1, "Number of threads. Leave as -1 to use as many as there are cores.")
	flagNumReplicas = flag.Int("num_replicas", 1, "Number of replicas.")
	flagPlatform    = flag.String("platform", "", "Platform to use, if empty uses the default one.")

	// Training hyperparameters:
	flagModel            = flag.String("model", "transformer", "Model type: bow or transformer.")
	flagOptimizer        = flag.String("optimizer", "adam", "Optimizer, options: adam or sgd.")
	flagNumSteps         = flag.Int("steps", 5000, "Number of gradient descent steps to perform")
	flagBatchSize        = flag.Int("batch", 32, "Batch size for training")
	flagLearningRate     = flag.Float64("learning_rate", 0.0001, "Initial learning rate.")
	flagL2Regularization = flag.Float64("l2_reg", 0, "L2 regularization on kernels. It doesn't interact well with --batch_norm or with --optimizer=adam.")
	flagNormalization    = flag.String("norm", "layer", "Type of normalization to use. Valid values are \"none\", \"batch\", \"layer\".")
	flagDropoutRate      = flag.Float64("dropout", 0.15, "Dropout rate")
	flagWordDropoutRate  = flag.Float64("word_dropout", 0, "Dropout rate for whole words of the input")

	// Model hyperparameters:
	flagMaxLen              = flag.Int("max_len", 200, "Maximum number of tokens to take from observation.")
	flagMaxVocab            = flag.Int("max_vocab", 20000, "Top most frequent words to consider, the rest is considered unknown.")
	flagTokenEmbeddingSize  = flag.Int("token_embed", 32, "Size of token embedding table. There are ~140K unique tokens")
	flagIncludeSeparators   = flag.Bool("include_separators", false, "If true include the word separator symbols in the tokens.")
	flagMaxAttLen           = flag.Int("max_att_len", 200, "Maximum attention length: input will be split in ranges of this size.")
	flagNumAttHeads         = flag.Int("att_heads", 2, "Number of attention heads, if --model=transformer.")
	flagNumAttLayers        = flag.Int("att_layers", 1, "Number of stacked attention layers, if --model=transformer.")
	flagAttKeyQueryEmbedDim = flag.Int("att_key_dim", 8, "Dimension of the Key/Query attention embedding.")
	flagNumHiddenLayers     = flag.Int("hidden_layers", 2, "Number of output hidden layers, stacked with residual connection.")
	flagNumNodes            = flag.Int("num_nodes", 32, "Number of nodes in output hidden layers.")

	// UI
	flagUseProgressBar = flag.Bool("bar", true, "If to display a progress bar during training")
	flagPlots          = flag.Bool("plots", true, "Plots during training: perform periodic evaluations, "+
		"save results if --checkpoint is set and draw plots, if in a Jupyter notebook.")
)

func main() {
	flag.Parse()
	imdb.IncludeSeparators = *flagIncludeSeparators
	if *flagUseUnsupervised && *flagMaskWordTask <= 0 {
		log.Fatal("--use_unsupervised is only useful with --mask_word_task=x (x > 0).")
	}

	// Validate and create --data directory.
	*flagDataDir = data.ReplaceTildeInDir(*flagDataDir)
	if !data.FileExists(*flagDataDir) {
		AssertNoError(os.MkdirAll(*flagDataDir, 0777))
	}

	AssertNoError(imdb.Download(*flagDataDir))
	trainModel()
}

func Sample() {
	ds := imdb.NewDataset("Test", imdb.Test, *flagMaxLen, 3, DType, true, nil)
	_, inputs, labels, err := ds.Yield()
	AssertNoError(err)
	labelsRef := labels[0].Local().AcquireData()
	defer labelsRef.Release()
	labelsData := shapes.CastAsDType(labelsRef.Flat(), shapes.Int64).([]int)
	for ii := 0; ii < 3; ii++ {
		fmt.Printf("\n%v : %s\n", labelsData[ii], imdb.InputToString(inputs[0], ii))
	}
	fmt.Println()
}

func trainModel() {
	// Manager handles creation of ML computation graphs, accelerator resources, etc.
	manager := BuildManager().NumThreads(*flagNumThreads).NumReplicas(*flagNumReplicas).Platform(*flagPlatform).Done()

	// Datasets.
	var trainDS, trainEvalDS, testEvalDS train.Dataset
	if *flagUseUnsupervised {
		trainDS = imdb.NewUnsupervisedDataset("unsupervised-train", *flagMaxLen, *flagBatchSize, DType, true, nil)
	} else {
		trainDS = imdb.NewDataset("train", imdb.Train, *flagMaxLen, *flagBatchSize, DType, true, nil)
	}
	trainEvalDS = imdb.NewDataset("train-eval", imdb.Train, *flagMaxLen, *flagBatchSize, DType, false, nil)
	testEvalDS = imdb.NewDataset("test-eval", imdb.Test, *flagMaxLen, *flagBatchSize, DType, false, nil)

	// Parallelize generation of batches.
	trainDS = data.Parallel(trainDS)
	trainEvalDS = data.Parallel(trainEvalDS)
	testEvalDS = data.Parallel(testEvalDS)

	// Metrics we are interested.
	meanAccuracyMetric := metrics.NewMeanBinaryLogitsAccuracy("Mean Accuracy", "#acc")
	movingAccuracyMetric := metrics.NewMovingAverageBinaryLogitsAccuracy("Moving Average Accuracy", "~acc", 0.01)

	// Context holds the variables and hyperparameters for the model.
	ctx := context.NewContext(manager)
	ctx.SetParam(optimizers.LearningRateKey, *flagLearningRate)
	ctx.SetParam(layers.L2RegularizationKey, *flagL2Regularization)
	if *flagEval {
		ctx = ctx.Checked(false)
	}

	// Checkpoints saving.
	var checkpoint *checkpoints.Handler
	if *flagCheckpoint != "" {
		var err error
		checkpoint, err = checkpoints.Build(ctx).DirFromBase(*flagCheckpoint, *flagDataDir).Keep(*flagCheckpointKeep).Done()
		AssertNoError(err)
	}

	// Create a train.Trainer: this object will orchestrate running the model, feeding
	// results to the optimizer, evaluating the metrics, etc. (all happens in trainer.TrainStep)
	loss := losses.BinaryCrossentropyLogits
	if *flagUseUnsupervised {
		loss = nil
	}
	trainer := train.NewTrainer(
		manager, ctx, modelGraph, loss,
		optimizers.MustOptimizerByName(*flagOptimizer),
		[]metrics.Interface{movingAccuracyMetric}, // trainMetrics
		[]metrics.Interface{meanAccuracyMetric})   // evalMetrics

	// Use standard training loop.
	if *flagTrain {
		// --train: train model
		loop := train.NewLoop(trainer)
		if *flagUseProgressBar {
			commandline.AttachProgressBar(loop) // Attaches a progress bar to the loop.
		}

		// Attach a checkpoint: checkpoint every 1 minute of training.
		if checkpoint != nil {
			period := time.Minute * 1
			train.PeriodicCallback(loop, period, true, "saving checkpoint", 100,
				func(loop *train.Loop, metrics []tensor.Tensor) error {
					fmt.Printf("\n[saving checkpoint@%d] [median train step (ms): %d]\n", loop.LoopStep, loop.MedianTrainStepDuration().Milliseconds())
					return checkpoint.Save()
				})
		}

		// Attach a margaid plots: plot points at exponential steps.
		// The points generated are saved along the checkpoint directory (if one is given).
		if *flagPlots {
			_ = margaid.NewDefault(loop, checkpoint.Dir(), 100, 1.1, trainEvalDS, testEvalDS)
		}

		// Loop for given number of steps.
		_, err := loop.RunSteps(trainDS, *flagNumSteps)
		AssertNoError(err)
		fmt.Printf("\t[Step %d] median train step: %d microseconds\n",
			loop.LoopStep, loop.MedianTrainStepDuration().Microseconds())
	}

	if *flagEval {
		// --eval: print an evaluation on train and test datasets.
		fmt.Println()
		err := commandline.ReportEval(trainer, trainEvalDS, testEvalDS)
		AssertNoError(err)
	}
}

// EmbedTokensGraph creates embeddings for tokens and returns them along with the mask of used tokens --
// set to false where padding was used.
func EmbedTokensGraph(ctx *context.Context, tokens *Node) (embed, mask *Node) {
	g := tokens.Graph()
	mask = NotEqual(tokens, ZerosLike(tokens)) // Mask of tokens actually used.

	// The token ids are indexed by frequency. Truncate to the vocabulary size considered, replacing
	// ids higher than that by 0.
	maxVocab := len(imdb.LoadedVocab.ListEntries)
	if maxVocab > *flagMaxVocab {
		maxVocab = *flagMaxVocab
	}

	// Limits tokens to the maxVocab.
	tokens = Where(GreaterOrEqual(tokens, Const(g, maxVocab)),
		MulScalar(OnesLike(tokens), float64(maxVocab-1)),
		tokens)

	// Embed tokens: shape=[batchSize, maxLen, embedDim]
	embed = layers.Embedding(ctx.In("tokens"), tokens, DType, maxVocab, *flagTokenEmbeddingSize)
	embed = Where(mask, embed, ZerosLike(embed))

	if *flagWordDropoutRate > 0 {
		shape := embed.Shape().Dimensions[:len(embed.Shape().Dimensions)-1]
		dropoutMask := Ones(g, shapes.Make(DType, shape...))
		dropoutMask = layers.Dropout(ctx, dropoutMask, ConstAsDType(g, DType, *flagWordDropoutRate))
		dropoutMask = ExpandDims(dropoutMask, -1)
		embed = Mul(embed, dropoutMask)
	}
	return
}

// modelGraph builds the model for our demo. It returns the logits, not the predictions, which works with most losses.
func modelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	_ = spec // Not used.
	tokens := inputs[0]
	embed, mask := EmbedTokensGraph(ctx, tokens)

	// Normalization function.
	if *flagModel == "bow" {
		// Bag-Of-Words model doesn't do anything, it's just the embedding table for each token.
	} else if *flagModel == "cnn" {
		embed = Conv1DGraph(ctx, embed, mask)
	} else if *flagModel == "transformer" {
		embed = TransformerGraph(ctx, tokens, embed, mask)
	} else {
		exceptions.Panicf("unknown model type %q, only types \"bow\", \"cnn\" and \"transformer\" are implemented", *flagModel)
	}

	// Sum-up per-token embeddings and do a FNN on the output. From now on, the dimensions are `[batch_dim, embed_dim]`
	// Notice we are not using mask.
	embed = ReduceMax(embed, 1)
	logits := ReadoutGraph(ctx, embed)
	return []*Node{logits}
}

// ReadoutGraph takes the embeddings after they have been pooled on the sequence axis, so shaped `[batch_size, embed_dim]`
// adds a FNN on top and readout the final logit.
func ReadoutGraph(ctx *context.Context, embed *Node) *Node {
	g := embed.Graph()
	var dropoutRate *Node
	if *flagDropoutRate > 0 {
		dropoutRate = ConstAsDType(g, DType, *flagDropoutRate)
	}

	// Output layers.
	for ii := 0; ii < *flagNumHiddenLayers; ii++ {
		ctx := ctx.In(fmt.Sprintf("output_dense_%d", ii))
		residual := embed
		if *flagDropoutRate > 0 {
			embed = layers.Dropout(ctx, embed, dropoutRate)
		}
		//embed = Tanh(embed)
		embed = Activation(embed)
		embed = layers.DenseWithBias(ctx, embed, *flagNumNodes)
		embed = Normalize(ctx, embed)
		if ii > 0 {
			// Add residual connection.
			embed = Add(embed, residual)
		}
	}

	// Final embed layer with dimension 1.
	{
		ctx := ctx.In("readout")
		if *flagDropoutRate > 0 {
			embed = layers.Dropout(ctx, embed, dropoutRate)
		}
		embed = Activation(embed)
		embed = layers.DenseWithBias(ctx, embed, 1)
	}
	return embed
}

// Activation function used for models.
func Activation(x *Node) *Node {
	return layers.Relu(x)
}

func Conv1DGraph(ctx *context.Context, embed, mask *Node) *Node {
	g := embed.Graph()
	// 1D Convolution:
	{
		ctx := ctx.In("conv1")
		embed = Activation(embed)
		embed = layers.Dropout(ctx, embed, ConstAsDType(g, DType, *flagDropoutRate))
		embed = layers.Convolution(ctx, embed).KernelSize(7).Filters(*flagTokenEmbeddingSize).Strides(3).Done()
		embed = Normalize(ctx, embed)
	}
	{
		ctx := ctx.In("conv2")
		embed = Activation(embed)
		embed = layers.Convolution(ctx, embed).KernelSize(7).Filters(*flagTokenEmbeddingSize).Strides(3).Done()
		embed = Normalize(ctx, embed)
	}
	return embed
}

// TransformerGraph is the part of the model that takes the word/token embeddings to a transformed
// embedding through attention ready to be pooled and read out.
func TransformerGraph(ctx *context.Context, tokens, embed, mask *Node) *Node {
	var newEmbed *Node
	if *flagMaxAttLen >= *flagMaxLen {
		newEmbed = TransformerLayers(ctx.In("transformer"), embed, mask)
		embed = Add(embed, newEmbed)
	} else {
		// Split embedding in multiple split embeddings and apply transformer in each of them.
		attLen := *flagMaxAttLen
		sequenceFrom := 0
		for {
			// x.shape = [batchSize, sequence, embedding]
			sequenceTo := sequenceFrom + attLen
			if sequenceTo > *flagMaxLen {
				sequenceTo = *flagMaxLen
				sequenceFrom = sequenceTo - attLen
			}
			// part = x[:, sequenceFrom:sequenceTo, :]
			residual := Slice(embed, AxisRange(), AxisRange(sequenceFrom, sequenceTo), AxisRange())
			partMask := Slice(mask, AxisRange(), AxisRange(sequenceFrom, sequenceTo))
			// Reuse "transformer" scope.
			part := TransformerLayers(ctx.In("transformer").Checked(false), residual, partMask)
			part = Add(residual, part)
			if newEmbed == nil {
				newEmbed = part
			} else {
				newEmbed = Add(newEmbed, part)
			}

			if sequenceTo == *flagMaxLen {
				// Reached end of parts.
				break
			}
			sequenceFrom += attLen - 20 // Attention window overlap 10 tokens among themselves.
		}
		embed = newEmbed // Notice shape changed to `[batchSize, maxAttLen, embedDim]`
	}

	if *flagMaskWordTask > 0 {
		MaskedWordTaskGraph(ctx.In("masked_word_task"), tokens, embed, mask,
			func(input, mask *Node) *Node {
				return TransformerLayers(ctx.In("transformer").Reuse(), input, mask)
			})
	}

	// Debugging.
	//train.AddPerStepUpdateGraphFn(ctx, g, func(ctx *context.Context, graph *Graph) {
	//	train.GetLosses(ctx, g).SetLogged("Loss")
	//})
	return embed
}

// TransformerLayers builds the stacked transformer layers for the model.
func TransformerLayers(ctx *context.Context, embed, mask *Node) *Node {
	g := embed.Graph()
	shape := embed.Shape()
	embedDim := shape.Dimensions[2]

	var dropoutRate *Node
	if *flagDropoutRate > 0 {
		dropoutRate = ConstAsDType(g, DType, *flagDropoutRate)
	}

	// Create positional embedding variable: it is 1 in every axis, but for the
	// sequence dimension -- there will be one embedding per position.
	// Shape: [1, maxLen, embedDim]
	posEmbedShape := shape.Copy()
	posEmbedShape.Dimensions[0] = 1
	posEmbedVar := ctx.VariableWithShape("positional", posEmbedShape)
	posEmbed := posEmbedVar.ValueGraph(g)
	embed = Add(embed, posEmbed) // Just add the embeddings, seems to work well.

	// Add the requested number of attention layers.
	for ii := 0; ii < *flagNumAttLayers; ii++ {
		// Each layer in its own scope.
		ctx := ctx.In(fmt.Sprintf("AttLayer_%d", ii))
		residual := embed
		embed = layers.MultiHeadAttention(ctx, embed, embed, embed, *flagNumAttHeads, *flagAttKeyQueryEmbedDim).
			SetKeyMask(mask).SetQueryMask(mask).
			SetOutputDim(embedDim).
			SetValueHeadDim(embedDim).Done()
		if *flagDropoutRate > 0 {
			embed = layers.Dropout(ctx.In("dropout_1"), embed, dropoutRate)
		}
		embed = Normalize(ctx.In("normalization_1"), embed)
		attentionOutput := embed

		// Transformers recipe: 2 dense layers after attention.
		embed = layers.Dense(ctx.In("ffn_1"), embed, true, embedDim)
		embed = Tanh(embed)
		embed = layers.Dense(ctx.In("ffn_2"), embed, true, embedDim)
		if *flagDropoutRate > 0 {
			embed = layers.Dropout(ctx.In("dropout_1"), embed, dropoutRate)
		}
		embed = Add(embed, attentionOutput)
		embed = Normalize(ctx.In("normalization_2"), embed)

		// Residual connection: not part of the usual transformer layer ...
		if ii > 0 {
			embed = Add(residual, embed)
		}
	}
	return embed
}

// MaskedWordTaskGraph builds the computation graph for the predicting hidden word unsupervised task.
func MaskedWordTaskGraph(ctx *context.Context, tokens, embed, mask *Node,
	transformerFn func(input, mask *Node) *Node) {
	g := embed.Graph()
	batchSize := embed.Shape().Dimensions[0]
	//maxSeqSize := embed.Shape().Dimensions[1]
	embedDim := embed.Shape().Dimensions[2]
	vocabSize := len(imdb.LoadedVocab.ListEntries)

	seqSize := ConvertType(mask, DType)
	seqSize = ReduceSum(seqSize, -1)
	//seqSize.SetLogged("0. seqSize")

	// choice: shape=[batch_size]
	choice := ctx.RandomUniform(g, seqSize.Shape())
	choice = Mul(seqSize, choice)
	choice = ConvertType(choice, shapes.I64)
	//choice.SetLogged("1. choice")

	// Find indices to gather the word token and later the word embedding.
	expandedChoice := ExpandDims(choice, -1)
	batchIndices := Iota(g, shapes.Make(shapes.I64, batchSize, 1), 0)
	indices := Concatenate([]*Node{batchIndices, expandedChoice}, -1) // shape=[batchSize, 2]
	wordToken := ExpandDims(Gather(tokens, indices), -1)              // [batchSize, 1]

	// wordMask: shape=[batch_size, seq_size]
	broadcastChoice := BroadcastToDims(expandedChoice, mask.Shape().Dimensions...)
	wordMask := Iota(g, shapes.Make(shapes.I64, mask.Shape().Dimensions...), 1)
	wordMask = Equal(wordMask, broadcastChoice)

	// wordMaskedEmbed: shape=[batch_size, seq_size, embedding_dim]
	// It will differ from embed only in chosen masked word position, for which it
	// takes some learned embedding to represent a masked word.
	maskedWordEmbeddingVar := ctx.VariableWithShape("masked_embedding", shapes.Make(DType /* batch */, 1 /* seq_size */, 1, embedDim))
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

// Normalize `x` according to `--norm` flag. Works for sequence nodes (rank-3) or plain feature nodes (rank-2).
func Normalize(ctx *context.Context, x *Node) *Node {
	switch *flagNormalization {
	case "layer":
		if x.Rank() == 3 {
			// Normalize sequence.
			return layers.LayerNormalization(ctx, x, -2, -1).
				LearnedOffset(true).LearnedScale(true).ScaleNormalization(true).Done()
		} else {
			// Normalize features only.
			return layers.LayerNormalization(ctx, x, -1).Done()
		}
	case "batch":
		return layers.BatchNormalization(ctx, x, -1).Done()
	case "none":
		return x
	}
	exceptions.Panicf("invalid normalization selected %q -- valid values are batch, layer, none", *flagNormalization)
	return nil
}
