package gnn

import (
	"fmt"
	. "github.com/gomlx/exceptions"
	mag "github.com/gomlx/gomlx/examples/ogbnmag"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
)

var (
	// ParamMagGnnNumMessages is the context parameter that defines the number of messages to
	// send in the GNN tree of nodes.
	// The default is 4.
	ParamMagGnnNumMessages = "mag_gnn_num_messages"
)

// getMagVar retrieves the static (not-learnable) OGBN-MAG variables -- e.g: the frozen papers embedding table.
func getMagVar(ctx *context.Context, g *Graph, name string) *Node {
	magVar := ctx.InspectVariable(mag.OgbnMagVariablesScope, name)
	if magVar == nil {
		Panicf("Missing OGBN-MAG dataset variables (%q), pls call UploadOgbnMagVariables() on context first.", name)
	}
	return magVar.ValueGraph(g)
}

// MagModelGraph builds a OGBN-MAG GNN model that sends [ParamMagGnnNumMessages] along its sampling
// strategy, and then adding a final layer on top of the seeds.
//
// It returns 3 tensors:
// * Predictions for all seeds shaped `Float32[BatchSize, mag.NumLabels]`.
// * Mask of the seeds, provided by the sampler, shaped `Bool[BatchSize]`.
func MagModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	g := inputs[0].Graph()
	optimizers.CosineAnnealingSchedule(ctx, g, shapes.F32)
	strategy := spec.(*sampler.Strategy)
	graphStates := FeaturePreprocessing(ctx, strategy, inputs)
	numMessages := context.GetParamOr(ctx, ParamMagGnnNumMessages, 4)
	for ii := range numMessages {
		GraphStateUpdate(ctx.In(fmt.Sprintf("round_%d", ii)), strategy, graphStates)
	}
	readoutState := graphStates[strategy.Seeds[0].Name]
	// Add one more hidden layer on the readout, by updating using itself as input.
	readoutState.Value = updateState(ctx.In("readout_hidden"), readoutState.Value, readoutState.Value, readoutState.Mask)
	readoutState.Value = layers.DenseWithBias(ctx.In("readout"), readoutState.Value, mag.NumLabels)
	return []*Node{readoutState.Value, readoutState.Mask}
}

// FeaturePreprocessing converts the `spec` and `inputs` given by the dataset into a map of node type name to
// its initial embeddings.
//
// TODO: Add dropout of embeddings. Both Institutions and FieldsOfStudy sometimes are associated to only one
//
//	author/paper, so it is reasonable to expect that during validation/testing it will see many embeddings
//	zero initialized.
func FeaturePreprocessing(ctx *context.Context, strategy *sampler.Strategy, inputs []*Node) (graphInputs map[string]*sampler.ValueMask[*Node]) {
	g := inputs[0].Graph()
	graphInputs = sampler.MapInputs[*Node](strategy, inputs)

	// Learnable embeddings context: zero initialized, and we should have dropout to have the model handle well
	// the cases of unknown (zero) embeddings.
	ctxEmbed := ctx.In("embeddings").Checked(false).WithInitializer(initializers.Zero)

	// Preprocess papers to its features --> these are in a frozen embedding table in the context as a frozen variable.
	papersEmbeddings := getMagVar(ctx, g, "PapersEmbeddings")
	for name, rule := range strategy.Rules {
		if rule.NodeTypeName == "papers" {
			// Gather values from frozen paperEmbeddings. Mask remains unchanged.
			graphInputs[name].Value = Gather(papersEmbeddings, ExpandDims(graphInputs[name].Value, -1))
		}
	}

	// Preprocess institutions to its embeddings.
	institutionsEmbedSize := context.GetParamOr(ctx, "InstitutionsEmbedSize", 16)
	for name, rule := range strategy.Rules {
		if rule.NodeTypeName == "institutions" {
			// Gather values from frozen paperEmbeddings. Mask remains unchanged.
			embedded := layers.Embedding(ctxEmbed.In("institutions"),
				graphInputs[name].Value, shapes.F32, mag.NumInstitutions, institutionsEmbedSize)
			embedded = Where(graphInputs[name].Mask, embedded, ZerosLike(embedded)) // Apply mask.
			graphInputs[name].Value = embedded
		}
	}

	// Preprocess "field of study" to its embeddings.
	fieldsOfStudyEmbedSize := context.GetParamOr(ctx, "FieldsOfStudyEmbedSize", 32)
	for name, rule := range strategy.Rules {
		if rule.NodeTypeName == "fields_of_study" {
			// Gather values from frozen paperEmbeddings. Mask remains unchanged.
			embedded := layers.Embedding(ctxEmbed.In("fields_of_study"),
				graphInputs[name].Value, shapes.F32, mag.NumFieldsOfStudy, fieldsOfStudyEmbedSize)
			embedded = Where(graphInputs[name].Mask, embedded, ZerosLike(embedded)) // Apply mask.
			graphInputs[name].Value = embedded
		}
	}

	// Preprocess "authors": it's purely latent -- meaning it has no information by itself, not even embeddings.
	// All the information will be propagated by the graph.
	// So their value (but not the mask) are replaced with nil.
	for name, rule := range strategy.Rules {
		if rule.NodeTypeName == "authors" {
			// Gather values from frozen paperEmbeddings. Mask remains unchanged.
			graphInputs[name].Value = nil
		}
	}
	return
}
