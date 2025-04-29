package ogbnmag

import (
	"fmt"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/examples/ogbnmag/gnn"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/ml/train/optimizers/cosineschedule"
	"github.com/gomlx/gopjrt/dtypes"
	"k8s.io/klog/v2"
)

var (
	// ParamEmbedDropoutRate adds an extra dropout to learning embeddings.
	// This may be important because many embeddings are seen only once, so likely in testing many will have never
	//  been seen, and we want the model learn how to handle lack of embeddings (zero initialized) well.
	ParamEmbedDropoutRate = "mag_embed_dropout_rate"

	// ParamSplitEmbedTablesSize will make embed tables share entries across these many entries.
	// Default is 1, which means no splitting.
	ParamSplitEmbedTablesSize = "mag_split_embed_tables"
)

// getMagVar retrieves the static (not-learnable) OGBN-MAG variables -- e.g: the frozen papers embedding table.
func getMagVar(ctx *context.Context, g *Graph, name string) *Node {
	magVar := ctx.GetVariableByScopeAndName(OgbnMagVariablesScope, name)
	if magVar == nil {
		Panicf("Missing OGBN-MAG dataset variables (%q), pls call UploadOgbnMagVariables() on context first.", name)
		panic(nil) // Quiet linter.
	}
	return magVar.ValueGraph(g)
}

// logitsGraph converts the readout state of the seed nodes to its logits.
func logitsGraph(ctx *context.Context, readout *Node) *Node {
	//useKan := context.GetParamOr(ctx, "kan", false)
	//if useKan {
	//	readout = kan.New(ctx.In("logits_kan"), readout, NumLabels).NumHiddenLayers(0, 0).Done()
	//} else {
	//	// Normal FNN
	//	readout = layers.DenseWithBias(ctx.In("logits"), readout, NumLabels)
	//}
	readout = layers.DenseWithBias(ctx.In("logits"), readout, NumLabels)
	return readout
}

// MagModelGraph builds a OGBN-MAG GNN model that sends [ParamNumGraphUpdates] along its sampling
// strategy, and then adding a final layer on top of the seeds.
//
// It returns 2 tensors:
// * Predictions for all seeds shaped `Float32[BatchSize, mag.NumLabels]` (or `Float16` or `Float64`).
// * Mask of the seeds, provided by the sampler, shaped `Bool[BatchSize]`.
func MagModelGraph(ctx *context.Context, spec any, inputs []*Node) []*Node {
	ctx = ctx.WithInitializer(initializers.GlorotUniformFn(ctx))
	dtype := getDType(ctx) // Default is Float32
	g := inputs[0].Graph()
	if klog.V(3).Enabled() {
		// The trace is used below to print the largest node.
		g.SetTraced(true)
	}

	lrDType := dtype
	if adamDType := context.GetParamOr(ctx, optimizers.ParamAdamDType, ""); adamDType != "" {
		var err error
		lrDType, err = dtypes.DTypeString(adamDType)
		if err != nil || !lrDType.IsFloat() {
			Panicf("Cannot parse hyperparameter %s=%q: %v", optimizers.ParamAdamDType, adamDType, err)
		}
	}
	cosineschedule.New(ctx, g, lrDType).FromContext().Done()

	// We disable checking for re-use of scopes because we deliberately reuse
	// kernels in our GNN.
	ctxModel := ctx.In("model").Checked(false)

	strategy := spec.(*sampler.Strategy)
	graphStates, _ := FeaturePreprocessing(ctxModel, strategy, inputs)

	if NanLogger != nil {
		fmt.Println("*** Using NanLogger ***")
	}
	gnn.NanLogger = NanLogger
	gnn.NodePrediction(ctxModel, strategy, graphStates)

	// Last layer outputs the logits for the `NumLabels` classes.
	readoutState := graphStates[strategy.Seeds[0].Name]
	readoutState.Value = logitsGraph(ctxModel, readoutState.Value)

	if klog.V(2).Enabled() {
		// Log the largest non-parameter node.
		var largest *Node
		var largestSize uintptr
		for _, node := range g.Nodes() {
			if node.Type() == NodeTypeParameter || node.NumOutputs() > 1 {
				continue
			}
			if largest == nil || node.Shape().Memory() > largestSize {
				largest = node
				largestSize = node.Shape().Memory()
			}
		}
		if largest != nil {
			klog.Infof("Largest node in graph: #%d %s", largest.Id(), largest)
			klog.V(3).Infof("\n%+v", largest.Trace())
		}
	}

	return []*Node{readoutState.Value, readoutState.Mask}
}

// FeaturePreprocessing converts the `spec` and `inputs` given by the dataset into a map of node type name to
// its initial embeddings.
//
//	author/paper, so it is reasonable to expect that during validation/testing it will see many embeddings
//	zero initialized.
func FeaturePreprocessing(ctx *context.Context, strategy *sampler.Strategy, inputs []*Node) (
	graphInputs map[string]*sampler.ValueMask[*Node], remainingInputs []*Node) {
	g := inputs[0].Graph()
	graphInputs, remainingInputs = sampler.MapInputsToStates[*Node](strategy, inputs)
	dtype := getDType(ctx)
	dtypeEmbed := dtype
	if dtype == dtypes.Float16 || dtype == dtypes.BFloat16 {
		// If we don't do this for Float16, on a 2080ti GPU, the training becomes 3 times slower. Gemini mentioned
		// that the RTX 30 series is better at "scattering" (used on the auto-differentiation of the "gathers" here),
		// and may be worth a try then. But for now, leave it as Float32. Notice this is only an issue on non-sorted
		// gathers/scatters, which is the case here (indices may come randomly).
		dtypeEmbed = dtypes.Float32
	}

	// Learnable embeddings context: it may benefit from dropout to have the model handle well
	// the cases of unknown (zero) embeddings.
	// They shouldn't be initialized with GlorotUniform, but instead with small random uniform values.
	ctxEmbed := ctx.In("embeddings").Checked(false).
		WithInitializer(initializers.RandomUniformFn(ctx, -0.05, 0.05))
	embedDropoutRate := context.GetParamOr(ctx, ParamEmbedDropoutRate, 0.0)

	// Preprocess papers to its features --> these are in a frozen embedding table in the context as a frozen variable.
	papersEmbeddings := getMagVar(ctx, g, "PapersEmbeddings")
	for name, rule := range strategy.Rules {
		if rule.NodeTypeName == "papers" {
			// Gather values from frozen paperEmbeddings. Mask remains unchanged.
			graphInputs[name].Value = Gather(papersEmbeddings, InsertAxes(graphInputs[name].Value, -1))
			if dtype != dtypeEmbed {
				graphInputs[name].Value = ConvertDType(graphInputs[name].Value, dtype)
			}
		}
	}

	// Preprocess institutions to its embeddings.
	institutionsEmbedSize := context.GetParamOr(ctx, "InstitutionsEmbedSize", 16)
	splitEmbedTables := context.GetParamOr(ctx, ParamSplitEmbedTablesSize, 2)
	for name, rule := range strategy.Rules {
		if rule.NodeTypeName == "institutions" {
			// Gather values from frozen paperEmbeddings. Mask remains unchanged.
			indices := DivScalar(graphInputs[name].Value, float64(splitEmbedTables))
			embedded := layers.Embedding(ctxEmbed.In("institutions"), indices,
				dtypeEmbed, (NumInstitutions+splitEmbedTables-1)/splitEmbedTables, institutionsEmbedSize, false)
			if graphInputs[name].Mask != nil {
				embedMask := layers.DropoutStatic(ctx, graphInputs[name].Mask, embedDropoutRate)
				embedded = Where(embedMask, embedded, ZerosLike(embedded)) // Apply mask.
			}
			graphInputs[name].Value = embedded
			if dtype != dtypeEmbed {
				graphInputs[name].Value = ConvertDType(graphInputs[name].Value, dtype)
			}
		}
	}

	// Preprocess "field of study" to its embeddings.
	fieldsOfStudyEmbedSize := context.GetParamOr(ctx, "FieldsOfStudyEmbedSize", 32)
	for name, rule := range strategy.Rules {
		if rule.NodeTypeName == "fields_of_study" {
			// Gather values from frozen paperEmbeddings. Mask remains unchanged.
			indices := DivScalar(graphInputs[name].Value, float64(splitEmbedTables))
			embedded := layers.Embedding(ctxEmbed.In("fields_of_study"),
				indices, dtypeEmbed, (NumFieldsOfStudy+splitEmbedTables-1)/splitEmbedTables,
				fieldsOfStudyEmbedSize, false)

			if graphInputs[name].Mask != nil {
				embedMask := layers.DropoutStatic(ctx, graphInputs[name].Mask, embedDropoutRate)
				embedded = Where(embedMask, embedded, ZerosLike(embedded)) // Apply mask.
			}
			graphInputs[name].Value = embedded
			if dtype != dtypeEmbed {
				graphInputs[name].Value = ConvertDType(graphInputs[name].Value, dtype)
			}
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
