package ogbnmag

// Implements OGBN-MAG model layer-wise inference.

import (
	"fmt"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/examples/notebook/gonb/plots"
	"github.com/gomlx/gomlx/examples/ogbnmag/gnn"
	"github.com/gomlx/gomlx/examples/ogbnmag/sampler"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/initializers"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/types/shapes"
	"k8s.io/klog/v2"
	"time"
)

// LayerWiseEvaluation returns the train, validation and test accuracy of the model, using layer-wise inference.
func LayerWiseEvaluation(ctx *context.Context, strategy *sampler.Strategy) (train, validation, test float64) {
	var predictionsT *tensors.Tensor
	exec := context.NewExec(ctx.Backend(), ctx.Reuse(), BuildLayerWiseInferenceModel(strategy, true))

	if klog.V(1).Enabled() {
		// Report timings.
		start := time.Now()
		exec.PreCompile()
		elapsed := time.Since(start)
		klog.Infof("Layer-wise inference elapsed time (computation graph compilation): %s\n", elapsed)

		start = time.Now()
		predictionsT = exec.Call()[0]
		elapsed = time.Since(start)
		klog.Infof("Layer-wise inference elapsed time (execution): %s\n", elapsed)
	} else {
		// Just call inference.
		predictionsT = exec.Call()[0]
	}

	predictions := predictionsT.Local().Value().([]int16)
	labels := PapersLabels.Local().FlatCopy().([]int32)
	return layerWiseCalculateAccuracies(predictions, labels)
}

func layerWiseCalculateAccuracies(predictions []int16, labels []int32) (train, validation, test float64) {
	splitVars := []*float64{&train, &validation, &test}
	for splitIdx, splitT := range []*tensors.Tensor{TrainSplit, ValidSplit, TestSplit} {
		split := splitT.Local().FlatCopy().([]int32)
		numCorrect := 0
		for _, paperIdx := range split {
			if int(predictions[paperIdx]) == int(labels[paperIdx]) {
				numCorrect++
			}
		}
		*splitVars[splitIdx] = float64(numCorrect) / float64(len(split))
	}
	return
}

func BuildLayerWiseCustomMetricFn(ctx *context.Context, strategy *sampler.Strategy) plots.CustomMetricFn {
	exec := context.NewExec(ctx.Backend(), ctx.Reuse(), BuildLayerWiseInferenceModel(strategy, true))
	ctx = ctx.Reuse()
	labels := PapersLabels.Local().FlatCopy().([]int32)
	return func(plotter plots.Plotter, step float64) error {
		predictions := exec.Call()[0].Local().Value().([]int16)
		train, validation, test := layerWiseCalculateAccuracies(predictions, labels)
		accuracies := []float64{train, validation, test}
		names := []string{"Train", "Validation", "Test"}
		for ii, acc := range accuracies {
			plotter.AddPoint(plots.Point{
				MetricName: fmt.Sprintf("%s: layer-wise eval", names[ii]),
				MetricType: "accuracy",
				Step:       step,
				Value:      acc,
			})
		}
		return nil
	}
}

// BuildLayerWiseInferenceModel returns a function that builds the OGBN-MAG GNN inference model,
// that expects to run inference on the whole dataset in one go.
//
// It takes as input the [sampler.Strategy], and returns a function that can be used with `context.NewExec`
// and executed with the values of the MAG graph. Batch size is irrelevant.
//
// The returned function returns the predictions for all seeds shaped `Int16[NumSeedNodes]` if `predictions == true`,
// or the readout layer shaped `Float32[NumSeedNodes, mag.NumLabels]` (or Float16) if `predictions == false`.
func BuildLayerWiseInferenceModel(strategy *sampler.Strategy, predictions bool) func(ctx *context.Context, g *Graph) *Node {
	return func(ctx *context.Context, g *Graph) *Node {
		ctx = ctx.WithInitializer(initializers.GlorotUniformFn(initializers.NoSeed))

		// Create inputs with all elements. Similar to the code in [sampler.Dataset.Yield].
		inputs := make([]*Node, 0, 5*len(strategy.Rules))
		inputs = createInputsWithAllStates(g, strategy, inputs)
		inputs = createEdgesInputs(ctx, g, strategy, inputs)

		// Input preprocessing and re-organize graph states into a map. Masks are dropped, it's assumed to be dense.
		maskedGraphStates, inputs := FeaturePreprocessing(ctx, strategy, inputs)
		graphStates := make(map[string]*Node, len(maskedGraphStates))
		for stateName, state := range maskedGraphStates {
			graphStates[stateName] = state.Value
		}
		edges, _ := sampler.MapInputsToEdges(strategy, inputs)

		// Create layer-wise inference graph.
		lw, err := gnn.LayerWiseGNN(ctx, strategy)
		if err != nil {
			panic(err)
		}
		lw.NodePrediction(ctx, graphStates, edges) // Last layer outputs the logits for the `NumLabels` classes.
		readoutState := graphStates[strategy.Seeds[0].Name]
		readoutState = layers.DenseWithBias(ctx.In("logits"), readoutState, NumLabels)
		if predictions {
			return ArgMax(readoutState, -1, dtypes.Int16)
		}
		return readoutState
	}
}

// createInputsWithAllStates with masks and degrees are set to nil and append to the [inputs] slice.
func createInputsWithAllStates(g *Graph, strategy *sampler.Strategy, inputs []*Node) []*Node {
	for _, seedsRule := range strategy.Seeds {
		inputs = append(inputs, IotaFull(g, shapes.Make(dtypes.Int32, int(seedsRule.NumNodes))))
		inputs = append(inputs, nil) // mask is nil
		inputs = recursivelyCreateInputsWithAllStates(g, seedsRule, inputs)
	}
	return inputs
}

func recursivelyCreateInputsWithAllStates(g *Graph, rule *sampler.Rule, inputs []*Node) []*Node {
	for _, subRule := range rule.Dependents {
		if subRule.NumNodes == 0 {
			Panicf("Rule %q has 0 nodes configured.", subRule.Name)
		}
		inputs = append(inputs, IotaFull(g, shapes.Make(dtypes.Int32, int(subRule.NumNodes))))
		inputs = append(inputs, nil) // mask is nil
		if subRule.Strategy.KeepDegrees {
			inputs = append(inputs, nil) // degree is nil
		}
		inputs = recursivelyCreateInputsWithAllStates(g, subRule, inputs)
	}
	return inputs
}

// createEdgesInputs create the edges pairs (source indices, target indices) for each of the edge rules (non-seed).
func createEdgesInputs(ctx *context.Context, g *Graph, strategy *sampler.Strategy, inputs []*Node) []*Node {
	edges := createEdgesIndices(ctx, g)
	for _, seedsRule := range strategy.Seeds {
		// seedRule doesn't have a connecting edge.
		inputs = recursivelyCreateEdgesInputs(g, seedsRule, edges, inputs)
	}
	return inputs
}

func createEdgesIndices(ctx *context.Context, g *Graph) map[string]sampler.EdgePair[*Node] {
	edges := make(map[string]sampler.EdgePair[*Node])
	for _, edgeName := range []string{"Writes", "AffiliatedWith", "Cites", "HasTopic"} {
		edgeVar := getMagVar(ctx, g, "Edges"+edgeName)
		edges[edgeName] = sampler.EdgePair[*Node]{
			SourceIndices: Slice(edgeVar, AxisRange(), AxisElem(0)),
			TargetIndices: Slice(edgeVar, AxisRange(), AxisElem(1)),
		}
	}
	return edges
}

func recursivelyCreateEdgesInputs(g *Graph, rule *sampler.Rule, edges map[string]sampler.EdgePair[*Node], inputs []*Node) []*Node {
	for _, subRule := range rule.Dependents {
		var (
			edgeName string
			reversed bool
		)
		if subRule.EdgeType == nil {
			// Identity rule: edges are 1-to-1 mapping.
			indices := IotaFull(g, shapes.Make(dtypes.Int32, int(subRule.NumNodes), 1))
			inputs = append(inputs, indices, indices)
		} else {
			// Normal edge.
			switch subRule.EdgeType.Name {
			case "writes":
				edgeName, reversed = "Writes", false
			case "writtenBy":
				edgeName, reversed = "Writes", true
			case "cites":
				edgeName, reversed = "Cites", false
			case "citedBy":
				edgeName, reversed = "Cites", true
			case "affiliatedWith":
				edgeName, reversed = "AffiliatedWith", false
			case "affiliations":
				edgeName, reversed = "AffiliatedWith", true
			case "hasTopic":
				edgeName, reversed = "HasTopic", false
			case "topicHasPapers":
				edgeName, reversed = "HasTopic", true
			default:
				Panicf("Unknown edge name %q, can't generate its inputs", rule.EdgeType.Name)
			}
			if !reversed {
				inputs = append(inputs, edges[edgeName].SourceIndices, edges[edgeName].TargetIndices)
			} else {
				inputs = append(inputs, edges[edgeName].TargetIndices, edges[edgeName].SourceIndices)
			}
		}
		inputs = recursivelyCreateEdgesInputs(g, subRule, edges, inputs)
	}
	return inputs
}
