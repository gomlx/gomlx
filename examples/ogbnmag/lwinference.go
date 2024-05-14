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
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"time"
)

func LayerWiseInference(ctx *context.Context, strategy *sampler.Strategy) tensor.Tensor {
	var predictionsT tensor.Tensor
	exec := context.NewExec(ctx.Manager(), ctx.Reuse(), BuildLayerWiseInferenceModel(strategy, true))
	for _ = range 1 {
		start := time.Now()
		predictionsT = exec.Call()[0]
		fmt.Printf("predicitons.shape=%s\n", predictionsT.Shape())
		elapsed := time.Since(start)
		fmt.Printf("\tElapsed time: %s\n", elapsed)
	}

	predictions := predictionsT.Local().Value().([]int16)
	labels := PapersLabels.Local().FlatCopy().([]int32)
	splitNames := []string{"Train", "Validation", "Test"}
	for splitIdx, splitT := range []tensor.Tensor{TrainSplit, ValidSplit, TestSplit} {
		split := splitT.Local().FlatCopy().([]int32)
		numCorrect := 0
		for _, paperIdx := range split {
			if int(predictions[paperIdx]) == int(labels[paperIdx]) {
				numCorrect++
			}
		}
		fmt.Printf("%s Accuracy: %.2f%%\n", splitNames[splitIdx], 100.0*float64(numCorrect)/float64(len(split)))
	}
	return predictionsT
}

// BuildLayerWiseInferenceModel returns a function that builds the OGBN-MAG GNN inference model,
// that expects to run inference on the whole dataset in one go.
//
// It takes as input the [sampler.Strategy], and returns a function that can be used with `context.NewExec`
// and executed with the values of the MAG graph.
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
			return ArgMax(readoutState, -1, shapes.Int16)
		}
		return readoutState
	}
}

// createInputsWithAllStates with masks and degrees are set to nil and append to the [inputs] slice.
func createInputsWithAllStates(g *Graph, strategy *sampler.Strategy, inputs []*Node) []*Node {
	for _, seedsRule := range strategy.Seeds {
		inputs = append(inputs, IotaFull(g, shapes.Make(shapes.I32, int(seedsRule.NumNodes))))
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
		inputs = append(inputs, IotaFull(g, shapes.Make(shapes.I32, int(subRule.NumNodes))))
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
			indices := IotaFull(g, shapes.Make(shapes.I32, int(subRule.NumNodes), 1))
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
