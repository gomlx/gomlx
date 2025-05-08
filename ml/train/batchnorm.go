package train

import (
	"io"
	"slices"

	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/pkg/errors"
)

const (
	// BatchNormalizationUpdatePhase is a graph parameter name used to indicate that the graph is only
	// running to update the mean/average parameters. See batchnorm.UpdateAveragesUpdate method.
	BatchNormalizationUpdatePhase = "batch_normalization_update_phase"
)

// BatchNormalizationAveragesUpdate runs through the dataset ds twice, updating the running averages for mean and variance.
//
// It's recommended one resets the batch normalization weights with layers.BatchNormalizationResetWeights before
// calling this function.
//
// Usually this method is not used directly, instead use batchnorm.UpdateAverages.
func (r *Trainer) BatchNormalizationAveragesUpdate(ds Dataset) {
	for phase := range 2 {
		// Reset models from previous phases.
		r.batchNormStepExecMap = make(map[any]*context.Exec)
		ds.Reset()
		r.resetEvalMetrics()
		count := 0
		for {
			spec, inputs, labels, err := ds.Yield()
			if err == io.EOF {
				break
			}
			if err != nil {
				panic(errors.Wrapf(err, "dataset returned an error during BatchNormalizationAveragesUpdate(phase=%d)", phase))
			}
			count++
			r.batchNormAveragesStep(phase, spec, inputs, labels)

			// Free inputs and labels after usage.
			for _, input := range inputs {
				input.FinalizeAll()
			}
			for _, label := range labels {
				label.FinalizeAll()
			}
		}
		if count == 0 {
			Panicf("BatchNormalizationAveragesUpdate: dataset yielded no batches, no data to calculate running mean/average")
		}
	}
}

// batchNormAveragesStep runs one forward step on the model, with the model frozen, except
// for non-gradient updated variables, like batch normalization moving averages.
func (r *Trainer) batchNormAveragesStep(phase int, spec any, inputs, labels []*tensors.Tensor) {
	lossAndMetrics := r.callGraphFn(r.batchNormsAverageStepGraphFn(phase), BatchNormAveragesType, spec, inputs, labels)
	for _, t := range lossAndMetrics {
		t.FinalizeAll()
	}

}

// batchNormsAverageStepGraph builds the graph to eval one step, in training mode, so variables are allowed to be updates.
// It is called by the context executor (Trainer.batchNormStepExecMap)
// inputsAndLabel[:-1] are the inputs, and inputsAndLabel[-1] is the labels batch.
func (r *Trainer) batchNormsAverageStepGraphFn(phase int) func(spec any, ctx *context.Context, inputs, labels []*graph.Node) (metrics []*graph.Node) {
	return func(spec any, ctx *context.Context, inputs, labels []*graph.Node) (metrics []*graph.Node) {
		g := inputs[0].Graph()
		ctx.SetTraining(g, false)
		ctx.SetGraphParam(g, BatchNormalizationUpdatePhase, phase)

		predictions := r.modelFn(ctx, spec, inputs)
		metrics = slices.Clone(predictions)
		if r.lossFn != nil {
			loss := r.lossFn(labels, predictions)
			if !loss.Shape().IsScalar() {
				loss = graph.ReduceAllMean(loss)
			}
			AddLoss(ctx, loss)
			metrics = append(metrics, loss)
		}
		return
	}
}
