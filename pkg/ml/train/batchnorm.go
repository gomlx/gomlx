package train

import (
	"io"
	"slices"

	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
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
//
// DistributedDatasets are not accepted yet -- open an issue if you need it.
func (r *Trainer) BatchNormalizationAveragesUpdate(ds Dataset) error {
	var err error
	for phase := range 2 {
		// Reset models from previous phases.
		r.batchNormStepExecMap = make(map[any]*context.Exec)
		ds.Reset()
		err = r.resetEvalMetrics()
		if err != nil {
			return err
		}
		count := 0
		for {
			spec, inputs, labels, err := ds.Yield()
			if err == io.EOF {
				break
			}
			if err != nil {
				return errors.Wrapf(err,
					"dataset returned an error during BatchNormalizationAveragesUpdate(phase=%d)", phase)
			}
			count++
			err = r.batchNormAveragesStep(phase, spec, inputs, labels)
			if err != nil {
				return errors.WithMessagef(err, "BatchNormalizationAveragesUpdate(phase=%d) failed", phase)
			}

			// Free inputs and labels after usage.
			for sliceIdx, slice := range [][]*tensors.Tensor{inputs, labels} {
				for i, t := range slice {
					err := t.FinalizeAll()
					if err != nil {
						return errors.WithMessagef(
							err, "finalizing %s tensor #%d of dataset %q after use in a distributed eval step",
							yieldInputTypeNames[sliceIdx], i, ds.Name())
					}
				}
			}
		}
		if count == 0 {
			return errors.Errorf("BatchNormalizationAveragesUpdate: dataset yielded no batches, no data to calculate " +
				"running mean/average")
		}
	}
	return nil
}

// batchNormAveragesStep runs one forward step on the model, with the model frozen, except
// for non-gradient updated variables, like batch normalization moving averages.
func (r *Trainer) batchNormAveragesStep(phase int, spec any, inputs, labels []*tensors.Tensor) error {
	lossAndMetrics, err := r.callGraphFn(
		r.batchNormsAverageStepGraphFn(phase),
		BatchNormAveragesType,
		r.batchNormStepExecMap,
		spec,
		inputs,
		labels,
	)
	if err != nil {
		return err
	}
	for _, t := range lossAndMetrics {
		t.MustFinalizeAll()
	}
	return nil
}

// batchNormsAverageStepGraph builds the graph to eval one step, in training mode, so variables are allowed to be updates.
// It is called by the context executor (Trainer.batchNormStepExecMap)
// inputsAndLabel[:-1] are the inputs, and inputsAndLabel[-1] is the labels batch.
func (r *Trainer) batchNormsAverageStepGraphFn(
	phase int,
) func(spec any, ctx *context.Context, inputs, labels []*graph.Node) (metrics []*graph.Node) {
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
