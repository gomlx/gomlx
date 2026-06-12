// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package train

import (
	"slices"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
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
// Usually this method is not used directly, instead use norm.UpdateBatchNormAverages.
//
// DistributedDatasets are not accepted yet -- open an issue if you need it.
func (r *Trainer) BatchNormalizationAveragesUpdate(ds Dataset) error {
	var err error
	for phase := range 2 {
		// Reset models from previous phases.
		r.batchNormStepExecMap = make(map[any]*model.Exec)
		err = r.resetEvalMetrics()
		if err != nil {
			return err
		}
		count := 0
		for batch, err := range ds.Iter() {
			if err != nil {
				return errors.Wrapf(err,
					"dataset returned an error during BatchNormalizationAveragesUpdate(phase=%d)", phase)
			}
			count++
			err = r.batchNormAveragesStep(phase, batch)
			if err != nil {
				return errors.WithMessagef(err, "BatchNormalizationAveragesUpdate(phase=%d) failed", phase)
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
func (r *Trainer) batchNormAveragesStep(phase int, batch Batch) error {
	lossAndMetrics, err := r.callGraphFn(
		r.batchNormsAverageStepGraphFn(phase),
		BatchNormAveragesType,
		r.batchNormStepExecMap,
		batch,
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
// It is called by the scope executor (Trainer.batchNormStepExecMap)
// inputsAndLabel[:-1] are the inputs, and inputsAndLabel[-1] is the labels batch.
func (r *Trainer) batchNormsAverageStepGraphFn(
	phase int,
) func(spec any, scope *model.Scope, inputs, labels []*graph.Node) (metrics []*graph.Node) {
	return func(spec any, scope *model.Scope, inputs, labels []*graph.Node) (metrics []*graph.Node) {
		g := inputs[0].Graph()
		scope.Store().SetTraining(g, false)
		scope.SetGraphParam(g, BatchNormalizationUpdatePhase, phase)

		predictions := r.modelFn(scope, spec, inputs)
		metrics = slices.Clone(predictions)
		if r.lossFn != nil {
			loss := r.lossFn(labels, predictions)
			if !loss.Shape().IsScalar() {
				loss = graph.ReduceAllMean(loss)
			}
			AddLoss(loss)
			metrics = append(metrics, loss)
		}
		return
	}
}
