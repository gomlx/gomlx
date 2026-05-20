// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package train

import (
	"io"
	"iter"
	"math"
	"slices"
	"time"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/distributed"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/core/tensors/dtensor"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/metrics"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/pkg/errors"
)

// Priority for hooks, the lowest values are run first. Defaults to 0, but negative
// values are ok.
type Priority int

// OnStartFn is the type of OnStart hooks.
type OnStartFn func(loop *Loop, ds Dataset) error

// OnStepFn is the type of OnStep hooks.
type OnStepFn func(loop *Loop, metrics []*tensors.Tensor) error

// OnEndFn is the type of OnEnd hooks.
type OnEndFn func(loop *Loop, metrics []*tensors.Tensor) error

// Loop will run a training loop, invoking Trainer.TrainStep every step,
// and calling the appropriate hooks.
//
// It also converts graph building errors thrown with `panic` and return them
// instead as normal errors.
//
// By itself it doesn't do much, but one can attach functionality to it, like
// checkpointing, plotting tools, early-stopping strategies, etc.
// It is simple and flexible to allow arbitrary tools to the training loop.
//
// The public attributes are meant for reading only, don't change them -- behavior
// can be undefined.
type Loop struct {
	// Trainer associated with this loop. In particular Trainer.TrainMetrics() and
	// Trainer.EvalMetrics() can be of interest.
	// TODO: make Trainer an interface, so Loop can work on custom trainers..
	Trainer *Trainer

	// LoopStep currently being executed.
	// It is initialized with the current store's `GlobalStep`, which will be 0 for a new model.
	//
	// Notice if using Trainer.AccumulateGradients: this is a measure of "train steps", and "global steps"
	// will be incremented only every "numAccumulatedGradients" train steps are made.
	LoopStep int

	// StartStep is the value of LoopStep at the start of a run (RunSteps or RunEpochs).
	//
	// It is only set and valid during a run (Loop.RunSteps or Loop.RunEpochs).
	StartStep int

	// EndStep is one-past the last step to be executed. If -1 the end step is not known (if
	// running till the end of the dataset). When running for multiple epochs (Loop.RunEpochs) it can
	// change during the run (after the first epoch, the value is extrapolated based on how many steps
	// have been run so far).
	//
	// It is only set and valid during a run (Loop.RunSteps or Loop.RunEpochs).
	EndStep int

	// Epoch is set when running Loop.RunEpochs() to the current running epoch, starting from 0.
	Epoch int

	// SharedData allows for cross-tools to publish and consume information. Keys (strings)
	// and semantics/type of their values are not specified by loop.
	SharedData map[string]any

	// trainStepDurations collected during training
	TrainStepDurations []time.Duration

	// Registered hooks.
	onStart *priorityHooks[*hookWithName[OnStartFn]]
	onStep  *priorityHooks[*hookWithName[OnStepFn]]
	onEnd   *priorityHooks[*hookWithName[OnEndFn]]

	// finalizeYieldedTrainTensors indicates whether the training datasets yielded tensors should be finalized.
	// True by default.
	finalizeYieldedTrainTensors bool
}

// NewLoop creates a new training loop trainer.
func NewLoop(trainer *Trainer) *Loop {
	loop := &Loop{
		Trainer:    trainer,
		SharedData: make(map[string]any),
		onStart:    newPriorityHooks[*hookWithName[OnStartFn]](),
		onStep:     newPriorityHooks[*hookWithName[OnStepFn]](),
		onEnd:      newPriorityHooks[*hookWithName[OnEndFn]](),
		LoopStep:   int(trainer.GlobalStep()),
	}
	if trainer.NumAccumulatingSteps() > 1 {
		loop.LoopStep = loop.LoopStep * trainer.NumAccumulatingSteps()
	}
	return loop
}

// TrainLastStepVarName is the name of the variable that holds the number of the target last GlobalStep.
// This variable is set by the Loop trainer, and may be -1 if the last train step is not known yet (for instance
// if using RunEpochs).
//
// It is stored in the TrainerAbsoluteScope.
const TrainLastStepVarName = "train_last_global_step"

// GetTrainLastStepVar returns the variable that holds the number of the target last GlobalStep.
// This variable is set by the Loop trainer, and may be -1 if the last train step is not known yet (for instance
// if using RunEpochs).
//
// The Store object will be read from the Graph.
//
// It is stored in the TrainerAbsoluteScope.
//
// This is a graph building function and so it may panic if the variable cannot be created.
func GetTrainLastStepVar(g *graph.Graph) *model.Variable {
	store := model.GetStore(g)
	if store == nil {
		panic(errors.Errorf("the current graph does not have an associated model.Store; to get one, you need to create the graph using a [model.Exec] (see [model.NewExec])"))
	}
	return store.Scope(TrainerAbsoluteScope).
		VariableWithValue(TrainLastStepVarName, int64(-1)).
		SetTrainable(false)
}

// start of loop, called by all looping methods.
//
// It calls the appropriate hooks.
func (loop *Loop) start(ds Dataset) error {
	for hook := range loop.onStart.All() {
		err := hook.fn(loop, ds)
		if err != nil {
			return errors.WithMessagef(err, "OnStart(hook %q)", hook.name)
		}
	}
	return nil
}

// step of loop, called by all looping methods.
// It calls the appropriate hooks.
func (loop *Loop) step(spec any, inputs, labels []*tensors.Tensor) (metrics []*tensors.Tensor, err error) {
	startTime := time.Now()
	defer func() {
		elapsed := time.Since(startTime)
		loop.TrainStepDurations = append(loop.TrainStepDurations, elapsed)
	}()

	metrics, err = loop.Trainer.TrainStep(spec, inputs, labels)
	if err != nil {
		return nil, err
	}

	// Free inputs and labels:
	if loop.finalizeYieldedTrainTensors {
		for sliceIdx, slice := range [][]*tensors.Tensor{inputs, labels} {
			for i, t := range slice {
				err := t.FinalizeAll()
				if err != nil {
					return nil, errors.WithMessagef(
						err, "finalizing tensor #%d of %s after use in a distributed train step",
						i, yieldInputTypeNames[sliceIdx])
				}
			}
		}
	}

	err = loop.postStep(metrics)
	if err != nil {
		return nil, err
	}

	return metrics, nil
}

// distributedStep of loop, called by all looping methods.
// It calls the appropriate hooks.
func (loop *Loop) distributedStep(
	strategy distributed.Strategy, deviceAssignment []compute.DeviceNum,
	spec any, inputs, labels []*dtensor.Tensor) (metrics []*tensors.Tensor, err error) {
	startTime := time.Now()
	defer func() {
		elapsed := time.Since(startTime)
		loop.TrainStepDurations = append(loop.TrainStepDurations, elapsed)
	}()

	metrics, err = loop.Trainer.DistributedTrainStep(strategy, deviceAssignment, spec, inputs, labels)
	if err != nil {
		return nil, err
	}

	// Free inputs and labels:
	if loop.finalizeYieldedTrainTensors {
		for sliceIdx, slice := range [][]*dtensor.Tensor{inputs, labels} {
			for i, dt := range slice {
				err := dt.Finalize()
				if err != nil {
					return nil, errors.WithMessagef(
						err, "finalizing tensor #%d of %s after use in a distributed train step",
						i, yieldInputTypeNames[sliceIdx])
				}
			}
		}
	}

	err = loop.postStep(metrics)
	if err != nil {
		return nil, err
	}

	return metrics, nil
}

// postStep materializes the metrics locally (and frees their on-device storage) and calls the onStep hooks.
// It also checks for NaN loss, and returns an error accordingly.
func (loop *Loop) postStep(metrics []*tensors.Tensor) error {
	// Free metrics on-device usage: on-device memory is at premium,
	// we want to immediately free things that are no longer used there.
	for _, m := range metrics {
		m.MaterializeLocal()
		err := m.InvalidateOnDevice()
		if err != nil {
			return err
		}
	}

	// Call "OnStep" hooks.
	for hook := range loop.onStep.All() {
		err := hook.fn(loop, metrics)
		if err != nil {
			return errors.WithMessagef(err, "train.Loop.OnStep(hook %q)", hook.name)
		}
	}

	batchLoss := shapes.ConvertTo[float64](metrics[0].Value())
	if math.IsNaN(batchLoss) {
		return errors.Errorf("batch loss is NaN, training interrupted")
	}
	if math.IsInf(batchLoss, 0) {
		return errors.Errorf("batch loss is infinity (%f), training interrupted", batchLoss)
	}
	return nil
}

// setLastStep, both the field in Loop but also the corresponding variable in the model.
func (loop *Loop) setLastStep(lastStep int) error {
	loop.EndStep = lastStep
	var endStepVar *model.Variable
	err := exceptions.TryCatch[error](func() {
		endStepVar = loop.Trainer.Store().Scope(TrainerAbsoluteScope).
			VariableWithValue(TrainLastStepVarName, int64(-1)).
			SetTrainable(false)
	})
	if err != nil {
		return err
	}
	return endStepVar.SetValue(tensors.FromScalar(int64(loop.EndStep)))
}

// end of loop, called by all looping methods.
// It calls the appropriate hooks.
func (loop *Loop) end(metrics []*tensors.Tensor) error {
	for hook := range loop.onEnd.All() {
		if err := hook.fn(loop, metrics); err != nil {
			return errors.WithMessagef(err, "OnEnd(hook %q)", hook.name)
		}
	}
	return nil
}

// finalizeYieldedTensors checks whether for this datasets the yielded tensors should be finalized.
func finalizeYieldedTensors(ds Dataset) bool {
	dsOwnership, ok := ds.(DatasetCustomOwnership)
	if !ok {
		// Default is true, if not otherwise configured.
		return true
	}
	return dsOwnership.IsOwnershipTransferred()
}

var yieldInputTypeNames = []string{"inputs", "labels"}

func checkYield(inputs, labels []*tensors.Tensor) error {
	// Check inputs and labels are valid.
	for inputTypeIdx, slice := range [][]*tensors.Tensor{inputs, labels} {
		for tensorIdx, t := range slice {
			if !t.Ok() {
				return errors.Errorf(
					"dataset yielded an invalid tensor (tensor #%d of %s), -- likely it has already been finalized "+
						"(freed). The training loop by default immediately frees the yielded tensor after use, so "+
						"it doesn't wait for the garbage collector. If the dataset is trying to reuse tensors, "+
						"they will become invalid and cause this error. If that is the case, consider implementing "+
						"the method FinalizeYieldsAfterUse() in your dataset, and return false.",
					tensorIdx, yieldInputTypeNames[inputTypeIdx],
				)
			}
		}
	}
	return nil
}

func checkDistributedYield(inputs, labels []*dtensor.Tensor) error {
	// Check inputs and labels are valid.
	for inputTypeIdx, slice := range [][]*dtensor.Tensor{inputs, labels} {
		for tensorIdx, dt := range slice {
			if !dt.Ok() {
				return errors.Errorf(
					"dataset yielded an invalid tensor (tensor #%d of %s), -- likely it has already been finalized "+
						"(freed). The training loop by default immediately frees the yielded tensor after use, so "+
						"it doesn't wait for the garbage collector. If the dataset is trying to reuse tensors, "+
						"they will become invalid and cause this error. If that is the case, consider implementing "+
						"the method FinalizeYieldsAfterUse() in your dataset, and return false.",
					tensorIdx, yieldInputTypeNames[inputTypeIdx],
				)
			}
		}
	}
	return nil
}

// freeMetrics finalizes the metrics and returns an error if any of them cannot be finalized.
func freeMetrics(metricsInterfaces []metrics.Interface, metrics []*tensors.Tensor) error {
	var err error
	for metricIdx, metric := range metrics {
		if metric != nil {
			err = metric.FinalizeAll()
			if err != nil {
				return errors.WithMessagef(err, "failed to finalize metric %q (#%d)",
					metricsInterfaces[metricIdx].Name(), metricIdx)
			}
		}
	}
	return nil
}

// RunToGlobalStep runs the loop until the target global step is reached.
// If targetGlobalStep is smaller than the current global step, it does nothing and returns nil metrics.
func (loop *Loop) RunToGlobalStep(ds Dataset, targetGlobalStep int) (metrics []*tensors.Tensor, err error) {
	scope := loop.Trainer.Store()
	var globalStep int
	err = exceptions.TryCatch[error](func() {
		globalStep = int(optimizer.GetGlobalStep(scope))
	})
	if err != nil {
		return nil, err
	}
	if targetGlobalStep <= globalStep {
		return nil, nil
	}
	steps := targetGlobalStep - globalStep
	return loop.RunSteps(ds, steps)
}

// RunSteps runs those many steps. StartStep and EndStep are adjusted to the current
// LoopStep, so it can be called multiple times, and it will simply pick up where it left of last time.
func (loop *Loop) RunSteps(ds Dataset, steps int) (metrics []*tensors.Tensor, err error) {
	if steps <= 0 {
		return nil, nil
	}
	if err = loop.Trainer.ResetTrainMetrics(); err != nil {
		return
	}
	loop.finalizeYieldedTrainTensors = finalizeYieldedTensors(ds)
	loop.StartStep = loop.LoopStep
	loop.setLastStep(loop.LoopStep + steps)
	err = loop.start(ds)
	if err != nil {
		return nil, err
	}

	if distributedDS, ok := ds.(DistributedDataset); ok {
		// Run steps in a distributed manner.
		metrics, err = loop.runStepsDistributed(distributedDS, steps)
	} else {
		// Run steps on a single device.
		metrics, err = loop.runStepsSingleDevice(ds, steps)

	}
	if err != nil {
		return nil, err
	}
	err = loop.end(metrics)
	if err != nil {
		return nil, errors.WithMessagef(err, "Loop.RunSteps(%d): failed end (GlobaStep=%d)", steps, loop.LoopStep)
	}
	return metrics, nil
}

func (loop *Loop) runStepsSingleDevice(ds Dataset, steps int) (metrics []*tensors.Tensor, err error) {
	loop.TrainStepDurations = make([]time.Duration, 0, steps)
	metricsInterfaces := loop.Trainer.Metrics()
	for loop.LoopStep = loop.StartStep; loop.LoopStep < loop.EndStep; loop.LoopStep++ {
		spec, inputs, labels, err := ds.Yield()
		if err != nil {
			if err == io.EOF {
				return nil, errors.Errorf(
					"reached Dataset end after %d steps (requested %d steps) -- did you mean to use "+
						"a different (looping) Dataset, or use Loop.RunEpochs() instead of Loop.RunSteps() ?",
					loop.LoopStep-loop.StartStep, steps)
			}
			return nil, errors.WithMessagef(err, "Loop.RunSteps(%d): failed reading from Dataset", steps)
		}

		// Check inputs and labels are valid.
		err = checkYield(inputs, labels)
		if err != nil {
			return nil, err
		}

		// Immediately free previous metrics values.
		if err := freeMetrics(metricsInterfaces, metrics); err != nil {
			return nil, err
		}

		// Execute the step.
		metrics, err = loop.step(spec, inputs, labels)
		if err != nil {
			return nil, errors.WithMessagef(err, "Loop.RunSteps(%d): failed TrainStep(LoopStep=%d)",
				steps, loop.LoopStep)
		}
	}
	return metrics, nil
}

func (loop *Loop) runStepsDistributed(ds DistributedDataset, steps int) (metrics []*tensors.Tensor, err error) {
	loop.TrainStepDurations = make([]time.Duration, 0, steps)
	metricsInterfaces := loop.Trainer.Metrics()
	strategy := ds.Strategy()
	deviceAssignment := ds.DeviceAssignment()
	for loop.LoopStep = loop.StartStep; loop.LoopStep < loop.EndStep; loop.LoopStep++ {
		spec, inputs, labels, err := ds.DistributedYield()
		if err != nil {
			if err == io.EOF {
				return nil, errors.Errorf(
					"reached Dataset end after %d steps (requested %d steps) -- did you mean to use "+
						"a different (looping) Dataset, or use Loop.RunEpochs() instead of Loop.RunSteps() ?",
					loop.LoopStep-loop.StartStep, steps)
			}
			return nil, errors.WithMessagef(err, "Loop.RunSteps(%d): failed reading from Dataset", steps)
		}

		// Check inputs and labels are valid.
		err = checkDistributedYield(inputs, labels)
		if err != nil {
			return nil, err
		}

		// Immediately free previous metrics values.
		if err := freeMetrics(metricsInterfaces, metrics); err != nil {
			return nil, err
		}

		// Execute the step.
		metrics, err = loop.distributedStep(strategy, deviceAssignment, spec, inputs, labels)
		if err != nil {
			return nil, errors.WithMessagef(err, "Loop.RunSteps(%d): failed TrainStep(LoopStep=%d)",
				steps, loop.LoopStep)
		}
	}
	return metrics, nil
}

// RunEpochs runs those many steps. StartStep is adjusted to the current
// LoopStep, so it can be called multiple times, and it will simply pick up
// where it left of last time.
func (loop *Loop) RunEpochs(ds Dataset, epochs int) (metrics []*tensors.Tensor, err error) {
	if err = loop.Trainer.ResetTrainMetrics(); err != nil {
		return
	}
	loop.finalizeYieldedTrainTensors = finalizeYieldedTensors(ds)
	loop.StartStep = loop.LoopStep
	loop.setLastStep(-1)
	loop.Epoch = 0

	err = loop.start(ds)
	if err != nil {
		return nil, err
	}
	// Loop over epochs:
	loop.TrainStepDurations = nil // Reset.
	for loop.Epoch = 0; loop.Epoch < epochs; loop.Epoch++ {
		yieldsPerEpoch := 0
		// Loop over one epoch:
		for {
			spec, inputs, labels, err := ds.Yield()
			if err != nil {
				if err == io.EOF {
					// End of epoch: estimate new last step (loop.EndStep) and reset.
					loop.setLastStep(loop.LoopStep + yieldsPerEpoch*(epochs-loop.Epoch-1))
					break
				}
				return nil, errors.WithMessagef(
					err,
					"Loop.RunEpochs(epoch %d of %d): failed reading from Dataset",
					loop.Epoch,
					epochs,
				)
			}

			// Check inputs and labels are valid.
			err = checkYield(inputs, labels)
			if err != nil {
				return nil, err
			}

			yieldsPerEpoch++

			// Immediately free any space being used.
			for _, metric := range metrics {
				metric.MustFinalizeAll()
			}
			metrics, err = loop.step(spec, inputs, labels)
			if err != nil {
				return nil, errors.WithMessagef(
					err,
					"Loop.RunEpochs(%d): failed reading from Dataset (LoopStep=%d)",
					epochs,
					loop.LoopStep,
				)
			}
			loop.LoopStep++
		}
		ds.Reset()
	}
	err = loop.end(metrics)
	if err != nil {
		return nil, errors.WithMessagef(err, "Loop.RunEpochs(%d): failed end (GlobaStep=%d)", epochs, loop.LoopStep)
	}
	return
}

// MedianTrainStepDuration returns the median duration of each training step.
func (loop *Loop) MedianTrainStepDuration() time.Duration {
	if len(loop.TrainStepDurations) == 0 {
		// Return something different from 0 to avoid division by 0.
		return time.Millisecond
	}

	times := slices.Clone(loop.TrainStepDurations)
	slices.Sort(times)
	return times[len(times)/2]
}

// OnStart adds a hook with given priority and name (for error reporting) to the start of a loop.
func (loop *Loop) OnStart(name string, priority Priority, fn OnStartFn) {
	loop.onStart.Add(priority, &hookWithName[OnStartFn]{
		name: name,
		fn:   fn,
	})
}

// OnStep adds a hook with given priority and name (for error reporting) to each step of a loop.
func (loop *Loop) OnStep(name string, priority Priority, fn OnStepFn) {
	loop.onStep.Add(priority, &hookWithName[OnStepFn]{
		name: name,
		fn:   fn,
	})
}

// OnEnd adds a hook with given priority and name (for error reporting) to the end of a loop.
func (loop *Loop) OnEnd(name string, priority Priority, fn OnEndFn) {
	loop.onEnd.Add(priority, &hookWithName[OnEndFn]{
		name: name,
		fn:   fn,
	})
}

// hookWithName stores a hook name and function.
type hookWithName[F any] struct {
	name string
	fn   F
}

// priorityHooks organizes hooks for type F per priority.
type priorityHooks[H any] struct {
	hooks map[Priority][]H
}

func newPriorityHooks[H any]() *priorityHooks[H] {
	return &priorityHooks[H]{
		hooks: make(map[Priority][]H),
	}
}

// Add hook at the given priority.
func (h *priorityHooks[H]) Add(priority Priority, hook H) {
	list := h.hooks[priority]
	list = append(list, hook)
	h.hooks[priority] = list
}

// All returns an iterator over all registered hooks in priority order.
func (h *priorityHooks[H]) All() iter.Seq[H] {
	return func(yield func(H) bool) {
		keys := make([]Priority, 0, len(h.hooks)) // Convert Priority to int so we can easily sort.
		for key := range h.hooks {
			keys = append(keys, key)
		}
		slices.Sort(keys)
		for _, key := range keys {
			for _, hook := range h.hooks[key] {
				if !yield(hook) {
					return
				}
			}
		}
	}
}
