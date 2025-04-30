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

package train

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
	"io"
	"math"
	"slices"
	"sort"
	"time"
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
	// Defaults to the current context's `GlobalStep`, which will be 0 for a new context.
	LoopStep int

	// StartStep is the value of LoopStep at the start of a run (RunSteps or RunEpochs).
	StartStep int

	// EndStep is one-past the last step to be executed. If -1 the end step is not known (if
	// running till the end of the dataset). When running for multiple epochs (Loop.RunEpochs) it can
	// change during the run (after the first epoch, the value is extrapolated based on how many steps
	// have been run so far).
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
	return &Loop{
		Trainer:    trainer,
		SharedData: make(map[string]any),
		onStart:    newPriorityHooks[*hookWithName[OnStartFn]](),
		onStep:     newPriorityHooks[*hookWithName[OnStepFn]](),
		onEnd:      newPriorityHooks[*hookWithName[OnEndFn]](),
		LoopStep:   int(optimizers.GetGlobalStep(trainer.Context())),
	}
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
// It is stored in the TrainerAbsoluteScope.
func GetTrainLastStepVar(ctx *context.Context) *context.Variable {
	ctxTrainer := ctx.InAbsPath(TrainerAbsoluteScope)
	return ctxTrainer.Checked(false).
		VariableWithValue(TrainLastStepVarName, int64(-1)).
		SetTrainable(false)
}

// start of loop, called by all looping methods.
//
// It calls the appropriate hooks.
func (loop *Loop) start(ds Dataset) (err error) {
	loop.onStart.Enumerate(func(hook *hookWithName[OnStartFn]) {
		if err != nil {
			// After the first error stop.
			return
		}
		err = hook.fn(loop, ds)
		if err != nil {
			err = errors.WithMessagef(err, "OnStart(hook %q)", hook.name)
		}
	})
	return
}

// step of loop, called by all looping methods.
// It calls the appropriate hooks.
func (loop *Loop) step(spec any, inputs, labels []*tensors.Tensor) (metrics []*tensors.Tensor, err error) {
	startTime := time.Now()
	defer func() {
		elapsed := time.Since(startTime)
		loop.TrainStepDurations = append(loop.TrainStepDurations, elapsed)
	}()

	err = TryCatch[error](func() { metrics = loop.Trainer.TrainStep(spec, inputs, labels) })
	if err != nil {
		return nil, err
	}

	// Free inputs and labels:
	if loop.finalizeYieldedTrainTensors {
		for _, input := range inputs {
			input.FinalizeAll()
		}
		for _, label := range labels {
			label.FinalizeAll()
		}
	}

	// Free metrics on-device usage: on-device memory being more at premium,
	// we want to immediately free things that are no longer used there.
	for _, m := range metrics {
		m.MaterializeLocal()
		m.InvalidateOnDevice()
	}

	// Call "OnStep" hooks.
	loop.onStep.Enumerate(func(hook *hookWithName[OnStepFn]) {
		if err != nil {
			// After the first error stop.
			return
		}
		err = hook.fn(loop, metrics)
		if err != nil {
			err = errors.WithMessagef(err, "train.Loop.OnStep(hook %q)", hook.name)
		}
	})
	if err != nil {
		return nil, err
	}

	batchLoss := shapes.ConvertTo[float64](metrics[0].Value())
	if math.IsNaN(batchLoss) {
		err = errors.Errorf("batch loss is NaN, training interrupted")
		return
	}
	if math.IsInf(batchLoss, 0) {
		err = errors.Errorf("batch loss is infinity (%f), training interrupted", batchLoss)
		return
	}
	return
}

// setLastStep, both the field in Loop but also the corresponding variable in the context.
func (loop *Loop) setLastStep(lastStep int) {
	loop.EndStep = lastStep
	endStepVar := GetTrainLastStepVar(loop.Trainer.Context())
	endStepVar.SetValue(tensors.FromScalar(int64(loop.EndStep)))
}

// end of loop, called by all looping methods.
// It calls the appropriate hooks.
func (loop *Loop) end(metrics []*tensors.Tensor) (err error) {
	loop.onEnd.Enumerate(func(hook *hookWithName[OnEndFn]) {
		if err != nil {
			// After the first error stop.
			return
		}
		err = hook.fn(loop, metrics)
		if err != nil {
			err = errors.WithMessagef(err, "OnEnd(hook %q)", hook.name)
		}
	})
	return
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

func checkYield(inputs, labels []*tensors.Tensor) error {
	// Check inputs and labels are valid.
	for _, slice := range [][]*tensors.Tensor{inputs, labels} {
		for _, t := range slice {
			if t.DType() == dtypes.InvalidDType {
				return errors.New(
					"dataset yielded an invalid tensor -- likely it has already been finalized (freed). " +
						"The training loop by default immediately frees the yielded tensor after use, so it doesn't " +
						"wait for the garbage collector. If the dataset is trying to reuse tensors, they will become " +
						"invalid and cause this error. If that is the case, consider implementing the method " +
						"FinalizeYieldsAfterUse() in your dataset, and return false.")
			}
		}
	}
	return nil
}

// RunSteps runs those many steps. StartStep and EndStep are adjusted to the current
// LoopStep, so it can be called multiple times, and it will simply pick up
// where it left of last time.
//
// Note: inputs and labels yielded by the dataset are immediately finalized (freed) after use in each step.
func (loop *Loop) RunSteps(ds Dataset, steps int) (metrics []*tensors.Tensor, err error) {
	if steps == 0 {
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
	loop.TrainStepDurations = make([]time.Duration, 0, steps)
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

		// Immediately free any space being used.
		for _, metric := range metrics {
			if metric != nil {
				metric.FinalizeAll()
			}
		}
		metrics, err = loop.step(spec, inputs, labels)
		if err != nil {
			return nil, errors.WithMessagef(err, "Loop.RunSteps(%d): failed TrainStep(LoopStep=%d)", steps, loop.LoopStep)
		}
	}
	for _, metric := range metrics {
		// Transfer results locally and immediately free on-device storage.
		metric.MaterializeLocal()
		metric.InvalidateOnDevice()
	}
	err = loop.end(metrics)
	if err != nil {
		return nil, errors.WithMessagef(err, "Loop.RunSteps(%d): failed end (GlobaStep=%d)", steps, loop.LoopStep)
	}
	return
}

// RunEpochs runs those many steps. StartStep is adjusted to the current
// LoopStep, so it can be called multiple times, and it will simply pick up
// where it left of last time.
// Loop.Epoch is set to the current running epoch. EndStep starts as -1 and will
// be adjusted to expectation after the first epoch, when one knows how many steps there are
// going to be.
// Dataset.Reset is called after each epoch (including the last).
//
// Note: inputs and labels yielded by the dataset are immediately finalized (freed) after use in each step.
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
				return nil, errors.WithMessagef(err, "Loop.RunEpochs(epoch %d of %d): failed reading from Dataset", loop.Epoch, epochs)
			}

			// Check inputs and labels are valid.
			err = checkYield(inputs, labels)
			if err != nil {
				return nil, err
			}

			yieldsPerEpoch++

			// Immediately free any space being used.
			for _, metric := range metrics {
				metric.FinalizeAll()
			}
			metrics, err = loop.step(spec, inputs, labels)
			if err != nil {
				return nil, errors.WithMessagef(err, "Loop.RunEpochs(%d): failed reading from Dataset (LoopStep=%d)", epochs, loop.LoopStep)
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

// MedianTrainStepDuration returns the median duration of each training step. It returns 1 millisecond
// if no training step was recorded (to avoid potential division by 0).
//
// It sorts and mutates loop.TrainStepDurations.
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
// The function `fn` is called after each `Trainer.TrainStep`.
func (loop *Loop) OnStep(name string, priority Priority, fn OnStepFn) {
	loop.onStep.Add(priority, &hookWithName[OnStepFn]{
		name: name,
		fn:   fn,
	})
}

// OnEnd adds a hook with given priority and name (for error reporting) to the end of a loop,
// after the last call to `Trainer.TrainStep`.
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

// Enumerate will call fn for all registered hooks in priority order.
func (h *priorityHooks[H]) Enumerate(fn func(hook H)) {
	keys := make([]Priority, 0, len(h.hooks)) // Convert Priority to int so we can easily sort.
	for key := range h.hooks {
		keys = append(keys, key)
	}
	sort.Slice(keys, func(i, j int) bool {
		return keys[i] < keys[j]
	})
	for _, key := range keys {
		for _, hook := range h.hooks[key] {
			fn(hook)
		}
	}
}
