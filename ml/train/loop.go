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
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	xslices "golang.org/x/exp/slices"
	"io"
	"math"
	"sort"
	"time"
)

// Priority for hooks, the lowest values are run first. Defaults to 0, but negative
// values are ok.
type Priority int

// OnStartFn is the type of OnStart hooks.
type OnStartFn func(loop *Loop, ds Dataset) error

// OnStepFn is the type of OnStep hooks.
type OnStepFn func(loop *Loop, metrics []tensor.Tensor) error

// OnEndFn is the type of OnEnd hooks.
type OnEndFn func(loop *Loop, metrics []tensor.Tensor) error

// Loop will run a training loop, invoking Trainer.TrainStep every step,
// and calling the appropriate hooks.
//
// In itself it doesn't do much, but one can attach functionality to it, like
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

	// LoopStep currently being executed. Defaults to 0. Notice this may not be in sync with model's
	// typical `GlobalStep` variable.
	LoopStep int

	// StartStep is the value of LoopStep at the start of a run (RunSteps or RunEpochs). At the first
	// run it wil be 0 (the default value for LoopStep) and if Loop.RunSteps (or Loop.RunEpochs) is called
	// multiple times, StartStep is reset to the last LoopStep value of the previous run.
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

	// finalize inputs and labels after they are used.
	finalizeInputs bool

	// trainStepDurations collected during training
	TrainStepDurations []time.Duration

	// Registered hooks.
	onStart *priorityHooks[*hookWithName[OnStartFn]]
	onStep  *priorityHooks[*hookWithName[OnStepFn]]
	onEnd   *priorityHooks[*hookWithName[OnEndFn]]
}

// NewLoop creates a new training loop trainer.
func NewLoop(trainer *Trainer) *Loop {
	return &Loop{
		Trainer:        trainer,
		SharedData:     make(map[string]any),
		onStart:        newPriorityHooks[*hookWithName[OnStartFn]](),
		onStep:         newPriorityHooks[*hookWithName[OnStepFn]](),
		onEnd:          newPriorityHooks[*hookWithName[OnEndFn]](),
		finalizeInputs: false,
	}
}

// FreeInputs configure the loop to free the inputs yielded by the dataset immediately -- as opposed to
// wait for garbage collection. This can increase efficiency and be required for large inputs,
// but if the inputs are being reused, this will break.
//
// It returns loop itself after it has been configured, so calls can be cascaded.
func (loop *Loop) FreeInputs() *Loop {
	loop.finalizeInputs = true
	return loop
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
func (loop *Loop) step(spec any, inputs, labels []tensor.Tensor) (metrics []tensor.Tensor, err error) {
	startTime := time.Now()
	defer func() {
		elapsed := time.Since(startTime)
		loop.TrainStepDurations = append(loop.TrainStepDurations, elapsed)
	}()
	checkTensor := func(t tensor.Tensor) {
		if t.Ok() {
			return
		}
		var msg string
		if loop.finalizeInputs {
			msg = " -- loop is configured to free the dataset input after each step, this may be interfering with contents"
		}
		err = errors.Wrapf(t.Error(), "invalid input/label, cannot train%s", msg)
	}
	for _, inputT := range inputs {
		checkTensor(inputT)
		if err != nil {
			return
		}
	}
	for _, labelT := range labels {
		checkTensor(labelT)
		if err != nil {
			return
		}
	}

	metrics, err = loop.Trainer.TrainStep(spec, inputs, labels)
	if err != nil {
		return nil, err
	}
	loop.onStep.Enumerate(func(hook *hookWithName[OnStepFn]) {
		if err != nil {
			// After the first error stop.
			return
		}
		err = hook.fn(loop, metrics)
		if err != nil {
			err = errors.WithMessagef(err, "OnStep(hook %q)", hook.name)
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

	// Immediately release used inputs and labels.
	if loop.finalizeInputs {
		for _, t := range inputs {
			t.FinalizeAll()
		}
		for _, t := range labels {
			t.FinalizeAll()
		}
	}
	return
}

// end of loop, called by all looping methods.
// It calls the appropriate hooks.
func (loop *Loop) end(metrics []tensor.Tensor) (err error) {
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

// ReadGlobalStep will read the global step from the context and initialize the LoopStep
// to that value.
// The default is to have the LoopStep counter always start from 0 -- independent of the model's GlobalStep.
func (loop *Loop) ReadGlobalStep(ctx *context.Context) {
	globalStepVar := optimizers.GetGlobalStepVar(ctx)
	loop.LoopStep = globalStepVar.Value().Value().(int)
}

// RunSteps runs those many steps. StartStep and EndStep are adjusted to the current
// LoopStep, so it can be called multiple times, and it will simply pick up
// where it left of last time.
func (loop *Loop) RunSteps(ds Dataset, steps int) (metrics []tensor.Tensor, err error) {
	if steps == 0 {
		return nil, nil
	}
	if err = loop.Trainer.ResetTrainMetrics(); err != nil {
		return
	}
	loop.StartStep = loop.LoopStep
	loop.EndStep = loop.LoopStep + steps
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
		metrics, err = loop.step(spec, inputs, labels)
		if err != nil {
			return nil, errors.WithMessagef(err, "Loop.RunSteps(%d): failed TrainStep(LoopStep=%d)", steps, loop.LoopStep)
		}
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
func (loop *Loop) RunEpochs(ds Dataset, epochs int) (metrics []tensor.Tensor, err error) {
	if err = loop.Trainer.ResetTrainMetrics(); err != nil {
		return
	}
	loop.StartStep = loop.LoopStep
	loop.EndStep = -1
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
			if err == io.EOF {
				// End of epoch: estimate new EndStep and reset.
				loop.EndStep = loop.LoopStep + yieldsPerEpoch*(epochs-loop.Epoch-1)
				break
			}
			yieldsPerEpoch++

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
		// Return something different than 0 to avoid division by 0.
		return time.Millisecond
	}

	times := slices.Copy(loop.TrainStepDurations)
	xslices.Sort(times)
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
