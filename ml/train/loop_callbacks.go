package train

import (
	"fmt"
	"math"
	"time"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/tensors"
)

// nTimes is used to implement NTimesDuringLoop.
type nTimes struct {
	n, nUsed int
	fn       OnStepFn
}

func (nT *nTimes) onStep(loop *Loop, metrics []*tensors.Tensor) error {
	stepsDone := (loop.LoopStep - loop.StartStep) + 1 // Current LoopStep just finished.
	if loop.EndStep < 0 {
		// End not known, run steps in powers of 2, starting at 128.
		if stepsDone < (128 << nT.nUsed) {
			return nil
		}
	} else if loop.LoopStep < loop.EndStep-1 { // Last step (LoopStep == EndStep-1) is always included.
		totalSteps := loop.EndStep - loop.StartStep
		stepsPerCall := float64(totalSteps) / float64(nT.n)
		if stepsPerCall > 1 && float64(nT.nUsed) > float64(stepsDone)/stepsPerCall {
			return nil
		}
	}

	// Call hook at this step.
	nT.nUsed++
	return nT.fn(loop, metrics)
}

func (nT *nTimes) onEnd(loop *Loop, metrics []*tensors.Tensor) error {
	return nil
}

// NTimesDuringLoop registers a OnStep hook on the loop that is called at most N times, split evenly
// across all steps.
//
// For Loop.RunEpochs it does not work perfectly even, at least until it knows what is the
// exact number of steps -- it may even call OnStepFn more than n times.
//
// It always calls `fn` at the very last step.
func NTimesDuringLoop(loop *Loop, n int, name string, priority Priority, fn OnStepFn) {
	nT := &nTimes{
		n:  n,
		fn: fn,
	}
	name = fmt.Sprintf("NTimesDuringLoop(%d): %s", n, name)
	loop.OnStep(name, priority, nT.onStep)
	loop.OnEnd(name, priority, nT.onEnd)
}

type everyNSteps struct {
	n, count int
	fn       OnStepFn
}

func (eN *everyNSteps) onStep(loop *Loop, metrics []*tensors.Tensor) error {
	eN.count++
	if eN.count%eN.n != 0 {
		return nil
	}
	return eN.fn(loop, metrics)
}

// EveryNSteps registers a OnStep hook on the loop that is called every N times.
//
// Notice that it does not call `fn` at the last step (except by coincidence).
func EveryNSteps(loop *Loop, n int, name string, priority Priority, fn OnStepFn) {
	eN := &everyNSteps{n: n, fn: fn}
	fullName := fmt.Sprintf("EveryNSteps(%d): %s", n, name)
	loop.OnStep(fullName, priority, eN.onStep)
}

type periodicCallback struct {
	last               time.Time
	period             time.Duration
	started, callOnEnd bool
	fn                 OnStepFn
}

func (p *periodicCallback) onStep(loop *Loop, metrics []*tensors.Tensor) error {
	if !p.started {
		// Start the clock.
		p.started = true
		p.last = time.Now()
		return nil
	}
	elapsed := time.Since(p.last)
	if elapsed < p.period {
		return nil
	}

	err := p.fn(loop, metrics)
	p.last = time.Now()
	return err
}

// PeriodicCallback registers an `OnStep` hook on the loop that is called every period of time.
// The period counts after the execution of `OnStep`: this discounts the time to run `OnStep` (in case it is expensive)
// and it discounts cases where the execution is paused. By other hand, OnStep is not executed exactly at every `period`
// time.
//
// If callOnEnd is set, it will also call at the end of the loop.
func PeriodicCallback(loop *Loop, period time.Duration, callOnEnd bool, name string, priority Priority, fn OnStepFn) {
	p := &periodicCallback{
		period:    period,
		callOnEnd: callOnEnd,
		fn:        fn,
	}
	fullName := fmt.Sprintf("PeriodicCallback(%s): %s", period, name)
	loop.OnStep(fullName, priority, p.onStep)
	if callOnEnd {
		loop.OnEnd(fullName, priority, func(loop *Loop, metrics []*tensors.Tensor) error { return p.fn(loop, metrics) })
	}
}

// ExponentialCallback registers an `OnStep` hook on the loop that is called at exponentially increasing number
// of steps in between, starting with startStep, and growing at geometric factor of exponentialFactor.
//
// If callOnEnd is set, it will also call at the end of the loop.
//
// Example: This will call at steps 100, 100+100*1.2 = 220, 220+100*1.2^2 = 364, ...
//
//	ExponentialCallback(loop, 100, 1.2, "my_callback", 100, myCallback)
func ExponentialCallback(loop *Loop, startStep int, exponentialFactor float64, callOnEnd bool,
	name string, priority Priority, fn OnStepFn) {
	if startStep == 0 || exponentialFactor <= 1 {
		exceptions.Panicf("Invalid parameters for ExponentialCallback(startStep=%d, exponentialFactor=%f), startStep must be > 0 and exponentialFactor must be > 1", startStep, exponentialFactor)
	}
	e := &exponentialCallback{
		startStep:         startStep,
		exponentialFactor: exponentialFactor,
		fn:                fn,
	}
	fullName := fmt.Sprintf("ExponentialCallback(%d, %f): %s", startStep, exponentialFactor, name)
	loop.OnStep(fullName, priority, e.onStep)
	if callOnEnd {
		loop.OnEnd(fullName, priority, func(loop *Loop, metrics []*tensors.Tensor) error { return e.fn(loop, metrics) })
	}

}

type exponentialCallback struct {
	startStep, currentStepSkip, nextStepToCall int
	exponentialFactor                          float64
	fn                                         OnStepFn
}

func (e *exponentialCallback) bump() {
	e.nextStepToCall += e.currentStepSkip
	e.currentStepSkip = int(math.Round(float64(e.currentStepSkip) * e.exponentialFactor))
}

func (e *exponentialCallback) findNextStepToCall(currentStep int) {
	e.currentStepSkip = e.startStep
	for currentStep >= e.nextStepToCall {
		e.bump()
	}
}

func (e *exponentialCallback) onStep(loop *Loop, metrics []*tensors.Tensor) error {
	if e.nextStepToCall == 0 {
		e.findNextStepToCall(loop.StartStep)
	}
	if loop.LoopStep < e.nextStepToCall {
		return nil
	}

	e.bump()
	return e.fn(loop, metrics)
}
