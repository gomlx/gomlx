// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package plot

import (
	"math"

	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

// PointFilter can change any [Point] arbitrarily. If it returns false it means the point should
// be dropped.
type PointFilter func(p *Point) bool

// Config holds the rendering-independent state and logic for scheduling metric collection
// against a [train.Loop] and persisting plot points to a file -- shared by
// [github.com/gomlx/gomlx/ui/gonb/plotly.PlotConfig] and
// [github.com/gomlx/gomlx/ui/gonb/margaid.Plots], which each embed a *Config (anonymously, so
// its exported fields/methods are promoted) and add their own rendering on top.
//
// Go embedding can't dispatch virtually: if Config called its own AddPoint, it could never reach
// the embedding type's rendering logic. Instead, Config is given an owner (implementing
// [Plotter]) at construction time via [NewConfig], and calls back through it.
type Config struct {
	owner Plotter

	// EvalDatasets registered to be evaluated at each collecting step (see Schedule* methods).
	EvalDatasets []train.Dataset

	// batchNormAveragesDS is used to update the batch normalization averages, if configured.
	batchNormAveragesDS train.Dataset

	// pointsAdded limits plotting only if enough points have been added.
	pointsAdded int

	// lastStepCollected avoids collecting the same step twice, in case metrics collection was
	// scheduled more than one way.
	lastStepCollected int

	customMetricFn CustomMetricFn

	// scheduledFinalPlot indicates attachOnEnd has already registered its loop.OnEnd callback.
	scheduledFinalPlot bool
	onEndName          string
	finalize           func()

	// filePath where to save data points to. Only used if not empty.
	filePath      string
	fileWriter    chan<- Point
	errFileWriter <-chan error
}

// NewConfig creates a Config that calls back into owner (the [Plotter] implementation embedding
// this Config) whenever it needs to add a point or signal a sample is done.
//
// onEndName and finalize are used when the first Schedule* call registers a loop.OnEnd callback
// (only once, no matter how many Schedule* methods are used): onEndName is the name passed to
// [train.Loop.OnEnd], and finalize is called before the file writer (if any) is closed -- it's
// where the owner does its final render.
func NewConfig(owner Plotter, onEndName string, finalize func()) *Config {
	return &Config{owner: owner, onEndName: onEndName, finalize: finalize}
}

// WithDatasets configures the datasets to evaluate at each collecting step (see Schedule* methods).
//
// It returns itself to allow cascading configuration method calls.
func (c *Config) WithDatasets(datasets ...train.Dataset) *Config {
	c.EvalDatasets = datasets
	return c
}

// WithBatchNormalizationAveragesUpdate configures a dataset to use to update the averages (of
// mean and variance) for batch normalization.
//
// The oneEpochDS dataset (typically, the same as a training data evaluation dataset) should be a
// 1-epoch training data dataset, and it can use evaluation batch sizes.
// If oneEpochDS is nil, it disables the updating of the averages.
//
// If the model is not using batch normalization, this is a no-op and nothing is executed.
//
// It returns itself to allow cascading configuration method calls.
func (c *Config) WithBatchNormalizationAveragesUpdate(oneEpochDS train.Dataset) *Config {
	c.batchNormAveragesDS = oneEpochDS
	return c
}

// WithCustomMetricFn registers the given function to run at every step it collects metrics.
// Only one function can be registered. Set to nil to reset.
//
// It returns itself to allow cascading configuration method calls.
func (c *Config) WithCustomMetricFn(fn CustomMetricFn) *Config {
	c.customMetricFn = fn
	return c
}

// StartWriting creates an asynchronous writer that persists every subsequent point (added
// through the owner's AddPoint / FilterAndWrite) to filePath.
func (c *Config) StartWriting(filePath string) {
	c.filePath = filePath
	c.fileWriter, c.errFileWriter = CreatePointsWriter(filePath)
}

// StopWriting indicates that no more points are coming. This closes the asynchronous job writing
// new points, blocking until it's done and logging (not returning) any write error, since it's
// meant to be called from a defer/OnEnd context where there's no one left to hand the error to.
func (c *Config) StopWriting() {
	if c.fileWriter == nil {
		return
	}
	close(c.fileWriter)
	c.fileWriter = nil
	err := <-c.errFileWriter
	if err != nil {
		klog.Errorf("Failed to write plots data: %+v", err)
	}
}

// FilterAndWrite validates pt (dropping NaN/Inf values or steps) and, if a file writer is
// attached (see StartWriting), persists it asynchronously.
//
// It returns false if pt should be dropped -- the owner's AddPoint should return immediately in
// that case, without adding the point to whatever it renders.
func (c *Config) FilterAndWrite(pt Point) bool {
	if math.IsNaN(pt.Value) || math.IsInf(pt.Value, 0) || math.IsNaN(pt.Step) || math.IsInf(pt.Step, 0) {
		return false
	}
	if c.fileWriter != nil {
		c.fileWriter <- pt
	}
	return true
}

// PointsAdded returns how many points (counted per the owner's own convention -- e.g. per
// distinct step, or via some other heuristic) have been added so far.
func (c *Config) PointsAdded() int {
	return c.pointsAdded
}

// IncrementPointsAdded increments the PointsAdded counter by one. Exposed for owners (like
// plotly.PlotConfig.LoadCheckpointData) that count points added by their own convention.
func (c *Config) IncrementPointsAdded() {
	c.pointsAdded++
}

// SetPointsAddedIfGreater raises PointsAdded to n, if n is greater than the current value.
// Exposed for owners (like margaid.Plots.PreloadFile) that derive an estimate of "how many
// points have been collected" some other way (e.g. from the minimum series length across all
// loaded metrics) rather than counting one by one.
func (c *Config) SetPointsAddedIfGreater(n int) {
	if n > c.pointsAdded {
		c.pointsAdded = n
	}
}

// MarkSampleDone is the generic part of [Plotter.DynamicSampleDone]: it counts pt as added if
// the sample was complete. The owner still needs to trigger its own rendering afterward.
func (c *Config) MarkSampleDone(incomplete bool) {
	if !incomplete {
		c.pointsAdded++
	}
}

// addMetrics is the collection callback used by every Schedule* method: it dedups by
// loop.LoopStep (in case metrics collection was scheduled more than one way), runs the optional
// custom-metric hook, and then records train+eval metrics through the owner.
func (c *Config) addMetrics(loop *train.Loop, metrics []*tensors.Tensor) error {
	if c.lastStepCollected >= loop.LoopStep {
		return nil
	}
	c.lastStepCollected = loop.LoopStep

	if c.customMetricFn != nil {
		if err := c.customMetricFn(c.owner, float64(loop.LoopStep)); err != nil {
			return errors.WithMessagef(err, "plot.Config CustomMetricFn returned an error at step %d", loop.LoopStep)
		}
	}

	return AddTrainAndEvalMetrics(c.owner, loop, metrics, c.EvalDatasets, c.batchNormAveragesDS)
}

// attachOnEnd registers a final call to finalize (given to NewConfig) when training finishes,
// followed by StopWriting. Only registers once, no matter how many Schedule* methods are used.
func (c *Config) attachOnEnd(loop *train.Loop) {
	if c.scheduledFinalPlot {
		return
	}
	c.scheduledFinalPlot = true
	loop.OnEnd(c.onEndName, 120, func(_ *train.Loop, _ []*tensors.Tensor) error {
		c.finalize()
		c.StopWriting()
		return nil
	})
}

// ScheduleExponential collection of plot points, starting at startStep and with an increasing
// step factor of stepFactor. Typical values here are 100 and 1.1. name is passed to
// [train.ExponentialCallback] (and, the first time any Schedule* method is called, to
// [train.Loop.OnEnd] as well).
//
// It returns itself to allow cascading configuration method calls.
func (c *Config) ScheduleExponential(loop *train.Loop, startStep int, stepFactor float64, name string) *Config {
	train.ExponentialCallback(loop, startStep, stepFactor, true, name, 0, c.addMetrics)
	c.attachOnEnd(loop)
	return c
}

// ScheduleNTimes collection of plot points, numPoints times over the loop. name is passed to
// [train.NTimesDuringLoop] (and, the first time any Schedule* method is called, to
// [train.Loop.OnEnd] as well).
//
// It returns itself to allow cascading configuration method calls.
func (c *Config) ScheduleNTimes(loop *train.Loop, numPoints int, name string) *Config {
	train.NTimesDuringLoop(loop, numPoints, name, 0, c.addMetrics)
	c.attachOnEnd(loop)
	return c
}

// ScheduleEveryNSteps collects metrics every n steps. name is passed to [train.EveryNSteps]
// (and, the first time any Schedule* method is called, to [train.Loop.OnEnd] as well).
//
// It returns itself to allow cascading configuration method calls.
func (c *Config) ScheduleEveryNSteps(loop *train.Loop, n int, name string) *Config {
	train.EveryNSteps(loop, n, name, 0, c.addMetrics)
	c.attachOnEnd(loop)
	return c
}
