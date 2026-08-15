// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package plotly uses GoNB plotly support (`github.com/janpfeifer/gonb/gonbui/plotly`) to plot both
// on dynamic plots while training or to quickly plot the results of a previously
// saved plot results in a checkpoint directory.
//
// In either case, it allows adding baseline plots of previous checkpoint.
//
// The advantage of `plotly` over `margaid` plots is that it uses JavaScript to make the plot interactive (it displays
// information on mouse hover).
//
// The disadvantage is that saving of the notebook doesn't include JavaScript -- but exporting to HTML does.
//
// See New to get started and an example.
package plotly

import (
	"fmt"
	"os"
	"path"

	grob "github.com/MetalBlueberry/go-plotly/generated/v2.34.0/graph_objects"
	ptypes "github.com/MetalBlueberry/go-plotly/pkg/types"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/gomlx/gomlx/ui/plot"
	"github.com/janpfeifer/gonb/gonbui"
	"github.com/janpfeifer/gonb/gonbui/dom"
	gonbplotly "github.com/janpfeifer/gonb/gonbui/plotly"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

var (
	// ParamPlots is the context parameter to trigger generating of plot points and
	// displaying them.
	// A boolean value that defaults to false.
	ParamPlots = "plots"
)

// onEndName is used both to name the collection callbacks registered on the train.Loop (see the
// Schedule* methods) and, once, to name the loop.OnEnd callback that triggers the final plot.
const onEndName = "plotly.DynamicPlot"

// PointFilter can change any [plot.Point] arbitrarily. If it returns false means the point should be dropped.
type PointFilter = plot.PointFilter

// PlotConfig hold the configuration object that will generate the plot. Create it with [New].
//
// The generic, rendering-independent parts (scheduling metric collection, persisting points to
// a file) live in the embedded *plot.Config -- see its documentation for those.
type PlotConfig struct {
	*plot.Config

	// figs is the list of figures, one per metric type.
	figs []*grob.Fig

	// metricsNames maps the metric name to the trace in the corresponding figure.
	// One per figure.
	metricsNamesToTrace []map[string]int

	// metricsTypesToFig maps a metric type to a figure index in `figs`.
	metricsTypesToFig map[string]int

	// gonbId of the `<div>` tag where to generate dynamic plots.
	gonbId string

	// finalPlot indicates whether the final plot has already been drawn.
	finalPlot bool
}

// New creates a new PlotConfig, that can be used to generate plots.
//
// This is used when configuring the train.Loop, and a typical use example, triggered by a "plots" parameter,
// might look like:
//
//	usePlots := model.GetParamOr(scope, plotly.ParamPlots, false)
//	...
//	if usePlots {
//		_ = plotly.New().
//			WithCheckpoint(checkpointHandler).
//			Dynamic().
//			WithDatasets(evalOnTrainDS, evalOnTestDS).
//			ScheduleExponential(loop, 200, 1.2).
//			WithBatchNormalizationAveragesUpdate(evalOnTrainDS)
//	}
func New() *PlotConfig {
	pc := &PlotConfig{
		metricsTypesToFig: make(map[string]int),
	}
	pc.Config = plot.NewConfig(pc, onEndName, func() {
		// Final plot: only called once to the transient plots.
		if pc.gonbId != "" && !pc.finalPlot {
			// Erase intermediary transient plots.
			pc.DynamicPlot(true)
			pc.finalPlot = true
		}
	})
	return pc
}

// WithDatasets configures the datasets to evaluate at each collecting step (see `Schedule*` methods).
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) WithDatasets(datasets ...train.Dataset) *PlotConfig {
	pc.Config.WithDatasets(datasets...)
	return pc
}

// WithBatchNormalizationAveragesUpdate configures a dataset to use to update the averages (of mean and variance)
// for batch normalization.
//
// The oneEpochDS dataset (typically, the same as a training data evaluation dataset) should be a 1-epoch training
// data dataset, and it can use evaluation batch sizes.
// If oneEpochDS is nil, it disabled the updating of the averages.
//
// If the model is not using batch normalization, this is a no-op and nothing is executed.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) WithBatchNormalizationAveragesUpdate(oneEpochDS train.Dataset) *PlotConfig {
	pc.Config.WithBatchNormalizationAveragesUpdate(oneEpochDS)
	return pc
}

// Dynamic sets plot to be dynamically updated as new data points are added. It's a no-op if not running in a GoNB notebook.
//
// It should be followed by a call to [ScheduleExponential] or [SchedulePeriodic] (or both) to schedule capturing
// points to plot, and [WithCheckpoint] to save the captured points.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) Dynamic() *PlotConfig {
	if !gonbui.IsNotebook {
		return pc
	}
	pc.gonbId = gonbui.UniqueId()
	if pc.PointsAdded() < 3 {
		// If we are having a dynamically updating plot, we reserve the transient HTML block
		// upfront -- otherwise it will interfere with the progressbar the first time it is displayed.
		gonbui.UpdateHTML(pc.gonbId, "(...collecting metrics, minimum 3 required to start plotting...)")
	} else {
		gonbui.UpdateHTML(pc.gonbId, "")
		pc.DynamicPlot(false)
	}
	return pc
}

// ScheduleExponential collection of plot points, starting at `startStep` and with an increasing step factor
// of `stepFactor`. Typical values where could be 100 and 1.1.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) ScheduleExponential(loop *train.Loop, startStep int, stepFactor float64) *PlotConfig {
	pc.Config.ScheduleExponential(loop, startStep, stepFactor, onEndName)
	return pc
}

// ScheduleNTimes collections of plot points.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) ScheduleNTimes(loop *train.Loop, numPoints int) *PlotConfig {
	pc.Config.ScheduleNTimes(loop, numPoints, onEndName)
	return pc
}

// ScheduleEveryNSteps to collect metrics.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) ScheduleEveryNSteps(loop *train.Loop, n int) *PlotConfig {
	pc.Config.ScheduleEveryNSteps(loop, n, onEndName)
	return pc
}

// WithCustomMetricFn registers the given function to run at every step it collects metrics.
// Only one function can be registered. Set to nil to reset.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) WithCustomMetricFn(fn plot.CustomMetricFn) *PlotConfig {
	pc.Config.WithCustomMetricFn(fn)
	return pc
}

// WithCheckpoint uses the checkpoint both to load data points and to save any new data points.
// Usually, used with [PlotConfig.Dynamic].
// If checkpoint is nil, it's a no-op.
//
// New data-points are saved asynchronously not to slow down training,
// but with the downside of potentially having I/O issues reported asynchronously.
//
// It returns itself to allow cascading configuration method calls.
func (pc *PlotConfig) WithCheckpoint(checkpointHandler *checkpoint.Handler) *PlotConfig {
	if checkpointHandler == nil {
		return pc
	}
	checkpointDir := checkpointHandler.Dir()

	// Ignore errors while loading: maybe nothing was written yet.
	_ = pc.LoadCheckpointData(checkpointDir)
	checkpointDir = fsutil.MustReplaceTildeInDir(checkpointDir)
	filePath := path.Join(checkpointDir, plot.TrainingPlotFileName)
	pc.Config.StartWriting(filePath)
	return pc
}

// LoadCheckpointData loads plotting data from a checkpoint path.
// Notice this only works if the model was trained with plotting, with the metrics saved into the file
// `training_plot_points.json` ([plot.TrainingPlotFileName]).
// Notice that if `dataDirOrFile` is a file, it reads from that instead.
//
// The `filters` are an optional list of filters to apply (in order) to each of the points read: it allows points
// to be modified arbitrarily -- in particular, useful to change names (like adding a prefix) of metrics or metrics
// types.
//
// Each filter can also remove points by returning false -- only points for which filters returned true are
// included.
func (pc *PlotConfig) LoadCheckpointData(dataDirOrFile string, filters ...PointFilter) error {
	dataDirOrFile = fsutil.MustReplaceTildeInDir(dataDirOrFile)
	fi, err := os.Stat(dataDirOrFile)
	if err != nil {
		return errors.Wrapf(err, "plotly.LoadCheckpointData(%q) cannot stat the file", dataDirOrFile)
	}
	var points []plot.Point
	if fi.IsDir() {
		points, err = plot.LoadPointsFromCheckpoint(dataDirOrFile)
	} else {
		points, err = plot.LoadPoints(dataDirOrFile)
	}
	if err != nil {
		return errors.WithMessagef(err, "plotly.LoadCheckpointData(%q) failed to load points", dataDirOrFile)
	}

	steps := sets.Make[float64]()
nextPoint:
	for _, point := range points {
		for _, filter := range filters {
			if !filter(&point) {
				continue nextPoint
			}
		}
		pc.AddPoint(point)
		if !steps.Has(point.Step) {
			pc.Config.IncrementPointsAdded() // Count the number of points per different value of the global step.
			steps.Insert(point.Step)
		}
	}
	return nil
}

// AddPoint add one point to the plots.
//
// Usually not called directly, instead use [LoadCheckpointData] or [Dynamic], which will attach to a
// training loop and call this automatically.
func (pc *PlotConfig) AddPoint(pt plot.Point) {
	if !pc.Config.FilterAndWrite(pt) {
		return
	}
	figIdx, found := pc.metricsTypesToFig[pt.MetricType]
	if !found {
		pc.figs = append(pc.figs, &grob.Fig{
			Layout: &grob.Layout{
				Title: &grob.LayoutTitle{
					Text: ptypes.S(pt.MetricType),
				},
				Xaxis: &grob.LayoutXaxis{
					Showgrid: ptypes.B(true),
					Type:     grob.LayoutXaxisTypeLog,
					Title: &grob.LayoutXaxisTitle{
						Text: "Steps",
					},
				},
				Yaxis: &grob.LayoutYaxis{
					Showgrid: ptypes.B(true),
					Type:     grob.LayoutYaxisTypeLog,
				},
				Legend: &grob.LayoutLegend{
					//Y:       -0.2,
					//X:       1.0,
					//X anchor: grob.LayoutLegendX anchorRight,
					//Y anchor: grob.LayoutLegendY anchorTop,
				},
			},
		})
		pc.metricsNamesToTrace = append(pc.metricsNamesToTrace, make(map[string]int))
		figIdx = len(pc.figs) - 1
		pc.metricsTypesToFig[pt.MetricType] = figIdx
	}
	fig := pc.figs[figIdx]
	metricNameToTrace := pc.metricsNamesToTrace[figIdx]

	traceIdx, found := metricNameToTrace[pt.MetricName]
	if !found {
		metricNameToTrace[pt.MetricName] = len(fig.Data)
		traceIdx = len(fig.Data)
		fig.Data = append(fig.Data, &grob.Scatter{
			Name: ptypes.S(pt.MetricName),
			Line: &grob.ScatterLine{
				Shape: grob.ScatterLineShapeLinear,
			},
			Mode: "lines+markers",
			X:    ptypes.DataArray([]float64{}),
			Y:    ptypes.DataArray([]float64{}),
		})
	}
	trace := fig.Data[traceIdx].(*grob.Scatter)
	Xs := trace.X.Value().([]float64)
	Xs = append(Xs, pt.Step)
	trace.X = ptypes.DataArray(Xs)
	Ys := trace.Y.Value().([]float64)
	Ys = append(Ys, pt.Value)
	trace.Y = ptypes.DataArray(Ys)
}

// DynamicSampleDone is called after all the data points recorded for this sample (evaluation at a time step).
// The value `incomplete` is set to true if any of the evaluations are NaN or infinite.
//
// If in a notebook, this would trigger redrawing of the plot.
//
// It implements [plot.Plotter]
func (pc *PlotConfig) DynamicSampleDone(incomplete bool) {
	pc.Config.MarkSampleDone(incomplete)
	if gonbui.IsNotebook && pc.gonbId != "" {
		pc.DynamicPlot(false)
	}
}

// Plot all figures with current data.
// If not in a notebook, this is a no-op.
func (pc *PlotConfig) Plot() error {
	if !gonbui.IsNotebook {
		return nil
	}
	for _, metricType := range xslices.SortedKeys(pc.metricsTypesToFig) {
		figIdx := pc.metricsTypesToFig[metricType]
		gonbui.DisplayHtmlf("<p><b>Metric: %s</b></p>\n", metricType)
		err := gonbplotly.DisplayFig(pc.figs[figIdx])
		if err != nil {
			return err
		}
	}
	return nil
}

// DynamicPlot is called everytime a new metrics comes in, if configured for dynamic updates.
// Usually, not called directly by the user. Simply use [Dynamic] and schedule updates and this function
// will be called automatically.
//
// If `final` is true, it clears the transient area, and it plots instead in the definitive version.
func (pc *PlotConfig) DynamicPlot(final bool) {
	if !gonbui.IsNotebook {
		return
	}
	if pc.gonbId == "" {
		return
	}
	if pc.PointsAdded() < 3 {
		return
	}
	if final == true {
		gonbui.UpdateHTML(pc.gonbId, "")
		err := pc.Plot()
		if err != nil {
			klog.Errorf("Failed to plot: %+v", err)
		}
		return
	}

	// Plot transient version.
	//gonbui.UpdateHtml(pc.gonbId, pc.PlotToHTML())
	elementId := gonbui.UniqueId()
	gonbui.UpdateHTML(pc.gonbId, fmt.Sprintf("<div id=%q></div>", elementId))

	for _, metricType := range xslices.SortedKeys(pc.metricsTypesToFig) {
		figIdx := pc.metricsTypesToFig[metricType]
		dom.Append(elementId, fmt.Sprintf("<p><b>Metric: %s</b></p>\n", metricType))
		err := gonbplotly.AppendFig(elementId, pc.figs[figIdx])
		if err != nil {
			klog.Errorf("Failed to plot: %+v", err)
		}
	}
	return
}
