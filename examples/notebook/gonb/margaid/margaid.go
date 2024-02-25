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

// Package margaid implements automatic plotting of all metrics registered in a trainer,
// using the Margaid library (https://github.com/erkkah/margaid/) to draw SVG, and
// GoNB (https://github.com/janpfeifer/gonb/) to display it in a Jupyter Notebook.
//
// Example 1: to dynamically plot `*flatNumPlotPoints` points during training, for all train metrics
// as well as eval metrics measured on the datasets `trainEvalDS` and `validationEvalDS`, use:
//
//	if *flagNumPlotPoints > 0 {
//	    margaid.New(1024, 400, trainEvalDS, validationEvalDS).DynamicUpdates().Attach(loop, *flagNumPlotPoints)
//	}
//
// Example 2: Plots loads and saves points to the model's checkpoint directory (to allow restarting training),
// and take points at an exponential rate (starting at 100 time steps, and increasing steps at 1.1x rate).
//
//	plots = margaid.New(1024, 400, trainEvalDS, validationDS)
//	if checkpoint != nil {
//		_, err := plots.WithFile(path.Join(checkpoint.Dir(), "training_plot_points.json"))
//		AssertNoError(err)
//	}
//	plots.DynamicUpdates()
//	train.ExponentialCallback(loop, 100, 1.1, true, "Plots", 0, plots.AddTrainAndEvalMetrics)
package margaid

import (
	"bytes"
	"encoding/json"
	"fmt"
	mg "github.com/erkkah/margaid"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/janpfeifer/gonb/gonbui"
	"github.com/pkg/errors"
	"io"
	"k8s.io/klog/v2"
	"math"
	"os"
	"path"
	"strings"
)

var (
	_ = gonbui.IsNotebook
	_ = mg.NewSeries
	_ = New

	// ParamPlots is the conventioned context parameter to trigger generating of plot points and
	// displaying them.
	// A boolean value that defaults to false.
	ParamPlots = "plots"
)

// Plots holds many plots for different metrics. They are organized per "metric type", where
// the metric type is a unit/quantity unique name. It's assumed that series of the same "metric type"
// can share the same Y-Axis and hence the same plot.
type Plots struct {
	// Image dimensions.
	Width, Height int

	// EvalDatasets will be evaluated with `train.Trainer.Eval()` and its metrics collected.
	EvalDatasets []train.Dataset

	// Plot per metric name.
	PerMetricType map[string]*Plot

	// UniqueID for dynamic plotting for GoNB.
	gonbID      string
	pointsAdded int

	// Default projection of the graph on X, Y axis.
	xProjection, yProjection mg.Projection

	// filePath where to load data points from and save to. Only used if not empty.
	filePath   string
	fileWriter chan PlotPoint

	evalLossMetricType string
}

// PlotPoint is used to save/load plot points. They reflect the parameters to Plots.AddPoint.
type PlotPoint struct {
	MetricName, MetricType string
	Step, Value            float64
}

// New creates new Margaid plots structure.
//
// It starts empty and can have the points added manually with Plots.AddPoint or automatically with Plots.Attach.
//
// Use Plots.Plot() to actually generate the plots.
func New(width, height int, evalDatasets ...train.Dataset) *Plots {
	return &Plots{
		Width:        width,
		Height:       height,
		EvalDatasets: evalDatasets,
		xProjection:  mg.Lin,
		yProjection:  mg.Lin,
	}
}

const DefaultFileName = "training_plot_points.json"

// NewDefault creates a new Margaid plots with the usual defaults.
// This "default" configuration may change over time, and it aims to work with the usual GoNB notebook, or if
// run from the command line, it simply save the data points for future plotting.
//
// It serves also as an example.
//
// Arguments:
//   - 'loop': train.Loop to attach itself to. It uses generates evaluations
//   - `dir`: directory where to save the plot data-points, with the file name DefaultFileName.
//   - `startStep` and `stepFactor`: when to add plot points.
//     The `stepFactor` defines the growth of steps between generating plot points.
//     Typical values here are `startStep=100` and `stepFactor=1.1`.
//   - `datasets`: Evaluated whenever plot points are added.
//
// This will automatically `Attach` to the given loop, so no need to call `Attach()` if using this
// constructor.
func NewDefault(loop *train.Loop, dir string, startStep int, stepFactor float64, datasets ...train.Dataset) *Plots {
	plots := New(1024, 400, datasets...).LogScaleX() // .LogScaleY()
	if dir != "" {
		// Save plot points.
		_, err := plots.WithFile(path.Join(dir, DefaultFileName))
		if err != nil {
			panic(err)
		}
	}
	plots.DynamicUpdates()

	// Only plot if (1) it's running in a notebook or if (B) it has a checkpoint directory, where those plot points
	// will be saved.
	if dir != "" || gonbui.IsNotebook {
		// Register plot points at exponential steps.
		train.ExponentialCallback(loop, startStep, stepFactor, true,
			"Monitor", 0, func(loop *train.Loop, metrics []tensor.Tensor) error {
				// Update plots with metrics.
				return plots.AddTrainAndEvalMetrics(loop, metrics)
			})
		plots.attachOnEnd(loop)
	}
	return plots
}

// Plot struct holds the series to different metrics that share the same Y axis.
// They are organized per name of the metric.
type Plot struct {
	MetricType string

	// Flat maps a plot name to
	PerName map[string]*mg.Series

	// allPoints collects all points from all series, to configure the axis.
	allPoints *mg.Series

	xProjection, yProjection mg.Projection
}

// WithFile uses the filePath both to load data points and to save any new data points.
//
// New data-points are saved asynchronously -- not to slow down training, with the downside of
// potentially having I/O issues reported asynchronously.
//
// Consider using DefaultFileName as the file name, if you don't have one.
//
// If used with DynamicUpdates, call this first, so when DynamicUpdates is called, and dynamic plot
// is immediately created.
func (ps *Plots) WithFile(filePath string) (*Plots, error) {
	_, err := ps.PreloadFile(filePath, nil)
	if err != nil && !os.IsNotExist(errors.Cause(err)) {
		return nil, err
	}

	// Create/append file with upcoming metrics.
	f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0664)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to open Plots file %q for append", filePath)
	}
	ps.fileWriter = make(chan PlotPoint, 100)
	go func(f *os.File, fileWriter <-chan PlotPoint) {
		enc := json.NewEncoder(f)
		errLogCount := 0
		errLogStep := 1
		for point := range fileWriter {
			err = enc.Encode(point)
			if err != nil {
				errLogCount++
				if errLogCount%errLogStep == 0 {
					klog.Errorf("failed (%d times) to write to Plots log file in %q: %+v", errLogCount, filePath, err)
					errLogStep *= 10
				}
			}
		}
		_ = f.Close()
	}(f, ps.fileWriter)
	return ps, nil
}

// PreloadFile uses the filePath both to load data points.
// Its metric names can be renamed with renameFn -- leave it as nil for no changes.
//
// If used with DynamicUpdates, call this first, so when DynamicUpdates is called, and dynamic plot
// is immediately created.
func (ps *Plots) PreloadFile(filePath string, renameFn func(metricName string) string) (*Plots, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read Plots file %q", filePath)
	}

	// Read previously stored points.
	dec := json.NewDecoder(f)
	var point PlotPoint
	for {
		err := dec.Decode(&point)
		if err == nil {
			if renameFn != nil {
				point.MetricName = renameFn(point.MetricName)
			}
			ps.AddPoint(point.MetricName, point.MetricType, point.Step, point.Value)
			continue
		}
		if err == io.EOF {
			break
		}
		return nil, errors.Wrapf(err, "error while decoding Plots file %q", filePath)
	}
	_ = f.Close()
	minPoints := ps.minPoints()
	if minPoints > ps.pointsAdded {
		ps.pointsAdded = minPoints
	}
	return ps, nil
}

// minPoints is the minimum number of points per metricName/metricType: used to decide if we can
// already generate plots.
func (ps *Plots) minPoints() int {
	minPoints := -1
	for _, plt := range ps.PerMetricType {
		for _, series := range plt.PerName {
			numPoints := series.Size()
			if minPoints < 0 || numPoints < minPoints {
				minPoints = numPoints
			}
		}
	}
	return minPoints
}

// Done indicates that no more points are coming. This closes the asynchronous job writing new points.
func (ps *Plots) Done() {
	if ps.fileWriter != nil {
		close(ps.fileWriter)
		ps.fileWriter = nil
	}
}

// LogScaleX sets Plots to use a log scale on the X-axis.
// If not set, it uses linear scale.
func (ps *Plots) LogScaleX() *Plots {
	ps.xProjection = mg.Log
	return ps
}

// LogScaleY sets Plots to use a log scale on the X-axis.
// If not set, it uses linear scale.
func (ps *Plots) LogScaleY() *Plots {
	ps.yProjection = mg.Log
	return ps
}

// DynamicUpdates configure the Plots to dynamically generate the plot as new points are included.
//
// It immediately plots a graph if there are already plot points loaded, or alternatively creates the displaying
// area informing that it is waiting to have enough points to plot.
//
// If not in a Jupyter Notebook (with GoNB kernel), this doesn't do anything.
func (ps *Plots) DynamicUpdates() *Plots {
	if !gonbui.IsNotebook {
		return ps
	}
	ps.gonbID = gonbui.UniqueId()
	if ps.pointsAdded < 3 {
		// If we are having a dynamically updating plot, we reserve the transient HTML block
		// upfront -- otherwise it will interfere with the progressbar the first time it is displayed.
		gonbui.UpdateHtml(ps.gonbID, "(...collecting metrics, minimum 3 required to start plotting...)")
	} else {
		ps.DynamicPlot(false)
	}
	return ps
}

// WithEvalLossType defines the name to be used by the loss on eval datasets.
// This is useful if one wants a separate plot for the loss in evaluation than the one used
// for training (called "loss") -- this is needed if training includes loss terms that don't
// show up in evaluation.
//
// A common choice of name here would be "eval_loss".
func (ps *Plots) WithEvalLossType(evalLossMetricType string) *Plots {
	ps.evalLossMetricType = evalLossMetricType
	return ps
}

// attachOnEnd of the loop to draw the final plot -- and clear the transient area if using dynamic plots.
func (ps *Plots) attachOnEnd(loop *train.Loop) {
	loop.OnEnd("margaid plots", 120, func(_ *train.Loop, _ []tensor.Tensor) error {
		// Final plot.
		if ps.gonbID != "" {
			// Erase intermediary transient plots.
			ps.DynamicPlot(true)
		} else {
			ps.Plot()
		}
		return nil
	})
}

// Attach plots to the given loop, collecting metric values for plot. For each EvalDatasets given
// to `Plots.New()`, their metrics are evaluated and also plotted.
// It automatically calls Plots.Plot at the end of the loop (`loop.OnEnd()`).
func (ps *Plots) Attach(loop *train.Loop, numPoints int) {
	train.NTimesDuringLoop(loop, numPoints, "margaid plots", 0, ps.AddTrainAndEvalMetrics)
	ps.attachOnEnd(loop)
}

// AddTrainAndEvalMetrics will add the given train metrics, and run `loop.Trainer.Eval()` on each of the
// datasets registered and include those metrics.
// Finally, if set for DynamicUpdates, it will update the plots.
//
// This function can be set as a callback to the `train.Loop`, at some desired frequency.
// It is used by Plots.Attach, for instance.
func (ps *Plots) AddTrainAndEvalMetrics(loop *train.Loop, trainMetrics []tensor.Tensor) error {
	// Training metrics are pre-generated and given.
	step := float64(loop.LoopStep)
	var incomplete bool
	for ii, desc := range loop.Trainer.TrainMetrics() {
		if desc.Name() == "Batch Loss" {
			// Skip the batch loss, that is not very informative -- it fluctuates a lot at each batch,
			// and the trainer always includes the moving average loss.
			continue
		}
		metric := shapes.ConvertTo[float64](trainMetrics[ii].Value())
		if math.IsNaN(metric) || math.IsInf(metric, 0) {
			incomplete = true
			continue
		}
		ps.AddPoint("Train: "+desc.Name(), desc.MetricType(), step, metric)
	}

	// Eval metrics, if given
	for _, ds := range ps.EvalDatasets {
		var evalMetrics []tensor.Tensor
		if err := exceptions.TryCatch[error](func() { evalMetrics = loop.Trainer.Eval(ds) }); err != nil {
			return err
		}
		for ii, desc := range loop.Trainer.EvalMetrics() {
			metric := shapes.ConvertTo[float64](evalMetrics[ii].Value())
			if math.IsNaN(metric) || math.IsInf(metric, 0) {
				incomplete = true
				continue
			}
			metricType := desc.MetricType()
			if ii == 0 && ps.evalLossMetricType != "" {
				metricType = ps.evalLossMetricType
			}
			ps.AddPoint(fmt.Sprintf("Eval on %s: %s", ds.Name(), desc.Name()), metricType, step, metric)
		}
	}

	if !incomplete {
		ps.pointsAdded++
	}
	if gonbui.IsNotebook && ps.gonbID != "" {
		ps.DynamicPlot(false)
	}
	return nil
}

// AddPoint adds a point for the given metric: `step` is the x-axis, and `value` is the y-axis.
// Metrics with the same type share the same plot and y-axis.
func (ps *Plots) AddPoint(metricName, metricType string, step, value float64) {
	if math.IsNaN(value) || math.IsInf(value, 0) || math.IsNaN(step) || math.IsInf(step, 0) {
		// Ignore invalid points.
		return
	}
	if ps.fileWriter != nil {
		// Save point asynchronously.
		ps.fileWriter <- PlotPoint{metricName, metricType, step, value}
	}
	if ps.PerMetricType == nil {
		ps.PerMetricType = make(map[string]*Plot)
	}
	p, found := ps.PerMetricType[metricType]
	if !found {
		p = &Plot{
			MetricType:  metricType,
			PerName:     make(map[string]*mg.Series),
			xProjection: ps.xProjection,
			yProjection: ps.yProjection,
		}
		ps.PerMetricType[metricType] = p
	}
	p.AddPoint(metricName, step, value)
}

// AddValues is a shortcut to add all `values` as y-coordinates, and it uses the indices
// of the values as x-coordinate.
// `metricName` and `metricType` defines the label for the y-values, and the type of metric
// (if there are more than one plot).
// If there is only one plot, you can leave them empty ("") and there will be no plot title
// nor labels.
func (ps *Plots) AddValues(metricName, metricType string, values []float64) {
	for ii, v := range values {
		ps.AddPoint(metricName, metricType, float64(ii), v)
	}
}

// AddPoint adds a point for the given metric. The `step` is the x-axis, and `value` is the y-axis.
func (p *Plot) AddPoint(metricName string, step, value float64) {
	s, found := p.PerName[metricName]
	if !found {
		s = mg.NewSeries(mg.Titled(metricName))
		p.PerName[metricName] = s
	}
	mgValue := mg.MakeValue(step, value)
	s.Add(mgValue)

	if p.allPoints == nil {
		p.allPoints = mg.NewSeries()
	}
	p.allPoints.Add(mgValue)
}

// Plot a graph for each metric type. You don't need to call this if you called Attach -- it is called automatically
// at the end of the loop.
func (ps *Plots) Plot() {
	if !gonbui.IsNotebook {
		return
	}
	for _, key := range slices.SortedKeys(ps.PerMetricType) {
		gonbui.DisplayHTML(ps.PerMetricType[key].PlotToHTML(ps.Width, ps.Height))
	}
}

// PlotToHTML is similar to Plot but instead returns the HTML that include all plots
// (one per metric type), which can be displayed in some different way.
func (ps *Plots) PlotToHTML() string {
	parts := make([]string, 0, len(ps.PerMetricType))
	for _, key := range slices.SortedKeys(ps.PerMetricType) {
		parts = append(parts, ps.PerMetricType[key].PlotToHTML(ps.Width, ps.Height))
	}
	return strings.Join(parts, "\n")
}

// DynamicPlot will plot on a transient area that gets overwritten each time there is a new data point.
// If final is true, it clears the transient area, and it plots instead in the definitive version.
func (ps *Plots) DynamicPlot(final bool) {
	if !gonbui.IsNotebook {
		return
	}
	if ps.gonbID == "" {
		return
	}
	if ps.pointsAdded < 3 {
		return
	}
	if final == true {
		gonbui.UpdateHTML(ps.gonbID, "")
		ps.Plot()
		return
	}

	// Plot transient version.
	gonbui.UpdateHTML(ps.gonbID, ps.PlotToHTML())
	return
}

// PlotToHTML all series for a metric type associated with Plot, returning the HTML code for it (which includes the SVG).
func (p *Plot) PlotToHTML(width, height int) string {
	if len(p.PerName) == 0 {
		return ""
	}
	allSeries := make([]*mg.Series, 0, len(p.PerName))
	for _, key := range slices.SortedKeys(p.PerName) {
		allSeries = append(allSeries, p.PerName[key])
	}
	diagram := mg.New(width, height,
		mg.WithAutorange(mg.XAxis, allSeries...),
		mg.WithProjection(mg.XAxis, p.xProjection),
		mg.WithAutorange(mg.YAxis, allSeries...),
		mg.WithProjection(mg.YAxis, p.yProjection),
		mg.WithInset(70),
		mg.WithPadding(2),
		mg.WithColorScheme(90),
		mg.WithBackgroundColor("#f8f8f8"),
	)
	for _, s := range allSeries {
		diagram.Line(s, mg.UsingAxes(mg.XAxis, mg.YAxis), mg.UsingMarker("square"), mg.UsingStrokeWidth(2))
	}
	diagram.Axis(p.allPoints, mg.XAxis, diagram.ValueTicker('f', 0, 10), false, "Steps")
	diagram.Axis(p.allPoints, mg.YAxis, diagram.ValueTicker('f', 3, 10), true, p.MetricType)
	diagram.Frame()
	if p.MetricType != "" {
		diagram.Title(fmt.Sprintf("%s metrics", p.MetricType))
	}
	if len(p.PerName) > 1 || slices.SortedKeys(p.PerName)[0] != "" {
		diagram.Legend(mg.BottomLeft)
	}
	buf := bytes.NewBuffer(nil)
	err := diagram.Render(buf)
	if err != nil {
		return fmt.Sprintf("%+v", errors.Wrapf(err, "failed to render plot for %q", p.MetricType))
	}
	return buf.String()
}
