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
// Example usage:
//
// ```go
//
// ```
package margaid

import (
	"bytes"
	"fmt"
	mg "github.com/erkkah/margaid"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/janpfeifer/gonb/gonbui"
	errors "github.com/pkg/errors"
	"strings"
)

var (
	_ = gonbui.IsNotebook
	_ = mg.NewSeries
	_ = New
)

// Plots holds many plots for different metrics. They are organized per "metric type", where
// metric type is a unit/quantity unique name. It's assumed that series of the same "metric type"
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
}

// New creates new Margaid plots structure.
//
// It starts empty and can have the points added manually with Plots.AddPoint or automatically with Plots.Attach.
//
// Use Plots.Plot() to actually generate the
func New(width, height int, evalDatasets ...train.Dataset) *Plots {
	return &Plots{
		Width:        width,
		Height:       height,
		EvalDatasets: evalDatasets,
	}
}

// Plot holds the series to different metrics that share the same Y axis. They are organized per
// name of the metric.
type Plot struct {
	MetricType string

	// Flat maps a plot name to
	PerName map[string]*mg.Series

	// allPoints collects all points from all series, to configure the axis.
	allPoints *mg.Series
}

// DynamicUpdates configure the Plots to dynamically generate the plot as new points are included.
// It starts displaying the plot as soon as there are at least 3 points. It returns itself, so calls
// can be cascaded.
func (ps *Plots) DynamicUpdates() *Plots {
	ps.gonbID = gonbui.UniqueID()
	return ps
}

// Attach plots to the given loop, collecting metric values for plot. For each EvalDatasets given
// to `Plots.New()`, their metrics are evaluated and also plotted. It automatically calls
// Plots.Plot at the end of the loop (`loop.OnEnd()`).
func (ps *Plots) Attach(loop *train.Loop, numPoints int) {
	if ps.gonbID != "" {
		// If we are having a dynamically updating plot, we reserve the transient HTML block
		// upfront -- otherwise it will interfere with the progressbar the first time it is displayed.
		gonbui.UpdateHTML(ps.gonbID, "(...collecting metrics, minimum 3 required to start plotting...)")
	}
	train.NTimesDuringLoop(loop, numPoints, "margaid plots", 0,
		func(loop *train.Loop, metrics []tensor.Tensor) error {
			// Training metrics are pre-generated and given.
			step := float64(loop.LoopStep)
			for ii, desc := range loop.Trainer.TrainMetrics() {
				metric := shapes.ConvertTo[float64](metrics[ii].Value())
				ps.AddPoint("Train: "+desc.Name(), desc.MetricType(), step, metric)
			}

			// Eval metrics, if given
			for _, ds := range ps.EvalDatasets {
				evalMetrics, err := loop.Trainer.Eval(ds)
				if err != nil {
					return err
				}
				for ii, desc := range loop.Trainer.EvalMetrics() {
					metric := shapes.ConvertTo[float64](evalMetrics[ii].Value())
					ps.AddPoint(fmt.Sprintf("Eval on %s: %s", ds.Name(), desc.Name()), desc.MetricType(), step, metric)
				}
			}

			ps.pointsAdded++
			if ps.gonbID != "" && ps.pointsAdded >= 3 {
				ps.DynamicPlot()
			}
			return nil
		})
	loop.OnEnd("margaid plots", 120, func(_ *train.Loop, _ []tensor.Tensor) error {
		if ps.gonbID != "" {
			// Erase intermediary transient plots.
			gonbui.UpdateHTML(ps.gonbID, "")
		}
		// Final plot.
		ps.Plot()
		return nil
	})
}

// AddPoint adds a point for the given metric. The x-axis is given by step and the y-axis is
// given by value.
func (ps *Plots) AddPoint(metricName, metricType string, step, value float64) {
	if ps.PerMetricType == nil {
		ps.PerMetricType = make(map[string]*Plot)
	}
	p, found := ps.PerMetricType[metricType]
	if !found {
		p = &Plot{MetricType: metricType, PerName: make(map[string]*mg.Series)}
		ps.PerMetricType[metricType] = p
	}
	p.AddPoint(metricName, step, value)
}

// AddPoint adds a point for the given metric. The x-axis is given by step and the y-axis is
// given by value.
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
	for _, key := range slices.SortedKeys(ps.PerMetricType) {
		gonbui.DisplayHTML(ps.PerMetricType[key].PlotToHTML(ps.Width, ps.Height))
	}
}

// DynamicPlot will plot on a transient area that gets overwritten each time there is a new data point.
func (ps *Plots) DynamicPlot() {
	if ps.gonbID == "" {
		return
	}
	parts := make([]string, 0, len(ps.PerMetricType))
	for _, key := range slices.SortedKeys(ps.PerMetricType) {
		parts = append(parts, ps.PerMetricType[key].PlotToHTML(ps.Width, ps.Height))
	}
	gonbui.UpdateHTML(ps.gonbID, strings.Join(parts, "\n"))
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
		mg.WithAutorange(mg.YAxis, allSeries...),
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
	diagram.Title(fmt.Sprintf("%s metrics", p.MetricType))
	diagram.Legend(mg.BottomLeft)
	buf := bytes.NewBuffer(nil)
	err := diagram.Render(buf)
	if err != nil {
		return fmt.Sprintf("%+v", errors.Wrapf(err, "failed to render plot for %q", p.MetricType))
	}
	return buf.String()
}
