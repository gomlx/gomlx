// Package plots define common types and utilities to the different plot libraries.
package plots

import (
	"encoding/json"
	"fmt"
	"io"
	"math"
	"os"
	"path"
	"slices"
	"sort"

	"github.com/charmbracelet/lipgloss"
	lgtable "github.com/charmbracelet/lipgloss/table"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/layers/batchnorm"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	types "github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
	"golang.org/x/exp/maps"
	"k8s.io/klog/v2"
)

// TrainingPlotFileName is the default file name within a checkpoint directory to store
// plot points collected during training.
const TrainingPlotFileName = "training_plot_points.json"

// Point represents a training plot point. It is used to save/load plots.
type Point struct {
	// MetricName of this point.
	MetricName string

	// Short name
	Short string

	// MetricType typically will be "loss", "accuracy".
	// It's used in plotting to aggregate similar metric types in the same plot.
	MetricType string

	// Step is the global step this metric was measured.
	// Usually, this is an int value, stored as a float64.
	Step float64

	// Value is the metric captured.
	Value float64
}

// Plotter is a generic plotter API, implemented by [margaid.Plots] and [plotly.PlotConfig].
type Plotter interface {
	// AddPoint to be drawn. One metric at a time.
	AddPoint(point Point)

	// DynamicSampleDone is called after all the data points recorded for this sample (evaluation at a time step).
	// The value `incomplete` is set to true if any of the evaluations are NaN or infinite.
	//
	// If in a notebook, this would trigger a redraw of the plot.
	DynamicSampleDone(incomplete bool)
}

// CustomMetricFn can be used by plotters to allow the user to subscribe a function to create arbitrary metrics to
// plot. Use
type CustomMetricFn func(plotter Plotter, step float64) error

// AddTrainAndEvalMetrics is used by plotters (see [margaid.PLots] and [plotly.PlotConfig]) to include the metrics
// generated given training, plus to run evaluation on the given datasets.
//
// Notice it evaluate on the datasets sequentially -- presumably the training could go in parallel if there is
// enough accelerator processing / memory. But this doesn't assume that.
//
// If batchNormAveragesDS is given, and the model uses batch normalization, it will first go over this dataset and
// update the averages of the mean and variance accordingly.
func AddTrainAndEvalMetrics(plotter Plotter, loop *train.Loop, trainMetrics []*tensors.Tensor,
	evalDatasets []train.Dataset, batchNormAveragesDS train.Dataset) error {
	if batchNormAveragesDS != nil {
		_, err := batchnorm.UpdateAverages(loop.Trainer, batchNormAveragesDS)
		if err != nil {
			return errors.WithMessagef(err, "Updating batch normalization averages before evaluation: ")
		}
	}

	// Training metrics are pre-generated and given.
	step := float64(loop.Trainer.GlobalStep())
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
		plotter.AddPoint(Point{
			MetricName: "Train: " + desc.Name(),
			Short:      fmt.Sprintf("T/%s", desc.ShortName()),
			MetricType: desc.MetricType(),
			Step:       step,
			Value:      metric})
	}

	// Eval metrics, if given
	for _, ds := range evalDatasets {
		evalMetrics, err := loop.Trainer.Eval(ds)
		if err != nil {
			return err
		}
		for ii, desc := range loop.Trainer.EvalMetrics() {
			metric := shapes.ConvertTo[float64](evalMetrics[ii].Value())
			if math.IsNaN(metric) || math.IsInf(metric, 0) {
				incomplete = true
				continue
			}
			metricType := desc.MetricType()
			dsShort := ds.Name()[:3]
			if sn, ok := ds.(train.HasShortName); ok {
				dsShort = sn.ShortName()
			}
			plotter.AddPoint(Point{
				MetricName: fmt.Sprintf("%s on %s", desc.Name(), ds.Name()),
				Short:      fmt.Sprintf("%s(%s)", desc.ShortName(), dsShort),
				MetricType: metricType,
				Step:       step,
				Value:      metric})
		}
	}

	plotter.DynamicSampleDone(incomplete)
	return nil
}

// LoadPointsFromCheckpoint loads all plot points saved during training in file [TrainingPlotFileName]
// in a checkpoint directory.
func LoadPointsFromCheckpoint(checkpointDir string) ([]Point, error) {
	checkpointDir = fsutil.MustReplaceTildeInDir(checkpointDir)
	filePath := path.Join(checkpointDir, TrainingPlotFileName)
	return LoadPoints(filePath)
}

// LoadPoints parses all plot points saved in the given file.
func LoadPoints(filePath string) ([]Point, error) {
	f, err := os.Open(filePath)
	if err != nil {
		return nil, errors.Wrapf(err, "failed to read Plots file %q", filePath)
	}

	// Read previously stored points.
	dec := json.NewDecoder(f)
	var point Point
	var points []Point
	for {
		err := dec.Decode(&point)
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, errors.Wrapf(err, "error while decoding plots file %q", filePath)
		}
		points = append(points, point)
	}
	_ = f.Close()
	return points, nil
}

// CreatePointsWriter creates a channel to write Point to the given file.
// It creates an errReport channel to report an error (or nil) back at the very end.
// If any error occurs, it stops writing, and will report the error back once pointWriter is closed.
func CreatePointsWriter(filePath string) (pointWriter chan<- Point, errReport <-chan error) {
	pointChan := make(chan Point, 100)
	pointWriter = pointChan
	errChan := make(chan error, 1)
	errReport = errChan
	go func() {
		// Create/append file with upcoming metrics.
		f, err := os.OpenFile(filePath, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0664)
		if err != nil {
			err = errors.Wrapf(err, "failed to open Plots file %q for append", filePath)
			klog.Errorf("Error: %v", err)
		}
		enc := json.NewEncoder(f)
		for point := range pointChan {
			if err == nil {
				err = enc.Encode(point)
				if err != nil {
					err = errors.Wrapf(err, "failed to encode point %v", point)
					klog.Errorf("Error: %v", err)
				} else {
				}
			}
		}
		if f != nil {
			if err == nil {
				err = f.Close()
			} else {
				_ = f.Close()
			}
		}
		errChan <- err
	}()
	return
}

// Points is a collection of Point objects organized by their Step value.
// It's a `map[float64][]Point` with several utility methods.
type Points map[float64][]Point

// NewPoints create a Points object from a collection of individual `Point`.
//
// See LoadPoints and LoadPointsFromCheckpoint if you want to read `rawPoints` from a file.
func NewPoints(rawPoints []Point) (points Points) {
	points = make(map[float64][]Point)
	for _, p := range rawPoints {
		points[p.Step] = append(points[p.Step], p)
	}
	return points
}

// Map executes the given function on all individual points, in `Step` order.
// Note that if `p.Step` change, it is not re-index.
//
// If you need to reindex the [Step] after the `Map` transformation, you may call [Extract]
// followed by [NewPoints] to create the re-indexed structure.
func (points Points) Map(fn func(p *Point)) {
	sortedKeys := maps.Keys(points)
	slices.Sort(sortedKeys)
	for _, step := range sortedKeys {
		stepPoints := points[step]
		for ii := range stepPoints {
			fn(&stepPoints[ii])
		}
	}
}

// Filter only keeps those points for which `fn` returns true, removing the other ones.
func (points Points) Filter(fn func(p Point) bool) {
	sortedKeys := maps.Keys(points)
	slices.Sort(sortedKeys)
	for _, step := range sortedKeys {
		stepPoints := points[step]
		newStepPoints := make([]Point, 0, len(stepPoints))
		for _, pt := range stepPoints {
			if fn(pt) {
				newStepPoints = append(newStepPoints, pt)
			}
		}
		if len(newStepPoints) == len(stepPoints) {
			continue // Nothing filtered.
		}
		if len(newStepPoints) == 0 {
			// Remove this step.
			delete(points, step)
		} else {
			points[step] = newStepPoints
		}
	}
}

// Extract converts the [Points] structure back to a list of individual points.
// The output is sorted by [Point.Step].
func (points Points) Extract() (rawPoints []Point) {
	points.Map(func(p *Point) {
		rawPoints = append(rawPoints, *p)
	})
	return
}

// Add `otherPoints` into this `Points` structure. `otherPoints` is unchanged.
// It does not check for duplicates, points from `otherPoints` are simply appended as is.
func (points Points) Add(otherPoints Points) {
	otherPoints.Map(func(p *Point) {
		points[p.Step] = append(points[p.Step], *p)
	})
}

// MetricsNames return the list of metrics names in the whole collection, sorted alphabetically by their type and
// then by their name.
func (points Points) MetricsNames() []string {
	metricNames := types.Make[string]()
	nameToType := make(map[string]string)
	points.Map(func(p *Point) {
		metricNames.Insert(p.MetricName)
		nameToType[p.MetricName] = p.MetricType
	})
	names := xslices.SortedKeys(metricNames)
	sort.SliceStable(names, func(i, j int) bool {
		return nameToType[names[i]] < nameToType[names[j]]
	})
	return names
}

// TableForMetrics returns a table with the first column being the `Step` followed
// by the columns given by the `metrics` names.
// If `metrics` is empty, it will include all metrics in the table.
func (points Points) TableForMetrics(metrics ...string) string {
	cellStyle := lipgloss.NewStyle().Padding(0, 1)
	headerStyle := lipgloss.NewStyle().Padding(0, 1).Bold(true).Reverse(true)
	table := lgtable.New().
		Border(lipgloss.RoundedBorder()).
		StyleFunc(func(row, col int) lipgloss.Style {
			if row == 0 {
				return headerStyle
			}
			return cellStyle
		})

	// Headers from metric names.
	if len(metrics) == 0 {
		metrics = points.MetricsNames()
	}
	headers := []string{"Step"}
	headers = append(headers, metrics...)
	table.Headers(headers...)

	// Add rows:
	sortedKeys := maps.Keys(points)
	slices.Sort(sortedKeys)
	for _, step := range sortedKeys {
		row := make([]string, 1+len(metrics))
		row[0] = fmt.Sprintf("%.0f", step)
		for _, pt := range points[step] {
			idx := slices.Index(metrics, pt.MetricName)
			if idx != -1 {
				row[idx+1] = fmt.Sprintf("%f", pt.Value)
			}
		}
		table.Row(row...)
	}
	return table.String()
}

func (points Points) String() string {
	return points.TableForMetrics()
}
