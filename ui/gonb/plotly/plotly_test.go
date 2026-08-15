// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package plotly

import (
	"math"
	"os"
	"path"
	"testing"

	grob "github.com/MetalBlueberry/go-plotly/generated/v2.34.0/graph_objects"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/gomlx/gomlx/ui/plot"
	"github.com/stretchr/testify/require"
)

// TestAddPoint_BuildsFigsAndTraces characterizes how AddPoint groups points into figures (one
// per MetricType) and traces (one per MetricName within a figure) -- this state (pc.figs,
// pc.metricsNamesToTrace, pc.metricsTypesToFig) must survive the Phase 0 refactor unchanged,
// since it's rendering-specific and stays on PlotConfig, not moved into plot.Config.
func TestAddPoint_BuildsFigsAndTraces(t *testing.T) {
	pc := New()
	pc.AddPoint(plot.Point{MetricName: "Train: Loss", Short: "T/loss", MetricType: "loss", Step: 0, Value: 1.0})
	pc.AddPoint(plot.Point{MetricName: "Train: Loss", Short: "T/loss", MetricType: "loss", Step: 1, Value: 0.5})
	pc.AddPoint(plot.Point{MetricName: "Eval: Accuracy", Short: "E/acc", MetricType: "accuracy", Step: 0, Value: 0.9})

	require.Len(t, pc.figs, 2, "expected one figure per metric type (loss, accuracy)")
	lossFigIdx := pc.metricsTypesToFig["loss"]
	accFigIdx := pc.metricsTypesToFig["accuracy"]
	require.NotEqual(t, lossFigIdx, accFigIdx)

	lossFig := pc.figs[lossFigIdx]
	require.Len(t, lossFig.Data, 1, "expected a single trace for the single 'Train: Loss' metric name")
	trace := lossFig.Data[0].(*grob.Scatter)
	xs := trace.X.Value().([]float64)
	ys := trace.Y.Value().([]float64)
	require.Equal(t, []float64{0, 1}, xs)
	require.Equal(t, []float64{1.0, 0.5}, ys)

	accFig := pc.figs[accFigIdx]
	require.Len(t, accFig.Data, 1)
}

// TestAddPoint_DropsInvalidPoints characterizes that NaN/Inf points (in either Value or Step)
// are silently dropped -- no figure, no trace, no file write.
func TestAddPoint_DropsInvalidPoints(t *testing.T) {
	pc := New()
	pc.AddPoint(plot.Point{MetricName: "m", MetricType: "t", Step: 0, Value: math.NaN()})
	pc.AddPoint(plot.Point{MetricName: "m", MetricType: "t", Step: 0, Value: math.Inf(1)})
	pc.AddPoint(plot.Point{MetricName: "m", MetricType: "t", Step: math.NaN(), Value: 1.0})
	require.Empty(t, pc.figs, "invalid points must not create any figure")
}

// TestAddPoint_WritesToFileWhenWriting characterizes that once a file writer is attached (as
// WithCheckpoint does), AddPoint also persists the point asynchronously; closing the writer
// (stopWriting) flushes it so it can be read back with plot.LoadPoints.
func TestAddPoint_WritesToFileWhenWriting(t *testing.T) {
	filePath := path.Join(t.TempDir(), plot.TrainingPlotFileName)
	pc := New()
	pc.Config.StartWriting(filePath)

	pc.AddPoint(plot.Point{MetricName: "m", MetricType: "t", Step: 0, Value: 1.0})
	pc.AddPoint(plot.Point{MetricName: "m", MetricType: "t", Step: 1, Value: 2.0})
	pc.StopWriting()

	points, err := plot.LoadPoints(filePath)
	require.NoError(t, err)
	require.Len(t, points, 2)
	require.Equal(t, 1.0, points[0].Value)
	require.Equal(t, 2.0, points[1].Value)
}

// TestLoadCheckpointData_RoundTrip characterizes LoadCheckpointData reading points back from a
// file (not a directory), applying filters in order (a point is dropped if any filter returns
// false), and counting pointsAdded once per distinct Step (not once per point).
func TestLoadCheckpointData_RoundTrip(t *testing.T) {
	filePath := path.Join(t.TempDir(), "points.json")
	w, errCh := plot.CreatePointsWriter(filePath)
	w <- plot.Point{MetricName: "Train: Loss", MetricType: "loss", Step: 0, Value: 1.0}
	w <- plot.Point{MetricName: "Eval: Loss", MetricType: "loss", Step: 0, Value: 1.1}
	w <- plot.Point{MetricName: "Train: Loss", MetricType: "loss", Step: 1, Value: 0.8}
	w <- plot.Point{MetricName: "Skip: Me", MetricType: "loss", Step: 1, Value: 99}
	close(w)
	require.NoError(t, <-errCh)

	pc := New()
	err := pc.LoadCheckpointData(filePath, func(p *plot.Point) bool {
		return p.MetricName != "Skip: Me"
	})
	require.NoError(t, err)

	require.Equal(t, 2, pc.PointsAdded(), "2 distinct steps (0 and 1), the dropped point at step 1 doesn't add a new step")
	lossFigIdx := pc.metricsTypesToFig["loss"]
	require.Len(t, pc.figs[lossFigIdx].Data, 2, "Train: Loss and Eval: Loss traces, Skip: Me filtered out")
}

// TestLoadCheckpointData_FromDirectory characterizes that a directory is resolved to
// plot.TrainingPlotFileName within it (LoadPointsFromCheckpoint), not treated as a file itself.
func TestLoadCheckpointData_FromDirectory(t *testing.T) {
	dir := t.TempDir()
	filePath := path.Join(dir, plot.TrainingPlotFileName)
	w, errCh := plot.CreatePointsWriter(filePath)
	w <- plot.Point{MetricName: "m", MetricType: "t", Step: 0, Value: 1.0}
	close(w)
	require.NoError(t, <-errCh)

	pc := New()
	require.NoError(t, pc.LoadCheckpointData(dir))
	require.Equal(t, 1, pc.PointsAdded())
}

// TestWithCheckpoint_NilHandler_NoOp characterizes that WithCheckpoint(nil) is documented as a
// no-op: no panic, pc is returned unchanged (no file writer attached, so a point added
// afterward isn't persisted -- StopWriting is a no-op with nothing to flush).
func TestWithCheckpoint_NilHandler_NoOp(t *testing.T) {
	pc := New().WithCheckpoint(nil)
	pc.StopWriting() // must not panic
}

// TestWithCheckpoint_WritesAndReloads exercises the real ml/model/checkpoint.Handler path
// end-to-end (not just a raw file path): points written through one PlotConfig must be visible
// to a second PlotConfig loading the same checkpoint directory.
func TestWithCheckpoint_WritesAndReloads(t *testing.T) {
	dir := t.TempDir()
	store := model.NewStore()
	handler, err := checkpoint.Build(store).Dir(dir).Done()
	require.NoError(t, err)

	pc := New().WithCheckpoint(handler)
	pc.AddPoint(plot.Point{MetricName: "m", MetricType: "t", Step: 0, Value: 1.0})
	pc.StopWriting()

	_, err = os.Stat(path.Join(dir, plot.TrainingPlotFileName))
	require.NoError(t, err, "WithCheckpoint must write to <dir>/training_plot_points.json")

	pc2 := New().WithCheckpoint(handler)
	require.Equal(t, 1, pc2.PointsAdded())
}
