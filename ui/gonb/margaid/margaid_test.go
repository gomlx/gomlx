// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package margaid

import (
	"math"
	"os"
	"path"
	"testing"

	stdplots "github.com/gomlx/gomlx/ui/plot"
	"github.com/stretchr/testify/require"
)

// TestAddPoint_BuildsSeries characterizes how AddPoint groups points into Plot structs (one per
// MetricType) and mg.Series (one per MetricName within a Plot) -- this state (ps.PerMetricType)
// must survive the Phase 0 refactor unchanged, since it's rendering-specific and stays on Plots,
// not moved into plot.Config.
func TestAddPoint_BuildsSeries(t *testing.T) {
	ps := New(1024, 400)
	ps.AddPoint(stdplots.Point{MetricName: "Train: Loss", MetricType: "loss", Step: 0, Value: 1.0})
	ps.AddPoint(stdplots.Point{MetricName: "Train: Loss", MetricType: "loss", Step: 1, Value: 0.5})
	ps.AddPoint(stdplots.Point{MetricName: "Eval: Accuracy", MetricType: "accuracy", Step: 0, Value: 0.9})

	require.Len(t, ps.PerMetricType, 2, "expected one Plot per metric type (loss, accuracy)")
	lossPlot := ps.PerMetricType["loss"]
	require.Len(t, lossPlot.PerName, 1)
	series := lossPlot.PerName["Train: Loss"]
	require.Equal(t, 2, series.Size())
	require.Equal(t, 1.0, series.MaxY())
	require.Equal(t, 0.5, series.MinY())

	accPlot := ps.PerMetricType["accuracy"]
	require.Len(t, accPlot.PerName, 1)
}

// TestAddPoint_DropsInvalidPoints characterizes that NaN/Inf points (in either Value or Step)
// are silently dropped -- no Plot entry, no file write.
func TestAddPoint_DropsInvalidPoints(t *testing.T) {
	ps := New(1024, 400)
	ps.AddPoint(stdplots.Point{MetricName: "m", MetricType: "t", Step: 0, Value: math.NaN()})
	ps.AddPoint(stdplots.Point{MetricName: "m", MetricType: "t", Step: 0, Value: math.Inf(1)})
	ps.AddPoint(stdplots.Point{MetricName: "m", MetricType: "t", Step: math.NaN(), Value: 1.0})
	require.Empty(t, ps.PerMetricType, "invalid points must not create any Plot entry")
}

// TestAddPoint_WritesToFileWhenWriting characterizes that once a file writer is attached (as
// WithFile does), AddPoint also persists the point asynchronously; closing the writer
// (stopWriting) flushes it so it can be read back with stdplots.LoadPoints.
func TestAddPoint_WritesToFileWhenWriting(t *testing.T) {
	filePath := path.Join(t.TempDir(), stdplots.TrainingPlotFileName)
	ps := New(1024, 400)
	ps.Config.StartWriting(filePath)

	ps.AddPoint(stdplots.Point{MetricName: "m", MetricType: "t", Step: 0, Value: 1.0})
	ps.AddPoint(stdplots.Point{MetricName: "m", MetricType: "t", Step: 1, Value: 2.0})
	ps.StopWriting()

	points, err := stdplots.LoadPoints(filePath)
	require.NoError(t, err)
	require.Len(t, points, 2)
	require.Equal(t, 1.0, points[0].Value)
	require.Equal(t, 2.0, points[1].Value)
}

// TestPreloadFile_RoundTrip characterizes PreloadFile reading points back from a file, applying
// renameFn to every MetricName, and updating pointsAdded to the minimum series length across all
// metric names/types once loading is done (a different heuristic than plotly's -- distinct steps
// seen while loading -- kept as-is since margaid never used that approach).
func TestPreloadFile_RoundTrip(t *testing.T) {
	filePath := path.Join(t.TempDir(), "points.json")
	w, errCh := stdplots.CreatePointsWriter(filePath)
	w <- stdplots.Point{MetricName: "Loss", MetricType: "loss", Step: 0, Value: 1.0}
	w <- stdplots.Point{MetricName: "Loss", MetricType: "loss", Step: 1, Value: 0.8}
	close(w)
	require.NoError(t, <-errCh)

	ps := New(1024, 400)
	got, err := ps.PreloadFile(filePath, func(metricName string) string {
		return "[m1] " + metricName
	})
	require.NoError(t, err)
	require.Same(t, ps, got)

	lossPlot := ps.PerMetricType["loss"]
	require.Contains(t, lossPlot.PerName, "[m1] Loss")
	require.Equal(t, 2, lossPlot.PerName["[m1] Loss"].Size())
	require.Equal(t, 2, ps.PointsAdded())
}

// TestPreloadFile_MissingFile_Errors characterizes that, unlike WithFile, PreloadFile itself
// does not tolerate a missing file -- it returns (nil, err) directly from os.Open's error.
func TestPreloadFile_MissingFile_Errors(t *testing.T) {
	ps := New(1024, 400)
	got, err := ps.PreloadFile(path.Join(t.TempDir(), "does-not-exist.json"), nil)
	require.Error(t, err)
	require.Nil(t, got)
}

// TestWithFile_WritesAndReloads exercises WithFile end-to-end: points written through one Plots
// must be visible to a second Plots preloading the same file.
func TestWithFile_WritesAndReloads(t *testing.T) {
	filePath := path.Join(t.TempDir(), stdplots.TrainingPlotFileName)

	ps, err := New(1024, 400).WithFile(filePath)
	require.NoError(t, err, "WithFile must tolerate the file not existing yet")
	ps.AddPoint(stdplots.Point{MetricName: "m", MetricType: "t", Step: 0, Value: 1.0})
	ps.StopWriting()

	_, statErr := os.Stat(filePath)
	require.NoError(t, statErr)

	ps2, err := New(1024, 400).WithFile(filePath)
	require.NoError(t, err)
	require.Equal(t, 1, ps2.PerMetricType["t"].PerName["m"].Size())
}
