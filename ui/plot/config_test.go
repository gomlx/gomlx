// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package plot

import (
	"math"
	"path"
	"testing"

	"github.com/stretchr/testify/require"
)

// fakePlotter is a minimal Plotter used to unit-test Config without needing a real rendering
// package (plotly/margaid) or a real train.Loop.
type fakePlotter struct {
	added []Point
	done  []bool
}

func (f *fakePlotter) AddPoint(pt Point)                 { f.added = append(f.added, pt) }
func (f *fakePlotter) DynamicSampleDone(incomplete bool) { f.done = append(f.done, incomplete) }

func TestConfig_WithDatasetsAndBatchNorm(t *testing.T) {
	c := NewConfig(&fakePlotter{}, "test", func() {})
	got := c.WithDatasets(nil, nil)
	require.Same(t, c, got, "WithDatasets must return itself for chaining")
	require.Len(t, c.EvalDatasets, 2)

	got2 := c.WithBatchNormalizationAveragesUpdate(nil)
	require.Same(t, c, got2)
}

func TestConfig_WithCustomMetricFn(t *testing.T) {
	c := NewConfig(&fakePlotter{}, "test", func() {})
	called := false
	got := c.WithCustomMetricFn(func(_ Plotter, _ float64) error {
		called = true
		return nil
	})
	require.Same(t, c, got)
	require.NoError(t, c.customMetricFn(c.owner, 0))
	require.True(t, called)
}

func TestConfig_FilterAndWrite_DropsInvalid(t *testing.T) {
	c := NewConfig(&fakePlotter{}, "test", func() {})
	require.False(t, c.FilterAndWrite(Point{Value: math.NaN()}))
	require.False(t, c.FilterAndWrite(Point{Value: math.Inf(1)}))
	require.False(t, c.FilterAndWrite(Point{Step: math.NaN(), Value: 1}))
	require.False(t, c.FilterAndWrite(Point{Step: math.Inf(-1), Value: 1}))
	require.True(t, c.FilterAndWrite(Point{Step: 0, Value: 1}))
}

func TestConfig_FilterAndWrite_WritesWhenOpen(t *testing.T) {
	filePath := path.Join(t.TempDir(), "points.json")
	c := NewConfig(&fakePlotter{}, "test", func() {})
	c.StartWriting(filePath)

	ok := c.FilterAndWrite(Point{MetricName: "m", MetricType: "t", Step: 0, Value: 1.0})
	require.True(t, ok)
	c.StopWriting()

	points, err := LoadPoints(filePath)
	require.NoError(t, err)
	require.Len(t, points, 1)
	require.Equal(t, 1.0, points[0].Value)
}

func TestConfig_StopWriting_Idempotent(t *testing.T) {
	c := NewConfig(&fakePlotter{}, "test", func() {})
	c.StopWriting() // no writer started: must not panic
	c.StartWriting(path.Join(t.TempDir(), "points.json"))
	c.StopWriting()
	c.StopWriting() // closing twice must not panic either
}

func TestConfig_PointsAdded(t *testing.T) {
	c := NewConfig(&fakePlotter{}, "test", func() {})
	require.Equal(t, 0, c.PointsAdded())
	c.IncrementPointsAdded()
	c.IncrementPointsAdded()
	require.Equal(t, 2, c.PointsAdded())

	c.MarkSampleDone(true) // incomplete: does not count
	require.Equal(t, 2, c.PointsAdded())
	c.MarkSampleDone(false)
	require.Equal(t, 3, c.PointsAdded())
}
