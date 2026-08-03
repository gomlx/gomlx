// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func TestResolveOutputFilePath_DefaultIsStableWithinASession(t *testing.T) {
	t.Parallel()
	pb := NewPlotBuilder(false, "", 0, "")
	checkpointPaths := []string{"/home/user/mnist_data/checkpoint"}

	// Calling it twice must return the exact same path -- that's the whole
	// point (see PlotBuilder's doc comment): a stable path lets repeated -loop
	// iterations overwrite the same file instead of a new one accumulating
	// every time. It does NOT need to be reproducible across separate
	// process runs -- just stable within one PlotBuilder's lifetime (per
	// maintainer feedback).
	path1 := pb.resolveOutputFilePath(checkpointPaths)
	path2 := pb.resolveOutputFilePath(checkpointPaths)
	require.Equal(t, path1, path2)
	require.True(t, filepath.IsAbs(path1))
}

func TestResolveOutputFilePath_CacheIgnoresCheckpointPathsArgument(t *testing.T) {
	t.Parallel()
	pb := NewPlotBuilder(false, "", 0, "")

	// Once cached, the *same PlotBuilder* reuses the same path regardless of
	// what checkpointPaths is passed on a later call -- correct, because in
	// real usage one PlotBuilder is used for one fixed set of checkpoint
	// paths for its entire lifetime; the cache is intentionally not keyed by
	// them.
	pathA := pb.resolveOutputFilePath([]string{"/checkpoints/model-a"})
	pathB := pb.resolveOutputFilePath([]string{"/checkpoints/model-b"})
	require.Equal(t, pathA, pathB)
}

func TestResolveOutputFilePath_ExplicitRelativePath(t *testing.T) {
	t.Parallel()
	pb := NewPlotBuilder(false, "plot.html", 0, "")
	got := pb.resolveOutputFilePath([]string{"/checkpoints/model-a"})
	require.Equal(t, "/checkpoints/model-a/plot.html", got)
}

func TestResolveOutputFilePath_ExplicitAbsolutePath(t *testing.T) {
	t.Parallel()
	// t.TempDir(), not a hardcoded "/tmp/..." literal: a Unix-style leading-slash
	// path isn't absolute by Windows' rules (filepath.IsAbs requires a drive
	// letter there), so a hardcoded literal passes on Unix and fails on the
	// Windows CI runner -- t.TempDir() is a real absolute path on whichever OS
	// the test is running on.
	absPath := filepath.Join(t.TempDir(), "my-plot.html")
	pb := NewPlotBuilder(false, absPath, 0, "")
	got := pb.resolveOutputFilePath([]string{"/checkpoints/model-a"})
	require.Equal(t, absPath, got)
}

func TestLatestMetricsModTime_NoFilesReturnsZero(t *testing.T) {
	t.Parallel()
	got := latestMetricsModTime([]string{filepath.Join(t.TempDir(), "does-not-exist")})
	require.True(t, got.IsZero())
}

func TestLatestMetricsModTime_ReturnsMostRecent(t *testing.T) {
	t.Parallel()
	dirA, dirB := t.TempDir(), t.TempDir()
	older := time.Now().Add(-time.Hour)
	newer := time.Now()

	writePoints := func(dir string, modTime time.Time) {
		filePath := filepath.Join(dir, "training_plot_points.json")
		require.NoError(t, os.WriteFile(filePath, []byte("{}\n"), 0644))
		require.NoError(t, os.Chtimes(filePath, modTime, modTime))
	}
	writePoints(dirA, older)
	writePoints(dirB, newer)

	got := latestMetricsModTime([]string{dirA, dirB})
	require.WithinDuration(t, newer, got, time.Second)
}

func TestLatestMetricsModTime_SkipsMissingFilesAmongMultiple(t *testing.T) {
	t.Parallel()
	dirWithFile := t.TempDir()
	modTime := time.Now()
	filePath := filepath.Join(dirWithFile, "training_plot_points.json")
	require.NoError(t, os.WriteFile(filePath, []byte("{}\n"), 0644))
	require.NoError(t, os.Chtimes(filePath, modTime, modTime))

	dirWithoutFile := filepath.Join(t.TempDir(), "no-training-data-yet")

	got := latestMetricsModTime([]string{dirWithoutFile, dirWithFile})
	require.WithinDuration(t, modTime, got, time.Second)
}

func TestInjectAutoRefresh_InsertsTag(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath, []byte("<!DOCTYPE html><html><head><title>x</title></head><body></body></html>"), 0644))

	require.NoError(t, injectAutoRefresh(htmlPath, 30*time.Second))

	content, err := os.ReadFile(htmlPath)
	require.NoError(t, err)
	require.Contains(t, string(content), `<meta http-equiv="refresh" content="30">`)
	// Must land right after <head>, before the rest of the original content.
	require.Contains(t, string(content), `<head><meta http-equiv="refresh" content="30"><title>x</title>`)
}

func TestInjectAutoRefresh_SubSecondPeriodClampsToOne(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath, []byte("<head></head>"), 0644))

	require.NoError(t, injectAutoRefresh(htmlPath, 200*time.Millisecond))

	content, err := os.ReadFile(htmlPath)
	require.NoError(t, err)
	require.Contains(t, string(content), `content="1"`)
}

func TestInjectAutoRefresh_NoHeadTagReturnsError(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath, []byte("<html><body>no head here</body></html>"), 0644))

	err := injectAutoRefresh(htmlPath, 30*time.Second)
	require.Error(t, err)
}

// TestDisambiguateShortLabels_NoCollision characterizes the common case: when every line's
// short is already unique within the metric type, nothing changes.
func TestDisambiguateShortLabels_NoCollision(t *testing.T) {
	t.Parallel()
	lines := []*plotLineInfo{
		{short: "T/loss"},
		{short: "T/~loss"},
	}
	disambiguateShortLabels(lines)
	require.Equal(t, "T/loss", lines[0].short)
	require.Equal(t, "T/~loss", lines[1].short)
}

// TestDisambiguateShortLabels_Collision reproduces the real bug found in the maintainer's
// FlowMatching fixture: "Mean Loss on train" and "Mean Loss on validation" both produce the
// Short "#loss()" (their eval dataset's ShortName came out empty), so without disambiguation
// they'd collide into a single series in vizb's chart (vizb groups by this label). Every line
// must end up with a distinct short after the call.
func TestDisambiguateShortLabels_Collision(t *testing.T) {
	t.Parallel()
	lines := []*plotLineInfo{
		{short: "#loss()", desc: "Mean Loss on train"},
		{short: "#loss()", desc: "Mean Loss on validation"},
		{short: "T/loss", desc: "Train: Loss"}, // Unrelated, must be untouched.
	}
	disambiguateShortLabels(lines)

	seen := make(map[string]bool)
	for _, line := range lines {
		require.False(t, seen[line.short], "short %q is not unique after disambiguation", line.short)
		seen[line.short] = true
	}
	require.Equal(t, "T/loss", lines[2].short, "non-colliding line must be left untouched")
	require.Contains(t, lines[0].short, "#loss()")
	require.Contains(t, lines[1].short, "#loss()")
}

// TestDisambiguateShortLabels_ThreeWayCollision checks the disambiguation counter handles more
// than two lines sharing the same short.
func TestDisambiguateShortLabels_ThreeWayCollision(t *testing.T) {
	t.Parallel()
	lines := []*plotLineInfo{
		{short: "x"}, {short: "x"}, {short: "x"},
	}
	disambiguateShortLabels(lines)
	seen := make(map[string]bool)
	for _, line := range lines {
		require.False(t, seen[line.short])
		seen[line.short] = true
	}
}

func TestBuildPageSidebarHTML_EmptyWhenNothingToShow(t *testing.T) {
	t.Parallel()
	require.Empty(t, buildPageSidebarHTML("", nil, nil))
}

func TestBuildPageSidebarHTML_TitleOnly(t *testing.T) {
	t.Parallel()
	got := buildPageSidebarHTML("My Title", nil, nil)
	require.Contains(t, got, `id="gomlx-sidebar"`)
	require.Contains(t, got, "<h1")
	require.Contains(t, got, "My Title")
	require.NotContains(t, got, "Models compared")
}

func TestBuildPageSidebarHTML_EscapesUserContent(t *testing.T) {
	t.Parallel()
	got := buildPageSidebarHTML(`<script>alert(1)</script>`, []legendEntry{{text: `<b>evil</b>`}}, nil)
	require.NotContains(t, got, "<script>")
	require.NotContains(t, got, "<b>evil</b>")
	require.Contains(t, got, "&lt;script&gt;")
}

func TestBuildPageSidebarHTML_ModelsAndDescriptions(t *testing.T) {
	t.Parallel()
	got := buildPageSidebarHTML("",
		[]legendEntry{{modelIdx: 0, text: "#1 model-a (path/a)"}, {modelIdx: 1, text: "#2 model-b (path/b)"}},
		[]legendEntry{{modelIdx: 0, text: "T/loss: Train: Loss"}})
	require.Contains(t, got, "Models compared")
	require.Contains(t, got, "#1 model-a (path/a)")
	require.Contains(t, got, "Metric labels")
	require.Contains(t, got, "T/loss: Train: Loss")
}

// TestBuildPageSidebarHTML_SwatchesOnlyInMultiModelMode characterizes that color swatches (a
// small colored <span>) only appear when modelEntries is non-empty -- single-model mode has
// nothing to color-code against, so entries stay plain, matching today's simpler common case.
func TestBuildPageSidebarHTML_SwatchesOnlyInMultiModelMode(t *testing.T) {
	t.Parallel()
	singleModel := buildPageSidebarHTML("", nil, []legendEntry{{text: "T/loss: Train: Loss"}})
	require.NotContains(t, singleModel, "border-radius:50%")

	multiModel := buildPageSidebarHTML("",
		[]legendEntry{{modelIdx: 0, text: "#1 model-a"}},
		[]legendEntry{{modelIdx: 0, text: "T/loss: Train: Loss"}})
	require.Contains(t, multiModel, "border-radius:50%")
	require.Contains(t, multiModel, modelColor(0))
}

func TestModelColor_DistinctAndStable(t *testing.T) {
	t.Parallel()
	require.NotEqual(t, modelColor(0), modelColor(1))
	require.Equal(t, modelColor(0), modelColor(0))
	// Cycles once we run out of palette entries, rather than panicking.
	require.NotPanics(t, func() { modelColor(len(modelColorPalette) + 3) })
}

func TestInjectPageExtras_NoOpWhenNothingToInject(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	original := "<html><head><title>Vizb</title></head><body><div id=\"app\"></div></body></html>"
	require.NoError(t, os.WriteFile(htmlPath, []byte(original), 0644))

	require.NoError(t, injectPageExtras(htmlPath, "", nil, nil))

	content, err := os.ReadFile(htmlPath)
	require.NoError(t, err)
	require.Equal(t, original, string(content))
}

func TestInjectPageExtras_TitleUpdatesTagAndInjectsHeading(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath,
		[]byte(`<html><head><title>Vizb</title></head><body><div id="app"></div></body></html>`), 0644))

	require.NoError(t, injectPageExtras(htmlPath, "FlowMatching results", nil, nil))

	content, err := os.ReadFile(htmlPath)
	require.NoError(t, err)
	require.Contains(t, string(content), "<title>FlowMatching results</title>")
	require.Contains(t, string(content), "#gomlx-sidebar")
	// The sidebar must land before vizb's own <div id="app">, which its JS fully replaces, and
	// the CSS giving #app room for the sidebar must land in <head>.
	require.Regexp(t, `(?s)<style>.*#app\{margin-left:300px\}.*</style>.*<div id="gomlx-sidebar"><h1>FlowMatching results</h1>.*<div id="app">`,
		string(content))
}

func TestInjectPageExtras_ModelsAndDescriptionsSurviveInBody(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath,
		[]byte(`<html><head><title>Vizb</title></head><body><div id="app"></div></body></html>`), 0644))

	require.NoError(t, injectPageExtras(htmlPath, "",
		[]legendEntry{{modelIdx: 0, text: "#1 model-a (dir-a)"}},
		[]legendEntry{{modelIdx: 0, text: "T/loss: Train: Loss"}}))

	content, err := os.ReadFile(htmlPath)
	require.NoError(t, err)
	require.Contains(t, string(content), "#1 model-a (dir-a)")
	require.Contains(t, string(content), "T/loss: Train: Loss")
}

func TestInjectPageExtras_NoHeadTagReturnsError(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath, []byte("<html>no head here<body><div id=\"app\"></div></body></html>"), 0644))

	err := injectPageExtras(htmlPath, "", []legendEntry{{text: "#1 model-a (dir-a)"}}, nil)
	require.Error(t, err)
}

func TestInjectPageExtras_NoBodyTagReturnsError(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath, []byte("<html><head><title>Vizb</title></head>no body here</html>"), 0644))

	err := injectPageExtras(htmlPath, "", []legendEntry{{text: "#1 model-a (dir-a)"}}, nil)
	require.Error(t, err)
}
