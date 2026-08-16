// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/gomlx/gomlx/ui/plot"
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

// TestPlot_SidebarReflectsCurrentMetricsAcrossRepeatedCalls characterizes the sidebar as dynamic
// across repeated Plot calls on the same PlotBuilder (the -loop use case: one PlotBuilder, called
// again each time training produces new metrics) -- it is NOT a live, in-browser element that
// reacts to which chart the viewer currently has selected (investigated separately; not currently
// built, see buildDescriptionEntries' per-metric-type grouping instead). What "dynamic" means
// here specifically: each call to Plot regenerates the sidebar from that call's own metricsOrder
// and points, so a metric type that only starts appearing partway through training (e.g. "loss"
// alone at first, "img_loss" joining later) shows up in the very next render -- the sidebar never
// gets stuck showing only what was true the first time Plot ran.
func TestPlot_SidebarReflectsCurrentMetricsAcrossRepeatedCalls(t *testing.T) {
	if err := checkVizbAvailable(); err != nil {
		t.Skip("vizb not installed, skipping integration test:", err)
	}
	dir := t.TempDir()
	metricsFile := filepath.Join(dir, plot.TrainingPlotFileName)
	older := time.Now().Add(-time.Hour)
	require.NoError(t, os.WriteFile(metricsFile, []byte("{}\n"), 0644))
	require.NoError(t, os.Chtimes(metricsFile, older, older))

	outPath := filepath.Join(dir, "plot.html")
	pb := NewPlotBuilder(false, outPath, 0, "")
	checkpointPaths := []string{dir}
	modelNames := []string{"model"}

	// First render: only "loss" has been logged so far (simulates early training).
	lossOrder := map[ModelNameAndMetric]int{
		{ModelName: "model", MetricName: "T/loss", MetricType: "loss"}: 0,
	}
	lossPoints := [][]plot.Point{{
		{MetricName: "Train: Loss", Short: "T/loss", MetricType: "loss", Step: 0, Value: 0.9},
	}}
	pb.Plot(checkpointPaths, modelNames, lossOrder, lossPoints)

	content, err := os.ReadFile(outPath)
	require.NoError(t, err)
	require.Contains(t, string(content), "Metric labels — loss")
	require.NotContains(t, string(content), "Metric labels — img_loss")

	// The metrics file changes (a new metric type starts appearing) -- bump its mtime so Plot's
	// caching guard (see PlotBuilder's doc comment) doesn't skip regenerating.
	newer := time.Now()
	require.NoError(t, os.Chtimes(metricsFile, newer, newer))

	// Second render: "img_loss" has now joined "loss" (simulates training progressing).
	allOrder := map[ModelNameAndMetric]int{
		{ModelName: "model", MetricName: "T/loss", MetricType: "loss"}:         0,
		{ModelName: "model", MetricName: "T/img_loss", MetricType: "img_loss"}: 1,
	}
	allPoints := [][]plot.Point{{
		{MetricName: "Train: Loss", Short: "T/loss", MetricType: "loss", Step: 0, Value: 0.9},
		{MetricName: "Train: Images Loss", Short: "T/img_loss", MetricType: "img_loss", Step: 0, Value: 0.5},
	}}
	pb.Plot(checkpointPaths, modelNames, allOrder, allPoints)

	content, err = os.ReadFile(outPath)
	require.NoError(t, err)
	require.Contains(t, string(content), "Metric labels — loss")
	require.Contains(t, string(content), "Metric labels — img_loss",
		"sidebar must pick up the newly-appeared metric type on the next render, not stay stuck with the first render's content")
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

// findLineByDesc returns the line whose desc matches, failing the test if there isn't exactly
// one -- disambiguateShortLabels sorts lines in place, so tests can't rely on index position to
// identify a particular line after calling it.
func findLineByDesc(t *testing.T, lines []*plotLineInfo, desc string) *plotLineInfo {
	t.Helper()
	var found *plotLineInfo
	for _, line := range lines {
		if line.desc == desc {
			require.Nil(t, found, "more than one line with desc %q", desc)
			found = line
		}
	}
	require.NotNil(t, found, "no line with desc %q", desc)
	return found
}

// TestDisambiguateShortLabels_Collision reproduces the real bug found in the maintainer's
// FlowMatching fixture: "Mean Loss on train" and "Mean Loss on validation" both produce the
// Short "#loss()" (their eval dataset's ShortName came out empty), so without disambiguation
// they'd collide into a single series in vizb's chart (vizb groups by this label). Every line
// must end up with a distinct short after the call, and coreShort must get the same suffix (used
// later to dedupe the sidebar glossary across models -- see buildDescriptionEntries).
func TestDisambiguateShortLabels_Collision(t *testing.T) {
	t.Parallel()
	lines := []*plotLineInfo{
		{short: "#loss()", coreShort: "#loss()", desc: "Mean Loss on train"},
		{short: "#loss()", coreShort: "#loss()", desc: "Mean Loss on validation"},
		{short: "T/loss", coreShort: "T/loss", desc: "Train: Loss"}, // Unrelated, must be untouched.
	}
	disambiguateShortLabels(lines)

	seen := make(map[string]bool)
	for _, line := range lines {
		require.False(t, seen[line.short], "short %q is not unique after disambiguation", line.short)
		seen[line.short] = true
		require.Equal(t, line.short, line.coreShort, "coreShort must get the same suffix as short")
	}
	require.Equal(t, "T/loss", findLineByDesc(t, lines, "Train: Loss").short, "non-colliding line must be left untouched")
	// Prefers the desc-derived "(train)"/"(validation)" suffix over a bare numeric counter, since a
	// bare "(1)"/"(2)" reads too easily as a per-model marker next to the ❶/❷ model-index markers
	// used elsewhere in the sidebar.
	require.Equal(t, "#loss() (train)", findLineByDesc(t, lines, "Mean Loss on train").short)
	require.Equal(t, "#loss() (validation)", findLineByDesc(t, lines, "Mean Loss on validation").short)
}

// TestDisambiguateShortLabels_FallsBackToNumericWhenDescHasNoDataset covers descs that don't fit
// the "<name> on <dataset>" shape disambiguationSuffix expects -- disambiguation must still make
// every short unique, just via a plain numeric counter instead.
func TestDisambiguateShortLabels_FallsBackToNumericWhenDescHasNoDataset(t *testing.T) {
	t.Parallel()
	lines := []*plotLineInfo{
		{short: "x", coreShort: "x", desc: "Batch Accuracy"},
		{short: "x", coreShort: "x", desc: "Moving Average Accuracy"},
	}
	disambiguateShortLabels(lines)

	seen := make(map[string]bool)
	for _, line := range lines {
		require.False(t, seen[line.short], "short %q is not unique after disambiguation", line.short)
		seen[line.short] = true
		require.Regexp(t, `^x \(\d+\)$`, line.short)
	}
}

// TestDisambiguateShortLabels_ConsistentAcrossModels is the regression test for the ordering bug
// found while implementing cross-model glossary deduplication: two "models" (❶/❷-prefixed short,
// simulating createPlotLines' output) each have their own train/validation collision on the same
// underlying metric. Without a deterministic sort first, Go's randomized map iteration order in
// createPlotLines could assign mismatched suffixes (e.g. model ❶'s train getting one suffix while
// model ❷'s train gets another), which would silently break buildDescriptionEntries' (coreShort,
// desc) matching. The same desc must always get the same suffix, regardless of model.
func TestDisambiguateShortLabels_ConsistentAcrossModels(t *testing.T) {
	t.Parallel()
	// Deliberately in a shuffled order that doesn't match desc's alphabetical order, to exercise
	// the internal sort rather than accidentally pass due to already-sorted input.
	lines := []*plotLineInfo{
		{short: "❷ #loss()", coreShort: "#loss()", desc: "Mean Loss on validation", modelIdx: 1},
		{short: "❶ #loss()", coreShort: "#loss()", desc: "Mean Loss on train", modelIdx: 0},
		{short: "❷ #loss()", coreShort: "#loss()", desc: "Mean Loss on train", modelIdx: 1},
		{short: "❶ #loss()", coreShort: "#loss()", desc: "Mean Loss on validation", modelIdx: 0},
	}
	disambiguateShortLabels(lines)

	findByModelAndDesc := func(modelIdx int, desc string) *plotLineInfo {
		for _, line := range lines {
			if line.modelIdx == modelIdx && line.desc == desc {
				return line
			}
		}
		t.Fatalf("no line with modelIdx=%d desc=%q", modelIdx, desc)
		return nil
	}

	trainModel0 := findByModelAndDesc(0, "Mean Loss on train")
	trainModel1 := findByModelAndDesc(1, "Mean Loss on train")
	require.Equal(t, trainModel0.coreShort, trainModel1.coreShort, "the same metric (train) must get the same suffix regardless of model")
	require.Equal(t, "#loss() (train)", trainModel0.coreShort)

	validationModel0 := findByModelAndDesc(0, "Mean Loss on validation")
	validationModel1 := findByModelAndDesc(1, "Mean Loss on validation")
	require.Equal(t, validationModel0.coreShort, validationModel1.coreShort, "the same metric (validation) must get the same suffix regardless of model")
	require.Equal(t, "#loss() (validation)", validationModel0.coreShort)
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
	got := buildPageSidebarHTML(`<script>alert(1)</script>`,
		[]legendEntry{{term: `<b>evil</b>`, detail: "d"}}, nil)
	require.NotContains(t, got, "<script>")
	require.NotContains(t, got, "<b>evil</b>")
	require.Contains(t, got, "&lt;script&gt;")
}

func TestBuildPageSidebarHTML_ModelsAndDescriptions(t *testing.T) {
	t.Parallel()
	got := buildPageSidebarHTML("",
		[]legendEntry{{term: "❶ model-a", detail: "path/a"}, {term: "❷ model-b", detail: "path/b"}},
		[]legendSection{{heading: "loss", entries: []legendEntry{{term: "T/loss", detail: "Train: Loss"}}}})
	require.Contains(t, got, "Models compared")
	require.Contains(t, got, "❶ model-a")
	require.Contains(t, got, "path/a")
	require.Contains(t, got, "Metric labels — loss")
	require.Contains(t, got, "T/loss")
	require.Contains(t, got, "Train: Loss")
}

// TestBuildPageSidebarHTML_GroupsDescriptionsByMetricType characterizes that each legendSection
// renders under its own "Metric labels — <heading>" heading, rather than one flat list -- so a
// reader looking at one chart isn't scrolling through every other chart's labels to find the ones
// that matter.
func TestBuildPageSidebarHTML_GroupsDescriptionsByMetricType(t *testing.T) {
	t.Parallel()
	got := buildPageSidebarHTML("", nil, []legendSection{
		{heading: "loss", entries: []legendEntry{{term: "T/loss", detail: "Train: Loss"}}},
		{heading: "img_loss", entries: []legendEntry{{term: "T/~img_loss", detail: "Train: Moving Images Loss"}}},
	})
	require.Contains(t, got, "Metric labels — loss")
	require.Contains(t, got, "Metric labels — img_loss")
	// The "loss" section's heading must come before its own entry, which must come before the
	// "img_loss" section's heading -- i.e. genuinely grouped, not just both headings present
	// anywhere in the output.
	lossHeading := strings.Index(got, "Metric labels — loss")
	lossEntry := strings.Index(got, "T/loss<")
	imgLossHeading := strings.Index(got, "Metric labels — img_loss")
	require.True(t, lossHeading < lossEntry && lossEntry < imgLossHeading)
}

// TestBuildPageSidebarHTML_EveryRowGetsASwatch characterizes the current swatch policy for
// "Metric labels": every row gets its own color, cycling by position -- including in single-model
// mode, since the swatch's job is telling metrics apart (mirroring vizb's own chart legend, which
// colors every series distinctly, and now actually uses the same colors -- see Plot), not
// indicating which model a row belongs to.
func TestBuildPageSidebarHTML_EveryRowGetsASwatch(t *testing.T) {
	t.Parallel()
	singleModel := buildPageSidebarHTML("", nil,
		[]legendSection{{heading: "loss", entries: []legendEntry{{term: "T/loss", detail: "Train: Loss"}}}})
	require.Contains(t, singleModel, "gomlx-swatch")

	multiModel := buildPageSidebarHTML("",
		[]legendEntry{{term: "❶ model-a"}, {term: "❷ model-b"}},
		[]legendSection{{heading: "loss", entries: []legendEntry{
			{term: "T/loss", detail: "Train: Loss"},
			{term: "T/~loss", detail: "Train: Moving Average Loss"},
		}}})
	require.Contains(t, multiModel, "gomlx-swatch")
	require.Contains(t, multiModel, swatchColor(0))
	require.Contains(t, multiModel, swatchColor(1))
}

// TestBuildPageSidebarHTML_ModelsHaveNoSwatch characterizes that "Models compared" rows never get
// a color swatch, unlike "Metric labels" rows -- swatchColor's cycling is scoped per list, so
// reusing it for models would make a model's dot coincidentally match an unrelated metric's color
// elsewhere on the page (metric colors now mirror the chart's own line colors, see Plot), implying
// a connection that doesn't exist. The ❶/❷ marker already uniquely identifies each model.
func TestBuildPageSidebarHTML_ModelsHaveNoSwatch(t *testing.T) {
	t.Parallel()
	got := buildPageSidebarHTML("",
		[]legendEntry{{term: "❶ model-a", detail: "path/a"}, {term: "❷ model-b", detail: "path/b"}},
		nil)
	require.NotContains(t, got, "gomlx-swatch")
}

func TestSwatchColor_DistinctAndStable(t *testing.T) {
	t.Parallel()
	require.NotEqual(t, swatchColor(0), swatchColor(1))
	require.Equal(t, swatchColor(0), swatchColor(0))
	// Cycles once we run out of palette entries, rather than panicking.
	require.NotPanics(t, func() { swatchColor(len(swatchPalette) + 3) })
}

// TestModelIndexMarker_CircledDigits characterizes the maintainer-requested "❶"/"❷" markers
// (instead of "#1"/"#2", which was ambiguous next to "#" already meaning "average" in these
// metric names) for the 1-10 range the circled-digit Unicode block covers.
func TestModelIndexMarker_CircledDigits(t *testing.T) {
	t.Parallel()
	require.Equal(t, "❶", modelIndexMarker(1))
	require.Equal(t, "❷", modelIndexMarker(2))
	require.Equal(t, "❿", modelIndexMarker(10))
}

// TestModelIndexMarker_FallsBackPastTen checks the bracket fallback beyond the circled-digit
// block's range, rather than an invalid/out-of-range character.
func TestModelIndexMarker_FallsBackPastTen(t *testing.T) {
	t.Parallel()
	require.Equal(t, "[11]", modelIndexMarker(11))
}

func TestBuildDescriptionEntries_DedupesAcrossModels(t *testing.T) {
	t.Parallel()
	// Same metric ("T/~loss": "Train: Moving Average Loss"), measured on two different models --
	// must collapse to one glossary entry, not one per model, per the maintainer's request.
	lines := []*plotLineInfo{
		{metricType: "loss", modelIdx: 0, coreShort: "T/~loss", desc: "Train: Moving Average Loss"},
		{metricType: "loss", modelIdx: 1, coreShort: "T/~loss", desc: "Train: Moving Average Loss"},
	}
	sections := buildDescriptionEntries(lines)
	require.Len(t, sections, 1)
	require.Equal(t, "loss", sections[0].heading)
	require.Len(t, sections[0].entries, 1)
	require.Equal(t, "T/~loss", sections[0].entries[0].term)
	require.Equal(t, "Train: Moving Average Loss", sections[0].entries[0].detail)
}

// TestBuildDescriptionEntries_GroupsByMetricType checks that lines from different metric types
// (e.g. "loss" vs "img_loss") land in separate sections, in first-encountered order, rather than
// one flat list -- Plot always calls createPlotLines/appends to allLines in
// createSortedMetricTypes' order, so that's also the sections' order.
func TestBuildDescriptionEntries_GroupsByMetricType(t *testing.T) {
	t.Parallel()
	lines := []*plotLineInfo{
		{metricType: "loss", coreShort: "T/loss", desc: "Train: Loss"},
		{metricType: "img_loss", coreShort: "T/~img_loss", desc: "Train: Moving Images Loss"},
		{metricType: "loss", coreShort: "T/~loss", desc: "Train: Moving Average Loss"},
	}
	sections := buildDescriptionEntries(lines)
	require.Len(t, sections, 2)
	require.Equal(t, "loss", sections[0].heading)
	require.Len(t, sections[0].entries, 2)
	require.Equal(t, "img_loss", sections[1].heading)
	require.Len(t, sections[1].entries, 1)
}

func TestBuildDescriptionEntries_DistinctMetricsStayDistinct(t *testing.T) {
	t.Parallel()
	// Different coreShort/desc pairs (even from the same model) must never collapse together.
	lines := []*plotLineInfo{
		{metricType: "loss", modelIdx: 0, coreShort: "#loss() (1)", desc: "Mean Loss on train"},
		{metricType: "loss", modelIdx: 0, coreShort: "#loss() (2)", desc: "Mean Loss on validation"},
	}
	sections := buildDescriptionEntries(lines)
	require.Len(t, sections, 1)
	require.Len(t, sections[0].entries, 2)
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
		[]legendEntry{{term: "❶ model-a", detail: "dir-a"}},
		[]legendSection{{heading: "loss", entries: []legendEntry{{term: "T/loss", detail: "Train: Loss"}}}}))

	content, err := os.ReadFile(htmlPath)
	require.NoError(t, err)
	require.Contains(t, string(content), "❶ model-a")
	require.Contains(t, string(content), "dir-a")
	require.Contains(t, string(content), "T/loss")
	require.Contains(t, string(content), "Train: Loss")
}

func TestInjectPageExtras_NoHeadTagReturnsError(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath, []byte("<html>no head here<body><div id=\"app\"></div></body></html>"), 0644))

	err := injectPageExtras(htmlPath, "", []legendEntry{{term: "❶ model-a", detail: "dir-a"}}, nil)
	require.Error(t, err)
}

func TestInjectPageExtras_NoBodyTagReturnsError(t *testing.T) {
	t.Parallel()
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath, []byte("<html><head><title>Vizb</title></head>no body here</html>"), 0644))

	err := injectPageExtras(htmlPath, "", []legendEntry{{term: "❶ model-a", detail: "dir-a"}}, nil)
	require.Error(t, err)
}
