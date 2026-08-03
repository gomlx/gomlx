// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"flag"
	"fmt"
	"html"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"slices"
	"strings"
	"time"

	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/support/sets"
	"github.com/gomlx/gomlx/ui/plot"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
)

var (
	flagPlot = flag.Bool("plot", false,
		fmt.Sprintf("Plots the metrics collected for plotting in file %q, using the vizb CLI tool "+
			"(https://vizb.goptics.org/ -- must be installed separately and present on PATH). "+
			"You can control which metrics to plot with -metrics_names and -metrics_types", plot.TrainingPlotFileName))
	flagBrowser    = flag.Bool("browser", true, "Opens the generated plots file in the default browser (once, even under -loop).")
	flagPlotOutput = flag.String("plot_output", "", "File to generate HTML file with plots. "+
		"**It is relative to the first dataset directory given**. "+
		"If empty (the default), a random path in the OS temp dir is picked once and reused for the "+
		"rest of the session -- so repeated runs (e.g. under -loop) overwrite the same file instead "+
		"of accumulating a new one each time.")
	flagPlotTitle = flag.String("plot_title", "", "Optional title for the generated plots page: "+
		"shown in the browser tab and as a heading atop the page. Handy when sharing the plot file "+
		"with others.")
)

// PlotBuilder renders training metrics to a self-contained HTML file using the `vizb` CLI tool,
// across possibly many calls to Plot within one process (e.g. once per -loop iteration).
//
// It exists (rather than a handful of package-level variables) so that: (1) every test can
// construct its own PlotBuilder and run in parallel, with no risk of one test's state leaking
// into another's; (2) the session state it carries -- has a plot ever been rendered yet, has the
// browser tab already been opened, what output path and last-render time are we using -- is
// visible in one place instead of scattered across the package; and (3) nothing stops two
// PlotBuilders (say, one per session in some future server-like use) from existing side by side.
//
// Construct one with NewPlotBuilder and call Plot as many times as needed.
type PlotBuilder struct {
	// Configuration, fixed for the lifetime of this PlotBuilder.
	openBrowser    bool
	outputOverride string        // corresponds to -plot_output; if empty, a random path is picked once (see cachedOutputPath) and reused.
	loopPeriod     time.Duration // corresponds to -loop; > 0 enables the auto-refresh <meta> tag.
	plotTitle      string        // corresponds to -plot_title; if empty, no title is added.

	// Session state, mutated across repeated Plot calls.
	browserOpenedOnce bool
	cachedOutputPath  string
	hasRenderedOnce   bool
	lastRenderModTime time.Time
}

// NewPlotBuilder creates a PlotBuilder configured to render plots across however many times
// Plot is called on it -- e.g. once per -loop iteration, in which case the resulting PlotBuilder
// must be created once and reused for every iteration, not recreated each time, since that's what
// makes the output path stable and the browser tab open only once.
func NewPlotBuilder(openBrowser bool, outputOverride string, loopPeriod time.Duration, plotTitle string) *PlotBuilder {
	return &PlotBuilder{
		openBrowser:    openBrowser,
		outputOverride: outputOverride,
		loopPeriod:     loopPeriod,
		plotTitle:      plotTitle,
	}
}

// createSortedMetricTypes collects all metric types and sort them.
func createSortedMetricTypes(metricsOrder map[ModelNameAndMetric]int) []string {
	metricTypesSet := sets.Make[string]()
	for info := range metricsOrder {
		metricTypesSet.Insert(info.MetricType)
	}
	numPlots := len(metricTypesSet)
	metricTypes := make([]string, 0, numPlots)
	for metricType := range metricTypesSet {
		metricTypes = append(metricTypes, metricType)
	}
	slices.Sort(metricTypes)
	return metricTypes
}

// plotLineInfo contains the information for a single line in a plot.
//
// short becomes the line's series label passed to vizb (`--select step,colN{short}`) -- vizb
// groups series by this label, so it must be unique among the lines of one metric type (see
// disambiguateShortLabels). desc is the long-form description shown in the legend box (e.g.
// "Train: Moving Average Loss" or "Mean Loss on train for model \"model-a\"") -- unlike short, it
// doesn't need to be unique. modelIdx (0-based index into the Plot call's modelNames) is used to
// color-code this line's legend entry so lines belonging to the same model are easy to spot in
// the sidebar at a glance -- see modelColor.
type plotLineInfo struct {
	short, desc   string
	modelIdx      int
	steps, values []float64
}

// createPlotLines for the given metric type.
//
// It returns one plotLineInfo per model x metric of the given metric type.
//
// The returned values and steps are sorted by steps.
func createPlotLines(metricType string, modelNames []string, metricsOrder map[ModelNameAndMetric]int, points [][]plot.Point, modelNamesToIndex map[string]int) []*plotLineInfo {
	var lines []*plotLineInfo
	for modelIdx, modelPoints := range points {
		modelName := modelNames[modelIdx]
		modelNum := modelNamesToIndex[modelName]

		// Group points by metric name.
		metricPoints := make(map[string]*plotLineInfo)
		for _, pt := range modelPoints {
			if pt.MetricType != metricType {
				continue
			}
			metricKey := ModelNameAndMetric{
				ModelName:  modelName,
				MetricName: pt.Short,
				MetricType: pt.MetricType,
			}
			if _, ok := metricsOrder[metricKey]; !ok {
				// Metric was not selected, skip.
				continue
			}
			info, exists := metricPoints[pt.MetricName]
			if !exists {
				info = &plotLineInfo{modelIdx: modelIdx}
				if len(modelNames) == 1 {
					info.short = pt.Short
					info.desc = pt.MetricName
				} else {
					info.short = fmt.Sprintf("#%d %s", modelNum, pt.Short)
					info.desc = fmt.Sprintf("%s for model %q", pt.MetricName, modelName)
				}
			}
			info.steps = append(info.steps, pt.Step)
			info.values = append(info.values, pt.Value)
			metricPoints[pt.MetricName] = info
		}

		// Sort points by steps.
		for _, info := range metricPoints {
			// Create the indices array.
			indices := xslices.Iota(0, len(info.steps))
			// Sort indices.
			slices.SortFunc(indices, func(i, j int) int {
				if info.steps[i] < info.steps[j] {
					return -1
				}
				if info.steps[i] > info.steps[j] {
					return 1
				}
				return 0
			})
			// Apply sorted order.
			steps := make([]float64, len(info.steps))
			values := make([]float64, len(info.values))
			for ii, idx := range indices {
				steps[ii] = info.steps[idx]
				values[ii] = info.values[idx]
			}
			info.steps = steps
			info.values = values
		}

		// Collect all lines.
		for _, info := range metricPoints {
			lines = append(lines, info)
		}
	}
	return lines
}

// disambiguateShortLabels ensures every line's short is unique within one metric type's line
// set -- vizb's chart renderer groups series by this label (the "type" field in its DataSet
// JSON), so two different metrics sharing an identical Short (e.g. when a training script's eval
// dataset doesn't provide a distinguishing ShortName -- confirmed to happen in practice) would
// otherwise silently collide into what the user perceives as one series instead of two, which is
// exactly what makes it look like same-metric-type lines (e.g. train vs validation loss) aren't
// being plotted together correctly.
func disambiguateShortLabels(lines []*plotLineInfo) {
	counts := make(map[string]int, len(lines))
	for _, line := range lines {
		counts[line.short]++
	}
	seen := make(map[string]int, len(lines))
	for _, line := range lines {
		if counts[line.short] <= 1 {
			continue // No collision: leave the common case untouched.
		}
		seen[line.short]++
		line.short = fmt.Sprintf("%s (%d)", line.short, seen[line.short])
	}
}

// Plot the models' metrics points, using the `vizb` CLI tool: one dataset per metric type,
// combined into a single self-contained HTML file. Safe to call repeatedly on the same
// PlotBuilder (e.g. once per -loop iteration) -- see PlotBuilder's doc comment.
func (pb *PlotBuilder) Plot(checkpointPaths []string, modelNames []string, metricsOrder map[ModelNameAndMetric]int, points [][]plot.Point) {
	if err := checkVizbAvailable(); err != nil {
		klog.Fatalf("%v", err)
	}

	outputFilePath := pb.resolveOutputFilePath(checkpointPaths)
	currentModTime := latestMetricsModTime(checkpointPaths)
	if pb.hasRenderedOnce && !currentModTime.After(pb.lastRenderModTime) {
		// Nothing new since the last render: regenerating would mean
		// re-invoking vizb as a subprocess per metric type, plus merge/ui --
		// real work, not free to redo on every -loop tick for no reason.
		fmt.Printf("\nNo new metrics since last update. Plot at:\t%s\n\n", outputFilePath)
		pb.openBrowserOnce(outputFilePath)
		return
	}

	metricTypes := createSortedMetricTypes(metricsOrder)
	modelNamesToIndex := make(map[string]int)
	for idx, name := range modelNames {
		modelNamesToIndex[name] = idx + 1
	}

	tmpDir, err := os.MkdirTemp("", "gomlx-vizb-*")
	if err != nil {
		panic(errors.Wrap(err, "failed to create temporary directory for vizb"))
	}
	defer func() { _ = os.RemoveAll(tmpDir) }()

	// One vizb DataSet JSON per metric type (e.g. "loss", "accuracy"). Also collect a short ->
	// long description entry per line, for the legend box (see buildPageSidebarHTML).
	jsonPaths := make([]string, 0, len(metricTypes))
	var descriptionEntries []legendEntry
	for _, metricType := range metricTypes {
		lines := createPlotLines(metricType, modelNames, metricsOrder, points, modelNamesToIndex)
		disambiguateShortLabels(lines)
		for _, line := range lines {
			descriptionEntries = append(descriptionEntries, legendEntry{
				modelIdx: line.modelIdx,
				text:     fmt.Sprintf("%s: %s", line.short, line.desc),
			})
		}
		jsonPath, err := runVizbLine(tmpDir, metricType, lines)
		if err != nil {
			panic(errors.Wrapf(err, "failed to build vizb dataset for metric type %q", metricType))
		}
		jsonPaths = append(jsonPaths, jsonPath)
	}
	slices.SortFunc(descriptionEntries, func(a, b legendEntry) int { return strings.Compare(a.text, b.text) })

	if err := mergeAndRenderVizb(tmpDir, jsonPaths, outputFilePath); err != nil {
		panic(errors.Wrap(err, "failed to render plots with vizb"))
	}

	var modelEntries []legendEntry
	if len(modelNames) > 1 {
		for idx, name := range modelNames {
			modelEntries = append(modelEntries, legendEntry{
				modelIdx: idx,
				text:     fmt.Sprintf("#%d %s (%s)", idx+1, name, checkpointPaths[idx]),
			})
		}
	}
	if err := injectPageExtras(outputFilePath, pb.plotTitle, modelEntries, descriptionEntries); err != nil {
		// Not fatal: the plots are still written and viewable, just without the title/legend
		// boxes -- the underlying chart data is unaffected.
		klog.Warningf("Failed to inject title/legend into %s: %v", outputFilePath, err)
	}
	pb.hasRenderedOnce = true
	pb.lastRenderModTime = currentModTime

	if pb.loopPeriod > 0 {
		if err := injectAutoRefresh(outputFilePath, pb.loopPeriod); err != nil {
			// Not fatal: the plots are still written and viewable, just without
			// self-refresh -- the user can reload the tab manually instead.
			klog.Warningf("Failed to inject auto-refresh into %s: %v", outputFilePath, err)
		}
	}

	fmt.Printf("\nPlots written to:\t%s\n\n", outputFilePath)
	pb.openBrowserOnce(outputFilePath)
}

// openBrowserOnce opens outputFilePath in the browser, but only the first time it's called on
// this PlotBuilder and only if openBrowser was requested at construction -- an already-open tab
// just needs the file underneath it to change, not a second tab.
func (pb *PlotBuilder) openBrowserOnce(outputFilePath string) {
	if pb.openBrowser && !pb.browserOpenedOnce {
		openBrowser(outputFilePath)
		pb.browserOpenedOnce = true
	}
}

// resolveOutputFilePath returns the path to write the plots HTML to.
//
// If -plot_output is set explicitly, it's used (resolved relative to the
// first checkpoint path, if not already absolute).
//
// Otherwise, a random path in the OS temp dir is picked *once* and cached
// for the remaining lifetime of this PlotBuilder (see cachedOutputPath) --
// it doesn't need to be deterministic across separate invocations of the
// tool, just stable across repeated calls within one session (in particular,
// each -loop iteration), so an already-open browser tab can be refreshed to
// see updated data instead of a new tab (and a new file) accumulating every
// time.
func (pb *PlotBuilder) resolveOutputFilePath(checkpointPaths []string) string {
	outputFilePath := pb.outputOverride
	if outputFilePath != "" {
		outputFilePath = fsutil.MustReplaceTildeInDir(outputFilePath)
		if !filepath.IsAbs(outputFilePath) {
			outputFilePath = path.Join(checkpointPaths[0], outputFilePath)
		}
		return outputFilePath
	}

	if pb.cachedOutputPath == "" {
		tmpFile, err := os.CreateTemp("", "gomlx-plots-*.html")
		if err != nil {
			panic(errors.Wrap(err, "failed to create temporary file for plots"))
		}
		_ = tmpFile.Close()
		pb.cachedOutputPath = tmpFile.Name()
	}
	return pb.cachedOutputPath
}

// latestMetricsModTime returns the most recent modification time across all
// checkpoints' training_plot_points.json files, or the zero Time if none of
// them exist yet.
func latestMetricsModTime(checkpointPaths []string) time.Time {
	var latest time.Time
	for _, dir := range checkpointPaths {
		filePath := path.Join(fsutil.MustReplaceTildeInDir(dir), plot.TrainingPlotFileName)
		info, err := os.Stat(filePath)
		if err != nil {
			continue
		}
		if info.ModTime().After(latest) {
			latest = info.ModTime()
		}
	}
	return latest
}

// injectAutoRefresh adds a <meta http-equiv="refresh"> tag to the generated
// HTML so an already-open browser tab reloads itself automatically after
// each -loop iteration, instead of requiring a manual refresh.
func injectAutoRefresh(htmlPath string, period time.Duration) error {
	content, err := os.ReadFile(htmlPath)
	if err != nil {
		return errors.Wrapf(err, "failed to read %q", htmlPath)
	}
	seconds := max(1, int(period.Seconds()))
	metaTag := fmt.Sprintf(`<meta http-equiv="refresh" content="%d">`, seconds)
	updated := bytes.Replace(content, []byte("<head>"), []byte("<head>"+metaTag), 1)
	if bytes.Equal(updated, content) {
		return errors.Errorf("could not find a <head> tag to inject auto-refresh into")
	}
	return os.WriteFile(htmlPath, updated, 0644)
}

// pageSidebarWidth is how much horizontal space the injected sidebar (see buildPageSidebarHTML)
// reserves for itself; vizb's own content (#app) gets an equal left margin so the sidebar doesn't
// overlap it.
const pageSidebarWidth = "300px"

// pageSidebarCSS styles the sidebar purely through the id="gomlx-sidebar" selector (chosen to
// not collide with anything vizb itself generates) plus a margin on #app -- vizb owns everything
// inside #app itself (see injectPageExtras), so this never needs to know anything about its
// internal structure, only that the id="app" element exists, which is stable across vizb
// versions. It's a fixed, independently-scrolling column so a long metric-labels list never
// pushes the actual charts below the fold -- collapses to a normal top block on narrow viewports.
const pageSidebarCSS = `
#gomlx-sidebar{position:fixed;top:0;left:0;width:` + pageSidebarWidth + `;height:100vh;overflow-y:auto;
box-sizing:border-box;padding:1em;font-family:sans-serif;font-size:0.85em;
border-right:1px solid #ccc;background:#fafafa;z-index:1000}
#gomlx-sidebar h1{font-size:1.1em;margin:0 0 0.6em 0}
#gomlx-sidebar h2{font-size:0.95em;margin:1em 0 0.3em 0}
#gomlx-sidebar ul{margin:0;padding-left:1.1em}
#gomlx-sidebar li{margin-bottom:0.2em;word-break:break-word}
#app{margin-left:` + pageSidebarWidth + `}
@media (max-width:900px){
#gomlx-sidebar{position:static;width:auto;height:auto;border-right:none;border-bottom:1px solid #ccc}
#app{margin-left:0}
}
@media (prefers-color-scheme:dark){
#gomlx-sidebar{background:#1e1e1e;border-color:#444;color:#eee}
}
`

// legendEntry is one row in a sidebar list. modelIdx (0-based index into the Plot call's
// modelNames) selects which color swatch to show next to it -- see modelColor -- so entries
// belonging to the same model are easy to spot at a glance; it's meaningless (ignored) in
// single-model mode, where there's nothing to color-code against.
type legendEntry struct {
	modelIdx int
	text     string
}

// modelColorPalette assigns each compared model its own consistent color, shown as a swatch next
// to every one of its entries in the sidebar (both "Models compared" and "Metric labels") -- the
// Okabe-Ito colorblind-safe categorical palette. This intentionally does NOT try to match vizb's
// own chart line colors: vizb's charts have their own live, in-browser theme switcher, so any
// color we set at generation time could be changed by the viewer at any moment -- the sidebar
// swatches are a self-contained, always-correct way to group entries by model, independent of
// whatever chart colors/theme the viewer currently has selected.
var modelColorPalette = []string{
	"#0072B2", // blue
	"#E69F00", // orange
	"#009E73", // green
	"#D55E00", // vermillion
	"#CC79A7", // purple
	"#56B4E9", // sky blue
}

// modelColor returns a stable, distinct color for modelIdx (0-based), cycling through
// modelColorPalette if there are more models than palette entries.
func modelColor(modelIdx int) string {
	return modelColorPalette[modelIdx%len(modelColorPalette)]
}

// buildPageSidebarHTML renders the optional visible title heading and, when there's anything to
// show, the models-compared list and the metric short-label -> long-description list, as one
// <div id="gomlx-sidebar"> fragment (styled by pageSidebarCSS). Returns "" if there's nothing to
// show at all -- the caller should skip injecting both this and pageSidebarCSS in that case.
//
// Each entry gets a color swatch matching its model (see modelColor), but only when modelEntries
// is non-empty -- i.e. only in multi-model mode, where there's actually something to distinguish;
// single-model mode shows plain entries, matching today's simpler common case.
func buildPageSidebarHTML(title string, modelEntries, descriptionEntries []legendEntry) string {
	if title == "" && len(modelEntries) == 0 && len(descriptionEntries) == 0 {
		return ""
	}
	showSwatches := len(modelEntries) > 0
	var b strings.Builder
	b.WriteString(`<div id="gomlx-sidebar">`)
	if title != "" {
		fmt.Fprintf(&b, `<h1>%s</h1>`, html.EscapeString(title))
	}
	writeList := func(heading string, entries []legendEntry) {
		if len(entries) == 0 {
			return
		}
		fmt.Fprintf(&b, `<h2>%s</h2><ul>`, html.EscapeString(heading))
		for _, entry := range entries {
			if showSwatches {
				fmt.Fprintf(&b, `<li><span style="display:inline-block;width:0.8em;height:0.8em;`+
					`border-radius:50%%;background:%s;margin-right:0.4em;vertical-align:middle"></span>%s</li>`,
					modelColor(entry.modelIdx), html.EscapeString(entry.text))
			} else {
				fmt.Fprintf(&b, `<li>%s</li>`, html.EscapeString(entry.text))
			}
		}
		b.WriteString(`</ul>`)
	}
	writeList("Models compared", modelEntries)
	writeList("Metric labels", descriptionEntries)
	b.WriteString(`</div>`)
	return b.String()
}

// injectPageExtras adds an optional page <title> and, if there's anything to show, a sidebar
// (see buildPageSidebarHTML/pageSidebarCSS) alongside vizb's own charts -- the sidebar is
// inserted right after <body>, before vizb's own <div id="app">, which its client-side JS fully
// owns and replaces on load (confirmed by inspecting the page's skeleton markup), so it survives
// untouched once the page hydrates; the CSS goes in <head>, styling #app purely by id so it never
// needs to know anything about vizb's internal DOM structure.
func injectPageExtras(htmlPath, title string, modelEntries, descriptionEntries []legendEntry) error {
	content, err := os.ReadFile(htmlPath)
	if err != nil {
		return errors.Wrapf(err, "failed to read %q", htmlPath)
	}

	if title != "" {
		// Best-effort: if vizb's default <title> text ever changes, just skip updating the
		// browser tab's title -- the visible <h1> heading in the sidebar is what matters most,
		// and doesn't depend on this succeeding.
		content = bytes.Replace(content, []byte("<title>Vizb</title>"), []byte("<title>"+html.EscapeString(title)+"</title>"), 1)
	}

	sidebar := buildPageSidebarHTML(title, modelEntries, descriptionEntries)
	if sidebar == "" {
		return os.WriteFile(htmlPath, content, 0644)
	}

	withCSS := bytes.Replace(content, []byte("<head>"), []byte("<head><style>"+pageSidebarCSS+"</style>"), 1)
	if bytes.Equal(withCSS, content) {
		return errors.Errorf("could not find a <head> tag to inject the sidebar's CSS into")
	}
	updated := bytes.Replace(withCSS, []byte("<body>"), []byte("<body>"+sidebar), 1)
	if bytes.Equal(updated, withCSS) {
		return errors.Errorf("could not find a <body> tag to inject the sidebar into")
	}
	return os.WriteFile(htmlPath, updated, 0644)
}

// openBrowser opens the given file in the default browser.
func openBrowser(fileName string) {
	var err error
	switch runtime.GOOS {
	case "linux":
		err = exec.Command("xdg-open", fileName).Start()
	case "windows":
		err = exec.Command("cmd", "/c", "start", fileName).Start()
	case "darwin":
		err = exec.Command("open", fileName).Start()
	default:
		err = fmt.Errorf("unsupported platform")
	}
	if err != nil {
		fmt.Printf("Error opening browser: %v\n", err)
	}
}
