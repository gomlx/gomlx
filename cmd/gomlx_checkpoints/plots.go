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

// modelIndexMarker returns a compact marker ("❶", "❷", ...) for the 1-based position of a
// compared model -- not "#1"/"#2", since "#" already means something else in these metric names.
// Falls back to "[N]" brackets past 10 (the circled-digit block doesn't continue further).
func modelIndexMarker(oneBasedIdx int) string {
	if oneBasedIdx >= 1 && oneBasedIdx <= 10 {
		return string(rune('❶' + oneBasedIdx - 1))
	}
	return fmt.Sprintf("[%d]", oneBasedIdx)
}

// plotLineInfo contains the information for a single line in a plot.
//
// short is the series label passed to vizb, unique per metric type (see disambiguateShortLabels)
// and model-prefixed in multi-model mode. coreShort is the same label without the model prefix,
// used to collapse duplicate sidebar entries across models (see buildDescriptionEntries). desc is
// the long-form description shown in the sidebar. metricType is which chart this line belongs to.
// modelIdx is which compared model produced it.
type plotLineInfo struct {
	metricType             string
	short, coreShort, desc string
	modelIdx               int
	steps, values          []float64
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
				info = &plotLineInfo{metricType: metricType, modelIdx: modelIdx, coreShort: pt.Short, desc: pt.MetricName}
				if len(modelNames) == 1 {
					info.short = pt.Short
				} else {
					info.short = fmt.Sprintf("%s %s", modelIndexMarker(modelNum), pt.Short)
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

// disambiguationSuffix derives a human-readable disambiguation suffix from a line's long
// description, e.g. " (train)" or " (validation)". Exploits the known shape of eval-metric
// descriptions built by ui/plot.AddTrainAndEvalMetrics: "<name> on <dataset>" (e.g. "Mean Loss on
// train" -- see ui/plot/plot.go). Returns "" if desc doesn't end in " on <something>", signaling
// the caller to fall back to a plain numeric suffix instead.
func disambiguationSuffix(desc string) string {
	const marker = " on "
	idx := strings.LastIndex(desc, marker)
	if idx < 0 {
		return ""
	}
	dataset := strings.TrimSpace(desc[idx+len(marker):])
	if dataset == "" {
		return ""
	}
	return fmt.Sprintf(" (%s)", dataset)
}

// disambiguateShortLabels ensures every line's short (and coreShort) is unique within one metric
// type's line set -- vizb's chart renderer groups series by short, so two different metrics
// sharing an identical Short would otherwise silently collide into one series.
//
// Prefers a suffix derived from desc (see disambiguationSuffix), e.g. "#loss() (train)" vs
// "#loss() (validation)", over a bare numeric counter, which reads too easily as a per-model
// marker next to the ❶/❷ markers used elsewhere. Falls back to a numeric counter when desc doesn't
// fit that shape, or when two colliding lines would otherwise derive the same suffix.
//
// Sorts lines by desc first: createPlotLines builds `lines` from a Go map, whose iteration order
// is randomized per run, so without a deterministic sort two models' independent collisions on the
// same underlying metric could get mismatched suffixes purely by chance -- silently breaking
// buildDescriptionEntries' cross-model dedup, which merges by exact (coreShort, desc) match.
func disambiguateShortLabels(lines []*plotLineInfo) {
	slices.SortFunc(lines, func(a, b *plotLineInfo) int { return strings.Compare(a.desc, b.desc) })

	counts := make(map[string]int, len(lines))
	for _, line := range lines {
		counts[line.short]++
	}
	seen := make(map[string]int, len(lines))                     // base short -> numeric fallback counter
	usedSuffixes := make(map[string]map[string]bool, len(lines)) // base short -> suffixes already assigned
	for _, line := range lines {
		base := line.short
		if counts[base] <= 1 {
			continue // No collision: leave the common case untouched.
		}
		suffix := disambiguationSuffix(line.desc)
		if suffix == "" || usedSuffixes[base][suffix] {
			seen[base]++
			suffix = fmt.Sprintf(" (%d)", seen[base])
		}
		if usedSuffixes[base] == nil {
			usedSuffixes[base] = make(map[string]bool)
		}
		usedSuffixes[base][suffix] = true
		line.short += suffix
		line.coreShort += suffix
	}
}

// buildMetricTypeSection collects one legendEntry per distinct (coreShort, desc) pair among a
// single metric type's lines, deduplicating entries that describe the same metric across multiple
// models, sorted by term then detail. Factored out of buildDescriptionEntries so Plot can compute
// the same ordered list before calling runVizbLine, to assign matching swatch/chart colors (§Plot).
func buildMetricTypeSection(lines []*plotLineInfo) []legendEntry {
	type entry struct {
		term, detail string
	}
	byKey := make(map[string]*entry, len(lines))
	var order []string
	for _, line := range lines {
		key := line.coreShort + "\x00" + line.desc
		if _, exists := byKey[key]; exists {
			continue
		}
		byKey[key] = &entry{term: line.coreShort, detail: line.desc}
		order = append(order, key)
	}
	entries := make([]legendEntry, len(order))
	for i, key := range order {
		e := byKey[key]
		entries[i] = legendEntry{term: e.term, detail: e.detail}
	}
	slices.SortFunc(entries, func(a, b legendEntry) int {
		if c := strings.Compare(a.term, b.term); c != 0 {
			return c
		}
		return strings.Compare(a.detail, b.detail)
	})
	return entries
}

// buildDescriptionEntries groups the "Metric labels" sidebar entries (see buildMetricTypeSection)
// into one legendSection per metric type, in first-encountered order -- so a reader looking at one
// chart finds only that chart's labels under its own heading.
func buildDescriptionEntries(allLines []*plotLineInfo) []legendSection {
	var metricTypeOrder []string
	byMetricType := make(map[string][]*plotLineInfo, len(allLines))
	for _, line := range allLines {
		if _, ok := byMetricType[line.metricType]; !ok {
			metricTypeOrder = append(metricTypeOrder, line.metricType)
		}
		byMetricType[line.metricType] = append(byMetricType[line.metricType], line)
	}
	sections := make([]legendSection, len(metricTypeOrder))
	for i, metricType := range metricTypeOrder {
		sections[i] = legendSection{heading: metricType, entries: buildMetricTypeSection(byMetricType[metricType])}
	}
	return sections
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

	// One vizb DataSet JSON per metric type (e.g. "loss", "accuracy"). Also collect every line
	// across every type, to build the sidebar from once the loop is done (see buildDescriptionEntries).
	jsonPaths := make([]string, 0, len(metricTypes))
	var allLines []*plotLineInfo
	for _, metricType := range metricTypes {
		lines := createPlotLines(metricType, modelNames, metricsOrder, points, modelNamesToIndex)
		disambiguateShortLabels(lines)
		allLines = append(allLines, lines...)

		// Assign each distinct metric a swatch color (matching the order it'll appear in the
		// sidebar, see buildMetricTypeSection), then look each line's color up by coreShort, so
		// lines for the same metric (e.g. one per model) share a color.
		colorByCoreShort := make(map[string]string, len(lines))
		for i, entry := range buildMetricTypeSection(lines) {
			colorByCoreShort[entry.term] = swatchColorHex(i)
		}
		lineColors := make([]string, len(lines))
		for i, line := range lines {
			lineColors[i] = colorByCoreShort[line.coreShort]
		}

		jsonPath, err := runVizbLine(tmpDir, metricType, lines, lineColors)
		if err != nil {
			panic(errors.Wrapf(err, "failed to build vizb dataset for metric type %q", metricType))
		}
		jsonPaths = append(jsonPaths, jsonPath)
	}
	descriptionEntries := buildDescriptionEntries(allLines)

	if err := mergeAndRenderVizb(tmpDir, jsonPaths, outputFilePath); err != nil {
		panic(errors.Wrap(err, "failed to render plots with vizb"))
	}

	var modelEntries []legendEntry
	if len(modelNames) > 1 {
		for idx, name := range modelNames {
			modelEntries = append(modelEntries, legendEntry{
				term:   fmt.Sprintf("%s %s", modelIndexMarker(idx+1), name),
				detail: checkpointPaths[idx],
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

// pageSidebarCSS styles the sidebar purely through the id="gomlx-sidebar" selector plus a margin
// on #app -- never anything about vizb's internal DOM, only that id="app" exists. A fixed,
// independently-scrolling column so a long metric-labels list never pushes the charts below the
// fold. Colors are the dataviz skill's validated colorblind-safe 8-hue palette, as custom
// properties so light/dark mode switches automatically.
const pageSidebarCSS = `
#gomlx-sidebar{
--gomlx-surface:#fcfcfb;--gomlx-text:#0b0b0b;--gomlx-text-secondary:#52514e;--gomlx-text-muted:#898781;
--gomlx-border:rgba(11,11,11,0.10);
--gomlx-series-1:#2a78d6;--gomlx-series-2:#eb6834;--gomlx-series-3:#1baf7a;--gomlx-series-4:#eda100;
--gomlx-series-5:#e87ba4;--gomlx-series-6:#008300;--gomlx-series-7:#4a3aa7;--gomlx-series-8:#e34948;
position:fixed;top:0;left:0;width:` + pageSidebarWidth + `;height:100vh;overflow-y:auto;box-sizing:border-box;
padding:1.5em 1.25em;background:var(--gomlx-surface);color:var(--gomlx-text);
font-family:system-ui,-apple-system,"Segoe UI",sans-serif;font-size:13px;line-height:1.45;
border-right:1px solid var(--gomlx-border);z-index:1000}
#gomlx-sidebar h1{font-size:1.05em;font-weight:600;margin:0 0 1em;letter-spacing:-0.01em}
#gomlx-sidebar h2{font-size:0.72em;font-weight:600;letter-spacing:0.06em;text-transform:uppercase;
color:var(--gomlx-text-muted);margin:1.5em 0 0.7em;padding-top:1.1em;border-top:1px solid var(--gomlx-border)}
#gomlx-sidebar h2:first-of-type{margin-top:0;padding-top:0;border-top:none}
#gomlx-sidebar dl{margin:0}
#gomlx-sidebar .gomlx-row{margin-bottom:0.9em}
#gomlx-sidebar .gomlx-row:last-child{margin-bottom:0}
#gomlx-sidebar dt{display:flex;align-items:center;gap:0.5em;font-weight:500;word-break:break-word}
#gomlx-sidebar dt code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em}
#gomlx-sidebar dd{margin:0.2em 0 0 1.15em;color:var(--gomlx-text-secondary);font-size:0.93em;word-break:break-word}
#gomlx-sidebar dd code{font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;font-size:0.92em}
#gomlx-sidebar .gomlx-swatch{flex:0 0 auto;width:0.6em;height:0.6em;border-radius:50%}
#app{margin-left:` + pageSidebarWidth + `}
@media (max-width:900px){
#gomlx-sidebar{position:static;width:auto;height:auto;border-right:none;border-bottom:1px solid var(--gomlx-border)}
#app{margin-left:0}
}
@media (prefers-color-scheme:dark){
#gomlx-sidebar{
--gomlx-surface:#1a1a19;--gomlx-text:#ffffff;--gomlx-text-secondary:#c3c2b7;--gomlx-text-muted:#898781;
--gomlx-border:rgba(255,255,255,0.10);
--gomlx-series-1:#3987e5;--gomlx-series-2:#d95926;--gomlx-series-3:#199e70;--gomlx-series-4:#c98500;
--gomlx-series-5:#d55181;--gomlx-series-6:#008300;--gomlx-series-7:#9085e9;--gomlx-series-8:#e66767}
}
`

// legendEntry is one row in a sidebar list: term is the short, scannable label (a model name, or
// a metric's short code); detail is the longer text underneath it (a checkpoint path, or a
// metric's full description).
type legendEntry struct {
	term, detail string
}

// legendSection is one heading plus its rows in the sidebar -- e.g. one metric type's ("loss",
// "img_loss", ...) glossary group within "Metric labels" (see buildDescriptionEntries), so a
// reader looking at one chart finds only that chart's labels under their own heading.
type legendSection struct {
	heading string
	entries []legendEntry
}

// swatchPalette is the fixed, colorblind-safe 8-hue categorical order (validated via the dataviz
// skill) used to color every "Metric labels" row -- each row gets its own color, matching vizb's
// own chart line for that metric (see swatchColorHex and Plot), not a color tied to model identity.
var swatchPalette = []string{
	"blue", "orange", "aqua", "yellow", "magenta", "green", "violet", "red",
}

// swatchPaletteHex holds swatchPalette's literal light-step hex values, in the same order -- for
// contexts needing a raw hex string instead of a CSS custom property (e.g. vizb's --theme flag).
// Must stay in sync with the light-mode "--gomlx-series-N" values in pageSidebarCSS.
var swatchPaletteHex = []string{
	"#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
}

// swatchColor returns the CSS custom property (see pageSidebarCSS) holding rowIdx's (0-based)
// color slot, cycling through swatchPalette if there are more rows than slots.
func swatchColor(rowIdx int) string {
	return fmt.Sprintf("var(--gomlx-series-%d)", rowIdx%len(swatchPalette)+1)
}

// swatchColorHex returns the literal hex color (see swatchPaletteHex) for rowIdx's (0-based) slot,
// cycling if there are more rows than slots -- the hex-string counterpart of swatchColor, for
// contexts (like vizb's --theme flag) that need a raw color rather than a CSS custom property.
func swatchColorHex(rowIdx int) string {
	return swatchPaletteHex[rowIdx%len(swatchPaletteHex)]
}

// buildPageSidebarHTML renders the optional title heading, the models-compared list, and one
// "Metric labels — <type>" section per metric type (see buildDescriptionEntries), as one
// <div id="gomlx-sidebar"> fragment. Returns "" if there's nothing to show.
//
// Each list is a <dl>: term (model name, or a metric's short code, code-formatted) paired with
// detail underneath (checkpoint path, or the metric's long description). "Metric labels" rows get
// a color swatch matching vizb's chart line for that metric (see Plot); "Models compared" rows
// deliberately don't, since the ❶/❷ marker already identifies each model and reusing the palette
// there would coincidentally match an unrelated metric's color.
func buildPageSidebarHTML(title string, modelEntries []legendEntry, descriptionSections []legendSection) string {
	hasDescriptions := false
	for _, section := range descriptionSections {
		if len(section.entries) > 0 {
			hasDescriptions = true
			break
		}
	}
	if title == "" && len(modelEntries) == 0 && !hasDescriptions {
		return ""
	}
	var b strings.Builder
	b.WriteString(`<div id="gomlx-sidebar">`)
	if title != "" {
		fmt.Fprintf(&b, `<h1>%s</h1>`, html.EscapeString(title))
	}
	wrap := func(s string, asCode bool) string {
		escaped := html.EscapeString(s)
		if asCode {
			return "<code>" + escaped + "</code>"
		}
		return escaped
	}
	writeList := func(heading string, entries []legendEntry, codeTerm, codeDetail, showSwatch bool) {
		if len(entries) == 0 {
			return
		}
		fmt.Fprintf(&b, `<h2>%s</h2><dl>`, html.EscapeString(heading))
		for i, entry := range entries {
			b.WriteString(`<div class="gomlx-row"><dt>`)
			if showSwatch {
				fmt.Fprintf(&b, `<span class="gomlx-swatch" style="background:%s"></span>`, swatchColor(i))
			}
			b.WriteString(wrap(entry.term, codeTerm))
			b.WriteString(`</dt>`)
			if entry.detail != "" {
				b.WriteString(`<dd>`)
				b.WriteString(wrap(entry.detail, codeDetail))
				b.WriteString(`</dd>`)
			}
			b.WriteString(`</div>`)
		}
		b.WriteString(`</dl>`)
	}
	writeList("Models compared", modelEntries, false, true, false) // no swatch, see doc comment above.
	for _, section := range descriptionSections {
		writeList("Metric labels — "+section.heading, section.entries, true, false, true)
	}
	b.WriteString(`</div>`)
	return b.String()
}

// injectPageExtras adds an optional page <title> and, if there's anything to show, a sidebar (see
// buildPageSidebarHTML/pageSidebarCSS) right after <body>, before vizb's own <div id="app">, which
// its client-side JS fully owns and replaces on load -- so the sidebar survives untouched once the
// page hydrates.
//
// Note: a hover-tooltip on vizb's own chart legend was investigated and deliberately not built --
// vizb's legend renders onto a <canvas>, not real DOM, so a title-attribute tooltip can never fire
// on it. The sidebar's "Metric labels" list is the answer to "see the full name" instead.
func injectPageExtras(htmlPath, title string, modelEntries []legendEntry, descriptionSections []legendSection) error {
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

	sidebar := buildPageSidebarHTML(title, modelEntries, descriptionSections)
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
