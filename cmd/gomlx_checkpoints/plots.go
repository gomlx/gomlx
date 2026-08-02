// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path"
	"path/filepath"
	"runtime"
	"slices"
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
func NewPlotBuilder(openBrowser bool, outputOverride string, loopPeriod time.Duration) *PlotBuilder {
	return &PlotBuilder{
		openBrowser:    openBrowser,
		outputOverride: outputOverride,
		loopPeriod:     loopPeriod,
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
type plotLineInfo struct {
	short, desc   string
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
				info = &plotLineInfo{}
				if len(modelNames) == 1 {
					info.short = pt.Short
					info.desc = fmt.Sprintf("%s: %s", pt.Short, pt.MetricName)
				} else {
					info.short = fmt.Sprintf("#%d %s", modelNum, pt.Short)
					info.desc = fmt.Sprintf("%s: %s for model %q", info.short, pt.MetricName, modelName)
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

	// One vizb DataSet JSON per metric type (e.g. "loss", "accuracy").
	jsonPaths := make([]string, 0, len(metricTypes))
	for _, metricType := range metricTypes {
		lines := createPlotLines(metricType, modelNames, metricsOrder, points, modelNamesToIndex)
		jsonPath, err := runVizbLine(tmpDir, metricType, lines)
		if err != nil {
			panic(errors.Wrapf(err, "failed to build vizb dataset for metric type %q", metricType))
		}
		jsonPaths = append(jsonPaths, jsonPath)
	}

	if err := mergeAndRenderVizb(tmpDir, jsonPaths, outputFilePath); err != nil {
		panic(errors.Wrap(err, "failed to render plots with vizb"))
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
