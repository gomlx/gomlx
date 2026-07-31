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

// browserOpenedOnce guards against opening a new browser tab on every -loop
// iteration: the same file gets regenerated each time, but the already-open
// tab just needs a refresh, not a new tab. Process-lifetime state, reset
// naturally on every new invocation of the tool.
var browserOpenedOnce bool

// cachedOutputPath holds the randomly-chosen output path (when -plot_output
// isn't set) for the lifetime of this process, so it doesn't need to be
// deterministic across separate invocations -- just stable for the duration
// of one session (in particular, across all -loop iterations of one run).
var cachedOutputPath string

// hasRenderedOnce and lastRenderModTime support skipping regeneration under
// -loop when the underlying metrics haven't changed since the last render --
// each regeneration re-invokes vizb as a subprocess per metric type, plus
// merge/ui, which isn't free to redo on every tick if nothing new landed.
var (
	hasRenderedOnce   bool
	lastRenderModTime time.Time
)

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

// BuildPlots from the models' metrics points, using the `vizb` CLI tool: one
// dataset per metric type, combined into a single self-contained HTML file.
func BuildPlots(checkpointPaths []string, modelNames []string, metricsOrder map[ModelNameAndMetric]int, points [][]plot.Point) {
	if err := checkVizbAvailable(); err != nil {
		klog.Fatalf("%v", err)
	}

	outputFilePath := resolveOutputFilePath(checkpointPaths)
	currentModTime := latestMetricsModTime(checkpointPaths)
	if hasRenderedOnce && !currentModTime.After(lastRenderModTime) {
		// Nothing new since the last render: regenerating would mean
		// re-invoking vizb as a subprocess per metric type, plus merge/ui --
		// real work, not free to redo on every -loop tick for no reason.
		fmt.Printf("\nNo new metrics since last update. Plot at:\t%s\n\n", outputFilePath)
		if *flagBrowser && !browserOpenedOnce {
			openBrowser(outputFilePath)
			browserOpenedOnce = true
		}
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
	hasRenderedOnce = true
	lastRenderModTime = currentModTime

	if *flagLoop > 0 {
		if err := injectAutoRefresh(outputFilePath, *flagLoop); err != nil {
			// Not fatal: the plots are still written and viewable, just without
			// self-refresh -- the user can reload the tab manually instead.
			klog.Warningf("Failed to inject auto-refresh into %s: %v", outputFilePath, err)
		}
	}

	fmt.Printf("\nPlots written to:\t%s\n\n", outputFilePath)
	if *flagBrowser && !browserOpenedOnce {
		openBrowser(outputFilePath)
		browserOpenedOnce = true
	}
}

// resolveOutputFilePath returns the path to write the plots HTML to.
//
// If -plot_output is set explicitly, it's used (resolved relative to the
// first checkpoint path, if not already absolute).
//
// Otherwise, a random path in the OS temp dir is picked *once* and cached
// for the remaining lifetime of this process (see cachedOutputPath) -- it
// doesn't need to be deterministic across separate invocations of the tool,
// just stable across repeated calls within one session (in particular, each
// -loop iteration), so an already-open browser tab can be refreshed to see
// updated data instead of a new tab (and a new file) accumulating every time.
func resolveOutputFilePath(checkpointPaths []string) string {
	outputFilePath := *flagPlotOutput
	if outputFilePath != "" {
		outputFilePath = fsutil.MustReplaceTildeInDir(outputFilePath)
		if !filepath.IsAbs(outputFilePath) {
			outputFilePath = path.Join(checkpointPaths[0], outputFilePath)
		}
		return outputFilePath
	}

	if cachedOutputPath == "" {
		tmpFile, err := os.CreateTemp("", "gomlx-plots-*.html")
		if err != nil {
			panic(errors.Wrap(err, "failed to create temporary file for plots"))
		}
		_ = tmpFile.Close()
		cachedOutputPath = tmpFile.Name()
	}
	return cachedOutputPath
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
