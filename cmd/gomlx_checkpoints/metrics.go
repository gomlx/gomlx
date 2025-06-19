package main

import (
	"flag"
	"fmt"
	"path"
	"regexp"
	"strings"

	"github.com/charmbracelet/lipgloss"
	"github.com/dustin/go-humanize"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gomlx/ui/plots"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
)

var (
	flagMetricsNames = flag.String("metrics_names", "", "Regular expression that if matches the name or short name, the metric is included.")
	flagMetricsTypes = flag.String("metrics_types", "", "Comma-separate list of metric types to include in metrics Reports. ")
)

func metrics(checkpointPaths, names []string) {
	numCheckpoints := len(names)
	checkpointPath := checkpointPaths[0]
	_ = numCheckpoints

	// Load metrics points from the checkpointPath.
	trainingMetricsPath := path.Join(checkpointPath, plots.TrainingPlotFileName)
	points := must.M1(plots.LoadPoints(trainingMetricsPath))
	if len(points) == 0 {
		klog.Errorf("No metrics found in %q", trainingMetricsPath)
	}

	var metricsNamesMatcher *regexp.Regexp
	if *flagMetricsNames != "" {
		var err error
		metricsNamesMatcher, err = regexp.Compile(*flagMetricsNames)
		if err != nil {
			klog.Fatalf("Failed to compile -metrics_names=%q matcher: %v", *flagMetricsNames, err)
		}
	}

	var metricsTypes types.Set[string]
	if *flagMetricsTypes != "" {
		metricsTypes = types.MakeSet[string]()
		for _, name := range strings.Split(*flagMetricsTypes, ",") {
			metricsTypes.Insert(name)
		}
	}
	metricsUsed := types.MakeSet[string]()
	nameToShort := make(map[string]string)
	shortToName := make(map[string]string)
	for _, point := range points {
		nameToShort[point.MetricName] = point.Short
		shortToName[point.Short] = point.MetricName
		if metricsNamesMatcher != nil || metricsTypes != nil {
			foundName := metricsNamesMatcher.MatchString(point.MetricName) || metricsNamesMatcher.MatchString(point.Short)
			foundType := metricsTypes != nil && metricsTypes.Has(point.MetricType)
			if !foundName && !foundType {
				continue
			}
		}
		metricsUsed.Insert(point.Short)
	}

	// map metric name to position in row, starting from 1 (position 0 is for the global set)
	metricsOrder := make(map[string]int)
	nextPos := 1
	if *flagMetricsNames != "" {
		// Start with the order given by the user:
		for _, name := range strings.Split(*flagMetricsNames, ",") {
			if short, found := nameToShort[name]; found {
				name = short
			}
			if metricsUsed.Has(name) {
				metricsOrder[name] = nextPos
				nextPos++
			}
		}
	}
	// Append all other metric names, in alphabetical order.
	for _, short := range xslices.SortedKeys(metricsUsed) {
		if _, found := metricsOrder[short]; found {
			continue // Already listed.
		}
		metricsOrder[short] = nextPos
		nextPos++
	}

	if *flagMetricsLabels {
		ReportMetricsLabels(shortToName, nameToShort, metricsOrder)
	}

	if *flagMetrics {
		ReportMetrics(checkpointPath, metricsUsed, metricsOrder, points)
	}
}

// ReportMetricsLabels list all metrics short and long names.
func ReportMetricsLabels(shortToName map[string]string, nameToShort map[string]string, metricsOrder map[string]int) {
	fmt.Println(titleStyle.Render("Metrics Labels"))
	table := newPlainTable(true, lipgloss.Center, lipgloss.Left)
	table.Headers("Short", "MetricName")
	rows := make([][]string, len(metricsOrder))
	for short, idx := range metricsOrder {
		name, found := shortToName[short]
		if !found {
			// metric manually selected by name:
			name = short
			short = nameToShort[name]
		}
		rows[idx-1] = []string{short, name}
	}
	for _, row := range rows {
		table.Row(row...)
	}
	fmt.Println(table.Render())
}

// ReportMetrics of the model.
func ReportMetrics(checkpointPath string, metricsUsed types.Set[string], metricsOrder map[string]int, points []plots.Point) {
	fmt.Println(titleStyle.Render(fmt.Sprintf("Metrics %q", checkpointPath)))
	table := newPlainTable(true, lipgloss.Right)
	header := make([]string, 1+len(metricsUsed))
	header[0] = "Global Step"
	for name, idx := range metricsOrder {
		header[idx] = name
	}
	table.Headers(header...)

	currentStep := int64(-1)
	var currentRow []string
	for _, point := range points {
		step := int64(point.Step)
		if step != currentStep {
			if currentStep != -1 {
				table.Row(currentRow...)
			}
			currentStep = step
			currentRow = make([]string, 1+len(metricsUsed))
			currentRow[0] = humanize.Comma(step)
		}
		idx, found := metricsOrder[point.Short]
		if found {
			var value string
			switch point.MetricType {
			case "accuracy":
				value = fmt.Sprintf("%.2f%%", 100.0*point.Value)
			default:
				value = fmt.Sprintf("%.3g", point.Value)
			}
			currentRow[idx] = value
		}
	}
	if currentStep != -1 {
		table.Row(currentRow...)
	}
	fmt.Println(table.Render())
}
