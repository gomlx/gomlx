package main

import (
	"flag"
	"fmt"
	"maps"
	"path"
	"regexp"
	"slices"
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
	_ = numCheckpoints

	// Load metrics points from the checkpointPath.
	points := make([][]plots.Point, numCheckpoints)
	foundSomething := false
	for ii, checkpointPath := range checkpointPaths {
		trainingMetricsPath := path.Join(checkpointPath, plots.TrainingPlotFileName)
		points[ii] = must.M1(plots.LoadPoints(trainingMetricsPath))
		if len(points) > 0 {
			foundSomething = true
		}
	}
	if !foundSomething {
		klog.Errorf("No metrics found in file %q in paths %v", plots.TrainingPlotFileName, checkpointPaths)
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
	nameToShort := make(map[string]string)
	shortToName := make(map[string]string)
	metricsUsed := types.MakeSet[NameMetric]()
	for modelIdx, pointsPerModel := range points {
		for _, point := range pointsPerModel {
			nameToShort[point.MetricName] = point.Short
			shortToName[point.Short] = point.MetricName
			if metricsNamesMatcher != nil || metricsTypes != nil {
				foundName := metricsNamesMatcher.MatchString(point.MetricName) || metricsNamesMatcher.MatchString(point.Short)
				foundType := metricsTypes != nil && metricsTypes.Has(point.MetricType)
				if !foundName && !foundType {
					continue
				}
			}
			metricsUsed.Insert(NameMetric{names[modelIdx], point.Short})
		}
	}

	// Map the metrics to the column number, starting from 1 (column 0 is for the global step)
	metricsInOrder := slices.SortedFunc(maps.Keys(metricsUsed), func(a, b NameMetric) int {
		if c := strings.Compare(a.Metric, b.Metric); c != 0 {
			return c
		}
		return strings.Compare(a.Name, b.Name)
	})
	metricsOrder := make(map[NameMetric]int, len(metricsInOrder))
	for idx, nameMetric := range metricsInOrder {
		metricsOrder[nameMetric] = idx + 1
	}

	if *flagMetricsLabels {
		ReportMetricsLabels(shortToName)
	}

	if *flagMetrics {
		ReportMetrics(names, metricsOrder, points)
	}
}

// NameMetric hold the pair of information.
type NameMetric struct{ Name, Metric string }

// ReportMetricsLabels list all metrics short and long names.
func ReportMetricsLabels(shortToName map[string]string) {
	fmt.Println(titleStyle.Render("Metrics Labels"))
	table := newPlainTable(true, lipgloss.Center, lipgloss.Left)
	table.Headers("Short", "MetricName")
	rows := make([][]string, 0, len(shortToName))
	for _, short := range xslices.SortedKeys(shortToName) {
		rows = append(rows, []string{short, shortToName[short]})
	}
	for _, row := range rows {
		table.Row(row...)
	}
	fmt.Println(table.Render())
}

// ReportMetrics of the model.
func ReportMetrics(names []string, metricsOrder map[NameMetric]int, points [][]plots.Point) {
	numCheckpoints := len(names)
	fmt.Println(titleStyle.Render("Metrics Table"))
	table := newPlainTable(true, lipgloss.Right)
	header := make([]string, 1+len(metricsOrder))
	header[0] = "Global Step"
	for nameMetric, idx := range metricsOrder {
		if numCheckpoints == 1 {
			header[idx] = nameMetric.Metric
		} else {
			header[idx] = fmt.Sprintf("%s: %s", nameMetric.Name, nameMetric.Metric)
		}
	}
	table.Headers(header...)

	// Checks whether the indices reached on the end of all points.
	pointsIndices := make([]int, numCheckpoints)
	isFinished := func() bool {
		for ii, points := range points {
			if pointsIndices[ii] < len(points) {
				return false
			}
		}
		return true
	}

	// Find the next global step to enumerate.
	nextGlobalStep := func() int64 {
		globalStep := int64(-1)
		for modelIdx, pointsPerModel := range points {
			if pointsIndices[modelIdx] < len(pointsPerModel) {
				point := pointsPerModel[pointsIndices[modelIdx]]
				if globalStep == -1 || int64(point.Step) < globalStep {
					globalStep = int64(point.Step)
				}
			}
		}
		return globalStep
	}

	// Loop one global step at a time:
	for !isFinished() {
		currentGlobalStep := nextGlobalStep()
		row := make([]string, 1+len(metricsOrder))
		row[0] = humanize.Comma(currentGlobalStep)
		for modelIdx, pointsPerModel := range points {
			// Consume all points for the currentGlobalStep.
			nameMetric := NameMetric{Name: names[modelIdx]}
			for pointsIndices[modelIdx] < len(pointsPerModel) {
				point := pointsPerModel[pointsIndices[modelIdx]]
				if int64(point.Step) != currentGlobalStep {
					// Nothing else in this globalStep for this model.
					break
				}
				pointsIndices[modelIdx]++ // We'll consume this point, move the head forward.
				nameMetric.Metric = point.Short
				colIdx, found := metricsOrder[nameMetric]
				if !found {
					// We are not printing this metric.
					continue
				}
				var value string
				switch point.MetricType {
				case "accuracy":
					value = fmt.Sprintf("%.2f%%", 100.0*point.Value)
				default:
					value = fmt.Sprintf("%.3g", point.Value)
				}
				row[colIdx] = value
			} // end for loop on all model points on currentGlobalStep.
		} // end for loop on all models.
		table.Row(row...)
	} // end scanning all points.
	fmt.Println(table.Render())
}
