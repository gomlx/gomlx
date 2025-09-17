package main

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"io"
	"os"
	"path"
	"slices"

	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gomlx/ui/plots"
	"github.com/janpfeifer/gonb/gonbui/plotly"
	"github.com/pkg/errors"

	grob "github.com/MetalBlueberry/go-plotly/generated/v2.34.0/graph_objects"
	ptypes "github.com/MetalBlueberry/go-plotly/pkg/types"
)

var (
	flagPlot = flag.Bool("plot", false,
		fmt.Sprintf("Plots the metrics collected for plotting in file %q. "+
			"You can control which metrics to plot with -metrics_names and -metrics_types", plots.TrainingPlotFileName))
)

// createModelNamesToIndex collects all model names (base name of the their directory), sort them, and returns a
// map from model name to their sorted index.
func createModelNamesToIndex(metricsOrder map[ModelNameAndMetric]int) (modelNamesToIndex map[string]int) {
	modelNamesToIndex = make(map[string]int)
	for info := range metricsOrder {
		name := path.Base(info.ModelName)
		modelNamesToIndex[name] = 0
	}
	{
		// Sort modelNames and generate index by sort order.
		sortedNames := xslices.SortedKeys(modelNamesToIndex)
		for idx, name := range sortedNames {
			modelNamesToIndex[name] = idx + 1
		}
	}
	return
}

// createSortedMetricTypes collects all metric types and sort them.
func createSortedMetricTypes(metricsOrder map[ModelNameAndMetric]int) []string {
	metricTypesSet := types.MakeSet[string]()
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
func createPlotLines(metricType string, modelNames []string, points [][]plots.Point, modelNamesToIndex map[string]int) []*plotLineInfo {
	var lines []*plotLineInfo
	for modelIdx, modelPoints := range points {
		modelName := path.Base(modelNames[modelIdx])
		modelNum := modelNamesToIndex[modelName]

		// Group points by metric name.
		metricPoints := make(map[string]*plotLineInfo)
		for _, pt := range modelPoints {
			if pt.MetricType != metricType {
				continue
			}
			info, exists := metricPoints[pt.MetricName]
			if !exists {
				shortName := fmt.Sprintf("#%d %s", modelNum, pt.Short)
				info = &plotLineInfo{
					short: shortName,
					desc:  fmt.Sprintf("%s: %s for model %q", shortName, pt.MetricName, modelName),
				}
			}
			info.steps = append(info.steps, pt.Step)
			info.values = append(info.values, pt.Value)
			metricPoints[pt.MetricName] = info
		}

		// Sort points by steps.
		for _, info := range metricPoints {
			// Create indices array.
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

// BuildPlots from the models' metrics points.
func BuildPlots(modelNames []string, metricsOrder map[ModelNameAndMetric]int, points [][]plots.Point) {
	metricTypes := createSortedMetricTypes(metricsOrder)
	numPlots := len(metricTypes)
	modelNamesToIndex := createModelNamesToIndex(metricsOrder)

	// Create one plot per metric type.
	serializedPlots := make([][]byte, 0, numPlots)
	for _, metricType := range metricTypes {
		fig := &grob.Fig{
			Layout: &grob.Layout{
				Title: &grob.LayoutTitle{
					Text: ptypes.S(metricType),
				},
				Xaxis: &grob.LayoutXaxis{
					Showgrid: ptypes.B(true),
					Type:     grob.LayoutXaxisTypeLog,
				},
				Yaxis: &grob.LayoutYaxis{
					Showgrid: ptypes.B(true),
					Type:     grob.LayoutYaxisTypeLog,
				},
				Legend: &grob.LayoutLegend{
					//Y:       -0.2,
					//X:       1.0,
					//X anchor: grob.LayoutLegendX anchorRight,
					//Y anchor: grob.LayoutLegendY anchorTop,
				},
			},
		}
		// Add scatter lines to the plot.
		lines := createPlotLines(metricType, modelNames, points, modelNamesToIndex)
		for _, line := range lines {
			fig.Data = append(fig.Data, &grob.Scatter{
				Name: ptypes.S(line.short),
				Line: &grob.ScatterLine{
					Shape: grob.ScatterLineShapeLinear,
				},
				Mode: "lines+markers",
				X:    ptypes.DataArray(line.steps),
				Y:    ptypes.DataArray(line.values),
			})
		}

		// Convert the plot to JSON and serialize it.
		figAsJSON, err := json.Marshal(fig)
		if err != nil {
			panic(errors.Wrapf(err, "failed to marshal plotly figure for metric type %q", metricType))
		}
		serializedPlots = append(serializedPlots, figAsJSON)
	}

	// Create temporary file for serializedPlots
	tmpFile, err := os.CreateTemp("", "gomlx-serializedPlots-*.html")
	if err != nil {
		panic(errors.Wrap(err, "failed to create temporary file for serializedPlots"))
	}

	// Write serializedPlots to a temporary file
	if err := PlotlyToHTMLFile(tmpFile.Name(), serializedPlots...); err != nil {
		panic(errors.Wrap(err, "failed to write serializedPlots to temporary file"))
	}

	fmt.Printf("\nPlots written to:\t%s\n\n", tmpFile.Name())
}

var (
	singleFileHTML = `<!DOCTYPE html>
	<head>
		<meta charset="utf-8">
		<script src="{{ .CDN }}"></script>
	</head>
	<body style="background-color: black;">
{{- range $i, $f := .Figures }}
		<div id="plot{{ $i }}"></div>
		{{ if not (eq $i (lastIdx $.Figures)) }}
		<hr style="border-color: gray;">
		{{ end }}
{{- end }}
	<script>
{{- range $i, $f := .Figures }}
		data = JSON.parse(atob('{{ $f }}'))
		Plotly.newPlot('plot{{ $i }}', data);
{{- end }}
	</script>
	</body>
</html>`
	singleFileHTMLTmpl = template.Must(template.New("plotly").Funcs(template.FuncMap{
		"lastIdx": func(a []string) int { return len(a) - 1 },
	}).Parse(singleFileHTML))
)

// WritePlotlyAsHTML renders the Plotly figures (given as JSON) to an HTML page that can be
// served or saved to a file.
func WritePlotlyAsHTML(w io.Writer, figuresAsJSON ...[]byte) error {
	data := &struct {
		CDN     string
		Figures []string
	}{
		CDN:     plotly.PlotlySrc,
		Figures: xslices.Map(figuresAsJSON, func(fig []byte) string { return base64.StdEncoding.EncodeToString(fig) }),
	}
	err := singleFileHTMLTmpl.Execute(w, data)
	if err != nil {
		return errors.Wrap(err, "failed to render plotly")
	}
	return nil
}

// PlotlyToHTMLFile renders the Plotly figure (given as JSON) to an HTML file.
func PlotlyToHTMLFile(fileName string, figuresAsJSon ...[]byte) error {
	f, err := os.Create(fileName)
	if err != nil {
		return errors.Wrapf(err, "failed to create file %q", fileName)
	}
	defer func() { _ = f.Close() }()
	return WritePlotlyAsHTML(f, figuresAsJSon...)
}
