package main

import (
	"encoding/base64"
	"encoding/json"
	"flag"
	"fmt"
	"html/template"
	"io"
	"os"
	"os/exec"
	"path"
	"runtime"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gomlx/ui/plots"
	"github.com/janpfeifer/gonb/gonbui/plotly"
	"github.com/pkg/errors"

	grob "github.com/MetalBlueberry/go-plotly/generated/v2.34.0/graph_objects"
	ptypes "github.com/MetalBlueberry/go-plotly/pkg/types"
)

// TODO:
//  - Fix mouse-over tooltip to be over the legend, and not over the points themselves. There is no support for this
//    in Plotly, but one can code one using javascript.
//  - Remove model index from legends, if there is only one model

var (
	flagPlot = flag.Bool("plot", false,
		fmt.Sprintf("Plots the metrics collected for plotting in file %q. "+
			"You can control which metrics to plot with -metrics_names and -metrics_types", plots.TrainingPlotFileName))
	flagBrowser    = flag.Bool("browser", true, "Opens the generated plots file in the default browser.")
	flagPlotOutput = flag.String("plot_output", "", "File to generate HTML file with plots. "+
		"If empty (the default) it will create a temporary file.")
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

// BuildPlots from the models' metrics points.
func BuildPlots(modelNames []string, metricsOrder map[ModelNameAndMetric]int, points [][]plots.Point) {
	metricTypes := createSortedMetricTypes(metricsOrder)
	numPlots := len(metricTypes)
	modelNamesToIndex := createModelNamesToIndex(metricsOrder)
	legendsHoverTexts := make([][]string, 0, numPlots)

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
					Title: &grob.LayoutXaxisTitle{
						Text: ptypes.S("Step"),
					},
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
				Template: plotly.PlotlyDarkTheme,
			},
		}
		// Add scatter lines to the plot.
		lines := createPlotLines(metricType, modelNames, points, modelNamesToIndex)
		lineHovers := make([]string, 0, len(lines))
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
			lineHovers = append(lineHovers, line.desc)
		}

		// Convert the plot to JSON and serialize it.
		figAsJSON, err := json.Marshal(fig)
		if err != nil {
			panic(errors.Wrapf(err, "failed to marshal plotly figure for metric type %q", metricType))
		}
		serializedPlots = append(serializedPlots, figAsJSON)
		legendsHoverTexts = append(legendsHoverTexts, lineHovers)
	}

	// Set the title.
	var title string
	if len(modelNames) == 1 {
		title = modelNames[0]
	} else {
		baseNames := xslices.Map(modelNames, path.Base)
		title = fmt.Sprintf("Models: %s", strings.Join(baseNames, ", "))
	}

	// Create a temporary file for the serializedPlots if needed.
	outputFilePath := *flagPlotOutput
	outputFilePath = data.ReplaceTildeInDir(outputFilePath)
	if outputFilePath == "" {
		tmpFile, err := os.CreateTemp("", "gomlx-serializedPlots-*.html")
		if err != nil {
			panic(errors.Wrap(err, "failed to create temporary file for serializedPlots"))
		}
		outputFilePath = tmpFile.Name()
	}

	// Write serializedPlots to a temporary file
	if err := PlotlyToHTMLFile(outputFilePath, title, serializedPlots, legendsHoverTexts); err != nil {
		panic(errors.Wrap(err, "failed to write serializedPlots to temporary file"))
	}

	fmt.Printf("\nPlots written to:\t%s\n\n", outputFilePath)
	if *flagBrowser {
		openBrowser(outputFilePath)
	}
}

var (
	singleFileHTML = `<!DOCTYPE html>
	<head>
		<meta charset="utf-8">
		<script src="{{ .CDN }}"></script>
		<style>
			body {
				background-color: #1a1a1a;
				color: #ffffff;
				font-family: 'Segoe UI', 'Arial', sans-serif;
				margin: 0;
				padding: 20px;
			}
			h1 {
				color: #00ffcc;
				text-align: center;
				font-weight: 300;
				margin-bottom: 40px;
			}
			hr {
				border: none;
				height: 1px;
				background: linear-gradient(90deg, transparent, #404040, transparent);
				margin: 30px 0;
			}
			.plot-container {
				background-color: #222222;
				border-radius: 8px;
				padding: 20px;
				margin: 20px 0;
				box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
			}
			#custom-tooltip {
				position: absolute; /* Allows us to position it with JS */
				display: none;      /* Start hidden */
				top: 0;
				left: 0;
				padding: 8px;
				background-color: #2a2a2a;
				color: #fff;
				border: 1px solid #555;
				border-radius: 4px;
				/* font-family: sans-serif; */
				font-size: 12px;
				z-index: 1000;      /* Ensure it appears on top of everything */
				pointer-events: none; /* VERY IMPORTANT: Prevents the tooltip from blocking mouse events on elements underneath it */
			}
		</style>
	</head>
	<body>
		<h1>{{ .Title }}</h1>
	    <div id="custom-tooltip"></div>
{{- range $i, $f := .Figures }}
		<div class="plot-container">
			<div id="plot{{ $i }}"></div>
		</div>
		<hr>
{{- end }}
	<script>
{{- range .Figures }}
const {{.LegendsVar}} = [
{{- range .LegendsHover }}
	"{{ . }}",
{{- end }}
];
{{- end }}

{{- range $i, $f := .Figures }}
		data = JSON.parse(atob('{{ $f.Figure }}'))
		Plotly.newPlot('plot{{ $i }}', data, {
			paper_bgcolor: '#222222',
			plot_bgcolor: '#222222',
			font: { color: '#ffffff' }
		}).then(() => {
			const tooltip = document.getElementById('custom-tooltip');
			const legendItems = document.querySelectorAll('#plot{{ $i }} .legend .traces');
			legendItems.forEach((item, index) => {

				// When the mouse enters the legend item...
				item.addEventListener('mouseover', (event) => {
					// Set the tooltip's text
					tooltip.innerHTML = {{$f.LegendsVar}}[index];
					// Make it visible
					tooltip.style.display = 'block';
				});
		
				// When the mouse moves over the legend item...
				item.addEventListener('mousemove', (event) => {
					// Update the tooltip's position to follow the cursor
					// The 10px offset prevents the tooltip from flickering
					tooltip.style.left = (event.pageX - tooltip.offsetWidth - 10) + 'px';
					//tooltip.style.left = (event.pageX + 10) + 'px';
					tooltip.style.top = (event.pageY + 10) + 'px';
				});
		
				// When the mouse leaves the legend item...
				item.addEventListener('mouseout', () => {
					// Hide the tooltip
					tooltip.style.display = 'none';
				});
			});
		});
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
func WritePlotlyAsHTML(w io.Writer, title string, figuresAsJSON [][]byte, legendHoverTests [][]string) error {
	type FigureData struct {
		Figure       string
		LegendsVar   template.JS
		LegendsHover []string
	}
	type RootData struct {
		CDN     string
		Title   string
		Figures []FigureData
	}
	data := &RootData{
		CDN:     plotly.PlotlySrc,
		Title:   title,
		Figures: make([]FigureData, len(figuresAsJSON)),
	}
	for i, fig := range figuresAsJSON {
		data.Figures[i].Figure = base64.StdEncoding.EncodeToString(fig)
		data.Figures[i].LegendsVar = template.JS(fmt.Sprintf("legendsHover%d", i))
		data.Figures[i].LegendsHover = legendHoverTests[i]
	}

	err := singleFileHTMLTmpl.Execute(w, data)
	if err != nil {
		return errors.Wrap(err, "failed to render plotly")
	}
	return nil
}

// PlotlyToHTMLFile renders the Plotly figure (given as JSON) to an HTML file.
func PlotlyToHTMLFile(fileName, title string, figuresAsJSon [][]byte, legendHoverTests [][]string) error {
	f, err := os.Create(fileName)
	if err != nil {
		return errors.Wrapf(err, "failed to create file %q", fileName)
	}
	defer func() { _ = f.Close() }()
	return WritePlotlyAsHTML(f, title, figuresAsJSon, legendHoverTests)
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
