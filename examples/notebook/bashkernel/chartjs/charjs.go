/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package chartjs

import (
	"encoding/hex"
	"fmt"
	"github.com/gomlx/gomlx/examples/notebook/bashkernel"
	"github.com/pkg/errors"
	"math/rand"
	"strings"
	"text/template"
	"unsafe"
)

// SrcUrl is the default URL for the ChartJSUrl.js library. It can be changed to
// reflect where the notebook will pull it from.
var SrcUrl = "https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.js"

var (
	// DefaultBackgroundColors to use on plots.
	DefaultBackgroundColors = []string{
		"rgba(255, 99, 132, 0.2)",
		"rgba(54, 162, 235, 0.2)",
		"rgba(255, 206, 86, 0.2)",
		"rgba(75, 192, 192, 0.2)",
		"rgba(153, 102, 255, 0.2)",
		"rgba(255, 159, 64, 0.2)",
	}
	DefaultBorderColors = []string{
		"rgba(255, 99, 132, 1)",
		"rgba(54, 162, 235, 1)",
		"rgba(255, 206, 86, 1)",
		"rgba(75, 192, 192, 1)",
		"rgba(153, 102, 255, 1)",
		"rgba(255, 159, 64, 1)",
	}
)

func MustParse(t *template.Template, text string) *template.Template {
	var err error
	t, err = t.Parse(text)
	if err != nil {
		panic(fmt.Sprintf("Failed to parse template: %v", err))
	}
	return t
}

// UniqueSuffix returns are unique suffix for JavaScript variable names.
func UniqueSuffix() string {
	rnd := rand.Uint32()
	src := unsafe.Slice((*byte)(unsafe.Pointer(&rnd)), unsafe.Sizeof(rnd))
	buf := make([]byte, hex.EncodedLen(len(src)))
	hex.Encode(buf, src)
	return strings.ToUpper(string(buf))
}

// LinePlot can build and dynamically update line plots.
type LinePlot struct {
	Id   string
	Data *LinePlotData

	ChartVar            string
	DivId, CanvasId     string
	JavascriptDisplayId string

	// Plot creation content.
	Content string

	// ChartJSUrl is set with SrcUrl.
	ChartJSUrl string

	// DivStyle is inlined in the enclosing `div` html tag.
	DivStyle string

	// CanvasStyle is inlined in the `canvas` html tag that holds the plot.
	CanvasStyle string
}

// Plot returns the body of the plot in HTML.
func (p *LinePlot) Plot() error {
	builder := &strings.Builder{}
	err := linePlotScriptTemplate.Execute(builder, p)
	if err != nil {
		return errors.Wrapf(err, "Failed to run LinePlot template")
	}
	html := builder.String()
	//fmt.Printf("html for notebook:\n%s\n", html)
	err = bashkernel.OutputHTML(html, "")
	if err != nil {
		return errors.WithMessagef(err, "Failed to output LinePlot contents to notebook")
	}
	return nil
}

// AddPoint executes Javascript code needed to dynamically add one point to the plot. They are only
// effectively displayed after Update is called.
func (p *LinePlot) AddPoint(lineIdx int, x, y float64) error {
	return bashkernel.OutputJavascript(
		fmt.Sprintf("%s.data.datasets[%d].data.push({x:%f, y:%f});", p.ChartVar, lineIdx, x, y),
		p.JavascriptDisplayId)
}

// Update refreshes the plot after updates with AddPoint (or others).
func (p *LinePlot) Update() error {
	return bashkernel.OutputJavascript(fmt.Sprintf("%s.update();", p.ChartVar),
		p.JavascriptDisplayId)
}

// XY represents one point.
type XY struct{ X, Y float64 }

type LineData struct {
	Label string

	// BackgroundColor, leave empty and one will be chosen for you.
	BackgroundColor string

	// BorderColor, leave empty and one will be chosen for you.
	BorderColor string

	// IsSecondAxis, if used, set also Has2Axis in LinePlotData.
	IsSecondAxis bool

	Points []XY
}

// LinePlotData contains potentially multiple lines sharing the same X,Y axis.
type LinePlotData struct {
	Title, Subtitle string
	HasTwoAxis      bool
	Lines           []*LineData
	XTitle          string
	YTitle, Y2Title string
}

func (l *LinePlotData) AddPoint(lineIdx int, x, y float64) {
	l.Lines[lineIdx].Points = append(l.Lines[lineIdx].Points, XY{x, y})
}

func NewLinePlot(data *LinePlotData) *LinePlot {
	id := UniqueSuffix()
	p := &LinePlot{
		Id:                  id,
		Data:                data,
		ChartJSUrl:          SrcUrl,
		ChartVar:            fmt.Sprintf("linePlot_%s", id),
		CanvasId:            fmt.Sprintf("line_plot_canvas_%s", id),
		DivId:               fmt.Sprintf("line_plot_div_%s", id),
		JavascriptDisplayId: fmt.Sprintf("js_cell_%s", id),
		CanvasStyle:         "",
		DivStyle:            "width: 80%;",
	}
	//fmt.Printf("\n\tChart: %s\n", p.ChartVar)
	backgroundColorPicks := 0
	borderColorPicks := 0
	for _, line := range p.Data.Lines {
		if line.BackgroundColor == "" {
			if backgroundColorPicks < len(DefaultBackgroundColors) {
				line.BackgroundColor = DefaultBackgroundColors[backgroundColorPicks]
				backgroundColorPicks++
			} else {
				line.BackgroundColor = "Gray"
			}
		}
		if line.BorderColor == "" {
			if borderColorPicks < len(DefaultBorderColors) {
				line.BorderColor = DefaultBorderColors[borderColorPicks]
				borderColorPicks++
			} else {
				line.BorderColor = "DarkGray"
			}
		}
	}
	return p
}

var linePlotScriptTemplate = MustParse(template.New("linePlotScriptTemplate"), `
<div id="{{.DivId}}" style="{{.DivStyle}}">
<canvas id="{{.CanvasId}}" style="{{.CanvasStyle}}"></canvas>
<script type="text/javascript">
var {{.ChartVar}};

function Create{{.ChartVar}}(Chart) {
	let canvasId = document.getElementById('{{.CanvasId}}');
	return new Chart(canvasId,{
		type: 'line',
		data: {
			datasets: [
{{with .Flat}}
{{range .Lines}}
				{
				label: '{{.Label}}',
				showLine: true,
				xAxisID: 'x',
{{if .IsSecondAxis}}
                yAxisID: 'y1',
{{end}}
				borderColor: '{{.BorderColor}}',
				backgroundColor: '{{.BackgroundColor}}',
				data: [
{{range .Points}}
					{ x: {{.X}}, y: {{.Y}} },
{{end}}
					]
				},
{{end}}
			]
		},
		options: {
			scales: {
				'x': {
					type: 'linear',
					display: true,
{{if .XTitle}}
					title: { 
						text: '{{.XTitle}}',
						display: true,
					},
{{end}}
				},
				'y': {
					type: 'linear',
					display: true,
                    position: 'left',
{{if .YTitle}}
					title: { 
						text: '{{.YTitle}}',
						display: true,
					},
{{end}}
				},
{{if .HasTwoAxis}}
				'y1': {
					type: 'linear',
					display: 'true',
					position: 'right',
{{if .Y2Title}}
					title: { 
						text: '{{.Y2Title}}',
						display: true,
					},
{{end}}
					grid: {
						drawOnChartArea: false, // only want the grid lines for one axis to show up.
					},
				}
{{end}}
			},
			plugins: {
{{if .Title}}
				title: {
					display: true,
					text: "{{.Title}}",
				},
{{end}}
{{if .Subtitle}}
				subtitle: {
					display: true,
					text: "{{.Subtitle}}",
				},
{{end}}
			},
		},
	});
}
{{end}}

if (typeof require === 'function') {
	// When Jupyter notebook is exported, it imports require.js, which will conflict with a direct import
	// of chart.js, so we need to do the following:
	require(["{{.ChartJSUrl}}"], (Chart) => {
		{{.ChartVar}} = Create{{.ChartVar}}(Chart);
	});
} else {
	// Within a Jupyter notebook, we need to manually dynamically load the javascript code:
	if (typeof Chart === 'function') {
		{{.ChartVar}} = Create{{.ChartVar}}(Chart);
	} else {
		let s = document.createElement("script");
		s.src = "{{.ChartJSUrl}}";
		s.type = "text/javascript";
		s.addEventListener("error", (ev) => {
			console.log("Error loading {{.ChartVar}}:", ev);
		});
		s.addEventListener("load", () => {
			console.log("{{.ChartVar}} loaded");
			{{.ChartVar}} = Create{{.ChartVar}}(Chart);
		});
		document.head.appendChild(s);	
	}
}
</script>
</div>
`)
