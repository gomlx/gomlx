package discretekan

import (
	"fmt"
	"strings"

	grob "github.com/MetalBlueberry/go-plotly/generated/v2.34.0/graph_objects"
	"github.com/MetalBlueberry/go-plotly/pkg/types"
	"github.com/gomlx/gopjrt/dtypes"
	gonbplotly "github.com/janpfeifer/gonb/gonbui/plotly"
	"github.com/janpfeifer/must"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"

	_ "github.com/gomlx/gomlx/backends/xla"
)

// Univariate graph function.
type Univariate func(x *Node) *Node

// Plot univariate function for values between
func Plot(name string, univariateFunctions ...Univariate) {
	backend := backends.MustNew()
	numPoints := 1000
	minX, maxX := -0.1, 1.1

	// Split names, if separate function names were provided.
	nameParts := strings.Split(name, ";")
	var fnNames []string
	if len(nameParts) > 1 {
		name = nameParts[0]
		fnNames = nameParts[1:]
	}

	fig := &grob.Fig{
		Layout: &grob.Layout{
			Title: &grob.LayoutTitle{
				Text: types.StringType(name),
			},
			Xaxis: &grob.LayoutXaxis{
				Showgrid: types.True,
				Type:     grob.LayoutXaxisTypeLinear,
			},
			Yaxis: &grob.LayoutYaxis{
				Showgrid: types.True,
				Type:     grob.LayoutYaxisTypeLinear,
			},
		},
	}
	lineWidth := 2.0
	if len(univariateFunctions) > 1 {
		lineWidth = 1.0
	}
	for fnIdx, fn := range univariateFunctions {
		exec := NewExec(backend, func(g *Graph) []*Node {
			inputs := Iota(g, shapes.Make(dtypes.Float64, numPoints), 0)
			inputs = MulScalar(inputs, (maxX-minX)/float64(numPoints-1))
			inputs = AddScalar(inputs, minX)
			outputs := fn(inputs)
			return []*Node{inputs, outputs}
		})
		results := exec.Call()
		inputs, outputs := results[0].Value().([]float64), results[1].Value().([]float64)
		var fnName string
		if len(fnNames) > fnIdx {
			fnName = fnNames[fnIdx]
		} else {
			fnName = fmt.Sprintf("#%d", fnIdx)
		}
		fig.Data = append(fig.Data,
			&grob.Scatter{
				Name: types.StringType(fnName),
				Line: &grob.ScatterLine{
					Shape: grob.ScatterLineShapeLinear,
					Width: types.N(lineWidth),
				},
				Mode: "lines",
				X:    types.DataArray(inputs),
				Y:    types.DataArray(outputs),
			})
	}
	must.M(gonbplotly.DisplayFig(fig))
}
