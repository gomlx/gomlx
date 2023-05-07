package graph

import (
	"testing"
)

func TestInterpolate(t *testing.T) {
	testFuncOneInput(t, "Interpolate(1D).Nearest().HalfPixelCenters(true).AlignCorner(false)",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][][]float32{{{0}, {3}, {6}}})
			output = Interpolate(input, NoInterpolation, 6, NoInterpolation).
				Nearest().HalfPixelCenters(false).AlignCorner(false).
				Done()
			return
		}, [][][]float32{{{0}, {0}, {3}, {3}, {6}, {6}}})

	testFuncOneInput(t, "Interpolate(1D).Bilinear().HalfPixelCenters(true).AlignCorner(false)",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][][]float32{{{0}, {3}, {6}}})
			output = Interpolate(input, NoInterpolation, 6, NoInterpolation).
				Bilinear().HalfPixelCenters(true).AlignCorner(false).
				Done()
			return
		}, [][][]float32{{{0}, {0.75}, {2.25}, {3.75}, {5.25}, {6}}})

	testFuncOneInput(t, "Interpolate(2D).Bilinear().HalfPixelCenters(true).AlignCorner(false)",
		func(g *Graph) (input, output *Node) {
			input = Const(g, [][][][]float32{{{{0}, {3}, {6}}, {{30}, {33}, {36}}, {{60}, {63}, {66}}}})
			output = Interpolate(input, NoInterpolation, 6, 6, NoInterpolation).
				Bilinear().HalfPixelCenters(true).AlignCorner(false).
				Done()
			return
		}, [][][][]float32{{
			{{0}, {0.75}, {2.25}, {3.75}, {5.25}, {6}},
			{{7.5}, {8.25}, {9.75}, {11.25}, {12.75}, {13.5}},
			{{22.5}, {23.25}, {24.75}, {26.25}, {27.75}, {28.5}},
			{{37.5}, {38.25}, {39.75}, {41.25}, {42.75}, {43.5}},
			{{52.5}, {53.25}, {54.75}, {56.25}, {57.75}, {58.5}},
			{{60}, {60.75}, {62.25}, {63.75}, {65.25}, {66}},
		}})
}
