package losses

import (
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"

	_ "github.com/gomlx/gomlx/backends/xla"
)

func TestPairwiseL2Distances(t *testing.T) {
	graphtest.RunTestGraphFn(t, "PairwiseL2Distances",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, [][]float32{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}}),
			}
			outputs = []*Node{
				pairwiseL2Distances(inputs[0], false),
				pairwiseL2Distances(inputs[0], true),
			}
			return
		}, []any{
			[][]float32{{0., 1.4142135, 1.4142135}, {1.4142135, 0., 1.4142135}, {1.4142135, 1.4142135, 0.}},
			[][]float32{{0, 2, 2}, {2, 0, 2}, {2, 2, 0}},
		}, -1)
}

func TestPairwiseCosineDistances(t *testing.T) {
	graphtest.RunTestGraphFn(t, "PairwiseCosineDistances",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, [][]float32{{1, 1, 1}, {0, 1, 0}, {1, 0, 0}}),
			}
			outputs = []*Node{
				pairwiseCosineDistances(inputs[0]),
			}
			return
		}, []any{
			[][]float32{{5.9604645e-08, 0.42264974, 0.42264974}, {0.42264974, 0., 1.}, {0.42264974, 1, 0}},
		}, -1)
}

func TestTripletLoss(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TripletLoss",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, [][]float32{{1}, {0}, {0}, {0}, {3}, {2}, {3}, {2}, {1}, {2}}), // labels
				Const(g, [][]float32{
					{0.08208963, 0.11788353, 0.46360782, 0.3360519, 0.2702437, 0.6951965},
					{0.598121, 0.14609586, 0.07872304, 0.949776, 0.41479972, 0.36961815},
					{0.11646613, 0.8878409, 0.4034519, 0.9632401, 0.6313564, 0.0198459},
					{0.03582959, 0.3428808, 0.843301, 0.6335877, 0.8623248, 0.16186231},
					{0.09054314, 0.746887, 0.56099737, 0.7181275, 0.60642695, 0.02207313},
					{0.2735666, 0.08748698, 0.13752021, 0.4570993, 0.8813543, 0.98528206},
					{0.5412437, 0.2382705, 0.6263132, 0.29713312, 0.9241606, 0.734765},
					{0.22289598, 0.84535605, 0.4398808, 0.5816502, 0.31203038, 0.5436755},
					{0.5512105, 0.6922551, 0.11149547, 0.6343566, 0.20425326, 0.3884894},
					{0.51529086, 0.35541356, 0.77092594, 0.3715265, 0.40550032, 0.7369012},
				}), // embeddings
			}
			outputs = []*Node{
				TripletLoss([]*Node{inputs[0]}, []*Node{inputs[1]}, 1.0, TripletLossDistanceL2),
			}
			return
		}, []any{
			float32(1.0418946),
		}, -1)
}

func TestTripletLossHard(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TripletLossHard",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, [][]float32{{1}, {0}, {0}, {0}, {3}, {2}, {3}, {2}, {1}, {2}}), // labels
				Const(g, [][]float32{
					{0.08208963, 0.11788353, 0.46360782, 0.3360519, 0.2702437, 0.6951965},
					{0.598121, 0.14609586, 0.07872304, 0.949776, 0.41479972, 0.36961815},
					{0.11646613, 0.8878409, 0.4034519, 0.9632401, 0.6313564, 0.0198459},
					{0.03582959, 0.3428808, 0.843301, 0.6335877, 0.8623248, 0.16186231},
					{0.09054314, 0.746887, 0.56099737, 0.7181275, 0.60642695, 0.02207313},
					{0.2735666, 0.08748698, 0.13752021, 0.4570993, 0.8813543, 0.98528206},
					{0.5412437, 0.2382705, 0.6263132, 0.29713312, 0.9241606, 0.734765},
					{0.22289598, 0.84535605, 0.4398808, 0.5816502, 0.31203038, 0.5436755},
					{0.5512105, 0.6922551, 0.11149547, 0.6343566, 0.20425326, 0.3884894},
					{0.51529086, 0.35541356, 0.77092594, 0.3715265, 0.40550032, 0.7369012},
				}), // embeddings
			}
			outputs = []*Node{
				TripletHardLoss([]*Node{inputs[0]}, []*Node{inputs[1]}, 1.0, false, TripletLossDistanceL2),
				TripletHardLoss([]*Node{inputs[0]}, []*Node{inputs[1]}, 1.0, true, TripletLossDistanceL2),
			}
			return
		}, []any{
			float32(1.5172637),
			float32(0.98718655),
		}, -1)
}

func TestTripletLossSemiHard(t *testing.T) {
	graphtest.RunTestGraphFn(t, "TripletLossSemiHard",
		func(g *Graph) (inputs, outputs []*Node) {
			inputs = []*Node{
				Const(g, [][]float32{{0}, {1}, {0}, {1}}),
				Const(g, [][]float32{
					{0.23, 0.75},
					{0.89, 0.41},
					{0.37, 0.62},
					{0.78, 0.24},
				}), // embeddings
				Const(g, [][]float32{{1}, {0}, {0}, {0}, {3}, {2}, {3}, {2}, {1}, {2}}), // labels
				Const(g, [][]float32{
					{0.08208963, 0.11788353, 0.46360782, 0.3360519, 0.2702437, 0.6951965},
					{0.598121, 0.14609586, 0.07872304, 0.949776, 0.41479972, 0.36961815},
					{0.11646613, 0.8878409, 0.4034519, 0.9632401, 0.6313564, 0.0198459},
					{0.03582959, 0.3428808, 0.843301, 0.6335877, 0.8623248, 0.16186231},
					{0.09054314, 0.746887, 0.56099737, 0.7181275, 0.60642695, 0.02207313},
					{0.2735666, 0.08748698, 0.13752021, 0.4570993, 0.8813543, 0.98528206},
					{0.5412437, 0.2382705, 0.6263132, 0.29713312, 0.9241606, 0.734765},
					{0.22289598, 0.84535605, 0.4398808, 0.5816502, 0.31203038, 0.5436755},
					{0.5512105, 0.6922551, 0.11149547, 0.6343566, 0.20425326, 0.3884894},
					{0.51529086, 0.35541356, 0.77092594, 0.3715265, 0.40550032, 0.7369012},
				}), // embeddings
			}
			outputs = []*Node{
				TripletSemiHardLoss([]*Node{inputs[0]}, []*Node{inputs[1]}, 1.0, false, TripletLossDistanceL2),
				TripletSemiHardLoss([]*Node{inputs[2]}, []*Node{inputs[3]}, 1.0, false, TripletLossDistanceL2),
			}
			return
		}, []any{
			float32(0.5914507),
			float32(0.93788296),
		}, -1)
}
