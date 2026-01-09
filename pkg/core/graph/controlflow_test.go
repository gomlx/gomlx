// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"testing"

	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

func TestWhile_SimpleCounting(t *testing.T) {
	graphtest.RunTestGraphFn(t, "While: sum 1 to 10",
		func(g *Graph) (inputs, outputs []*Node) {
			// Compute sum 1+2+...+10 = 55
			// State: [counter, sum]

			// Condition: counter <= 10
			cond := NewClosure(g, func(g *Graph) []*Node {
				counter := Parameter(g, "counter", shapes.Scalar[int32]())
				_ = Parameter(g, "sum", shapes.Scalar[int32]())
				return []*Node{LessOrEqual(counter, Const(g, int32(10)))}
			})

			// Body: counter++, sum += counter
			body := NewClosure(g, func(g *Graph) []*Node {
				counter := Parameter(g, "counter", shapes.Scalar[int32]())
				sum := Parameter(g, "sum", shapes.Scalar[int32]())
				newCounter := Add(counter, Const(g, int32(1)))
				newSum := Add(sum, counter)
				return []*Node{newCounter, newSum}
			})

			// Initial state: counter=1, sum=0
			results := While(cond, body,
				Const(g, int32(1)),
				Const(g, int32(0)))

			// Return sum (index 1)
			return nil, []*Node{results[1]}
		},
		[]any{int32(55)},
		0,
	)
}

func TestIf_SelectBranch(t *testing.T) {
	graphtest.RunTestGraphFn(t, "If: select true branch",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, float32(10))
			pred := GreaterThan(x, Const(g, float32(5)))

			trueBranch := NewClosure(g, func(g *Graph) []*Node {
				return []*Node{Const(g, float32(1))}
			})
			falseBranch := NewClosure(g, func(g *Graph) []*Node {
				return []*Node{Const(g, float32(-1))}
			})

			results := If(pred, trueBranch, falseBranch)
			return nil, []*Node{results[0]}
		},
		[]any{float32(1)},
		0,
	)
}

func TestIf_SelectBranchFalse(t *testing.T) {
	graphtest.RunTestGraphFn(t, "If: select false branch",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, float32(3))
			pred := GreaterThan(x, Const(g, float32(5)))

			trueBranch := NewClosure(g, func(g *Graph) []*Node {
				return []*Node{Const(g, float32(1))}
			})
			falseBranch := NewClosure(g, func(g *Graph) []*Node {
				return []*Node{Const(g, float32(-1))}
			})

			results := If(pred, trueBranch, falseBranch)
			return nil, []*Node{results[0]}
		},
		[]any{float32(-1)},
		0,
	)
}

func TestSort_Ascending(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Sort: ascending",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{5, 2, 8, 1, 9, 3})
			result := Sort(x, 0, true)
			return nil, []*Node{result}
		},
		[]any{[]float32{1, 2, 3, 5, 8, 9}},
		0,
	)
}

func TestSort_Descending(t *testing.T) {
	graphtest.RunTestGraphFn(t, "Sort: descending",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{5, 2, 8, 1, 9, 3})
			result := Sort(x, 0, false)
			return nil, []*Node{result}
		},
		[]any{[]float32{9, 8, 5, 3, 2, 1}},
		0,
	)
}

func TestSortFunc_WithComparator(t *testing.T) {
	graphtest.RunTestGraphFn(t, "SortFunc: custom comparator",
		func(g *Graph) (inputs, outputs []*Node) {
			x := Const(g, []float32{5, 2, 8, 1, 9, 3})

			comparator := NewClosure(g, func(g *Graph) []*Node {
				lhs := Parameter(g, "lhs", shapes.Scalar[float32]())
				rhs := Parameter(g, "rhs", shapes.Scalar[float32]())
				return []*Node{LessThan(lhs, rhs)}
			})

			results := SortFunc(comparator, 0, false, x)
			return nil, []*Node{results[0]}
		},
		[]any{[]float32{1, 2, 3, 5, 8, 9}},
		0,
	)
}

func TestWhile_Factorial(t *testing.T) {
	graphtest.RunTestGraphFn(t, "While: factorial 5! = 120",
		func(g *Graph) (inputs, outputs []*Node) {
			// Compute 5! = 120
			// State: [n, result]

			// Condition: n > 1
			cond := NewClosure(g, func(g *Graph) []*Node {
				n := Parameter(g, "n", shapes.Scalar[int32]())
				_ = Parameter(g, "result", shapes.Scalar[int32]())
				return []*Node{GreaterThan(n, Const(g, int32(1)))}
			})

			// Body: result *= n, n--
			body := NewClosure(g, func(g *Graph) []*Node {
				n := Parameter(g, "n", shapes.Scalar[int32]())
				result := Parameter(g, "result", shapes.Scalar[int32]())
				newResult := Mul(result, n)
				newN := Sub(n, Const(g, int32(1)))
				return []*Node{newN, newResult}
			})

			// Initial: n=5, result=1
			results := While(cond, body,
				Const(g, int32(5)),
				Const(g, int32(1)))

			return nil, []*Node{results[1]}
		},
		[]any{int32(120)},
		0,
	)
}
