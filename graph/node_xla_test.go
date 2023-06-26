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

package graph

import (
	"fmt"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/stretchr/testify/require"
	"testing"
)

type graphFnOneInputToTest func(g *Graph) (input, output *Node)

func testFuncOneInput(t *testing.T, testName string, graphFn graphFnOneInputToTest, want any) {
	fmt.Printf("%s\n", testName)
	manager := buildTestManager()
	g := manager.NewGraph(testName)
	input, output := graphFn(g)
	g.Compile(input, output)
	g.MustOk()
	tuple := g.Run(nil)
	if !tuple.Ok() {
		t.Fatalf("Failed to run graph: %+v", tuple.Error())
	}
	results := tuple.SplitTuple()
	fmt.Printf("\t%s(%s) = %s\n", testName, results[0].Local().GoStr(), results[1].Local().GoStr())
	if !slices.SlicesInDelta(results[1].Local().Value(), want, slices.Epsilon) {
		t.Errorf("%s(%v): want=%v, got=%v", testName, results[0].Local(), want, results[1].Local().GoStr())
	}
}

func TestSliceXLA(t *testing.T) {
	manager := buildTestManager()
	g := manager.NewGraph("iota0")
	numbers := Iota(g, shapes.Make(shapes.F64, 9), 0)
	numbers = ReshapeWithShape(numbers, shapes.Make(shapes.F64, 3, 3))
	SliceXLA(numbers, []int{1, 1}, []int{2, 3})
	g.MustCompile()
	got := g.MustRun(nil).Local().Value()
	want := [][]float64{{4, 5}}
	if !slices.DeepSliceCmp(got, want, slices.Equal[float64]) {
		t.Fatalf("Iota: want %v, got %v", want, got)
	}
}

func TestGatherXLA(t *testing.T) {
	manager := buildTestManager()
	g := manager.NewGraph("iota0")
	// numbers=(Float64)[5 3]: [[0 1 2] [3 4 5] [6 7 8] [9 10 11] [12 13 14]]
	numbers := ReshapeWithShape(Iota(g, shapes.Make(shapes.F64, 5*3), 0), shapes.Make(shapes.F64, 5, 3))
	indices := Const(g, [][]int{{2}, {0}})
	gather := gatherXLA(numbers, indices, 1,
		/* offsetDims */ []int{1},
		/* collapsed_slice_dims */ []int{0},
		/* start_index_map */ []int{0},
		/* slice_sizes */ []int{1, 3}, false)
	g.MustCompile(gather)
	if g.Error() != nil {
		t.Fatalf("Failed to create graph: %v", g.Error())
	}
	got := g.MustRun(nil).Local()
	fmt.Printf("\tgatherXLA=%v\n", got)
	want := [][]float64{{6, 7, 8}, {0, 1, 2}}
	if !slices.DeepSliceCmp(got.Value(), want, slices.Equal[float64]) {
		t.Fatalf("gatherXLA: want %v, got %v", want, got)
	}
}

func TestSelectAndScatterWithGeneralPaddingXLA(t *testing.T) {
	testFuncOneInput(t, "selectAndScatterWithGeneralPaddingXLA()",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(shapes.Float64, 1, 6, 1))
			source := Add(IotaFull(g, shapes.Make(shapes.Float64, 1, 2, 1)), Const(g, 1.0))
			output = selectAndScatterWithGeneralPaddingXLA(input, source, []int{1, 3, 1}, []int{1, 3, 1}, nil)
			return
		}, [][][]float64{{{0}, {0}, {1}, {0}, {0}, {2}}})
}

func TestDotGeneralXLA(t *testing.T) {
	testFuncOneInput(t, "dotGeneralXLA(lhs=Iota([3,4]), rhs=0.1*Ones([3,4]))",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(shapes.F32, 3, 4))
			input = OnePlus(input)
			rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, 3, 4)), 0.1)
			output = dotGeneralXLA(input, []int{1}, []int{0}, rhs, []int{1}, []int{0})
			return
		}, []float32{1, 2.6, 4.2})

	testFuncOneInput(t, "dotGeneralXLA(lhs=Iota([3,2,4]), rhs=0.1*Ones([3,5,4]))",
		func(g *Graph) (input, output *Node) {
			input = IotaFull(g, shapes.Make(shapes.F32, 3, 2, 4))
			input = OnePlus(input)
			rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, 3, 5, 4)), 0.1)
			output = dotGeneralXLA(input, []int{2}, []int{0}, rhs, []int{2}, []int{0})
			return
		}, [][][]float32{
			{
				{1, 1, 1, 1, 1}, {2.6, 2.6, 2.6, 2.6, 2.6},
			}, {
				{4.2, 4.2, 4.2, 4.2, 4.2}, {5.8, 5.8, 5.8, 5.8, 5.8},
			}, {
				{7.4, 7.4, 7.4, 7.4, 7.4}, {9, 9, 9, 9, 9},
			}})
}

// allPermutations lists all permutations of a list with n elements. So for n=2, it returns [][]int{{0,1}, {1, 0}},
// and so on.
func allPermutations(n int) (permutations [][]int) {
	return allPermutationsRecursive(n, slices.Iota(0, n))
}

// allPermutationsRecursive is used by allPermutations to build all variations.
func allPermutationsRecursive(reserve int, remainder []int) (permutations [][]int) {
	if len(remainder) == 1 {
		// There is only one choice left, return it.
		result := make([]int, 0, reserve)
		result = append(result, remainder[0])
		return [][]int{result}
	}
	for ii, choice := range remainder {
		subRemainder := make([]int, 0, len(remainder)-1)
		subRemainder = append(subRemainder, remainder[:ii]...)
		subRemainder = append(subRemainder, remainder[ii+1:]...)
		tails := allPermutationsRecursive(reserve, subRemainder)
		if permutations == nil {
			permutations = make([][]int, 0, len(tails)*len(remainder)) // Create space for factorial number of elements.
		}
		for _, tail := range tails {
			permutation := append(tail, choice)
			permutations = append(permutations, permutation)
		}
	}
	return
}

// reversePermutation returns the permutation that reverses the given one.
func reversePermutation(permutation []int) []int {
	reverse := make([]int, len(permutation))
	for ii, jj := range permutation {
		reverse[jj] = ii
	}
	return reverse
}

func TestGradDotGeneralXLABatchContracting(t *testing.T) {
	manager := buildTestManager()

	dimensions := []int{2, 3, 4}
	lhsPermutations := allPermutations(len(dimensions))
	rhsPermutations := allPermutations(len(dimensions))

	// Test batch and contracting dimensions only. We are going to use
	// one batch axis and 2 contracting for the lhs (left-hand-side) and
	// (right-hand-side).
	//
	// We run one time per possible permutation (6 x 6 = 36 combinations in total).
	for _, lhsPermutation := range lhsPermutations {
		revLhsPermutation := reversePermutation(lhsPermutation)
		for _, rhsPermutation := range rhsPermutations {
			fmt.Println()
			fmt.Println()
			fmt.Println()
			revRhsPermutation := reversePermutation(rhsPermutation)
			testFn := func(g *Graph) []*Node { // It returns: lhs, rhs, dot, grad_lhs, grad_rhs
				lhs := IotaFull(g, shapes.Make(shapes.F32, dimensions...))
				lhs = OnePlus(lhs)
				lhs = TransposeAllDims(lhs, lhsPermutation...)
				rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, dimensions...)), 0.1)
				rhs = TransposeAllDims(rhs, rhsPermutation...)

				lhsBatchAxes := gatherSlice([]int{0}, revLhsPermutation)
				lhsContractingAxes := gatherSlice([]int{1, 2}, revLhsPermutation)
				fmt.Printf("\tlhs: p:%v, rev:%v, batch: %v, contracting: %v\n", lhsPermutation, revLhsPermutation, lhsBatchAxes, lhsContractingAxes)
				fmt.Printf("\t\tlhs.shape=%s\n", lhs.Shape())
				rhsBatchAxes := gatherSlice([]int{0}, revRhsPermutation)
				rhsContractingAxes := gatherSlice([]int{1, 2}, revRhsPermutation)
				fmt.Printf("\trhs: p:%v, rev:%v, batch: %v, contracting: %v\n", rhsPermutation, revRhsPermutation, rhsBatchAxes, rhsContractingAxes)
				fmt.Printf("\t\trhs.shape=%s\n", rhs.Shape())
				dot := dotGeneralXLA(lhs, lhsContractingAxes, lhsBatchAxes, rhs, rhsContractingAxes, rhsBatchAxes)
				require.Truef(t, g.Ok(), "TestGradDotGeneralXLA failed to generate DotGeneral: %+v", g.Error())
				fmt.Printf("\t\tdot.shape=%s\n", dot.Shape())
				// loss is the product of dot and iota (increasing numbers), all reduced sum.
				incremental := OnePlus(IotaFull(g, dot.Shape()))
				loss := ReduceAllSum(Mul(incremental, dot))
				grads := Gradient(loss, lhs, rhs)
				require.Truef(t, g.Ok(), "TestGradDotGeneralXLA failed to generate gradient: %+v", g.Error())
				grads[0] = TransposeAllDims(grads[0], revLhsPermutation...)
				grads[1] = TransposeAllDims(grads[1], revRhsPermutation...)
				return []*Node{lhs, rhs, dot, grads[0], grads[1]}
			}

			exec := NewExec(manager, testFn)
			parts, err := exec.Call()
			require.NoError(t, err, "Executing TestGradDotGeneralXLA failed")
			fmt.Printf("Executing TestGradDotGeneralXLA:\n")
			for ii, name := range []string{"lhs", "rhs", "dot", "grad_lhs", "grad_rhs"} {
				fmt.Printf("\t%s: %s\n", name, parts[ii].Local().GoStr())
			}

			// require.InDeltaSlice is not working for some reason.
			dot := parts[2].Value()
			wantDot := []float32{7.8, 22.2}
			require.Truef(t, slices.DeepSliceCmp(dot, wantDot, slices.Close[float32]), "DotGeneral results don't match, want %v", wantDot)
			gradLhs := parts[3].Value()
			wantGradLhs := [][][]float32{
				{{0.1, 0.1, 0.1, 0.1}, {0.1, 0.1, 0.1, 0.1}, {0.1, 0.1, 0.1, 0.1}},
				{{0.2, 0.2, 0.2, 0.2}, {0.2, 0.2, 0.2, 0.2}, {0.2, 0.2, 0.2, 0.2}},
			}
			require.Truef(t, slices.DeepSliceCmp(gradLhs, wantGradLhs, slices.Close[float32]),
				"Grad lhs (%v) doesn't match (%v)", gradLhs, wantGradLhs)
			gradRhs := parts[4].Value()
			wantGradRhs := [][][]float32{
				{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
				{{26, 28, 30, 32}, {34, 36, 38, 40}, {42, 44, 46, 48}},
			}
			require.Truef(t, slices.DeepSliceCmp(gradRhs, wantGradRhs, slices.Close[float32]),
				"Grad rhs (%v) doesn't match (%v)", gradRhs, wantGradRhs)
		}
	}
}

func TestGradDotGeneralXLABatchContractingCrossing(t *testing.T) {
	manager := buildTestManager()

	lhsDimensions := []int{4, 2, 5}
	rhsDimensions := []int{4, 3, 5}
	lhsPermutations := allPermutations(len(lhsDimensions))
	rhsPermutations := allPermutations(len(rhsDimensions))
	//lhsPermutations := [][]int{IotaSlice(0, len(lhsDimensions))}
	//rhsPermutations := [][]int{IotaSlice(0, len(rhsDimensions))}

	// Test batch, contraction and crosses axes simultaneously -- one of each.
	//
	// We run one time per possible permutation (6 x 6 = 36 combinations in total).
	for _, lhsPermutation := range lhsPermutations {
		revLhsPermutation := reversePermutation(lhsPermutation)
		for _, rhsPermutation := range rhsPermutations {
			fmt.Println()
			fmt.Println()
			revRhsPermutation := reversePermutation(rhsPermutation)
			testFn := func(g *Graph) []*Node { // It returns: lhs, rhs, dot, grad_lhs, grad_rhs
				lhs := IotaFull(g, shapes.Make(shapes.F32, lhsDimensions...))
				lhs = OnePlus(lhs)
				lhs = TransposeAllDims(lhs, lhsPermutation...)
				rhs := MulScalar(Ones(g, shapes.Make(shapes.F32, rhsDimensions...)), 0.1)
				rhs = TransposeAllDims(rhs, rhsPermutation...)

				lhsBatchAxes := gatherSlice([]int{0}, revLhsPermutation)
				lhsContractingAxes := gatherSlice([]int{2}, revLhsPermutation)
				fmt.Printf("\tlhs: p:%v, rev:%v, batch: %v, contracting: %v\n", lhsPermutation, revLhsPermutation, lhsBatchAxes, lhsContractingAxes)
				fmt.Printf("\t\tlhs.shape=%s\n", lhs.Shape())
				rhsBatchAxes := gatherSlice([]int{0}, revRhsPermutation)
				rhsContractingAxes := gatherSlice([]int{2}, revRhsPermutation)
				fmt.Printf("\trhs: p:%v, rev:%v, batch: %v, contracting: %v\n", rhsPermutation, revRhsPermutation, rhsBatchAxes, rhsContractingAxes)
				fmt.Printf("\t\trhs.shape=%s\n", rhs.Shape())
				dot := dotGeneralXLA(lhs, lhsContractingAxes, lhsBatchAxes, rhs, rhsContractingAxes, rhsBatchAxes)
				require.Truef(t, g.Ok(), "TestGradDotGeneralXLA failed to generate DotGeneral: %+v", g.Error())
				fmt.Printf("\t\tdot.shape=%s\n", dot.Shape())

				// loss is the product of dot and iota (increasing numbers), all reduced sum.
				incremental := OnePlus(IotaFull(g, dot.Shape()))
				loss := ReduceAllSum(Mul(incremental, dot))
				grads := Gradient(loss, lhs, rhs)
				require.Truef(t, g.Ok(), "TestGradDotGeneralXLA failed to generate gradient: %+v", g.Error())
				grads[0] = TransposeAllDims(grads[0], revLhsPermutation...)
				grads[1] = TransposeAllDims(grads[1], revRhsPermutation...)
				return []*Node{lhs, rhs, dot, grads[0], grads[1]}
			}

			exec := NewExec(manager, testFn)
			parts, err := exec.Call()
			require.NoError(t, err, "Executing TestGradDotGeneralXLA failed")
			fmt.Printf("Executing TestGradDotGeneralXLA:\n")
			for ii, name := range []string{"lhs", "rhs", "dot", "grad_lhs", "grad_rhs"} {
				fmt.Printf("\t%s: %s\n", name, parts[ii].Local().GoStr())
			}

			// require.InDeltaSlice is not working for some reason.
			dot := parts[2].Value()
			wantDot := [][][]float32{
				{{1.5, 1.5, 1.5}, {4, 4, 4}},
				{{6.5, 6.5, 6.5}, {9, 9, 9}},
				{{11.5, 11.5, 11.5}, {14, 14, 14}},
				{{16.5, 16.5, 16.5}, {19, 19, 19}},
			}
			require.Truef(t, slices.DeepSliceCmp(dot, wantDot, slices.Close[float32]), "DotGeneral results don't match, want %v", wantDot)
			gradLhs := parts[3].Value()
			wantGradLhs := [][][]float32{
				{{0.6, 0.6, 0.6, 0.6, 0.6}, {1.5, 1.5, 1.5, 1.5, 1.5}},
				{{2.4, 2.4, 2.4, 2.4, 2.4}, {3.3, 3.3, 3.3, 3.3, 3.3}},
				{{4.2, 4.2, 4.2, 4.2, 4.2}, {5.1, 5.1, 5.1, 5.1, 5.1}},
				{{6, 6, 6, 6, 6}, {6.9, 6.9, 6.9, 6.9, 6.9}},
			}
			require.Truef(t, slices.DeepSliceCmp(gradLhs, wantGradLhs, slices.Close[float32]),
				"Grad lhs (%v) doesn't match (%v)", gradLhs, wantGradLhs)
			gradRhs := parts[4].Value()
			wantGradRhs := [][][]float32{
				{{25, 30, 35, 40, 45}, {32, 39, 46, 53, 60}, {39, 48, 57, 66, 75}},
				{{237, 254, 271, 288, 305}, {264, 283, 302, 321, 340}, {291, 312, 333, 354, 375}},
				{{689, 718, 747, 776, 805}, {736, 767, 798, 829, 860}, {783, 816, 849, 882, 915}},
				{{1381, 1422, 1463, 1504, 1545}, {1448, 1491, 1534, 1577, 1620}, {1515, 1560, 1605, 1650, 1695}},
			}
			require.Truef(t, slices.DeepSliceCmp(gradRhs, wantGradRhs, slices.Close[float32]),
				"Grad rhs (%v) doesn't match (%v)", gradRhs, wantGradRhs)
		}
	}
}
