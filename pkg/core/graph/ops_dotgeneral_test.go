package graph_test

import (
	"fmt"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/dtypes"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/stretchr/testify/require"
)

func TestDot(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	t.Run("Product", func(t *testing.T) {
		g := NewGraph(backend, t.Name())

		// Shape: [batch=4, dims=3]
		inputs := Const(g, [][]float32{{1.1, 2.2, 3.3}, {11, 22, 33}, {111, 222, 333}, {1111, 2222, 3333}})
		// Layer 0: outputShapes [3, 2], that is the inputNodes have dim=3, and should output dims=2
		w0 := Const(g, [][]float32{{1, 0}, {1, -1}, {-1, 1}})
		// DotProduct(inputNodes, w0) -> outputShapes [batch=4, dims=2]
		Dot(inputs, w0).Product() // The last node created in the graph is taken as output by default.
		got := compileRunAndTakeFirst(t, g)
		want := tensors.FromValue([][]float32{{0, 1.1}, {0, 11}, {0, 111}, {0, 1111}})
		if !want.InDelta(got, Epsilon) {
			fmt.Printf("%s\n", g)
			fmt.Printf("\tResult=%v\n", got)
			t.Errorf("Wanted %v, got %v", want, got)
		}
	})

	t.Run("Einsum", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "MatrixMul",
			func(g *Graph) (inputs, outputs []*Node) {
				lhs := IotaFull(g, shapes.Make(dtypes.Float32, 2, 4))
				lhs = OnePlus(lhs)
				rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 4, 3)), 0.1)
				inputs = []*Node{lhs, rhs}
				outputs = []*Node{Einsum("ij,jk->ik", lhs, rhs)}
				return
			}, []any{[][]float32{{1, 1, 1}, {2.6, 2.6, 2.6}}}, Epsilon)
		graphtest.RunTestGraphFn(t, "DotProduct",
			func(g *Graph) (inputs, outputs []*Node) {
				lhs := IotaFull(g, shapes.Make(dtypes.Float32, 4))
				lhs = OnePlus(lhs)
				rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 4)), 0.1)
				inputs = []*Node{lhs, rhs}
				outputs = []*Node{Einsum("i,i->", lhs, rhs)}
				return
			}, []any{float32(1)}, Epsilon)
		graphtest.RunTestGraphFn(t, "OuterProduct",
			func(g *Graph) (inputs, outputs []*Node) {
				lhs := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 4)))
				rhs := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 3)))
				inputs = []*Node{lhs, rhs}
				outputs = []*Node{Einsum("i,j->ij", lhs, rhs)}
				return
			}, []any{[][]float32{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}, {4, 8, 12}}}, Epsilon)
		graphtest.RunTestGraphFn(t, "BatchMatrixMul",
			func(g *Graph) (inputs, outputs []*Node) {
				lhs := IotaFull(g, shapes.Make(dtypes.Float32, 5, 2, 4))
				lhs = OnePlus(lhs)
				rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 5, 4, 3)), 0.1)
				inputs = []*Node{lhs, rhs}
				outputs = []*Node{Einsum("bij,bjk->bik", lhs, rhs)}
				return
			}, []any{[][][]float32{
				{{1, 1, 1}, {2.6, 2.6, 2.6}},
				{{4.2, 4.2, 4.2}, {5.8, 5.8, 5.8}},
				{{7.4, 7.4, 7.4}, {9, 9, 9}},
				{{10.6, 10.6, 10.6}, {12.2, 12.2, 12.2}},
				{{13.8, 13.8, 13.8}, {15.4, 15.4, 15.4}}},
			}, Epsilon)
		graphtest.RunTestGraphFn(t, "BatchMatrixMulTransposed",
			func(g *Graph) (inputs, outputs []*Node) {
				lhs := IotaFull(g, shapes.Make(dtypes.Float32, 5, 2, 4))
				lhs = OnePlus(lhs)
				rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 5, 4, 3)), 0.1)
				inputs = []*Node{lhs, rhs}
				outputs = []*Node{Einsum("bij,bjk->ikb", lhs, rhs)}
				return
			}, []any{[][][]float32{
				{{1, 4.2, 7.4, 10.6, 13.8},
					{1, 4.2, 7.4, 10.6, 13.8},
					{1, 4.2, 7.4, 10.6, 13.8}},
				{{2.6, 5.8, 9, 12.2, 15.4},
					{2.6, 5.8, 9, 12.2, 15.4},
					{2.6, 5.8, 9, 12.2, 15.4}},
			}}, Epsilon)
	})

	t.Run("EinsumAxes", func(t *testing.T) {
		graphtest.RunTestGraphFn(t, "MatrixMul",
			func(g *Graph) (inputs, outputs []*Node) {
				lhs := IotaFull(g, shapes.Make(dtypes.Float32, 2, 4))
				lhs = OnePlus(lhs)
				rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 4, 3)), 0.1)
				inputs = []*Node{lhs, rhs}
				outputs = []*Node{EinsumAxes(lhs, rhs, [][2]int{{1, 0}}, nil)}
				return
			}, []any{[][]float32{{1, 1, 1}, {2.6, 2.6, 2.6}}}, Epsilon)
		graphtest.RunTestGraphFn(t, "DotProduct",
			func(g *Graph) (inputs, outputs []*Node) {
				lhs := IotaFull(g, shapes.Make(dtypes.Float32, 4))
				lhs = OnePlus(lhs)
				rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 4)), 0.1)
				inputs = []*Node{lhs, rhs}
				outputs = []*Node{EinsumAxes(lhs, rhs, [][2]int{{0, 0}}, nil)}
				return
			}, []any{float32(1)}, Epsilon)
		graphtest.RunTestGraphFn(t, "OuterProduct",
			func(g *Graph) (inputs, outputs []*Node) {
				lhs := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 4)))
				rhs := OnePlus(IotaFull(g, shapes.Make(dtypes.Float32, 3)))
				inputs = []*Node{lhs, rhs}
				outputs = []*Node{EinsumAxes(lhs, rhs, nil, nil)}
				return
			}, []any{[][]float32{{1, 2, 3}, {2, 4, 6}, {3, 6, 9}, {4, 8, 12}}}, Epsilon)
		graphtest.RunTestGraphFn(t, "BatchMatrixMul",
			func(g *Graph) (inputs, outputs []*Node) {
				lhs := IotaFull(g, shapes.Make(dtypes.Float32, 5, 2, 4))
				lhs = OnePlus(lhs)
				rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 5, 4, 3)), 0.1)
				inputs = []*Node{lhs, rhs}
				outputs = []*Node{EinsumAxes(lhs, rhs, [][2]int{{2, 1}}, [][2]int{{0, 0}})}
				return
			}, []any{[][][]float32{
				{{1, 1, 1}, {2.6, 2.6, 2.6}},
				{{4.2, 4.2, 4.2}, {5.8, 5.8, 5.8}},
				{{7.4, 7.4, 7.4}, {9, 9, 9}},
				{{10.6, 10.6, 10.6}, {12.2, 12.2, 12.2}},
				{{13.8, 13.8, 13.8}, {15.4, 15.4, 15.4}}},
			}, Epsilon)
	})

	t.Run("General", func(t *testing.T) {
		testFuncOneInput(t, "Dot.(lhs=Iota([3,4]), rhs=0.1*Ones([3,4])).General()",
			func(g *Graph) (input, output *Node) {
				input = IotaFull(g, shapes.Make(dtypes.Float32, 3, 4))
				input = OnePlus(input)
				rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 3, 4)), 0.1)
				output = DotGeneral(input, []int{1}, []int{0}, rhs, []int{1}, []int{0})
				return
			}, []float32{1, 2.6, 4.2})

		testFuncOneInput(t, "Dot.(lhs=Iota([3,2,4]), rhs=0.1*Ones([3,5,4])).General()",
			func(g *Graph) (input, output *Node) {
				input = IotaFull(g, shapes.Make(dtypes.Float32, 3, 2, 4))
				input = OnePlus(input)
				rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, 3, 5, 4)), 0.1)
				output = DotGeneral(input, []int{2}, []int{0}, rhs, []int{2}, []int{0})
				return
			}, [][][]float32{
				{
					{1, 1, 1, 1, 1}, {2.6, 2.6, 2.6, 2.6, 2.6},
				}, {
					{4.2, 4.2, 4.2, 4.2, 4.2}, {5.8, 5.8, 5.8, 5.8, 5.8},
				}, {
					{7.4, 7.4, 7.4, 7.4, 7.4}, {9, 9, 9, 9, 9},
				}})
	})
}

// allPermutations lists all permutations of a list with n elements. So for n=2, it returns [][]int{{0,1}, {1, 0}},
// and so on.
func allPermutations(n int) (permutations [][]int) {
	return allPermutationsRecursive(n, xslices.Iota(0, n))
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

func TestGradientDot(t *testing.T) {
	graphtest.RunTestGraphFn(t, "dot(vector,vector)", func(g *Graph) (inputs, outputs []*Node) {
		v1 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(2)))
		v2 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(3)))
		output := Dot(v1, v2).Product()
		gradients := Gradient(output, v1, v2)
		inputs = []*Node{v1, v2}
		outputs = append([]*Node{output}, gradients...)
		return
	}, []any{
		float32(24),           // dot product output
		[]float32{3, 3, 3, 3}, // gradient with respect to v1
		[]float32{2, 2, 2, 2}, // gradient with respect to v2
	}, Epsilon)

	graphtest.RunTestGraphFn(t, "dot(matrix,vector)", func(g *Graph) (inputs, outputs []*Node) {
		v1 := Add(Iota(g, MakeShape(F32, 2, 4), 0), Const(g, float32(2)))
		v2 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(3)))
		output := Dot(v1, v2).Product()
		sum := ReduceAllSum(output)
		gradients := Gradient(sum, v1, v2)
		inputs = []*Node{v1, v2}
		outputs = append([]*Node{output}, gradients...)
		return
	}, []any{
		[]float32{24, 36},                       // dot product output
		[][]float32{{3, 3, 3, 3}, {3, 3, 3, 3}}, // gradient with respect to v1
		[]float32{5, 5, 5, 5},                   // gradient with respect to v2
	}, Epsilon)

	graphtest.RunTestGraphFn(t, "dot(matrix,matrix)", func(g *Graph) (inputs, outputs []*Node) {
		v1 := Add(Iota(g, MakeShape(F32, 2, 4), 0), Const(g, float32(2)))
		v2 := Add(Iota(g, MakeShape(F32, 4, 1), 0), Const(g, float32(1)))
		output := Dot(v1, v2).Product()
		require.NoError(t, output.Shape().CheckDims(2, 1))
		sum := ReduceAllSum(output)
		gradients := Gradient(sum, v1, v2)
		inputs = []*Node{v1, v2}
		outputs = append([]*Node{output}, gradients...)
		return
	}, []any{
		[][]float32{{20}, {30}},                 // dot product output
		[][]float32{{1, 2, 3, 4}, {1, 2, 3, 4}}, // gradient with respect to v1
		[][]float32{{5}, {5}, {5}, {5}},         // gradient with respect to v2
	}, Epsilon)

	// backend for the other tests.
	backend := graphtest.BuildTestBackend()

	t.Run("General()-batch-contracting", func(t *testing.T) {
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
				revRhsPermutation := reversePermutation(rhsPermutation)
				testFn := func(g *Graph) []*Node { // It returns: lhs, rhs, dot, grad_lhs, grad_rhs
					lhs := IotaFull(g, shapes.Make(dtypes.Float32, dimensions...))
					lhs = OnePlus(lhs)
					lhs = TransposeAllAxes(lhs, lhsPermutation...)
					rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, dimensions...)), 0.1)
					rhs = TransposeAllAxes(rhs, rhsPermutation...)

					lhsBatchAxes := xslices.Gather(revLhsPermutation, []int{0})
					lhsContractingAxes := xslices.Gather(revLhsPermutation, []int{1, 2})
					fmt.Printf("\t\tlhs: p:%v, rev:%v, batch: %v, contracting: %v\n", lhsPermutation, revLhsPermutation, lhsBatchAxes, lhsContractingAxes)
					fmt.Printf("\tlhs.shape=%s\n", lhs.Shape())
					rhsBatchAxes := xslices.Gather(revRhsPermutation, []int{0})
					rhsContractingAxes := xslices.Gather(revRhsPermutation, []int{1, 2})
					fmt.Printf("\t\trhs: p:%v, rev:%v, batch: %v, contracting: %v\n", rhsPermutation, revRhsPermutation, rhsBatchAxes, rhsContractingAxes)
					fmt.Printf("\trhs.shape=%s\n", rhs.Shape())
					dot := DotGeneral(lhs, lhsContractingAxes, lhsBatchAxes, rhs, rhsContractingAxes, rhsBatchAxes)
					fmt.Printf("\t\tdot.shape=%s\n", dot.Shape())
					// loss is the product of dot and iota (increasing numbers), all reduced sum.
					incremental := OnePlus(IotaFull(g, dot.Shape()))
					loss := ReduceAllSum(Mul(incremental, dot))
					grads := Gradient(loss, lhs, rhs)
					grads[0] = TransposeAllAxes(grads[0], revLhsPermutation...)
					grads[1] = TransposeAllAxes(grads[1], revRhsPermutation...)
					fmt.Printf("\tDone graph\n")
					return []*Node{lhs, rhs, dot, grads[0], grads[1]}
				}

				exec := MustNewExec(backend, testFn)
				fmt.Printf("Executing GradDotGeneralBatchContracting:\n")
				parts := exec.MustExec()
				for ii, name := range []string{"lhs", "rhs", "dot", "grad_lhs", "grad_rhs"} {
					fmt.Printf("\t%s: %s\n", name, parts[ii].GoStr())
				}

				// require.InDeltaSlice is not working for some reason.
				dot := parts[2].Value()
				wantDot := []float32{7.8, 22.2}
				require.Truef(t, xslices.DeepSliceCmp(dot, wantDot, xslices.Close[float32]), "DotGeneral results don't match, want %v", wantDot)
				gradLhs := parts[3].Value()
				wantGradLhs := [][][]float32{
					{{0.1, 0.1, 0.1, 0.1}, {0.1, 0.1, 0.1, 0.1}, {0.1, 0.1, 0.1, 0.1}},
					{{0.2, 0.2, 0.2, 0.2}, {0.2, 0.2, 0.2, 0.2}, {0.2, 0.2, 0.2, 0.2}},
				}
				require.Truef(t, xslices.DeepSliceCmp(gradLhs, wantGradLhs, xslices.Close[float32]),
					"Grad lhs (%v) doesn't match (%v)", gradLhs, wantGradLhs)
				gradRhs := parts[4].Value()
				wantGradRhs := [][][]float32{
					{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}},
					{{26, 28, 30, 32}, {34, 36, 38, 40}, {42, 44, 46, 48}},
				}
				require.Truef(t, xslices.DeepSliceCmp(gradRhs, wantGradRhs, xslices.Close[float32]),
					"Grad rhs (%v) doesn't match (%v)", gradRhs, wantGradRhs)
			}
		}
	})

	t.Run("GeneralBatchContractingCrossing", func(t *testing.T) {
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
					lhs := IotaFull(g, shapes.Make(dtypes.Float32, lhsDimensions...))
					lhs = OnePlus(lhs)
					lhs = TransposeAllAxes(lhs, lhsPermutation...)
					rhs := MulScalar(Ones(g, shapes.Make(dtypes.Float32, rhsDimensions...)), 0.1)
					rhs = TransposeAllAxes(rhs, rhsPermutation...)

					lhsBatchAxes := xslices.Gather(revLhsPermutation, []int{0})
					lhsContractingAxes := xslices.Gather(revLhsPermutation, []int{2})
					fmt.Printf("\t\tlhs: p:%v, rev:%v, batch: %v, contracting: %v\n", lhsPermutation, revLhsPermutation, lhsBatchAxes, lhsContractingAxes)
					fmt.Printf("\tlhs.shape=%s\n", lhs.Shape())
					rhsBatchAxes := xslices.Gather(revRhsPermutation, []int{0})
					rhsContractingAxes := xslices.Gather(revRhsPermutation, []int{2})
					fmt.Printf("\t\trhs: p:%v, rev:%v, batch: %v, contracting: %v\n", rhsPermutation, revRhsPermutation, rhsBatchAxes, rhsContractingAxes)
					fmt.Printf("\trhs.shape=%s\n", rhs.Shape())
					dot := DotGeneral(lhs, lhsContractingAxes, lhsBatchAxes, rhs, rhsContractingAxes, rhsBatchAxes)
					fmt.Printf("\tdot.shape=%s\n", dot.Shape())

					// loss is the product of dot and iota (increasing numbers), all reduced sum.
					incremental := OnePlus(IotaFull(g, dot.Shape()))
					loss := ReduceAllSum(Mul(incremental, dot))
					grads := Gradient(loss, lhs, rhs)
					grads[0] = TransposeAllAxes(grads[0], revLhsPermutation...)
					grads[1] = TransposeAllAxes(grads[1], revRhsPermutation...)
					return []*Node{lhs, rhs, dot, grads[0], grads[1]}
				}

				exec := MustNewExec(backend, testFn)
				parts := exec.MustExec()
				fmt.Printf("Executing TestGradDotGeneral:\n")
				for ii, name := range []string{"lhs", "rhs", "dot", "grad_lhs", "grad_rhs"} {
					fmt.Printf("\t%s: %s\n", name, parts[ii].GoStr())
				}

				// require.InDeltaSlice is not working for some reason.
				dot := parts[2].Value()
				wantDot := [][][]float32{
					{{1.5, 1.5, 1.5}, {4, 4, 4}},
					{{6.5, 6.5, 6.5}, {9, 9, 9}},
					{{11.5, 11.5, 11.5}, {14, 14, 14}},
					{{16.5, 16.5, 16.5}, {19, 19, 19}},
				}
				require.Truef(t, xslices.DeepSliceCmp(dot, wantDot, xslices.Close[float32]), "DotGeneral results don't match, want %v", wantDot)
				gradLhs := parts[3].Value()
				wantGradLhs := [][][]float32{
					{{0.6, 0.6, 0.6, 0.6, 0.6}, {1.5, 1.5, 1.5, 1.5, 1.5}},
					{{2.4, 2.4, 2.4, 2.4, 2.4}, {3.3, 3.3, 3.3, 3.3, 3.3}},
					{{4.2, 4.2, 4.2, 4.2, 4.2}, {5.1, 5.1, 5.1, 5.1, 5.1}},
					{{6, 6, 6, 6, 6}, {6.9, 6.9, 6.9, 6.9, 6.9}},
				}
				require.Truef(t, xslices.DeepSliceCmp(gradLhs, wantGradLhs, xslices.Close[float32]),
					"Grad lhs (%v) doesn't match (%v)", gradLhs, wantGradLhs)
				gradRhs := parts[4].Value()
				wantGradRhs := [][][]float32{
					{{25, 30, 35, 40, 45}, {32, 39, 46, 53, 60}, {39, 48, 57, 66, 75}},
					{{237, 254, 271, 288, 305}, {264, 283, 302, 321, 340}, {291, 312, 333, 354, 375}},
					{{689, 718, 747, 776, 805}, {736, 767, 798, 829, 860}, {783, 816, 849, 882, 915}},
					{{1381, 1422, 1463, 1504, 1545}, {1448, 1491, 1534, 1577, 1620}, {1515, 1560, 1605, 1650, 1695}},
				}
				require.Truef(t, xslices.DeepSliceCmp(gradRhs, wantGradRhs, xslices.Close[float32]),
					"Grad rhs (%v) doesn't match (%v)", gradRhs, wantGradRhs)
			}
		}
	})

}

func TestGradientDotConfig(t *testing.T) {
	graphtest.RunTestGraphFn(t, "dot_with_accumulator_dtype", func(g *Graph) (inputs, outputs []*Node) {
		// lhs and rhs are Float32.
		// We set AccumulatorDType to Float64.
		// We explicitly set OutputDType to Float32 to check that VJP handles the mismatch between
		// computation precision (F64) and adjoint/output precision (F32).
		v1 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(2)))
		v2 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(3)))

		// Dot product with F64 accumulator.
		output := Dot(v1, v2).WithAccumulatorDType(dtypes.Float64).WithOutputDType(dtypes.Float32).Product()

		// Calculate gradients.
		gradients := Gradient(output, v1, v2)

		inputs = []*Node{v1, v2}
		outputs = append([]*Node{output}, gradients...)
		return
	}, []any{
		float32(24),           // dot product output (Float32)
		[]float32{3, 3, 3, 3}, // gradient with respect to v1 (Float32)
		[]float32{2, 2, 2, 2}, // gradient with respect to v2 (Float32)
	}, Epsilon)

	graphtest.RunTestGraphFn(t, "dot_with_output_dtype", func(g *Graph) (inputs, outputs []*Node) {
		// lhs and rhs are Float32.
		// We set OutputDType to Float64.
		v1 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(2)))
		v2 := Mul(Ones(g, MakeShape(F32, 4)), Const(g, float32(3)))

		// Dot product with F64 output.
		output := Dot(v1, v2).WithOutputDType(dtypes.Float64).Product()

		// Calculate gradients.
		gradients := Gradient(output, v1, v2)

		inputs = []*Node{v1, v2}
		outputs = append([]*Node{output}, gradients...)
		return
	}, []any{
		float64(24),           // dot product output (Float64)
		[]float32{3, 3, 3, 3}, // gradient with respect to v1 (Float32)
		[]float32{2, 2, 2, 2}, // gradient with respect to v2 (Float32)
	}, Epsilon)
}
