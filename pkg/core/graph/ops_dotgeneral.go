package graph

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

// DotBuilder is a builder for the DotGeneral operation.
type DotBuilder struct {
	lhs    *Node
	rhs    *Node
	config backends.DotGeneralConfig
}

// Dot returns a builder for a generic, highly configurable "DotGeneral" operation.
// It returns a DotBuilder object, and once configured, one can call:
//
//   - Product(): the simplest form, it performs dot product of vectors, or vector-matrix/matrix-vector or simple
//     matrix multiplications, for lhs/rhs ranks <= 2. Similar to numpy.dot.
//   - MatMul(): it assumes a batch dimension to the lhs.
//   - Einsum(): it performs a generic einsum operation, given an equation.
//   - DotGeneral(): it performs an arbitrary "DotGeneral" operation, where contracting and batch axes can be selected,
//     the remainder axes are assumed to be cross-product.
func Dot(lhs, rhs *Node) *DotBuilder {
	return &DotBuilder{
		lhs: lhs,
		rhs: rhs,
	}
}

// WithAccumulatorDType sets the accumulator data type for the DotGeneral operation.
//
// Support for accumulator dtypes is very backend dependent. For example, XLA only supports
// mixed dtypes if accumulation is in F32.
//
// GoMLX's Dot() operation will attempt to recover from an error and pre-convert the operands (lhs, rhs)
// to the accumulator dtype if the backend yields a "not-implemented" error.
func (b *DotBuilder) WithAccumulatorDType(dtype dtypes.DType) *DotBuilder {
	b.config.AccumulatorDType = dtype
	return b
}

// WithOutputDType sets the output data type for the DotGeneral operation.
//
// Some combinations of input/accumulator/output dtypes may be invalid and yield errors -- in which case simply
// use ConvertDType after the DotGeneral operation.
func (b *DotBuilder) WithOutputDType(dtype dtypes.DType) *DotBuilder {
	b.config.OutputDType = dtype
	return b
}

// DotProduct performs a generic linear dot-product operation, extended to handle matrices (rank-2 lhs and/or rhs),
// similar to numpy.dot.
//
// The exact semantics of this operation depend on the ranks of the operands:
//
//   - DotProduct([n], [n]) -> scalar
//   - DotProduct([m, k], [k]) -> vector [m]
//   - DotProduct([k], [k, n]) -> vector [n]
//   - DotProduct([m, k], [k, n]) -> matrix [m x n]
//
// This is an alias for Dot(lhs, rhs).Product().
// Use Dot() instead if you need further configuration.
func DotProduct(lhs, rhs *Node) *Node {
	return Dot(lhs, rhs).Product()
}

// Product performs a generic linear dot-product operation, extended to handle matrices (rank-2 lhs and/or rhs),
// similar to numpy.dot.
//
// The exact semantics of this operation depend on the ranks of the operands:
//
//   - Dot([n], [n]).Product() -> scalar
//   - Dot([m, k], [k]).Product() -> vector [m]
//   - Dot([k], [k, n]).Product() -> vector [n]
//   - Dot([m, k], [k, n]).Product() -> matrix [m x n]
func (b *DotBuilder) Product() *Node {
	var lhsContractingAxes, rhsContractingAxes []int
	switch {
	case b.lhs.Rank() == 1 && b.rhs.Rank() == 1:
		// Contracting both vectors.
		lhsContractingAxes = []int{0}
		rhsContractingAxes = []int{0}
	case b.lhs.Rank() == 2 && b.rhs.Rank() == 1:
		// Contract rhs vector and lhs columns (axis 1).
		lhsContractingAxes = []int{1}
		rhsContractingAxes = []int{0}
	case b.lhs.Rank() == 1 && b.rhs.Rank() == 2:
		// Contract lhs vector and rhs rows (axis 0).
		lhsContractingAxes = []int{0}
		rhsContractingAxes = []int{0}
	case b.lhs.Rank() == 2 && b.rhs.Rank() == 2:
		// Traditional matrix multiplication:
		lhsContractingAxes = []int{1}
		rhsContractingAxes = []int{0}
	default:
		exceptions.Panicf(
			"Default (simple) Dot() requires matrices or vectors as input, got invalid ranks: lhs=%v, rhs=%v",
			b.lhs.Shape(), b.rhs.Shape())
	}
	return b.General(lhsContractingAxes, nil, rhsContractingAxes, nil)
}

// General takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension, and finally the 'rhs' non-contracting/non-batch dimension.
// It provides the basic means of implementing Einsum.
func (b *DotBuilder) General(
	lhsContractingAxes, lhsBatchAxes []int,
	rhsContractingAxes, rhsBatchAxes []int,
) *Node {
	_ = validateBuildingGraphFromInputs(b.lhs, b.rhs)
	lhsContractingAxes = adjustAxesToRank(b.lhs.Rank(), lhsContractingAxes, "lhsContractingAxes")
	lhsBatchAxes = adjustAxesToRank(b.lhs.Rank(), lhsBatchAxes, "lhsBatchAxes")
	rhsContractingAxes = adjustAxesToRank(b.rhs.Rank(), rhsContractingAxes, "rhsContractingAxes")
	rhsBatchAxes = adjustAxesToRank(b.rhs.Rank(), rhsBatchAxes, "rhsBatchAxes")

	var output *Node
	err := exceptions.TryCatch[error](func() {
		output = backendDotGeneral(b.lhs, lhsContractingAxes, lhsBatchAxes, b.rhs, rhsContractingAxes, rhsBatchAxes,
			b.config)
	})
	if err == nil {
		return output
	}
	if !backends.IsNotImplemented(err) || (b.config.AccumulatorDType == 0 && b.config.OutputDType == 0) {
		panic(err)
	}

	// Decompose the conversion of accumulator and output dtypes.
	lhs, rhs := b.lhs, b.rhs
	if b.config.AccumulatorDType != 0 {
		lhs = ConvertDType(b.lhs, b.config.AccumulatorDType)
		rhs = ConvertDType(b.rhs, b.config.AccumulatorDType)
	}
	output = backendDotGeneral(lhs, lhsContractingAxes, lhsBatchAxes, rhs, rhsContractingAxes, rhsBatchAxes,
		backends.DotGeneralConfig{})

	outputDType := b.lhs.DType() // Default to the same as the input.
	if b.config.OutputDType != 0 {
		outputDType = b.config.OutputDType
	}
	// Conversion may happen if accumulatorDType != outputDType. It's a no-op if there is no conversion.
	output = ConvertDType(output, outputDType)
	return output
}

// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//   - Just aligned (batch axes), so the output has the same axes as the inputs. The dimensions
//     must match in lhs and rhs.
//   - Crossed (default), in which case the output is the combination (concatenation) of the
//     dimensions.
//   - Contracted (contracting axes), where the output does multiply the values and reduce sum
//     those dimensions.
//
// It follows that the resulting dimension number starts with the batch dimension, then the 'lhs'
// non-contracting/non-batch dimension, and finally the 'rhs' non-contracting/non-batch dimension.
// It provides the basic means of implementing Einsum.
//
// It's an alias to Dot(lhs, rhs).General(lhsContractingAxes, lhsBatchAxes, rhsContractingAxes, rhsBatchAxes).
// Use Dot() instead if you need further configuration.
func DotGeneral(
	lhs *Node,
	lhsContractingAxes, lhsBatchAxes []int,
	rhs *Node,
	rhsContractingAxes, rhsBatchAxes []int,
) *Node {
	return Dot(lhs, rhs).General(lhsContractingAxes, lhsBatchAxes, rhsContractingAxes, rhsBatchAxes)
}

// MatMul is the `numpy.matmul` equivalent, for those used to that.
//
// It is similar to Dot but extends to allow for more batch dimensions in lhs or rhs operand, and
// does broadcasting (of all but the last 2 axes) according to the numpy broadcasting rules.
//
// It's popular hence it is here, but full of edge cases, consider using Dot().General() instead.
func (b *DotBuilder) MatMul() *Node {
	lhs, rhs := b.lhs, b.rhs
	_ = validateBuildingGraphFromInputs(lhs, rhs)
	if lhs.Rank() == 0 || rhs.Rank() == 0 {
		exceptions.Panicf("MatMul expects two tensors with rank > 0, got ranks %d and %d", lhs.Rank(), rhs.Rank())
	}
	if lhs.Rank() <= 2 && rhs.Rank() <= 2 {
		return b.Product()
	}

	// Special case when one of operands is a vector.
	if lhs.Rank() == 1 {
		return DotGeneral(lhs, []int{0}, nil, rhs, []int{rhs.Rank() - 2}, nil)
	}
	if rhs.Rank() == 1 {
		return DotGeneral(lhs, []int{lhs.Rank() - 1}, nil, rhs, []int{0}, nil)
	}

	// Trivial and most common case: right-hand-side is simply a linear transformation (matrix) on the last axis of lhs.
	if rhs.Rank() == 2 {
		return DotGeneral(lhs, []int{lhs.Rank() - 1}, nil, rhs, []int{rhs.Rank() - 2}, nil)
	}

	// Generic version, that will include broadcasting: we will use Einsum (and not DotGeneral) because it will do
	// the final transposing of the axes, where needed.
	//
	// . All axes before the last 2 are "batch":
	// . If one of the axes is not present, it should be effectively broadcast on the other:
	// . Batch axes appear first, then axis lhs[rank-2] and then rhs[rank-1].
	const lhsIdx, rhsIdx = 0, 1
	rhsRemap := make(map[int]int)              // Maps a rhs axis to match a lhs axis.
	rhsRemap[rhs.Rank()-2] = lhs.Rank() - 1    // The contracting axes.
	var lhsSqueezedAxes, rhsSqueezedAxes []int // Axes with dimension 1 that will be broadcast, they are squeezed away.
	letterForAxis := func(side int, axis int) string {
		var offset int
		if side == lhsIdx {
			if slices.Index(lhsSqueezedAxes, axis) >= 0 {
				// Axis has been dropped.
				return ""
			}
		} else {
			if slices.Index(rhsSqueezedAxes, axis) >= 0 {
				// Axis has been dropped.
				return ""
			}
			if lhsAxis, found := rhsRemap[axis]; found {
				// Use the lhs axis letter for the rhsAxis: either they are contracting or they are a batch dimension.
				axis = lhsAxis
				offset = 0
			} else {
				offset = lhs.Rank()
			}
		}
		return string('a' + rune(offset+axis))
	}
	var outputAxesLetters strings.Builder

	var lhsBatchAxes, rhsBatchAxes []int
	minRank := min(rhs.Rank(), lhs.Rank())
	for axis := lhs.Rank() - minRank; axis < lhs.Rank()-2; axis++ {
		lhsBatchAxes = append(lhsBatchAxes, axis)
	}
	for axis := rhs.Rank() - minRank; axis < rhs.Rank()-2; axis++ {
		rhsBatchAxes = append(rhsBatchAxes, axis)
	}

	// First axes of the output are the "batch" axes not present in the other side.
	// Only one of the two for-loop belows will run.
	for axis := range lhs.Rank() - minRank {
		outputAxesLetters.WriteString(letterForAxis(lhsIdx, axis))
	}
	for axis := range rhs.Rank() - minRank {
		outputAxesLetters.WriteString(letterForAxis(rhsIdx, axis))
	}

	// Process common batch axes:
	for idx := range len(lhsBatchAxes) {
		leftAxis := lhsBatchAxes[idx]
		rightAxis := rhsBatchAxes[idx]
		leftAxisDim := lhs.Shape().Dimensions[leftAxis]
		rightAxisDim := rhs.Shape().Dimensions[rightAxis]
		if leftAxisDim == rightAxisDim {
			// Same batch axis on both sides:
			rhsRemap[rightAxis] = leftAxis
			outputAxesLetters.WriteString(letterForAxis(lhsIdx, leftAxis))
			continue
		}
		if leftAxisDim != 1 && rightAxisDim != 1 {
			exceptions.Panicf("MatMul cannot match batch dimensions of lhs (left-hand-side) axis #%d (dim=%d) "+
				" and rhs (right-hand-side) axis #%d (dim=%d), for lhs.shape=%s and rhs.shape=%s",
				leftAxis, leftAxisDim, rightAxis, rightAxisDim, lhs.Shape(), rhs.Shape())
		}
		if leftAxisDim == 1 {
			lhsSqueezedAxes = append(lhsSqueezedAxes, leftAxis)
			outputAxesLetters.WriteString(letterForAxis(rhsIdx, rightAxis))
		} else { // rightAxisDim == 1
			rhsSqueezedAxes = append(rhsSqueezedAxes, rightAxis)
			outputAxesLetters.WriteString(letterForAxis(lhsIdx, leftAxis))
		}
	}

	// Final output axes
	outputAxesLetters.WriteString(letterForAxis(lhsIdx, lhs.Rank()-2))
	outputAxesLetters.WriteString(letterForAxis(rhsIdx, rhs.Rank()-1))

	// List lhs and rhs axes as letters:
	var lhsLetters, rhsLetters string
	for axis := range lhs.Rank() {
		lhsLetters += letterForAxis(lhsIdx, axis)
	}
	for axis := range rhs.Rank() {
		rhsLetters += letterForAxis(rhsIdx, axis)
	}

	// Squeeze unused 1-dimensional axes:
	lhsSqueezed := lhs
	if len(lhsSqueezedAxes) > 0 {
		lhsSqueezed = Squeeze(lhs, lhsSqueezedAxes...)
	}
	rhsSqueezed := rhs
	if len(rhsSqueezedAxes) > 0 {
		rhsSqueezed = Squeeze(rhs, rhsSqueezedAxes...)
	}

	equation := fmt.Sprintf("%s,%s->%s", lhsLetters, rhsLetters, outputAxesLetters.String())
	return Einsum(equation, lhsSqueezed, rhsSqueezed)
}

// MatMul is the `numpy.matmul` equivalent, for those used to that.
//
// It is similar to Dot but extends to allow for more batch dimensions in lhs or rhs operand, and
// does broadcasting (of all but the last 2 axes) according to the numpy broadcasting rules.
//
// It's popular hence it is here, but full of edge cases, consider using Dot().General() (or DotGeneral()) instead.
//
// This is an alias for Dot(lhs, rhs).MatMul().
// Use Dot() instead if you need further configuration.
func MatMul(lhs, rhs *Node) *Node {
	return Dot(lhs, rhs).MatMul()
}

// Einsum evaluates the "Einstein summation" various types of products (inner/outer/batched)
// between 2 tensors, on arbitrary dimensions.
// This version uses a textual description on how to manipulate the axes.
// See EinsumAxes for a version where the axes are given numerically.
//
// This is inspired on numpy Einsum, a description of which can be seen in
// https://stackoverflow.com/questions/26089893/understanding-numpys-einsum/33641428#33641428.
//
// The equation string describes what to do with each dimension, for each operand,
// separated by ",", and the format of the result after the "->" describes what is to be made
// for each dimension.
//
// Examples:
//
// * `Einsum("ij,jk->ik", matrixA, matrixB)` performs the usual matrix multiplication.
// * `Einsum("bij,bjk->bik", batchedMatrixA, batchedMatrixB)` performs a batched matrix multiplication.
// * `Einsum("i,i->", vectorA, vectorB)` performs a dot product.
// * `Einsum("i,j->ij", vectorA, vectorB)` performs an outer (cross) product between two vectors.
//
// It also works for higher dimension tensors. Dimensions missing on the output (after "->") are
// reduce-summed.
//
// More examples in TensorFlow documentation:
// https://www.tensorflow.org/api_docs/python/tf/einsum
//
// Notice though that this Einsum is only defined for operations between 2 operands:
//
// - `lhs`: left-hand-side operand.
// - `rhs`: right-hand-side operand.
//
// Important note: the order of the operands can have a dramatic impact on the speed of the multiplications.
// consider trying both sides.
//
// This is an alias for Dot(lhs, rhs).Einsum(equation).
// Use Dot() instead if you need further configuration.
func Einsum(equation string, lhs, rhs *Node) *Node {
	return Dot(lhs, rhs).Einsum(equation)
}

// Einsum evaluates the "Einstein summation" various types of products (inner/outer/batched)
// between 2 tensors, on arbitrary dimensions.
// This version uses a textual description on how to manipulate the axes.
// See EinsumAxes for a version where the axes are given numerically.
//
// This is inspired on numpy Einsum, a description of which can be seen in
// https://stackoverflow.com/questions/26089893/understanding-numpys-einsum/33641428#33641428.
//
// The equation string describes what to do with each dimension, for each operand,
// separated by ",", and the format of the result after the "->" describes what is to be made
// for each dimension.
//
// Examples:
//
// * `Dot(matrixA, matrixB).Einsum("ij,jk->ik")` performs the usual matrix multiplication.
// * `Dot(batchedMatrixA, batchedMatrixB).Einsum("bij,bjk->bik")` performs a batched matrix multiplication.
// * `Dot(vectorA, vectorB).Einsum("i,i->")` performs a dot product.
// * `Dot(vectorA, vectorB).Einsum("i,j->ij")` performs an outer (cross) product between two vectors.
//
// It also works for higher dimension tensors. Dimensions missing on the output (after "->") are
// reduce-summed.
//
// More examples in TensorFlow documentation:
// https://www.tensorflow.org/api_docs/python/tf/einsum
//
// Notice though that this Einsum is only defined for operations between 2 operands:
//
// - `lhs`: left-hand-side operand.
// - `rhs`: right-hand-side operand.
//
// Important note: the order of the operands can have a dramatic impact on the speed of the multiplications.
// consider trying both sides.
func (b *DotBuilder) Einsum(equation string) *Node {
	lhs, rhs := b.lhs, b.rhs
	_ = validateBuildingGraphFromInputs(lhs, rhs)

	// Parse equation.
	inOutParts := strings.Split(equation, "->")
	if len(inOutParts) != 2 {
		exceptions.Panicf(
			"Einsum(%q) missing or too many \"->\" separating inputNodes from outputs, there must be only one",
			equation,
		)
	}
	outputDesc, err := newEinsumOperandDesc(inOutParts[1])
	if err != nil {
		panic(err)
	}
	equationInputs := strings.Split(inOutParts[0], ",")
	if len(equationInputs) != 2 {
		exceptions.Panicf(
			"Einsum(%q) equation describes %d operands (separated by \",\"), but 2 operands (lhs and rhs) required",
			equation,
			len(equationInputs),
		)
	}
	operandsDesc := make([]einsumOperandDesc, 2)
	for ii, str := range equationInputs {
		operandsDesc[ii], err = newEinsumOperandDesc(str)
		if err != nil {
			panic(errors.WithMessagef(err, "when parsing operand %d", ii))
		}
	}

	// First, independently contract axes that only appear in one operand and not in the output.
	for opIdx, opPtr := range []**Node{&lhs, &rhs} {
		var newDesc einsumOperandDesc
		var contracting []int
		thisDesc, otherDesc := operandsDesc[opIdx], operandsDesc[1-opIdx]
		for axisIdx, axis := range thisDesc {
			if otherDesc.hasAxis(axis) || outputDesc.hasAxis(axis) {
				newDesc = append(newDesc, axis)
				continue
			}
			contracting = append(contracting, axisIdx)
		}
		if len(contracting) > 0 {
			//operandNames := []string{"lhs", "rhs"}
			//fmt.Printf("\tEinsum: independently contracting dimensions (%s): %v\n", operandNames[opIdx], contracting)
			// Contract dimensions.
			*opPtr = ReduceSum(*opPtr, contracting...)
			operandsDesc[opIdx] = newDesc
		}
	}

	// Calculate parameters for the dotGeneralXLA, and the order of its output — if
	// the order of `DotGeneral`'s output is different from the requested in `outputDesc`
	// we need to do a final transposition of the axes.
	lhsDesc := operandsDesc[0]
	rhsDesc := operandsDesc[1]
	var lhsBatchAxes, lhsContractingAxes, rhsBatchAxes, rhsContractingAxes []int
	var outputBatchAxes, outputCrossAxes einsumOperandDesc // dotGeneralXLA order of outputs.

	// Start from lhs: all axes that feature in both `lhs` and `rhs` are already taken care in
	// this loop.
	for lhsAxisIdx, axis := range lhsDesc {
		if rhsDesc.hasAxis(axis) {
			rhsAxisIdx := rhsDesc.axisIndex(axis)
			if outputDesc.hasAxis(axis) {
				// Batch dimension.
				lhsBatchAxes = append(lhsBatchAxes, lhsAxisIdx)
				rhsBatchAxes = append(rhsBatchAxes, rhsAxisIdx)
				outputBatchAxes = append(outputBatchAxes, axis)
			} else {
				// Contracting dimension.
				lhsContractingAxes = append(lhsContractingAxes, lhsAxisIdx)
				rhsContractingAxes = append(rhsContractingAxes, rhsAxisIdx)
			}
		} else {
			// Axis only exists on lhs and in the output: because axes that only
			// exist in one operand and nowhere else have already been contracted
			// earlier.
			//
			// This is a cross/outer product axes, the default for dotGeneralXLA.
			outputCrossAxes = append(outputCrossAxes, axis)
		}
	}

	// Loop in rhs: only missing those axes that only feature in rhs.
	for _, axis := range rhsDesc {
		if !lhsDesc.hasAxis(axis) {
			// This is a cross/outer product axes, the default for dotGeneralXLA.
			outputCrossAxes = append(outputCrossAxes, axis)
		}
	}

	// dotGeneralXLA will calculate the einsum, but the output may still be on the wrong
	// order.
	dotOutputDesc := outputBatchAxes
	if len(outputCrossAxes) > 0 {
		dotOutputDesc = append(dotOutputDesc, outputCrossAxes...)
	}

	output := DotGeneral(lhs, lhsContractingAxes, lhsBatchAxes,
		rhs, rhsContractingAxes, rhsBatchAxes)

	// Calculate the target permutation.
	permutation := make([]int, 0, output.Rank())
	hasPermutation := false
	for toAxisIdx, axis := range outputDesc {
		fromAxisIdx := dotOutputDesc.axisIndex(axis)
		permutation = append(permutation, fromAxisIdx)
		if fromAxisIdx != toAxisIdx {
			hasPermutation = true
		}
	}
	if hasPermutation {
		output = TransposeAllAxes(output, permutation...)
	}
	return output
}

type einsumOperandDesc []rune

func newEinsumOperandDesc(str string) (einsumOperandDesc, error) {
	e := make(einsumOperandDesc, 0, len(str))
	for _, r := range str {
		if e.hasAxis(r) {
			return nil, errors.Errorf("operands description (%q) has axis %q appearing more than once", str, r)
		}
		e = append(e, r)
	}
	return e, nil
}

func (e einsumOperandDesc) hasAxis(axis rune) bool {
	return slices.Contains(e, axis)
}

func (e einsumOperandDesc) axisIndex(axis rune) int {
	for ii, r := range e {
		if r == axis {
			return ii
		}
	}
	return -1
}

// EinsumAxes evaluates the "Einstein summation" various types of products (inner/outer/batched)
// between 2 tensors, on arbitrary dimensions. Similar to Einsum, but it uses the explicit numeric
// axis, as opposed to a textual description.
//
// There are two operands: `lhs` (left-hand-side) and `rhs` (right-hand-side). The default for
// every axis is to do a cross-product, and the resulting tensor will have the concatenated shape (`lhs`
// dimensions first then `rhs` dimensions).
//
// One can specify contractionAxes, pairs of axes (each pair with one index in the lhs and rhs operands)
// to be contracted: these dimensions will multiplied and summed one at a time. That's what happens in
// the usual "dot product."
//
// One can also specify batchAxes, pairs of axes (each pair with one index in the lhs and rhs operands)
// to be considered as independently, as a batch dimension. These dimensions will show up in the same
// position as the `lhs`.
//
// Examples:
//
//   - `EinsumAxes(matrixA, matrixB, [][2]int{{1, 0}}, nil)` performs the usual matrix multiplication, where
//     we contract axis 1 of `matrixA` with axis 0 of `matrixB`.
//   - `EinsumAxes(batchedMatrixA, batchedMatrixB, [][2]int{{2, 1}}, [][2]int{{0, 0}})` is similar, but we
//     use axis 0 of both inputNodes as a batch, and following 2 axes as a matrix multiplication.
//   - `EinsumAxes(vectorA, vectorB, nil, nil)` performs an outer (cross) product -- no contractions, no batch.
//   - `EinsumAxes(vectorA, vectorB, [][2]int{{0, 0}}, nil)` performs a dot product and returns a scalar.
//
// Important note: the order of the operands can have a dramatic impact on the speed of the multiplications.
// Consider trying both sides.
//
// This is an alias for Dot(lhs, rhs).EinsumAxes(contractingAxes, batchAxes).
// Use Dot() instead if you need further configuration.
func EinsumAxes(lhs, rhs *Node, contractingAxes, batchAxes [][2]int) (output *Node) {
	return Dot(lhs, rhs).EinsumAxes(contractingAxes, batchAxes)
}

// EinsumAxes evaluates the "Einstein summation" various types of products (inner/outer/batched)
// between 2 tensors, on arbitrary dimensions. Similar to Einsum, but it uses the explicit numeric
// axis, as opposed to a textual description.
//
// There are two operands: `lhs` (left-hand-side) and `rhs` (right-hand-side). The default for
// every axis is to do a cross-product, and the resulting tensor will have the concatenated shape (`lhs`
// dimensions first then `rhs` dimensions).
//
// One can specify contractionAxes, pairs of axes (each pair with one index in the lhs and rhs operands)
// to be contracted: these dimensions will multiplied and summed one at a time. That's what happens in
// the usual "dot product."
//
// One can also specify batchAxes, pairs of axes (each pair with one index in the lhs and rhs operands)
// to be considered as independently, as a batch dimension. These dimensions will show up in the same
// position as the `lhs`.
//
// Examples:
//
//   - `Dot(matrixA, matrixB).EinsumAxes([][2]int{{1, 0}}, nil)` performs the usual matrix multiplication, where
//     we contract axis 1 of `matrixA` with axis 0 of `matrixB`.
//   - `Dot(batchedMatrixA, batchedMatrixB).EinsumAxes([][2]int{{2, 1}}, [][2]int{{0, 0}})` is similar, but we
//     use axis 0 of both inputNodes as a batch, and following 2 axes as a matrix multiplication.
//   - `Dot(vectorA, vectorB).EinsumAxes(nil, nil)` performs an outer (cross) product -- no contractions, no batch.
//   - `Dot(vectorA, vectorB).EinsumAxes([][2]int{{0, 0}}, nil)` performs a dot product and returns a scalar.
//
// Important note: the order of the operands can have a dramatic impact on the speed of the multiplications.
// Consider trying both sides.
func (b *DotBuilder) EinsumAxes(contractingAxes, batchAxes [][2]int) (output *Node) {
	lhs, rhs := b.lhs, b.rhs
	_ = validateBuildingGraphFromInputs(lhs, rhs)
	lhsRank := lhs.Rank()
	rhsRank := rhs.Rank()

	// Create function to process both, contractingAxes and batchAxes.
	lhsSeen := sets.Make[int](lhsRank)
	rhsSeen := sets.Make[int](rhsRank)
	normalizePairs := func(name string, pairs [][2]int) (lhsAxes, rhsAxes []int) {
		if len(pairs) == 0 {
			return
		}
		lhsAxes = make([]int, 0, len(contractingAxes))
		rhsAxes = make([]int, 0, len(contractingAxes))
		for _, pair := range pairs {
			lhsAxis := MustAdjustAxis(pair[0], lhs)
			if lhsSeen.Has(lhsAxis) {
				exceptions.Panicf(
					"EinsumAxes %s axis for left-hand-side operand is duplicate -- each axis can only be contracted or batch once: %v",
					name,
					pairs,
				)
			}
			lhsSeen.Insert(lhsAxis)

			rhsAxis := MustAdjustAxis(pair[1], rhs)
			if rhsSeen.Has(rhsAxis) {
				exceptions.Panicf(
					"EinsumAxes %s axis for right-hand-side operand is duplicate -- each axis can only be contracted or batch once: %v",
					name,
					pairs,
				)
			}
			rhsSeen.Insert(rhsAxis)

			lhsAxes = append(lhsAxes, lhsAxis)
			rhsAxes = append(rhsAxes, rhsAxis)
		}
		return
	}

	lhsContractingAxes, rhsContractingAxes := normalizePairs("contractingAxes", contractingAxes)
	lhsBatchAxes, rhsBatchAxes := normalizePairs("batchAxes", batchAxes)

	// Execute DotGeneral with parameters.
	return b.General(lhsContractingAxes, lhsBatchAxes, rhsContractingAxes, rhsBatchAxes)
}

// crossAxes list all axes not included in contracting or batch: these are the dimensions that DotGeneral
// will do a "cross" (combine all variations for the lhs and rhs, effectively concatenating the dimensions).
func dotCrossAxes(input *Node, contractingAxes, batchAxes []int) (crossAxes []int) {
	rank := input.Rank()
	used := make([]bool, rank)
	for _, axis := range contractingAxes {
		used[axis] = true
	}
	for _, axis := range batchAxes {
		used[axis] = true
	}

	crossAxes = make([]int, 0, rank-len(contractingAxes)-len(batchAxes))
	for ii := range used {
		if !used[ii] {
			crossAxes = append(crossAxes, ii)
		}
	}
	return
}

// dotGeneralVJP generates the gradient with respect to the lhs (left-hand-side) and rhs (right-hand-side) operands.
func dotGeneralVJP(node, v *Node, _ shapes.Shape) []*Node {
	params := node.inputs.(*nodeInputsDotGeneral)
	lhs, rhs := params.lhs, params.rhs
	lhsCrossAxes := dotCrossAxes(lhs, params.lhsContractingAxes, params.lhsBatchAxes)
	rhsCrossAxes := dotCrossAxes(rhs, params.rhsContractingAxes, params.rhsBatchAxes)

	// Identify in which DType to do the calculations:
	// 1. If AccumulatorDType is given, use it.
	// 2. If OutputDType is given, use it (assumed higher precision than input).
	// 3. Else use v.DType() (which should be the output type of the forward operation).
	computationDType := v.DType()
	if params.config.AccumulatorDType != dtypes.InvalidDType {
		computationDType = params.config.AccumulatorDType
	} else if params.config.OutputDType != dtypes.InvalidDType {
		computationDType = params.config.OutputDType
	}

	// Gradient with respect to lhs:
	gradFn := func(thisInput *Node, thisBatchAxes, thisContractingAxes, thisCrossAxes []int, thisCrossesFirst bool,
		otherInput *Node, otherBatchAxes, otherContractingAxes, otherCrossAxes []int) *Node {
		_ = thisInput
		// Axes counts:
		numBatchAxes := len(thisBatchAxes)             // == len(otherBatchAxes)
		numContractionAxes := len(thisContractingAxes) // == len(otherContractingAxes)
		//numCrossedAxes := len(thisCrossAxes) + len(otherCrossAxes)

		// Project output (of the DotGeneral) shaped v to "this" (this node) shapes.

		// * Add back contracted dimensions, with size 1.
		//   thisVJP shapes will be [batch_dims..., lhs_cross_dims..., rhs_cross_dims..., 1 x (numContractionAxes)].
		thisVJP := v
		if thisVJP.DType() != computationDType {
			thisVJP = ConvertDType(thisVJP, computationDType)
		}
		if numContractionAxes > 0 {
			thisVJP = InsertAxes(thisVJP, xslices.SliceWithValue(numContractionAxes, -1)...)
		}

		// * Project other operand with contracted dimensions.
		//   otherProjected shapes for this=lhs will be [batch_dims..., 1 x (this_cross_dims), rhs_cross_dims, contracted_dims]
		otherProjected := otherInput
		if otherProjected.DType() != computationDType {
			otherProjected = ConvertDType(otherProjected, computationDType)
		}
		otherRank := otherProjected.Rank()
		{
			permutations := make([]int, 0, otherRank)
			permutations = append(permutations, otherBatchAxes...)
			permutations = append(permutations, otherCrossAxes...)
			permutations = append(permutations, otherContractingAxes...)
			changed := false
			for ii, axis := range permutations {
				if ii != axis {
					changed = true
				}
			}
			if changed {
				otherProjected = TransposeAllAxes(otherProjected, permutations...)
			}
			// Add placeholder axes (of dimension 1) for the crosses from "this".
			if len(thisCrossAxes) > 0 {
				pos := numBatchAxes // Where axes for thisCrossesAxes will be inserted.
				if !thisCrossesFirst {
					pos += len(otherCrossAxes)
				}
				otherProjected = InsertAxes(otherProjected, xslices.SliceWithValue(len(thisCrossAxes), pos)...)
			}
		}

		// * Multiply the contracted dimension by otherProjected: this will expand the contracted dimensions.
		thisVJP = Mul(thisVJP, otherProjected)

		// * Contract the otherCrossAxes, since those dimensions should exist in the final thisVJP — these
		//   cross-axes came from the "other" input.
		if len(otherCrossAxes) > 0 {
			pos := numBatchAxes
			if thisCrossesFirst {
				pos += len(thisCrossAxes)
			}
			thisVJP = ReduceSum(thisVJP, xslices.Iota(pos, len(otherCrossAxes))...)
		}

		// * Transpose thisVJP axes back to its inputNodes.
		thisRank := thisVJP.Rank()
		{
			permutation := make([]int, thisRank)
			for ii, axis := range thisBatchAxes {
				permutation[axis] = ii
			}
			for ii, axis := range thisCrossAxes {
				permutation[axis] = ii + numBatchAxes
			}
			for ii, axis := range thisContractingAxes {
				permutation[axis] = ii + numBatchAxes + len(thisCrossAxes)
			}
			thisVJP = TransposeAllAxes(thisVJP, permutation...)
		}
		if thisVJP.DType() != thisInput.DType() {
			thisVJP = ConvertDType(thisVJP, thisInput.DType())
		}
		return thisVJP
	}

	return []*Node{
		gradFn(lhs, params.lhsBatchAxes, params.lhsContractingAxes, lhsCrossAxes, true, rhs, params.rhsBatchAxes, params.rhsContractingAxes, rhsCrossAxes),  // grad wrt lhs
		gradFn(rhs, params.rhsBatchAxes, params.rhsContractingAxes, rhsCrossAxes, false, lhs, params.lhsBatchAxes, params.lhsContractingAxes, lhsCrossAxes), // grad wrt rhs
	}
}
