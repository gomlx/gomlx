package xla

import (
	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/go-xla/pkg/types/dtypes"
	"github.com/gomlx/gomlx/backends"
)

// DotGeneralConfig represents the configuration to use for DotGeneral.
// StableHLO has lots of options (see github.com/gomlx/go-xla/pkg/stablehlo.DotGeneral),
// and here is what we expose for now.
type DotGeneralConfig struct {
	// UseTF32 specifies whether to use tf32 (a truncated float32 that NVidia CUDA PJRT is able to use)
	// when doing float32 dot general.
	UseTF32 bool
}

// DotGeneral takes as input lhs (left-hand-side) and rhs (right-hand-side) specifications
// for a general vector product -- a generalized "Einsum". Each axis can be:
//
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
func (b *Builder) DotGeneral(lhs backends.Op, lhsContractingAxes, lhsBatchAxes []int, rhs backends.Op, rhsContractingAxes []int, rhsBatchAxes []int) (backends.Op, error) {
	nodes, err := b.verifyAndCastValues("Dot", lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode := nodes[0]
	rhsNode := nodes[1]

	config := b.backend.DotGeneralConfig
	dtype := lhsNode.shape.DType

	dotGeneralBuilder := stablehlo.DotGeneral(lhsNode.value, lhsContractingAxes, lhsBatchAxes, rhsNode.value, rhsContractingAxes, rhsBatchAxes)
	if config.UseTF32 && dtype == dtypes.Float32 {
		dotGeneralBuilder.Algorithm(&stablehlotypes.DotGeneralAlgorithm{
			LhsPrecisionType:           stablehlotypes.FloatPrecisionType{TF32: true},
			RhsPrecisionType:           stablehlotypes.FloatPrecisionType{TF32: true},
			AccumulationType:           stablehlotypes.FloatPrecisionType{DType: dtype},
			LhsComponentCount:          1,
			RhsComponentCount:          1,
			NumPrimitiveOperations:     1,
			AllowImpreciseAccumulation: false,
		})
	}
	value, err := dotGeneralBuilder.Done()
	if err != nil {
		return nil, err
	}
	return b.newNode(value), nil
}
