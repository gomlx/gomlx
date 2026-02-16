// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
)

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
func (f *Function) DotGeneral(
	lhs backends.Value, lhsContractingAxes, lhsBatchAxes []int,
	rhs backends.Value, rhsContractingAxes []int, rhsBatchAxes []int,
	config backends.DotGeneralConfig) (backends.Value, error) {
	nodes, err := f.verifyAndCastValues("Dot", lhs, rhs)
	if err != nil {
		return nil, err
	}
	lhsNode := nodes[0]
	rhsNode := nodes[1]

	dtype := lhsNode.shape.DType
	dotGeneralBuilder := stablehlo.DotGeneral(
		lhsNode.value, lhsContractingAxes, lhsBatchAxes,
		rhsNode.value, rhsContractingAxes, rhsBatchAxes)

	// Set algorithm based on config.
	useTF32 := dtype == dtypes.F32 && f.builder.backend.DotGeneralUseTF32
	if useTF32 || config.AccumulatorDType != dtypes.InvalidDType {
		var algo stablehlotypes.DotGeneralAlgorithm
		algo.LhsComponentCount = 1
		algo.RhsComponentCount = 1
		algo.NumPrimitiveOperations = 1
		algo.AllowImpreciseAccumulation = false

		if config.AccumulatorDType != dtypes.InvalidDType {
			algo.AccumulationType = stablehlotypes.FloatPrecisionType{DType: DTypeToXLA(config.AccumulatorDType)}
		} else {
			algo.AccumulationType = stablehlotypes.FloatPrecisionType{DType: DTypeToXLA(dtype)}
		}

		useTF32 := f.builder.backend.DotGeneralUseTF32
		if useTF32 && dtype == dtypes.Float32 {
			algo.LhsPrecisionType = stablehlotypes.FloatPrecisionType{TF32: true}
			algo.RhsPrecisionType = stablehlotypes.FloatPrecisionType{TF32: true}
		} else {
			algo.LhsPrecisionType = stablehlotypes.FloatPrecisionType{DType: DTypeToXLA(dtype)}
			algo.RhsPrecisionType = stablehlotypes.FloatPrecisionType{DType: DTypeToXLA(dtype)}
		}
		dotGeneralBuilder.Algorithm(&algo)
	}
	if config.OutputDType != dtypes.InvalidDType {
		dotGeneralBuilder.OutputDType(DTypeToXLA(config.OutputDType))
	}
	value, err := dotGeneralBuilder.Done()
	if err != nil {
		return nil, err
	}
	return f.newNode(value), nil
}
