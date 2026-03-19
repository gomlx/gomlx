// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/pkg/errors"
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
//
// The XLA implementation only supports accumulation in F32 (if different than the input dtypes).
// So when it receives a different accumulation dtype, it simply converts the inputs to F32.
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
	accumulationDType := dtype
	if config.AccumulatorDType != dtypes.InvalidDType {
		accumulationDType = config.AccumulatorDType
		if accumulationDType != dtypes.F32 {
			// XLA only supports accumulation in F32 (if different than the input dtypes).
			// For other accumulation dtypes, we convert the inputs to the type.
			var err error
			lhs, err = f.ConvertDType(lhs, accumulationDType)
			if err != nil {
				return nil, errors.WithMessagef(err, "failed to convert lhs to accumulation dtype")
			}
			rhs, err = f.ConvertDType(rhs, accumulationDType)
			if err != nil {
				return nil, errors.WithMessagef(err, "failed to convert rhs to accumulation dtype")
			}
			dtype = accumulationDType
		}
	}

	dotGeneralBuilder := stablehlo.DotGeneral(
		lhsNode.value, lhsContractingAxes, lhsBatchAxes,
		rhsNode.value, rhsContractingAxes, rhsBatchAxes)

	// Set algorithm based on config: this is tricky, the "dot_algorithm" is only supported by the CPU PJRT,
	// and not very well supported by CUDA. TPU support is unknown.
	//
	// So we make a simplification here: we only set the "dot_algorithm" is the plugin is not CUDA.
	useTF32 := accumulationDType == dtypes.F32 && f.builder.backend.DotGeneralUseTF32
	if useTF32 || accumulationDType != dtype {
		if !f.builder.backend.plugin.IsCUDA() {
			// Set "dot_algorithm"
			var algo stablehlotypes.DotGeneralAlgorithm
			algo.LhsComponentCount = 1
			algo.RhsComponentCount = 1
			algo.NumPrimitiveOperations = 1
			algo.AllowImpreciseAccumulation = false
			algo.AccumulationType = stablehlotypes.FloatPrecisionType{DType: DTypeToXLA(accumulationDType)}

			useTF32 := f.builder.backend.DotGeneralUseTF32
			if useTF32 && accumulationDType == dtypes.Float32 {
				algo.LhsPrecisionType = stablehlotypes.FloatPrecisionType{TF32: true}
				algo.RhsPrecisionType = stablehlotypes.FloatPrecisionType{TF32: true}
				algo.AccumulationType.TF32 = true
			} else {
				algo.LhsPrecisionType = stablehlotypes.FloatPrecisionType{DType: DTypeToXLA(accumulationDType)}
				algo.RhsPrecisionType = stablehlotypes.FloatPrecisionType{DType: DTypeToXLA(accumulationDType)}
			}
			dotGeneralBuilder.Algorithm(&algo)
		} else {
			// Set "precision_config"
			precisionConfig := stablehlotypes.DotGeneralPrecisionHighest
			dotGeneralBuilder.Precision(precisionConfig, precisionConfig)
		}
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
