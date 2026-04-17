// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package xla

import (
	"reflect"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/go-xla/pkg/stablehlo"
	stablehlotypes "github.com/gomlx/go-xla/pkg/types"
	"github.com/gomlx/gomlx/backends"
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
	}

	// Introduce a dependency to create a schedule barrier to prevent the compiler to try
	// to start processing weights too early (like converting them to the accumulationDType).
	// This was causing a massive increased in memory usage: in a multi-layer model (e.g.: KaLM-Gemma3 with 48 layers)
	// it was attempting to re-layout all the variables in parallel, and required temporary
	// space to all of them at the same time, as opposed to sequentially, one layer at a time, in which
	// case only one temporary buffer is needed at a time.
	//
	// The CUDA backend does not seem to need this (likely they do something like that internally).
	isCPU := f.builder.backend.plugin.IsCPU()
	lhsReady, rhsReady := lhsNode, rhsNode
	if accumulationDType != dtype && isCPU {
		lhsReady, rhsReady, err = dotGeneralAddDependency(f, lhsReady, rhsReady)
		if err != nil {
			return nil, err
		}
	}

	if accumulationDType != dtype {
		if accumulationDType != dtypes.F32 {
			// XLA only supports accumulation in F32 (if different than the input dtypes).
			// For other accumulation dtypes, we convert the inputs to the type.
			var err error
			lhsReadyValue, err := f.ConvertDType(lhsReady, accumulationDType)
			if err != nil {
				return nil, errors.WithMessagef(err, "failed to convert lhs to accumulation dtype")
			}
			rhsReadyValue, err := f.ConvertDType(rhsReady, accumulationDType)
			if err != nil {
				return nil, errors.WithMessagef(err, "failed to convert rhs to accumulation dtype")
			}
			lhsReady, rhsReady = lhsReadyValue.(*Node), rhsReadyValue.(*Node)
			dtype = accumulationDType
		}
	}

	dotGeneralBuilder := stablehlo.DotGeneral(
		lhsReady.value, lhsContractingAxes, lhsBatchAxes,
		rhsReady.value, rhsContractingAxes, rhsBatchAxes)

	// Set algorithm based on config: this is tricky, the "dot_algorithm" is only supported by the CPU PJRT,
	// and not very well supported by CUDA. TPU support is unknown.
	//
	// So we make a simplification here: we only set the "dot_algorithm" is the plugin is not CUDA.
	useTF32 := accumulationDType == dtypes.F32 && f.builder.backend.DotGeneralUseTF32
	if useTF32 || accumulationDType != dtype {
		isCUDA := f.builder.backend.plugin.IsCUDA()
		if isCUDA {
			// Set "precision_config"
			precisionConfig := stablehlotypes.DotGeneralPrecisionHighest
			dotGeneralBuilder.Precision(precisionConfig, precisionConfig)

		} else {
			// For all other PJRTs, set "dot_algorithm" instead.
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

// dotGeneralAddDependency creates a rhs to lhs dependency.
//
// This can create huge temporary space saving: this way the XLA compiler will only need
// a temporary buffer to convert the input to the accumulation dtype once the other operand
// is ready, instead of having to keep alive the temporary buffer for all weights (that come
// in as variables) immediately (and simulatneously) from the start of the graph.
//
// See: https://github.com/openxla/stablehlo/issues/2923
func dotGeneralAddDependency(f *Function, lhs, rhs *Node) (*Node, *Node, error) {
	isSwapped := lhs.value.OpName() == stablehlo.InputParameterName
	if isSwapped {
		lhs, rhs = rhs, lhs
	}

	// Create a fake barrier, by creating a dependency:
	starts := make([]int, lhs.shape.Rank())
	limits := xslices.SliceWithValue(lhs.shape.Rank(), 1)
	dependency, err := f.Slice(lhs, starts, limits, nil)
	if err != nil {
		return nil, nil, errors.WithMessagef(err, "failed to create fake barrier for DotGeneral operands")
	}
	dependency, err = f.Reshape(dependency) // Convert to scalar.
	if err != nil {
		return nil, nil, errors.WithMessagef(err, "failed to create fake barrier for DotGeneral operands")
	}
	zeroFlatV := reflect.MakeSlice(reflect.SliceOf(lhs.shape.DType.GoType()), 1, 1)
	zero, err := f.Constant(zeroFlatV.Interface())
	if err != nil {
		return nil, nil, errors.WithMessagef(err, "failed to create fake barrier for DotGeneral operands")
	}
	dependency, err = f.Mul(dependency, zero)
	if err != nil {
		return nil, nil, errors.WithMessagef(err, "failed to create fake barrier for DotGeneral operands")
	}
	rhsReady, err := f.Add(rhs, dependency) // Adds a dependency from the weights to the LHS.
	if err != nil {
		return nil, nil, errors.WithMessagef(err, "failed to create fake barrier for DotGeneral operands")
	}

	lhsReady := backends.Value(lhs)
	if isSwapped {
		lhsReady, rhsReady = rhsReady, lhsReady
	}
	return lhsReady.(*Node), rhsReady.(*Node), nil
}
