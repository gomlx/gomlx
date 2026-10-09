// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package peft implements parameter-efficient fine-tuning layers for GoMLX.
package peft

import (
	"math"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/ml/model"
	. "github.com/gomlx/gomlx/support/exceptions"
)

const (
	// ParamRank is the LoRA rank. Zero disables the adapter, which is the default.
	ParamRank = "peft_rank"
	// ParamAlpha is the LoRA scaling numerator. It defaults to ParamRank.
	ParamAlpha = "peft_alpha"
	// ParamDropout is the LoRA dropout rate. It defaults to zero.
	ParamDropout = "peft_dropout"
	// ParamBias controls whether the base projection bias is trainable.
	ParamBias = "peft_bias"
)

// BiasMode controls whether a projection bias is trainable with its LoRA weights.
type BiasMode uint8

const (
	BiasNone BiasMode = iota
	BiasAll
	// BiasLoRAOnly is equivalent to BiasAll for a directly constructed PEFT layer.
	BiasLoRAOnly
)

type config struct {
	rank           int
	alpha, dropout float32
	bias           BiasMode
	layout         compute.DenseLayout
}

func configFromScope(scope *model.Scope) config {
	rank := model.GetParamOr(scope, ParamRank, 0)
	return config{
		rank:    rank,
		alpha:   model.GetParamOr(scope, ParamAlpha, float32(rank)),
		dropout: model.GetParamOr(scope, ParamDropout, float32(0)),
		bias:    model.GetParamOr(scope, ParamBias, BiasNone),
		layout:  compute.DenseLayoutInputOutputs,
	}
}

func (c config) validate() {
	if c.rank < 0 {
		Panicf("peft: %s must be non-negative, got %d", ParamRank, c.rank)
	}
	if math.IsNaN(float64(c.alpha)) || math.IsInf(float64(c.alpha), 0) || c.alpha < 0 {
		Panicf("peft: %s must be finite and non-negative, got %g", ParamAlpha, c.alpha)
	}
	if math.IsNaN(float64(c.dropout)) || math.IsInf(float64(c.dropout), 0) || c.dropout < 0 || c.dropout >= 1 {
		Panicf("peft: %s must be in [0, 1), got %g", ParamDropout, c.dropout)
	}
	if c.bias > BiasLoRAOnly {
		Panicf("peft: invalid %s value %d", ParamBias, c.bias)
	}
	if c.layout != compute.DenseLayoutInputOutputs && c.layout != compute.DenseLayoutOutputsInput {
		Panicf("peft: invalid dense weight layout %v", c.layout)
	}
}
