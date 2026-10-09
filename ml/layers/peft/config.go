// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package peft implements parameter-efficient fine-tuning layers for GoMLX.
package peft

import (
	"math"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/ml/model"
	. "github.com/gomlx/gomlx/support/exceptions"
)

const (
	// ParamAdapter is the name of the LoRA adapter to use.
	// If empty (""), PEFT is inactive and base layers are trainable as normal.
	// If set to "off", "disabled", "false", or "none", base layers are frozen without adding any LoRA adapters.
	// Otherwise, it defines the sub-scope name for the LoRA adapter variables (e.g. "lora", "task_a").
	ParamAdapter = "peft_adapter"

	// ParamTargetModules specifies substrings of scope paths to which LoRA adapters should be applied.
	// Can be a []string or a comma/semicolon/space-separated string (e.g. "query,value").
	// Unmatched linear projections have their base weights frozen, but no adapter added.
	// If empty, all linear projections with peft_adapter set will receive adapters.
	ParamTargetModules = "peft_target_modules"

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
	adapter        string
	targetModules  []string
	rank           int
	alpha, dropout float32
	bias           BiasMode
	layout         compute.DenseLayout
}

func configFromScope(scope *model.Scope) config {
	rank := model.GetParamOr(scope, ParamRank, 0)
	adapter := model.GetParamOr(scope, ParamAdapter, "")
	targetModulesRaw := model.GetParamOr[any](scope, ParamTargetModules, nil)
	biasRaw := model.GetParamOr[any](scope, ParamBias, BiasNone)

	return config{
		adapter:       adapter,
		targetModules: ParseTargetModules(targetModulesRaw),
		rank:          rank,
		alpha:         model.GetParamOr(scope, ParamAlpha, float32(rank)),
		dropout:       model.GetParamOr(scope, ParamDropout, float32(0)),
		bias:          parseBiasMode(biasRaw),
		layout:        compute.DenseLayoutInputOutputs,
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

// IsAdapterDisabled returns true if the adapter name explicitly disables adapter creation
// (i.e. "off", "disabled", "false", or "none", case-insensitive).
// Note: An empty adapter name means PEFT is inactive, not disabled/freeze-only.
func IsAdapterDisabled(adapterName string) bool {
	s := strings.ToLower(strings.TrimSpace(adapterName))
	return s == "off" || s == "disabled" || s == "false" || s == "none"
}

// ParseTargetModules converts a target modules value (either []string or string) into a normalized string slice.
func ParseTargetModules(val any) []string {
	if val == nil {
		return nil
	}
	switch v := val.(type) {
	case []string:
		var res []string
		for _, s := range v {
			if trimmed := strings.TrimSpace(s); trimmed != "" {
				res = append(res, trimmed)
			}
		}
		return res
	case string:
		fields := strings.FieldsFunc(v, func(r rune) bool {
			return r == ',' || r == ';' || r == ' ' || r == '\t'
		})
		var res []string
		for _, s := range fields {
			if trimmed := strings.TrimSpace(s); trimmed != "" {
				res = append(res, trimmed)
			}
		}
		return res
	default:
		return nil
	}
}

// MatchesTargetModule returns true if targets is empty or if any target is contained within scopePath.
func MatchesTargetModule(scopePath string, targets []string) bool {
	if len(targets) == 0 {
		return true
	}
	for _, target := range targets {
		if strings.Contains(scopePath, target) {
			return true
		}
	}
	return false
}

func parseBiasMode(val any) BiasMode {
	switch v := val.(type) {
	case BiasMode:
		return v
	case string:
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "all":
			return BiasAll
		case "lora_only", "lora-only", "loraonly":
			return BiasLoRAOnly
		default:
			return BiasNone
		}
	case int:
		return BiasMode(v)
	default:
		return BiasNone
	}
}
