// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package peft implements parameter-efficient fine-tuning layers for GoMLX.
//
// It owns the GoMLX-native LoRA and NF4 QLoRA graph layers. Model-family
// registries, Hugging Face transport, tokenizer policy, and dataset policy
// remain outside this package.
package peft

import (
	"errors"
	"math"
	"strings"

	"github.com/gomlx/compute"
)

var (
	// ErrInvalidRank reports a non-positive adapter rank.
	ErrInvalidRank = errors.New("peft: rank must be positive")
	// ErrInvalidAlpha reports a non-finite or negative adapter scale.
	ErrInvalidAlpha = errors.New("peft: alpha must be finite and non-negative")
	// ErrInvalidDropout reports a dropout rate outside [0, 1).
	ErrInvalidDropout = errors.New("peft: dropout must be in [0, 1)")
	// ErrMissingTargetModules reports a missing injection target.
	ErrMissingTargetModules = errors.New("peft: target modules are required")
	// ErrInvalidBiasMode reports an unsupported bias configuration.
	ErrInvalidBiasMode = errors.New("peft: invalid bias mode")
	// ErrNoTargetModules reports an injection plan that selected nothing.
	ErrNoTargetModules = errors.New("peft: no target modules matched")
	// ErrInvalidNF4Weight reports invalid packed NF4 storage.
	ErrInvalidNF4Weight = errors.New("peft: invalid NF4 weight")
	// ErrInvalidQLoRAConfig reports invalid QLoRA-specific settings.
	ErrInvalidQLoRAConfig = errors.New("peft: invalid QLoRA configuration")
	// ErrInvalidWeightLayout reports an unsupported dense-weight layout.
	ErrInvalidWeightLayout = errors.New("peft: invalid dense weight layout")
)

// BiasMode controls which host biases are trainable with an adapter.
type BiasMode uint8

const (
	// BiasNone freezes host bias variables.
	BiasNone BiasMode = iota
	// BiasAll trains every selected module's host bias.
	BiasAll
	// BiasLoRAOnly trains a bias only when its module has a LoRA layer.
	BiasLoRAOnly
)

// Config configures a LoRA adapter. TargetModules accepts an exact module name
// or a dot-separated suffix, such as "q_proj" matching "layers.0.q_proj".
type Config struct {
	Rank          int
	Alpha         float32
	Dropout       float32
	TargetModules []string
	Bias          BiasMode
	// WeightLayout describes the frozen base projection. The zero value is
	// GoMLX's standard [input, output] layout.
	WeightLayout compute.DenseLayout
}

// Validate reports invalid configuration without changing a host model.
func (c Config) Validate() error {
	if c.Rank <= 0 {
		return ErrInvalidRank
	}
	if !finite(c.Alpha) || c.Alpha < 0 {
		return ErrInvalidAlpha
	}
	if !finite(c.Dropout) || c.Dropout < 0 || c.Dropout >= 1 {
		return ErrInvalidDropout
	}
	if len(c.TargetModules) == 0 {
		return ErrMissingTargetModules
	}
	for _, target := range c.TargetModules {
		if target == "" {
			return ErrMissingTargetModules
		}
	}
	if c.Bias > BiasLoRAOnly {
		return ErrInvalidBiasMode
	}
	if c.WeightLayout != compute.DenseLayoutInputOutputs && c.WeightLayout != compute.DenseLayoutOutputsInput {
		return ErrInvalidWeightLayout
	}
	return nil
}

// Matches reports whether a module name is selected by targets.
func Matches(module string, targets []string) bool {
	for _, target := range targets {
		if module == target || strings.HasSuffix(module, "."+target) {
			return true
		}
	}
	return false
}

// QLoRAConfig configures an NF4 base projection and its LoRA update.
// GoMLX's native QLoRA path uses NF4; int4-specific layouts remain backend
// integrations rather than a second graph-layer representation.
type QLoRAConfig struct {
	LoRA           Config
	BlockSize      int
	DoubleQuant    bool
	ScaleBlockSize int
}

// Validate reports invalid QLoRA configuration.
func (c QLoRAConfig) Validate() error {
	if err := c.LoRA.Validate(); err != nil {
		return err
	}
	// NF4Weight has one canonical [input, output] storage layout. Unlike
	// Linear, QLoRA does not reinterpret an existing dense variable.
	if c.LoRA.WeightLayout != compute.DenseLayoutInputOutputs {
		return ErrInvalidQLoRAConfig
	}
	if c.BlockSize <= 0 || (c.DoubleQuant && c.ScaleBlockSize <= 0) {
		return ErrInvalidQLoRAConfig
	}
	return nil
}

func finite(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}

func cloneConfig(config Config) Config {
	config.TargetModules = append([]string(nil), config.TargetModules...)
	return config
}

func cloneQLoRAConfig(config QLoRAConfig) QLoRAConfig {
	config.LoRA = cloneConfig(config.LoRA)
	return config
}
