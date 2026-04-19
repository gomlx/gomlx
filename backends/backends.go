// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package backends is an alias to compute package. It's here for historical reasons.
//
// Deprecated: use the [compute] package instead.
package backends

import "github.com/gomlx/compute"

// Backend represents a compute backend, capabable of building, compiling, transferring data to/from and executing a
// computation graph.
//
// Deprecated: it's just an alias to [compute.Backend], use that instead.
type Backend = compute.Backend

// DeviceNum represents which device holds a buffer or should execute a computation. It's up to the backend to interpret
// it, but it should be between 0 and Backend.NumDevices.
//
// Deprecated: it's just an alias to [compute.DeviceNum], use that instead.
type DeviceNum = compute.DeviceNum

// Buffer represents actual data (a tensor) stored in the accelerator that is actually going to execute the graph.
// It's used as input/output of computation execution. A Buffer is always associated to a DeviceNum, even if there is
// only one.
//
// It is opaque from GoMLX perspective, but it cannot be mixed -- a Buffer returned by one backend can't be used with
// another backend.
//
// Deprecated: it's just an alias to [compute.Buffer], use that instead.
type Buffer = compute.Buffer

// Value represents an intermediate value in a computation graph.
//
// Deprecated: it's just an alias to [compute.Value], use that instead.
type Value = compute.Value

// ActivationType specifies the activation function for fused operations.
//
// Deprecated: it's just an alias to [compute.ActivationType], use that instead.
type ActivationType = compute.ActivationType

const (
	ActivationNone      = compute.ActivationNone
	ActivationRelu      = compute.ActivationRelu
	ActivationTanh      = compute.ActivationTanh
	ActivationGelu      = compute.ActivationGelu
	ActivationSilu      = compute.ActivationSilu
	ActivationHardSwish = compute.ActivationHardSwish
)

// GGMLQuantType identifies the specific GGML block quantization format.
//
// Deprecated: it's just an alias to [compute.GGMLQuantType], use that instead.
type GGMLQuantType = compute.GGMLQuantType

const (
	GGMLQ4_0  = compute.GGMLQ4_0
	GGMLQ8_0  = compute.GGMLQ8_0
	GGMLIQ4NL = compute.GGMLIQ4NL
	GGMLQ2_K  = compute.GGMLQ2_K
	GGMLQ3_K  = compute.GGMLQ3_K
	GGMLQ4_K  = compute.GGMLQ4_K
	GGMLQ5_K  = compute.GGMLQ5_K
	GGMLQ6_K  = compute.GGMLQ6_K
)

// Quantization specifies the quantization for fused operations.
//
// Deprecated: it's just an alias to [compute.Quantization], use that instead.
type Quantization = compute.Quantization

// QuantizationScheme identifies the quantization scheme for fused operations.
//
// Deprecated: it's just an alias to [compute.QuantizationScheme], use that instead.
type QuantizationScheme = compute.QuantizationScheme

const (
	QuantLinear = compute.QuantLinear
	QuantNF4    = compute.QuantNF4
	QuantGGML   = compute.QuantGGML
)

// IQ4NLLookupTable contains the 16 fixed IQ4_NL non-linear dequantization values.
//
// Deprecated: it's just an alias to [compute.IQ4NLLookupTable], use that instead.
var IQ4NLLookupTable = compute.IQ4NLLookupTable

// NF4LookupTable contains the 16 fixed QLoRA NormalFloat4 dequantization values.
//
// Deprecated: it's just an alias to [compute.NF4LookupTable], use that instead.
var NF4LookupTable = compute.NF4LookupTable

// AxesLayout specifies the ordering of axes in 4D attention tensors.
//
// Deprecated: it's just an alias to [compute.AxesLayout], use that instead.
type AxesLayout = compute.AxesLayout

const (
	AxesLayoutBHSD = compute.AxesLayoutBHSD
	AxesLayoutBSHD = compute.AxesLayoutBSHD
)

// DefaultConfig is the name of the default backend configuration to use if specified.
//
// Deprecated: it's just an alias to [compute.DefaultConfig], use that instead.
var DefaultConfig = compute.DefaultConfig

// ConfigEnvVar is the name of the environment variable with the default backend configuration to use:
// "GOMLX_BACKEND".
//
// Deprecated: it's just an alias to [compute.ConfigEnvVar], use that instead.
const ConfigEnvVar = compute.ConfigEnvVar

// MustNew returns a new default Backend or panics if it fails.
//
// The default is:
//
// 1. The environment $GOMLX_BACKEND (ConfigEnvVar) is used as a configuration if defined.
// 2. Next, it uses the variable DefaultConfig as the configuration.
// 3. The first registered backend is used with an empty configuration.
//
// It fails if no backends were registered.
//
// Deprecated: use [compute.MustNew] instead.
func MustNew() compute.Backend {
	return compute.MustNew()
}

// New returns a new default backend.
//
// Deprecated: use [compute.New] instead.
func New() (compute.Backend, error) {
	return compute.New()
}

// NewWithConfig takes a configuration string formated as
//
// The format of config is "<backend_name>:<backend_configuration>".
// The "<backend_name>" is the name of a registered backend (e.g.: "xla") and
// "<backend_configuration>" is backend-specific (e.g.: for xla backend, it is the PJRT plugin name).
//
// Deprecated: use [compute.NewWithConfig] instead.
func NewWithConfig(config string) (compute.Backend, error) {
	return compute.NewWithConfig(config)
}
