// Copyright 2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package attention

// LayerType defines whether a layer is local (sliding window) or global.
type LayerType int

//go:generate go tool enumer -type LayerType -output=gen_layertype_enumer.go layertype.go

const (
	// GlobalLayer uses full sequence attention.
	GlobalLayer LayerType = iota
	// LocalLayer uses sliding window attention.
	LocalLayer
)
