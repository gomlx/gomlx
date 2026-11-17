// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph

import (
	"reflect"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/require"
)

// TestRNGBitGeneratorFallbackExact verifies that the decomposed pure-graph fallback for every
// counter-based algorithm (including Philox and ThreeFry) matches the native Go backend bit-for-bit
// across Uint8, Uint16, Uint32, and Uint64.
func TestRNGBitGeneratorFallbackExact(t *testing.T) {
	goBackend := testutil.GetOfficialBackend("go")
	algos := []RNGAlgorithm{RNGPhilox, RNGThreeFry, RNGSplitMix64, RNGChaCha8}
	outShapes := []shapes.Shape{
		shapes.Make(dtypes.Uint8, 7),
		shapes.Make(dtypes.Uint16, 7),
		shapes.Make(dtypes.Uint32, 7),
		shapes.Make(dtypes.Uint64, 5),
		shapes.Make(dtypes.Uint32),
		shapes.Make(dtypes.Uint32, 0),
	}

	for _, algo := range algos {
		for _, outShape := range outShapes {
			t.Run(algo.String()+"/"+outShape.String(), func(t *testing.T) {
				nativeOuts := MustNewExec(goBackend, func(g *Graph) (*Node, *Node) {
					s0 := RNGStateFromSeedForGraph(g, algo, 777)
					return RNGBitGenerator(algo, s0, outShape)
				}).MustCall()

				fallbackOuts := MustNewExec(goBackend, func(g *Graph) (*Node, *Node) {
					s0 := RNGStateFromSeedForGraph(g, algo, 777)
					return rngBitGeneratorFallback(s0, outShape, algo)
				}).MustCall()

				require.Truef(t, reflect.DeepEqual(nativeOuts[0].Value(), fallbackOuts[0].Value()),
					"state mismatch: fallback=%v, native=%v", fallbackOuts[0].Value(), nativeOuts[0].Value())
				if outShape.Size() > 0 {
					require.Truef(t, reflect.DeepEqual(nativeOuts[1].Value(), fallbackOuts[1].Value()),
						"values mismatch: fallback=%v, native=%v", fallbackOuts[1].Value(), nativeOuts[1].Value())
				}
			})
		}
	}
}
