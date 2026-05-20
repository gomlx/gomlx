// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package train

import (
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/support/testutil"
	"github.com/stretchr/testify/assert"
)

func TestStoreErrors(t *testing.T) {
	backend := testutil.BuildTestBackend()
	g := graph.NewGraph(backend, "test")

	// GetTrainLastStepVar should panic when graph has no associated store.
	assert.Panics(t, func() {
		GetTrainLastStepVar(g)
	})

	// ExecPerStepUpdateGraphFn should panic when graph has no associated store.
	assert.Panics(t, func() {
		ExecPerStepUpdateGraphFn(g)
	})
}
