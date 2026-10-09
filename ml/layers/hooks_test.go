// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package layers

import (
	"testing"

	"github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/assert"
)

func TestPostLinearHooks(t *testing.T) {
	ClearPostLinearHooks()
	defer ClearPostLinearHooks()

	var executionOrder []string

	RegisterPostLinearHook(10, func(scope *model.Scope, input, output *graph.Node) *graph.Node {
		executionOrder = append(executionOrder, "late")
		return output
	})
	RegisterPostLinearHook(-10, func(scope *model.Scope, input, output *graph.Node) *graph.Node {
		executionOrder = append(executionOrder, "early")
		return output
	})
	RegisterPostLinearHook(0, func(scope *model.Scope, input, output *graph.Node) *graph.Node {
		executionOrder = append(executionOrder, "default-1")
		return output
	})
	RegisterPostLinearHook(0, func(scope *model.Scope, input, output *graph.Node) *graph.Node {
		executionOrder = append(executionOrder, "default-2")
		return output
	})

	ApplyPostLinearHooks(nil, nil, nil)

	assert.Equal(t, []string{"early", "default-1", "default-2", "late"}, executionOrder)
}
