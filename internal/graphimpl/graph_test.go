// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graphimpl_test

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/internal/graphimpl"
)

func TestGraphState(t *testing.T) {
	g := graphimpl.NewGraphImpl(nil, "TestGraphState")

	type key1 struct{}
	type key2 struct{}
	type key3 struct{} // Non-existent key

	val1 := "value1"
	val2 := 42

	if got := g.State(key1{}); got != nil {
		t.Errorf("Expected nil for key1{}, got %v", got)
	}

	g.AttachState(key1{}, val1)
	g.AttachState(key2{}, val2)

	if got := g.State(key1{}); got != val1 {
		t.Errorf("Expected %v for key1{}, got %v", val1, got)
	}
	if got := g.State(key2{}); got != val2 {
		t.Errorf("Expected %v for key2{}, got %v", val2, got)
	}
	if got := g.State(key3{}); got != nil {
		t.Errorf("Expected nil for key3{}, got %v", got)
	}

	g.AttachState(key1{}, nil)
	if got := g.State(key1{}); got != nil {
		t.Errorf("Expected nil after deleting key1{}, got %v", got)
	}
}
