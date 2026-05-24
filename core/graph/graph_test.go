// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package graph_test

import (
	"flag"
	"fmt"
	"os"
	"testing"

	"k8s.io/klog/v2"

	"github.com/gomlx/go-xla/compute/xla"
	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/graph"
)

func init() {
	klog.InitFlags(nil)
}

func must(err error) {
	if err != nil {
		klog.Errorf("Failed with error: %+v", err)
		panic(err)
	}
}

func must1[T any](value T, err error) T {
	must(err)
	return value
}

func TestMain(m *testing.M) {
	fmt.Println(">> TestMain():")
	flag.Parse()

	// Auto-install XLA
	err := xla.AutoInstall()
	if err != nil {
		klog.Fatalf("Failed to auto-install XLA PJRT: %+v", err)
	}

	exitCode := m.Run()
	fmt.Println(">> TestMain(): finished")
	os.Exit(exitCode)
}

func TestGraphState(t *testing.T) {
	g := graph.NewGraph(nil, "TestGraphState")

	type key1 struct{}
	type key2 struct{}
	type key3 struct{} // Non-existent key

	val1 := "value1"
	val2 := 42

	// Initially, state should be nil for keys
	if got := g.State(key1{}); got != nil {
		t.Errorf("Expected nil for key1{}, got %v", got)
	}

	// Attach state
	g.AttachState(key1{}, val1)
	g.AttachState(key2{}, val2)

	// Retrieve state
	if got := g.State(key1{}); got != val1 {
		t.Errorf("Expected %v for key1{}, got %v", val1, got)
	}
	if got := g.State(key2{}); got != val2 {
		t.Errorf("Expected %v for key2{}, got %v", val2, got)
	}

	// Check non-existent key returns nil
	if got := g.State(key3{}); got != nil {
		t.Errorf("Expected nil for key3{}, got %v", got)
	}

	// Test that attaching nil deletes the key
	g.AttachState(key1{}, nil)
	if got := g.State(key1{}); got != nil {
		t.Errorf("Expected nil after deleting key1{}, got %v", got)
	}
}
