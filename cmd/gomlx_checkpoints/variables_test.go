// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"os"
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
	"github.com/stretchr/testify/require"
)

func TestListVariables(t *testing.T) {
	store := model.NewStore()
	s1 := store.Scope("scope1")
	_ = s1.VariableWithValue("varA", tensors.FromScalar(1.0))
	_ = s1.VariableWithValue("varB", tensors.FromScalar(2.0))
	s2 := store.Scope("scope2")
	_ = s2.VariableWithValue("varC", tensors.FromScalar(3.0))

	// Ensure calling ListVariables doesn't panic and executes cleanly.
	require.NotPanics(t, func() {
		ListVariables(store.RootScope())
	})
}

func TestPerturbVars(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "test_perturb")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	// Create context and add variable.
	scope := model.NewStore()
	_ = scope.VariableWithValue("test_var", tensors.FromScalarAndDimensions(1.0, 100, 100))
	checkpointHandler, err := checkpoint.Build(scope).Dir(tmpDir).Keep(-1).Done()
	require.NoError(t, err)
	require.NoError(t, checkpointHandler.Save())

	// Perturb variables.
	const perturbAmount = 0.1
	PerturbVars(tmpDir, perturbAmount)

	// Load and check perturbed values.
	newCtx := model.NewStore()
	_, err = checkpoint.Build(newCtx).Dir(tmpDir).Immediate().Done()
	require.NoError(t, err)
	perturbedT, err := newCtx.GetVariable("test_var").Value()
	require.NoError(t, err)
	var lowerCount, higherCount int
	tensors.ConstFlatData(perturbedT, func(flat []float64) {
		for _, v := range flat {
			require.Greater(t, v, 1.0-perturbAmount)
			require.Less(t, v, 1.0+perturbAmount)
			if v < 1.0 {
				lowerCount++
			} else if v > 1.0 {
				higherCount++
			}
		}
	})
	var totalCount = perturbedT.Shape().Size()
	fmt.Printf("Total count: %d\n", totalCount)
	fmt.Printf("Lower count: %d\n", lowerCount)
	fmt.Printf("Higher count: %d\n", higherCount)
	// At least 99% of the values must have changed.
	require.Greater(t, lowerCount+higherCount, 99*totalCount/100)

	// The difference of values moving up and down < 10%.
	diffCount := lowerCount - higherCount
	if diffCount < 0 {
		diffCount = -diffCount
	}
	require.Less(t, diffCount, 10*totalCount/100)
}
