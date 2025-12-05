package main

import (
	"fmt"
	"os"
	"testing"

	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/stretchr/testify/require"
)

func TestPerturbVars(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "test_perturb")
	require.NoError(t, err)
	defer os.RemoveAll(tmpDir)

	// Create context and add variable.
	ctx := context.New()
	_ = ctx.VariableWithValue("test_var", tensors.FromScalarAndDimensions(1.0, 100, 100))
	checkpoint, err := checkpoints.Build(ctx).Dir(tmpDir).Keep(-1).Done()
	require.NoError(t, err)
	require.NoError(t, checkpoint.Save())

	// Perturb variables.
	const perturbAmount = 0.1
	PerturbVars(tmpDir, perturbAmount)

	// Load and check perturbed values.
	newCtx := context.New()
	_, err = checkpoints.Build(newCtx).Dir(tmpDir).Immediate().Done()
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
