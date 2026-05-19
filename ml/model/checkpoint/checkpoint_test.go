// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package checkpoint

// TODO:
// * Test that previously loaded variables -- not used by the Context -- are also saved.
// * Test what happens with saving/loading of objects in Params: do they need to be filtered?

import (
	"crypto/rand"
	"fmt"
	"io"
	"os"
	"testing"

	"github.com/gomlx/compute/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/layers/regularizers"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/train/optimizer"
	"github.com/gomlx/gomlx/support/testutil"
)

func TestCheckpoints(t *testing.T) {
	backend := testutil.BuildTestBackend()

	// Graph function to test: it simply creates, increments and returns the global step.
	testGraphFn := func(scope *model.Scope, g *Graph) *Node {
		return optimizer.IncrementGlobalStep(scope, g, dtypes.Float64)
	}
	var dir string
	{
		// Build model, checkpoint a few times.
		store := model.NewStore()
		store.SetParam("learning_rate", 0.01)
		store.SetParam(regularizers.ParamL2, 0.001)
		store.SetParam(regularizers.ParamL1, 1.0e7)
		rootScope := store.RootScope()
		rootScope.In("layer_1").SetParam(regularizers.ParamL2, 0.004)
		checkpointHandler := Build(store).TempDir("", "test_checkpoints_").
			Keep(3).MustDone()
		assert.Equal(t, 0, checkpointHandler.checkpointsCount)
		dir = checkpointHandler.Dir()
		fmt.Printf("Checkpoint directory: %s\n", dir)
		e := model.MustNewExec(backend, store, testGraphFn)
		for ii := range 10 {
			results := e.MustExec()
			globalStep := tensors.ToScalar[float64](results[0])
			assert.Equal(t, float64(ii)+1, globalStep, "LoopStep")
			assert.NoError(t, checkpointHandler.Save(), "Saving checkpoint")
		}

		// Check the correct number of checkpoints (3) remain.
		list, err := checkpointHandler.ListCheckpoints()
		assert.NoError(t, err)
		assert.Len(t, list, 3, "Number of remaining checkpoints")
		assert.Equal(t, 10, checkpointHandler.checkpointsCount)
		assert.Equal(t, 9, maxCheckPointCountFromCheckpoints(list))
	}

	// Test loading of variables and parameters.
	{
		// Build model, checkpoint a few times.
		store := model.NewStore()
		store.SetParam("learning_rate", 5.0)       // Value should be overwritten when loading.
		store.SetParam(regularizers.ParamL1, 17.0) // Value should NOT be overwritten when loading.
		root := store.RootScope()
		root.In("layer_1").SetParam(regularizers.ParamL2, 0.004)
		checkpointHandler := Build(store).Dir(dir).Keep(3).ExcludeParams(regularizers.ParamL1).MustDone()

		lr, found := store.GetParam("learning_rate")
		assert.True(t, found, "learning_rate should be set")
		assert.Equal(t, 0.01, lr.(float64), "Params[learning_rate]")
		assert.Equal(t, 17.0, model.GetParamOr(root, regularizers.ParamL1, 0.0))

		var l2 any
		l2, found = root.GetParam(regularizers.ParamL2)
		assert.Truef(t, found, "%s should have been set", regularizers.ParamL2)
		assert.Equal(t, 0.001, l2.(float64), "(Scope=%s) Params[%s]", root.Scope(), regularizers.ParamL2)
		l2, found = root.At("layer_1").GetParam(regularizers.ParamL2)
		assert.Truef(t, found, "%s should have been set", regularizers.ParamL2)
		assert.Equal(t, 0.004, l2.(float64), "Params[%s]", regularizers.ParamL2)

		// If we are lazy loading, no variable should be listed.
		for v := range root.IterVariables() {
			fmt.Printf("\tvariable %q -> %s\n", v.Path(), v.Shape())
			t.Fail()
		}

		// Check that the variable (global_step) is listed, if we look for it.
		v := root.GetVariable(optimizer.GlobalStepVariableName)
		require.NoError(t, v.Shape().Check(dtypes.Int64))

		// Re-execute testGraphFn: it should load global step at 10, increment and return it at 11.
		e := model.MustNewExec(backend, store, testGraphFn)
		results := e.MustExec()
		globalStep := tensors.ToScalar[float64](results[0])
		assert.Equal(t, 11.0, globalStep, "Re-loaded global step")
		assert.NoError(t, checkpointHandler.Save(), "Saving checkpoint")

		// Check the correct number of checkpoints (3) remain.
		list, err := checkpointHandler.ListCheckpoints()
		assert.NoError(t, err)
		assert.Len(t, list, 3, "Number of remaining checkpoints")
	}

	// Test that immediate form also loads the variables correctly.
	{
		store := model.NewStore()
		root := store.RootScope()
		_ = Build(store).Dir(dir).Keep(3).ExcludeParams(regularizers.ParamL1).Immediate().MustDone()

		// Check that the only variable ("global_step") is present.
		count := 0
		for v := range root.IterVariables() {
			// fmt.Printf("\tvariable %q -> %s\n", v.ScopeAndName(), v.Shape())
			require.NoError(t, v.Shape().Check(dtypes.Int64))
			count++
		}
		require.Equal(t, 1, count, "Number of variables should have been one: global_step")

		v := root.GetVariable(optimizer.GlobalStepVariableName)
		require.NoError(t, v.Shape().Check(dtypes.Int64))
		e := model.MustNewExec(backend, store, testGraphFn)
		results := e.MustExec()
		globalStep := tensors.ToScalar[float64](results[0])
		assert.Equal(t, 12.0, globalStep, "Re-loaded global step")
	}

	// Test that one can embed the checkpoints.
	var (
		jsonBlob, binBlob []byte
	)
	{
		// Read the whole checkpoint to a variable -- similar to embedding it.
		store := model.NewStore()
		handler := Build(store).Dir(dir).Keep(3).ExcludeParams(regularizers.ParamL1).Immediate().MustDone()
		checkpoints, err := handler.ListCheckpoints()
		require.NoError(t, err)
		lastCheckpoint := checkpoints[len(checkpoints)-1]
		jsonBlob, err = os.ReadFile(model.JoinPath(dir, lastCheckpoint+JsonNameSuffix))
		require.NoError(t, err)
		binBlob, err = os.ReadFile(model.JoinPath(dir, lastCheckpoint+BinDataSuffix))
	}
	{
		// Check that reading from the variable works.
		store := model.NewStore()
		_, err := Build(store).FromEmbed(string(jsonBlob), binBlob).Immediate().Done()
		require.NoError(t, err)

		// Check that the only variable ("global_step") is present.
		count := 0
		for v := range store.IterVariables() {
			fmt.Printf("\tFromEmbed: variable %q: %s -> %s\n", v.Path(), v.Shape(), v.MustValue())
			require.NoError(t, v.Shape().Check(dtypes.Int64))
			require.Equal(t, "/global_step", v.Path(), "Variable name")
			require.Equal(t, int64(11), tensors.ToScalar[int64](v.MustValue()), "Variable value")
			count++
		}
		require.Equal(t, 1, count, "Number of variables should have been one: global_step")
	}

	// Remove test directory.
	if t.Failed() {
		fmt.Printf("Temporary directory with saved context: %s\n", dir)
	} else {
		assert.NoErrorf(t, os.RemoveAll(dir), "Removing directory used for testing %q", dir)
	}
}

func TestMergedCheckpoints(t *testing.T) {
	backend := testutil.BuildTestBackend()
	var dir string
	{
		store := model.NewStore()
		checkpointHandler := Build(store).TempDir("", "test_checkpoints_").Keep(2).MustDone()
		dir = checkpointHandler.Dir()
		globalStepV := optimizer.GetGlobalStepVar(store)
		err := globalStepV.SetValue(tensors.FromValue(int64(1)))
		require.NoError(t, err)
		xV := store.VariableWithValue("x", []float64{1.0, 1.0, 1.0})
		yV := store.VariableWithValue("y", [][]float32{{4.0}, {4.0}})
		require.NoError(t, checkpointHandler.Save())

		err = globalStepV.SetValue(tensors.FromValue(int64(10)))
		require.NoError(t, err)
		err = xV.SetValue(tensors.FromValue([]float64{3.0, 3.0, 3.0}))
		require.NoError(t, err)
		err = yV.SetValue(tensors.FromValue([][]float32{{6.0}, {6.0}}))
		require.NoError(t, err)
		require.NoError(t, checkpointHandler.Save())
	}
	{
		// Check that the values were averaged:
		store := model.NewStore()
		_ = Build(store).Dir(dir).Keep(2).TakeMean(-1, backend).MustDone()
		globalStep := optimizer.GetGlobalStep(store)
		assert.Equal(t, int64(10), globalStep, "GlobalStep")
		xV := store.VariableWithValue("x", []float64{1.0, 1.0, 1.0})
		// Assume X will be loaded with the mean of the previous 2 checkpoints:
		assert.Equal(t, []float64{2.0, 2.0, 2.0}, xV.MustValue().Value(), "X")
		yV := store.VariableWithValue("y", [][]float32{{4.0}, {4.0}})
		assert.Equal(t, [][]float32{{5.0}, {5.0}}, yV.MustValue().Value(), "Y")
	}

	if t.Failed() {
		fmt.Printf("Temporary directory with saved context: %s\n", dir)
	} else {
		assert.NoErrorf(t, os.RemoveAll(dir), "Removing directory used for testing %q", dir)
	}
}

func TestParams(t *testing.T) {
	var (
		dir                            string
		xFloat64, xFloat32, xInt, xStr = 0.01, float32(7.1), 11, "bar"
		xInts                          = []int{13, 17, 19}
		xStrs                          = []string{"a", "b", "c"}
	)

	{
		// Build model, checkpoint a few times.
		store := model.NewStore()
		store.SetParam("xFloat64", xFloat64)
		store.SetParam("xFloat32", xFloat32)
		store.SetParam("xInt", xInt)
		store.SetParam("xInts", xInts)
		fooScope := store.Scope("foo") // Some different scope.
		fooScope.SetParam("xStr", xStr)
		fooScope.SetParam("xStrs", xStrs)

		checkpointHandler := Build(store).TempDir("", "test_checkpoints_").Keep(3).MustDone()
		dir = checkpointHandler.Dir()
		require.NoError(t, checkpointHandler.Save())
	}

	// Test loading of values
	{
		// Build model, checkpoint a few times.
		store := model.NewStore()
		_ = Build(store).Dir(dir).Keep(3).MustDone()

		got, found := store.GetParam("xFloat64")
		require.True(t, found)
		assert.Equal(t, xFloat64, got)

		got, found = store.GetParam("xFloat32")
		assert.True(t, found)
		assert.Equal(t, xFloat32, got)

		got, found = store.GetParam("xInt")
		assert.True(t, found)
		assert.Equal(t, xInt, got)

		got, found = store.GetParam("xInts")
		assert.True(t, found)
		assert.Equal(t, xInts, got)

		fooScope := store.Scope("foo")
		got, found = fooScope.GetParam("xStr")
		assert.True(t, found)
		assert.Equal(t, xStr, got)

		got, found = fooScope.GetParam("xStrs")
		assert.True(t, found)
		assert.Equal(t, xStrs, got)
	}

	if t.Failed() {
		fmt.Printf("Temporary directory with saved context: %s\n", dir)
	} else {
		assert.NoErrorf(t, os.RemoveAll(dir), "Removing directory used for testing %q", dir)
	}
}

func Test_compressedBin(t *testing.T) {
	dirT := t.TempDir()
	const size = 10240

	sZip := model.JoinPath(dirT, "test.zip.bin")
	s := model.JoinPath(dirT, "test.bin")
	buffer := make([]byte, size)
	_, _ = rand.Read(buffer)
	_ = os.WriteFile(s, buffer, 0666)

	wr, err := getSaveVarFiles(sZip, BinGZIP)
	require.NoError(t, err)
	_, err = wr.Write(buffer)
	require.NoError(t, err)
	_ = wr.Flush()
	_ = wr.Close()

	f, err := os.Open(sZip)
	require.NoError(t, err)
	defer func() { _ = f.Close() }()
	zRd, err := getLoadVarFilesFromReader(f)
	require.NoError(t, err)
	b1, err := io.ReadAll(zRd)
	require.NoError(t, err)
	require.Equal(t, buffer, b1)

	f1, err := os.Open(s)
	require.NoError(t, err)
	defer func() { _ = f1.Close() }()
	zRd, err = getLoadVarFilesFromReader(f1)
	require.NoError(t, err)
	b2, err := io.ReadAll(zRd)
	require.NoError(t, err)
	require.Equal(t, buffer, b2)
}

func Test_uncompressedBin(t *testing.T) {
	dirT := t.TempDir()
	const size = 10240

	sZip := model.JoinPath(dirT, "test.zip.bin")
	buffer := make([]byte, size)
	_, _ = rand.Read(buffer)

	wr, err := getSaveVarFiles(sZip, BinUncompressed)
	require.NoError(t, err)
	_, err = wr.Write(buffer)
	require.NoError(t, err)
	_ = wr.Flush()
	_ = wr.Close()

	buffer2, err := os.ReadFile(sZip)
	require.NoError(t, err)
	require.Equal(t, buffer, buffer2)
}
