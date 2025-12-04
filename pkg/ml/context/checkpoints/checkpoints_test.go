/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package checkpoints

// TODO:
// * Test that previously loaded variables -- not used by the Context -- are also saved.
// * Test what happens with saving/loading of objects in Params: do they need to be filtered?

import (
	"crypto/rand"
	"fmt"
	"io"
	"os"
	"path"
	"testing"

	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/layers/regularizers"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
)

func TestCheckpoints(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	// Graph function to test: it simply creates, increments and returns the global step.
	testGraphFn := func(ctx *context.Context, g *Graph) *Node {
		return optimizers.IncrementGlobalStepGraph(ctx, g, dtypes.Float64)
	}
	var dir string
	{
		// Build model, checkpoint a few times.
		ctx := context.New()
		ctx.SetParam("learning_rate", 0.01)
		ctx.SetParam(regularizers.ParamL2, 0.001)
		ctx.SetParam(regularizers.ParamL1, 1.0e7)
		ctx.In("layer_1").SetParam(regularizers.ParamL2, 0.004)
		checkpoint := Build(ctx).TempDir("", "test_checkpoints_").
			Keep(3).MustDone()
		assert.Equal(t, 0, checkpoint.checkpointsCount)
		dir = checkpoint.Dir()
		fmt.Printf("Checkpoint directory: %s\n", dir)
		e := context.MustNewExec(backend, ctx, testGraphFn)
		for ii := 0; ii < 10; ii++ {
			results := e.MustExec()
			globalStep := tensors.ToScalar[float64](results[0])
			assert.Equal(t, float64(ii)+1, globalStep, "LoopStep")
			assert.NoError(t, checkpoint.Save(), "Saving checkpoint")
		}

		// Check the correct number of checkpoints (3) remain.
		list, err := checkpoint.ListCheckpoints()
		assert.NoError(t, err)
		assert.Len(t, list, 3, "Number of remaining checkpoints")
		assert.Equal(t, 10, checkpoint.checkpointsCount)
		assert.Equal(t, 9, maxCheckPointCountFromCheckpoints(list))
	}

	// Test loading of variables and parameters.
	{
		// Build model, checkpoint a few times.
		ctx := context.New()
		ctx.SetParam("learning_rate", 5.0)       // Value should be overwritten when loading.
		ctx.SetParam(regularizers.ParamL1, 17.0) // Value should NOT be overwritten when loading.
		checkpoint := Build(ctx).Dir(dir).Keep(3).ExcludeParams(regularizers.ParamL1).MustDone()

		lr, found := ctx.GetParam("learning_rate")
		assert.True(t, found, "learning_rate should be set")
		assert.Equal(t, 0.01, lr.(float64), "Params[learning_rate]")
		assert.Equal(t, 17.0, context.GetParamOr(ctx, regularizers.ParamL1, 0.0))

		var l2 any
		l2, found = ctx.GetParam(regularizers.ParamL2)
		assert.Truef(t, found, "%s should have been set", regularizers.ParamL2)
		assert.Equal(t, 0.001, l2.(float64), "(Scope=%s) Params[%s]", ctx.Scope(), regularizers.ParamL2)
		l2, found = ctx.In("layer_1").GetParam(regularizers.ParamL2)
		assert.Truef(t, found, "%s should have been set", regularizers.ParamL2)
		assert.Equal(t, 0.004, l2.(float64), "Params[%s]", regularizers.ParamL2)

		// If we are lazy loading, no variable should be listed.
		for v := range ctx.IterVariables() {
			fmt.Printf("\tvariable %q -> %s\n", v.ScopeAndName(), v.Shape())
			t.Fail()
		}

		// Check that the variable (global_step) is listed, if we look for it.
		v := ctx.GetVariable(optimizers.GlobalStepVariableName)
		require.NoError(t, v.Shape().Check(dtypes.Int64))

		// Re-execute testGraphFn: it should load global step at 10, increment and return it at 11.
		e := context.MustNewExec(backend, ctx, testGraphFn)
		results := e.MustExec()
		globalStep := tensors.ToScalar[float64](results[0])
		assert.Equal(t, 11.0, globalStep, "Re-loaded global step")
		assert.NoError(t, checkpoint.Save(), "Saving checkpoint")

		// Check the correct number of checkpoints (3) remain.
		list, err := checkpoint.ListCheckpoints()
		assert.NoError(t, err)
		assert.Len(t, list, 3, "Number of remaining checkpoints")
	}

	// Test that immediate form also loads the variables correctly.
	{
		ctx := context.New()
		_ = Build(ctx).Dir(dir).Keep(3).ExcludeParams(regularizers.ParamL1).Immediate().MustDone()

		// Check that the only variable ("global_step") is present.
		count := 0
		for v := range ctx.IterVariables() {
			// fmt.Printf("\tvariable %q -> %s\n", v.ScopeAndName(), v.Shape())
			require.NoError(t, v.Shape().Check(dtypes.Int64))
			count++
		}
		require.Equal(t, 1, count, "Number of variables should have been one: global_step")

		v := ctx.GetVariable(optimizers.GlobalStepVariableName)
		require.NoError(t, v.Shape().Check(dtypes.Int64))
		e := context.MustNewExec(backend, ctx, testGraphFn)
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
		ctx := context.New()
		handler := Build(ctx).Dir(dir).Keep(3).ExcludeParams(regularizers.ParamL1).Immediate().MustDone()
		checkpoints, err := handler.ListCheckpoints()
		require.NoError(t, err)
		lastCheckpoint := checkpoints[len(checkpoints)-1]
		jsonBlob, err = os.ReadFile(path.Join(dir, lastCheckpoint+JsonNameSuffix))
		require.NoError(t, err)
		binBlob, err = os.ReadFile(path.Join(dir, lastCheckpoint+BinDataSuffix))
	}
	{
		// Check that reading from the variable works.
		ctx := context.New()
		_, err := Build(ctx).FromEmbed(string(jsonBlob), binBlob).Immediate().Done()
		require.NoError(t, err)

		// Check that the only variable ("global_step") is present.
		count := 0
		for v := range ctx.IterVariables() {
			fmt.Printf("\tFromEmbed: variable %q: %s -> %s\n", v.ScopeAndName(), v.Shape(), v.MustValue())
			require.NoError(t, v.Shape().Check(dtypes.Int64))
			require.Equal(t, "/global_step", v.ScopeAndName(), "Variable name")
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
	backend := graphtest.BuildTestBackend()
	var dir string
	{
		ctx := context.New().Checked(false)
		checkpoint := Build(ctx).TempDir("", "test_checkpoints_").Keep(2).MustDone()
		dir = checkpoint.Dir()
		globalStepV := optimizers.GetGlobalStepVar(ctx)
		globalStepV.MustSetValue(tensors.FromValue(1))
		xV := ctx.VariableWithValue("x", []float64{1.0, 1.0, 1.0})
		yV := ctx.VariableWithValue("y", [][]float32{{4.0}, {4.0}})
		require.NoError(t, checkpoint.Save())

		globalStepV.MustSetValue(tensors.FromValue(10))
		xV.MustSetValue(tensors.FromValue([]float64{3.0, 3.0, 3.0}))
		yV.MustSetValue(tensors.FromValue([][]float32{{6.0}, {6.0}}))
		require.NoError(t, checkpoint.Save())
	}
	{
		// Check that the values were averaged:
		ctx := context.New().Checked(false)
		_ = Build(ctx).Dir(dir).Keep(2).TakeMean(-1, backend).MustDone()
		globalStep := optimizers.GetGlobalStep(ctx)
		assert.Equal(t, int64(10), globalStep, "GlobalStep")
		xV := ctx.VariableWithValue("x", []float64{1.0, 1.0, 1.0})
		// Assume X will be loaded with the mean of the previous 2 checkpoints:
		assert.Equal(t, []float64{2.0, 2.0, 2.0}, xV.MustValue().Value(), "X")
		yV := ctx.VariableWithValue("y", [][]float32{{4.0}, {4.0}})
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
		ctx := context.New()
		ctx.SetParam("xFloat64", xFloat64)
		ctx.SetParam("xFloat32", xFloat32)
		ctx.SetParam("xInt", xInt)
		ctx.SetParam("xInts", xInts)
		ctx = ctx.In("foo") // Some different scope.
		ctx.SetParam("xStr", xStr)
		ctx.SetParam("xStrs", xStrs)

		checkpoint := Build(ctx).TempDir("", "test_checkpoints_").Keep(3).MustDone()
		dir = checkpoint.Dir()
		require.NoError(t, checkpoint.Save())
	}

	// Test loading of values
	{
		// Build model, checkpoint a few times.
		ctx := context.New()
		_ = Build(ctx).Dir(dir).Keep(3).MustDone()

		got, found := ctx.GetParam("xFloat64")
		require.True(t, found)
		assert.Equal(t, xFloat64, got)

		got, found = ctx.GetParam("xFloat32")
		assert.True(t, found)
		assert.Equal(t, xFloat32, got)

		got, found = ctx.GetParam("xInt")
		assert.True(t, found)
		assert.Equal(t, xInt, got)

		got, found = ctx.GetParam("xInts")
		assert.True(t, found)
		assert.Equal(t, xInts, got)

		ctx = ctx.In("foo")
		got, found = ctx.GetParam("xStr")
		assert.True(t, found)
		assert.Equal(t, xStr, got)

		got, found = ctx.GetParam("xStrs")
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

	sZip := path.Join(dirT, "test.zip.bin")
	s := path.Join(dirT, "test.bin")
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

	sZip := path.Join(dirT, "test.zip.bin")
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
