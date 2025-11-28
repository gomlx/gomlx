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

package graph_test

import (
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/graph/graphtest"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func EuclideanDistance(a, b *Node) *Node {
	return Sqrt(ReduceAllSum(Square(Sub(a, b))))
}

func TestExec(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	testForDim := func(exec *Exec, dim int) {
		a := make([]float32, dim)
		b := xslices.SliceWithValue(dim, float32(1))
		outputs := exec.MustExec(a, b)
		if len(outputs) != 1 {
			t.Fatalf("Failed to %q.MustExec(), returned %d elements, wanted exactly 1.", exec.Name(), len(outputs))
		}
		got := tensors.ToScalar[float32](outputs[0])
		want := float32(math.Sqrt(float64(dim)))
		require.InDeltaf(t, want, got, Epsilon, "EuclideanDistance(%v to %v): want %.5f, got %.5f", a, b, want, got)
	}

	t.Run("VariousDims", func(t *testing.T) {
		dist := MustNewExec(backend, EuclideanDistance).SetMaxCache(10)
		fmt.Printf("\tExec name: %s\n", dist.Name())
		for dim := 1; dim <= 5; dim++ {
			testForDim(dist, dim)
		}
	})

	// Check that different types will fail.
	t.Run("InvalidDTypes", func(t *testing.T) {
		dist := MustNewExec(backend, EuclideanDistance)
		a := []float64{0, 0}
		b := []float32{1, 1}
		var results []*tensors.Tensor
		require.Panicsf(t, func() { results = dist.MustExec(a, b) },
			"EuclideanDistance(%v:%v, %v:%v) should have failed, got %+v",
			reflect.TypeOf(a), a, reflect.TypeOf(b), b, results)
	})

	// Check different shapes will fail.
	t.Run("InvalidShapes", func(t *testing.T) {
		dist := MustNewExec(backend, EuclideanDistance)
		a := []float32{0, 0, 0}
		b := []float32{1, 1}
		results, err := dist.Exec(a, b)
		require.Errorf(t, err,
			"EuclideanDistance(%v:%v, %v:%v) should have failed, got %+v",
			reflect.TypeOf(a), a, reflect.TypeOf(b), b, results)
	})

	// Check out-of-cache failure.
	t.Run("OutOfCache", func(t *testing.T) {
		dist := MustNewExec(backend, EuclideanDistance).SetMaxCache(10)
		for dim := range 10 {
			testForDim(dist, dim+1)
		}

		// Try with a different dtype (float64) so that we run out of cache.
		a := []float64{0, 0}
		b := []float64{1, 1}
		results, err := dist.Exec(a, b)
		require.Errorf(t, err, "EuclideanDistance(%v:%v, %v:%v) should have failed, got %+v",
			reflect.TypeOf(a), a, reflect.TypeOf(b), b, results)
		require.ErrorContainsf(t, err, "maximum cache",
			"EuclideanDistance(%v:%v, %v:%v) failed on something that was not cache: %+v",
			reflect.TypeOf(a), a, reflect.TypeOf(b), b, err)
	})

	// Check different shapes will fail.
	t.Run("InvalidConversion", func(t *testing.T) {
		dist := MustNewExec(backend, EuclideanDistance)
		a := [][]float32{{0}, {0}, {0}}
		b := [][]float32{{0}, {0}, {0, 1}} // Inconsistent shape for b, the conversion to Tensor should fail.
		results, err := dist.Exec(a, b)
		fmt.Printf("- Expected error: %v\n", err)
		require.Errorf(t, err,
			"EuclideanDistance(%v:%v, %v:%v) should have failed, got %+v",
			reflect.TypeOf(a), a, reflect.TypeOf(b), b, results)
	})

	t.Run("AddAndSub", func(t *testing.T) {
		addAndSubGraph := func(a, b *Node) (sum, add *Node) {
			return Add(a, b), Sub(a, b)
		}
		addAndSub := MustNewExec(backend, addAndSubGraph)
		a := []float32{2, 2}
		b := []float32{1, 1}
		var outputs []*tensors.Tensor
		require.NotPanicsf(t, func() { outputs = addAndSub.MustExec(a, b) },
			"%q(%v:%v, %v:%v) failed", addAndSub.Name(), reflect.TypeOf(a), a, reflect.TypeOf(b), b)
		add, sub := outputs[0].Value(), outputs[1].Value()
		wantAdd, wantSub := []float32{3, 3}, b
		require.Equalf(t, wantAdd, add, "%q(%v:%v, %v:%v) got (%v, %v), but wanted (%v, %v)", addAndSub.Name(), reflect.TypeOf(a), a, reflect.TypeOf(b), b, add, sub, wantAdd, wantSub)
		require.Equalf(t, wantSub, sub, "%q(%v:%v, %v:%v) got (%v, %v), but wanted (%v, %v)", addAndSub.Name(), reflect.TypeOf(a), a, reflect.TypeOf(b), b, add, sub, wantAdd, wantSub)
	})

	t.Run("IotaMatrix", func(t *testing.T) {
		iotaMatrix := MustExecOnce(backend, func(g *Graph) *Node { return IotaFull(g, shapes.Make(dtypes.Int8, 2, 2)) })
		require.Equal(t, [][]int8{{0, 1}, {2, 3}}, iotaMatrix.Value())
	})
}

func TestDonate(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	deviceNum := backends.DeviceNum(0)
	g := NewGraph(backend, "TestDonate")
	x := Parameter(g, "x", shapes.Make(dtypes.Float64))
	p1 := AddScalar(x, 1)
	require.NotPanics(t, func() { g.Compile(p1) })

	input := tensors.FromValue(5.0)

	// Do not donate input:
	output := g.Run(input)[0]
	require.Equal(t, 6.0, output.Value())
	require.True(t, input.Ok()) // input should still be valid.
	require.True(t, input.IsOnDevice(0))

	// Donate input: make sure input is not shared.
	input = tensors.FromValue(5.0)
	input.MustMaterializeOnDevice(backend, false, deviceNum)
	fmt.Printf("TestDonate: IsShared=%v\n", input.IsShared())
	buf, err := DonateTensorBuffer(input, backend, 0)
	require.NoError(t, err)
	output = g.Run(buf)[0]
	require.Equal(t, 6.0, output.Value())
	require.True(t, input.Ok()) // input should still be valid, since local copy stays alive.
	require.False(t, input.IsOnDevice(0))

	// Donate input with shared buffer:
	input = tensors.FromValue(11.0)
	input.MustMaterializeOnDevice(backend, true, deviceNum)
	fmt.Printf("TestDonate (shared requested): IsShared=%v\n", input.IsShared())
	buf, err = DonateTensorBuffer(input, backend, deviceNum)
	require.NoError(t, err)
	output = g.Run(buf)[0]
	require.Equal(t, 12.0, output.Value())
	require.False(t, input.Ok()) // input is no longer valid, since there are no local copies.
}

const scalarParamName = "scalar"

func addScalarTest(x *Node) *Node {
	sideParam := Parameter(x.Graph(), scalarParamName, shapes.Make(dtypes.Float64))
	return Add(x, sideParam)
}

func TestExecWithSideParams(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	scalarBuffer, err := tensors.FromValue(3.0).DonateBuffer(backend, 0)
	require.NoError(t, err)
	setSideParams := func(g *Graph, inputBuffers []backends.Buffer, donate []bool) error {
		node := g.GetParameterByName(scalarParamName)
		handle := node.GetParameterHandle()
		inputBuffers[handle] = scalarBuffer
		donate[handle] = false
		return nil
	}

	addScalar := MustNewExec(backend, addScalarTest).SetSideParamsHook(setSideParams)
	x := []float64{1, 2}
	want := []float64{4, 5}
	got := addScalar.MustExec(x)[0]
	require.Equal(t, want, got.Value(), "addScalar(%v, 3): got %v, wanted %v", x, got, want)

	scalarBuffer, err = tensors.FromValue(10.0).DonateBuffer(backend, 0)
	require.NoError(t, err)
	want = []float64{11, 12}
	got, err = addScalar.Exec1(x)
	require.NoError(t, err)
	require.Equal(t, want, got.Value(), "addScalar(%v, 10): got %v, wanted %v", x, got, want)

	x = []float64{0, 1, 2}
	want = []float64{10, 11, 12}
	got, err = addScalar.Exec1(x)
	require.NoError(t, err)
	require.Equal(t, want, got.Value(), "addScalar(%v, 10): got %v, wanted %v", x, got, want)
}

func concatGraph(nodes []*Node) *Node {
	return Concatenate(nodes, -1)
}

func addSubGraph(a, b *Node) []*Node {
	return []*Node{
		Add(a, b),
		Sub(a, b),
	}
}

func TestExecWithSlices(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	concat := MustNewExecAny(backend, concatGraph)

	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{10}, {20}}
	{
		got := concat.MustExec(a, b)[0]
		want := [][]float64{{1, 2, 10}, {3, 4, 20}}
		require.Equalf(t, want, got.Value(), "concat([%v, %v]): got %v, wanted %v", a, b, got, want)
	}

	c := [][]float64{{100, 101}, {200, 201}}
	{
		got := concat.MustExec(a, b, c)[0]
		want := [][]float64{{1, 2, 10, 100, 101}, {3, 4, 20, 200, 201}}
		require.Equalf(t, want, got.Value(), "concat([%v, %v, %v]): got %v, wanted %v", a, b, c, got, want)
	}

	addSub := MustNewExecAny(backend, addSubGraph)
	{
		got := addSub.MustExec(c, a)
		want0 := [][]float64{{101, 103}, {203, 205}}
		want1 := [][]float64{{99, 99}, {197, 197}}
		require.Equalf(t, want0, got[0].Value(), "addSub([%v, %v]): got[0]=%v, wanted %v", c, a, got[0], want0)
		require.Equalf(t, want1, got[1].Value(), "addSub([%v, %v]): got[1]=%v, wanted %v", c, a, got[1], want1)

		// Test that call with list of tensors also work.
		got = addSub.MustExec([]*tensors.Tensor{tensors.FromValue(c), tensors.FromValue(a)})
		require.Equalf(t, want0, got[0].Value(), "addSub([%v, %v]): got[0]=%v, wanted %v", c, a, got[0], want0)
		require.Equalf(t, want1, got[1].Value(), "addSub([%v, %v]): got[1]=%v, wanted %v", c, a, got[1], want1)
	}
}

func concatWithLoggedFirstNodeGraph(nodes []*Node) *Node {
	nodes[0].SetLogged("first concat node")
	return Concatenate(nodes, -1)
}

func TestExecWithLogger(t *testing.T) {
	manager := graphtest.BuildTestBackend()

	concat := MustNewExecAny(manager, concatWithLoggedFirstNodeGraph)
	var firstNodeValue *tensors.Tensor
	concat.SetNodeLogger(func(_ *Graph, messages []string, values []*tensors.Tensor, nodes []NodeId) {
		if len(messages) != 1 {
			t.Fatalf("Only one node marked for logging, got %d logged nodes", len(messages))
		}
		firstNodeValue = values[0]
		fmt.Printf("\tLogger: (node #%d) %s: %v\n", nodes[0], messages[0], values[0])
	})

	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{10}, {20}}
	{
		got := concat.MustExec(a, b)[0]
		want := [][]float64{{1, 2, 10}, {3, 4, 20}}
		require.Equalf(t, want, got.Value(), "concat(%v, %v): got %v, wanted %v", a, b, got, want)

		// firstNodeValue must have been set by our custom logger, concat.MustExec().
		require.Equalf(t, a, firstNodeValue.Value(), "concat(%v, %v): got first node %v, wanted %v", a, b, firstNodeValue, a)
	}
}

func TestExecWithNoInputs(t *testing.T) {
	backend := graphtest.BuildTestBackend()
	matrixInitFn := MustNewExec(backend, func(g *Graph) *Node {
		return IotaFull(g, shapes.Make(dtypes.Int64, 3, 3))
	})
	results := matrixInitFn.MustExec()
	got := results[0].Value()
	want := [][]int64{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}
	assert.Equal(t, want, got)
}

// TestExecUnusedInput checks that it should work if an input is not used in the computation.
func TestExecUnusedInput(t *testing.T) {
	backend := graphtest.BuildTestBackend()

	// One of two variables is not used.
	unusedInputFn := MustNewExec(backend, func(x, y *Node) *Node {
		return OnePlus(x)
	})
	_, err := unusedInputFn.Exec(0, 1)
	require.NoError(t, err)

	// x is only used to get the Graph object, but not its value.
	unusedInputFn = MustNewExec(backend, func(x *Node) *Node {
		return IotaFull(x.Graph(), shapes.Make(dtypes.Int32, 3))
	})
	_, err = unusedInputFn.Exec(0)
	require.NoError(t, err)
}
