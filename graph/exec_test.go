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
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"math"
	"reflect"
	"testing"
)

func EuclideanDistance(a, b *Node) *Node {
	return Sqrt(ReduceAllSum(Square(Sub(a, b))))
}

func TestExec(t *testing.T) {
	manager := buildTestManager()
	dist := NewExec(manager, EuclideanDistance).SetMaxCache(10)
	fmt.Printf("\tExec name: %s\n", dist.Name())
	testForDim := func(dim int) {
		a := make([]float32, dim)
		b := make([]float32, dim)
		for ii := range b {
			b[ii] = 1
		}
		slice := dist.Call(a, b)
		if len(slice) == 0 {
			t.Fatalf("Failed to %q.Call(), returned %d elements, wanted 1 only.", dist.Name(), len(slice))
		}
		result := slice[0]
		got := result.Value().(float32)
		want := float32(math.Sqrt(float64(dim)))
		if !xslices.Close[float32](want, got) {
			t.Fatalf("EuclideanDistance(%v to %v): want %.5f, got %.5f", a, b, want, got)
		}
	}
	for dim := 1; dim <= 5; dim++ {
		testForDim(dim)
	}

	// Check that different types will fail.
	{
		a := []float64{0, 0}
		b := []float32{1, 1}
		var results []tensors.Tensor
		require.Panicsf(t, func() { results = dist.Call(a, b) },
			"EuclideanDistance(%v:%v, %v:%v) should have failed, got %+v",
			reflect.TypeOf(a), a, reflect.TypeOf(b), b, results)
	}

	// Check different shapes will fail.
	{
		a := []float32{0, 0, 0}
		b := []float32{1, 1}
		var results []tensors.Tensor
		require.Panicsf(t, func() { results = dist.Call(a, b) },
			"EuclideanDistance(%v:%v, %v:%v) should have failed, got %+v",
			reflect.TypeOf(a), a, reflect.TypeOf(b), b, results)
	}

	// Check that we have another 5 different shapes before we maxed out the cache.
	for dim := 6; dim <= 10; dim++ {
		testForDim(dim)
	}

	// Try a different shape (float64) so that we run out of cache.
	{
		a := []float64{0, 0}
		b := []float64{1, 1}
		var results []tensors.Tensor
		err := TryCatch[error](func() { results = dist.Call(a, b) })
		require.Errorf(t, err, "EuclideanDistance(%v:%v, %v:%v) should have failed, got %+v",
			reflect.TypeOf(a), a, reflect.TypeOf(b), b, results)
		require.ErrorContainsf(t, err, "maximum cache",
			"EuclideanDistance(%v:%v, %v:%v) failed on something that was not cache: %+v",
			reflect.TypeOf(a), a, reflect.TypeOf(b), b, err)
	}

	addAndSubGraph := func(a, b *Node) (sum, add *Node) {
		return Add(a, b), Sub(a, b)
	}

	addAndSub := NewExec(manager, addAndSubGraph)
	{
		a := []float32{2, 2}
		b := []float32{1, 1}
		var outputs []tensors.Tensor
		require.NotPanicsf(t, func() { outputs = addAndSub.Call(a, b) },
			"%q(%v:%v, %v:%v) failed", addAndSub.Name(), reflect.TypeOf(a), a, reflect.TypeOf(b), b)
		add, sub := outputs[0].Value(), outputs[1].Value()
		wantAdd, wantSub := []float32{3, 3}, b
		if !xslices.DeepSliceCmp(add, wantAdd, xslices.Equal[float32]) || !xslices.DeepSliceCmp(sub, wantSub, xslices.Equal[float32]) {
			t.Fatalf("%q(%v:%v, %v:%v) got (%v, %v), but wanted (%v, %v)", addAndSub.Name(), reflect.TypeOf(a), a, reflect.TypeOf(b), b, add, sub, wantAdd, wantSub)
		}
	}
}

const scalarParamName = "scalar"

func addScalarTest(x *Node) *Node {
	return Add(x, x.Graph().Parameter(scalarParamName, MakeShape(F64)))
}

func TestExecWithSideParams(t *testing.T) {
	manager := buildTestManager()
	scalar := tensors.FromValue(3.0).Device(manager, manager.DefaultDeviceNum())
	setSideParams := func(g *Graph, tensors []*tensors.Device) {
		node := g.GetParameterByName(scalarParamName)
		tensors[node.GetParameterHandle()] = scalar
	}
	addScalar := NewExec(manager, addScalarTest).SetSideParamsHook(setSideParams)

	x := []float64{1, 2}
	want := []float64{4, 5}
	got := addScalar.Call(x)
	if !xslices.DeepSliceCmp(want, got[0].Value(), xslices.Equal[float64]) {
		t.Fatalf("addScalar(%v, 3): got %v, wanted %v", x, got, want)
	}

	scalar = tensors.FromValue(10.0).Device(manager, manager.DefaultDeviceNum())
	want = []float64{11, 12}
	got = addScalar.Call(x)
	if !xslices.DeepSliceCmp(want, got[0].Value(), xslices.Equal[float64]) {
		t.Fatalf("addScalar(%v, 10): got %v, wanted %v", x, got, want)
	}

	x = []float64{0, 1, 2}
	want = []float64{10, 11, 12}
	got = addScalar.Call(x)
	if !xslices.DeepSliceCmp(want, got[0].Value(), xslices.Equal[float64]) {
		t.Fatalf("addScalar(%v, 10): got %v, wanted %v", x, got, want)
	}
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
	manager := buildTestManager()
	concat := NewExecAny(manager, concatGraph)

	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{10}, {20}}
	{
		results := concat.Call(a, b)
		got := results[0]
		want := [][]float64{{1, 2, 10}, {3, 4, 20}}
		if !xslices.DeepSliceCmp(want, got.Value(), xslices.Equal[float64]) {
			t.Fatalf("concat(%v, %v): got %v, wanted %v", a, b, got, want)
		}
	}

	c := [][]float64{{100, 101}, {200, 201}}
	{
		results := concat.Call(a, b, c)
		got := results[0]
		want := [][]float64{{1, 2, 10, 100, 101}, {3, 4, 20, 200, 201}}
		if !xslices.DeepSliceCmp(want, got.Value(), xslices.Equal[float64]) {
			t.Fatalf("concat(%v, %v, %v): got %v, wanted %v", a, b, c, got, want)
		}
	}

	addSub := NewExecAny(manager, addSubGraph)
	{
		gotTuple := addSub.Call(c, a)
		want0 := [][]float64{{101, 103}, {203, 205}}
		want1 := [][]float64{{99, 99}, {197, 197}}
		if !xslices.DeepSliceCmp(want0, gotTuple[0].Value(), xslices.Equal[float64]) {
			t.Errorf("addSub(%v, %v)[0]: got %v, wanted %v", c, a, gotTuple[0].Local(), want0)
		}
		if !xslices.DeepSliceCmp(want1, gotTuple[1].Value(), xslices.Equal[float64]) {
			t.Errorf("addSub(%v, %v)[1]: got %v, wanted %v", c, a, gotTuple[1].Local(), want1)
		}

		// Test that call with list of tensors also work.
		gotTuple = addSub.Call([]tensors.Tensor{tensors.FromValue(c), tensors.FromValue(a)})
		if !xslices.DeepSliceCmp(want0, gotTuple[0].Value(), xslices.Equal[float64]) {
			t.Errorf("addSub(%v, %v)[0]: got %v, wanted %v", c, a, gotTuple[0].Local(), want0)
		}
		if !xslices.DeepSliceCmp(want1, gotTuple[1].Value(), xslices.Equal[float64]) {
			t.Errorf("addSub(%v, %v)[1]: got %v, wanted %v", c, a, gotTuple[1].Local(), want1)
		}
	}
}

func concatWithLoggedFirstNodeGraph(nodes []*Node) *Node {
	nodes[0].SetLogged("first concat node")
	return Concatenate(nodes, -1)
}

func TestExecWithLogger(t *testing.T) {
	manager := buildTestManager()

	concat := NewExecAny(manager, concatWithLoggedFirstNodeGraph)
	var firstNodeValue tensors.Tensor
	concat.SetNodeLogger(func(_ *Graph, messages []string, values []tensors.Tensor, nodes []NodeId) {
		if len(messages) != 1 {
			t.Fatalf("Only one node marked for logging, got %d logged nodes", len(messages))
		}
		firstNodeValue = values[0]
		fmt.Printf("\tLogger: (node #%d) %s: %v\n", nodes[0], messages[0], values[0].Local())
	})

	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{10}, {20}}
	{
		results := concat.Call(a, b)
		got := results[0]
		want := [][]float64{{1, 2, 10}, {3, 4, 20}}
		if !xslices.DeepSliceCmp(want, got.Value(), xslices.Equal[float64]) {
			t.Fatalf("concat(%v, %v): got %v, wanted %v", a, b, got, want)
		}
		// firstNodeValue must have been set by our custom logger, concat.Call().
		if !xslices.DeepSliceCmp(a, firstNodeValue.Value(), xslices.Equal[float64]) {
			t.Fatalf("concat(%v, %v): got first node %v, wanted %v", a, b, firstNodeValue, a)
		}
	}
}

func TestExecWithNoInputs(t *testing.T) {
	manager := buildTestManager()
	matrixInitFn := NewExec(manager, func(g *Graph) *Node {
		return IotaFull(g, shapes.Make(dtypes.Int64, 3, 3))
	})
	results := matrixInitFn.Call()
	got := results[0].Value()
	want := [][]int64{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}
	assert.Equal(t, want, got)
}
