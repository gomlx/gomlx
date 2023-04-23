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
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"math"
	"reflect"
	"strings"
	"testing"
)

func EuclideanDistance(a, b *Node) *Node {
	diff := Sub(a, b)
	return Sqrt(ReduceAllSum(Mul(diff, diff)))
}

func TestExec(t *testing.T) {
	manager := BuildManager().MustDone()
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
		if result.Error() != nil {
			t.Fatalf("Failed to %q.Call(): %+v", dist.Name(), result.Error())
		}
		got := result.Value().(float32)
		want := float32(math.Sqrt(float64(dim)))
		if !slices.Close[float32](want, got) {
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
		result := dist.Call(a, b)[0]
		if result == nil || result.Error() == nil {
			t.Fatalf("EuclideanDistance(%v:%v, %v:%v) should have failed, got %+v", reflect.TypeOf(a), a, reflect.TypeOf(b), b, result)
		}
	}

	// Check different shapes will fail.
	{
		a := []float32{0, 0, 0}
		b := []float32{1, 1}
		result := dist.Call(a, b)[0]
		if result == nil || result.Error() == nil {
			t.Fatalf("EuclideanDistance(%v:%v, %v:%v) should have failed, got %+v", reflect.TypeOf(a), a, reflect.TypeOf(b), b, result)
		}
	}

	// Check that we have another 3 different shapes before we maxed out the cache.
	for dim := 6; dim <= 8; dim++ {
		testForDim(dim)
	}

	// Try a different shape (float64) and we should have run out of cache.
	{
		a := []float64{0, 0}
		b := []float64{1, 1}
		result := dist.Call(a, b)[0]
		if result == nil || result.Error() == nil {
			t.Fatalf("EuclideanDistance(%v:%v, %v:%v) should have failed, got %+v", reflect.TypeOf(a), a, reflect.TypeOf(b), b, result)
		}
		if !strings.Contains(result.Error().Error(), "maximum cache") {
			t.Fatalf("EuclideanDistance(%v:%v, %v:%v) failed on something that was not cache: %+v", reflect.TypeOf(a), a, reflect.TypeOf(b), b, result.Error())
		}
	}

	addAndSubGraph := func(a, b *Node) (sum, add *Node) {
		return Add(a, b), Sub(a, b)
	}

	addAndSub := NewExec(manager, addAndSubGraph)
	{
		a := []float32{2, 2}
		b := []float32{1, 1}
		outputs := addAndSub.Call(a, b)
		if outputs == nil {
			t.Fatalf("%q(%v:%v, %v:%v) got nil", addAndSub.Name(), reflect.TypeOf(a), a, reflect.TypeOf(b), b)
		}
		add, sub := outputs[0].Value(), outputs[1].Value()
		wantAdd, wantSub := []float32{3, 3}, b
		if !slices.DeepSliceCmp(add, wantAdd, slices.Equal[float32]) || !slices.DeepSliceCmp(sub, wantSub, slices.Equal[float32]) {
			t.Fatalf("%q(%v:%v, %v:%v) got (%v, %v), but wanted (%v, %v)", addAndSub.Name(), reflect.TypeOf(a), a, reflect.TypeOf(b), b, add, sub, wantAdd, wantSub)
		}
	}
}

const scalarParamName = "scalar"

func addScalarTest(x *Node) *Node {
	return Add(x, x.Graph().Parameter(scalarParamName, MakeShape(F64)))
}

func TestExecWithSideParams(t *testing.T) {
	manager := BuildManager().MustDone()
	scalar := tensor.FromValue(3.0).Device(manager, manager.DefaultDeviceNum())
	setSideParams := func(g *Graph, tensors []*tensor.Device) {
		node := g.ParameterByName(scalarParamName)
		tensors[node.ParameterHandle()] = scalar
	}
	addScalar := NewExec(manager, addScalarTest).SetSideParamsHook(setSideParams)

	x := []float64{1, 2}
	want := []float64{4, 5}
	got := addScalar.Call(x)[0]
	if !slices.DeepSliceCmp(want, got.Value(), slices.Equal[float64]) {
		t.Fatalf("addScalar(%v, 3): got %v, wanted %v", x, got, want)
	}

	scalar = tensor.FromValue(10.0).Device(manager, manager.DefaultDeviceNum())
	want = []float64{11, 12}
	got = addScalar.Call(x)[0]
	if !slices.DeepSliceCmp(want, got.Value(), slices.Equal[float64]) {
		t.Fatalf("addScalar(%v, 10): got %v, wanted %v", x, got, want)
	}

	x = []float64{0, 1, 2}
	want = []float64{10, 11, 12}
	got = addScalar.Call(x)[0]
	if !slices.DeepSliceCmp(want, got.Value(), slices.Equal[float64]) {
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
	manager := BuildManager().MustDone()
	concat, err := NewExecAny(manager, concatGraph)
	if err != nil {
		t.Fatalf("Failed to create concatGraph: %+v", err)
	}

	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{10}, {20}}
	{
		got := concat.Call(a, b)[0]
		want := [][]float64{{1, 2, 10}, {3, 4, 20}}
		if !slices.DeepSliceCmp(want, got.Value(), slices.Equal[float64]) {
			t.Fatalf("concat(%v, %v): got %v, wanted %v", a, b, got, want)
		}
	}

	c := [][]float64{{100, 101}, {200, 201}}
	{
		got := concat.Call(a, b, c)[0]
		want := [][]float64{{1, 2, 10, 100, 101}, {3, 4, 20, 200, 201}}
		if !slices.DeepSliceCmp(want, got.Value(), slices.Equal[float64]) {
			t.Fatalf("concat(%v, %v, %v): got %v, wanted %v", a, b, c, got, want)
		}
	}

	addSub, err := NewExecAny(manager, addSubGraph)
	if err != nil {
		t.Fatalf("Failed to create addSubGraph: %+v", err)
	}
	{
		gotTuple := addSub.Call(c, a)
		want0 := [][]float64{{101, 103}, {203, 205}}
		want1 := [][]float64{{99, 99}, {197, 197}}
		if !slices.DeepSliceCmp(want0, gotTuple[0].Value(), slices.Equal[float64]) {
			t.Errorf("addSub(%v, %v)[0]: got %v, wanted %v", c, a, gotTuple[0].Local(), want0)
		}
		if !slices.DeepSliceCmp(want1, gotTuple[1].Value(), slices.Equal[float64]) {
			t.Errorf("addSub(%v, %v)[1]: got %v, wanted %v", c, a, gotTuple[1].Local(), want1)
		}
	}
}

func concatWithLoggedFirstNodeGraph(nodes []*Node) *Node {
	nodes[0].SetLogged("first concat node")
	return Concatenate(nodes, -1)
}

func TestExecWithLogger(t *testing.T) {
	manager := BuildManager().MustDone()

	concat, err := NewExecAny(manager, concatWithLoggedFirstNodeGraph)
	if err != nil {
		t.Fatalf("Failed to create concatWithLoggedFirstNodeGraph: %+v", err)
	}

	var firstNodeValue tensor.Tensor
	concat.SetNodeLogger(func(messages []string, values []tensor.Tensor) {
		if len(messages) != 1 {
			t.Fatalf("Only one node marked for logging, got %d logged nodes", len(messages))
		}
		firstNodeValue = values[0]
		fmt.Printf("\tLogger: %s: %v\n", messages[0], values[0].Local())
	})

	a := [][]float64{{1, 2}, {3, 4}}
	b := [][]float64{{10}, {20}}
	{
		got := concat.Call(a, b)[0]
		want := [][]float64{{1, 2, 10}, {3, 4, 20}}
		if !slices.DeepSliceCmp(want, got.Value(), slices.Equal[float64]) {
			t.Fatalf("concat(%v, %v): got %v, wanted %v", a, b, got, want)
		}
		// firstNodeValue must have been set by our custom logger, concat.Call().
		if !slices.DeepSliceCmp(a, firstNodeValue.Value(), slices.Equal[float64]) {
			t.Fatalf("concat(%v, %v): got first node %v, wanted %v", a, b, firstNodeValue, a)
		}
	}
}

func TestExecWithNoInputs(t *testing.T) {
	manager := BuildManager().MustDone()
	matrixInitFn := NewExec(manager, func(g *Graph) *Node {
		return IotaFull(g, shapes.Make(shapes.Int64, 3, 3))
	})
	got := matrixInitFn.Call()[0]
	want := [][]int{{0, 1, 2}, {3, 4, 5}, {6, 7, 8}}
	if !slices.DeepSliceCmp(want, got.Value(), slices.Equal[int]) {
		t.Fatalf("matrixInitFn(): got %v, wanted %v", got, want)
	}
}
