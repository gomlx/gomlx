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

package collector

import (
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/require"
	"testing"
)

func testDC(t *testing.T, manager *Manager, dc *DataCollector) {
	e := NewExec(manager, func(x *Node) *Node {
		g := x.Graph()
		y := Mul(x, Const(g, shapes.CastAsDType(10, x.DType())))
		dc.Collect(y, "y")
		return y
	})
	err := dc.AttachToExecutor(e)
	if err != nil {
		t.Fatalf("Fail to attach DataCollector to executor: %+v", err)
	}
	for ii := 0; ii < 5; ii++ {
		result, err := e.Call(ii)
		require.NoError(t, err)
		got := tensor.ToScalar[int](result[0])
		want := ii * 10
		if got != want {
			t.Fatalf("wrong execution result: wanted %d, got %d", want, got)
		}
	}
	allSeries := dc.GetAllSeriesNames()
	if len(allSeries) != 1 || allSeries[0] != "y" {
		t.Fatalf("GetAllSeriesNames(): wanted [\"y\"], got %v", allSeries)
	}
}

func TestDataCollector_CollectAll(t *testing.T) {
	manager := BuildManager().Platform("Host").MustDone()
	dc := NewDataCollector().CollectAll()
	testDC(t, manager, dc)
	got := dc.GetSeriesValues("y")
	want := []float64{0, 10, 20, 30, 40}
	if !slices.DeepSliceCmp(got, want, slices.Equal[float64]) {
		t.Fatalf("wrong values collected: wanted %v, got %v", want, got)
	}
}

func TestDataCollector_EveryNSteps(t *testing.T) {
	manager := BuildManager().Platform("Host").MustDone()
	dc := NewDataCollector().EveryNSteps(2)
	testDC(t, manager, dc)
	got := dc.GetSeriesValues("y")
	want := []float64{0, 20, 40}
	if !slices.DeepSliceCmp(got, want, slices.Equal[float64]) {
		t.Fatalf("wrong values collected: wanted %v, got %v", want, got)
	}
}

func TestDataCollector_KeepNPoints(t *testing.T) {
	manager := BuildManager().Platform("Host").MustDone()
	dc := NewDataCollector().KeepNPoints(3)
	testDC(t, manager, dc)
	got := dc.GetSeriesValues("y")
	want := []float64{0, 40} // Whenever it reaches 3 elements, it is split, so the result will never actually have 3 elements.
	if !slices.DeepSliceCmp(got, want, slices.Equal[float64]) {
		t.Fatalf("wrong values collected: wanted %v, got %v", want, got)
	}
}
