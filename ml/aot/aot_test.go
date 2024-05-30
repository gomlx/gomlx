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

package aot_test

import (
	"fmt"
	"os"
	"testing"

	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/graph/graphtest"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/gomlx/gomlx/xla"
	"google3/third_party/golang/testify/assert/assert"
	"github.com/stretchr/testify/require"
)

// EuclideanDistance is the "model" function we want to AOT compile, as an example.
// This is the simplest version, where there are no variables.
func EuclideanDistance(x, y *Node) *Node {
	return Sqrt(ReduceAllSum(Square(Sub(x, y))))
}

func TestAOTCompileModel(t *testing.T) {
	t.Skip()
	manager := graphtest.BuildTestManager()
	exec := NewExec(manager, EuclideanDistance)

	// Build graph with inputs shaped float32[3]. We don't care about the results,
	// actually we are only interested on the graph.
	_, g := exec.CallWithGraph([]float32{0.0, 0.0}, []float32{1.0, 1.0})
	fmt.Printf("Graph:\n%s\n\n", g)
	fmt.Printf("StableHLO version: %s\n", xla.StableHLOCurrentVersion())
	stableHLO := g.ConvertToStableHLO()
	hloV1 := stableHLO.String()
	fmt.Printf("StableHLO:\n%s\n", hloV1)

	fStableHLO, err := os.CreateTemp("", "gomlx_aot_stable_hlo_")
	fStableHLOName := fStableHLO.Name()
	require.NoError(t, err)
	err = stableHLO.SerializeWithVersion(fStableHLO.Fd(), xla.StableHLOCurrentVersion())
	require.NoError(t, err)
	_ = fStableHLO.Close()
	fmt.Printf("StableHLO serialized to %s\n", fStableHLO.Name())

	var serialized []byte
	serialized, err = os.ReadFile(fStableHLOName)
	require.NoError(t, err)
	stableHLO2, err := xla.NewStableHLOFromSerialized(serialized)
	require.NoError(t, err)
	hloV2 := stableHLO2.String()
	fmt.Printf("StableHLO2:\n%s\n", hloV2)
	assert.Equal(t, hloV1, hloV2)

	var aotCompilationResult []byte
	require.NotPanicsf(t, func() { aotCompilationResult = g.AOTCompile() },
		"Failed to run Graph.AOTCompile() on EucliadianDistance model")
	fmt.Printf("Compilation Result: %d bytes\n", len(aotCompilationResult))

	fAot, err := os.CreateTemp("", "gomlx_aot_test_")
	require.NoError(t, err)
	_, _ = fAot.Write(aotCompilationResult)
	require.NoError(t, fAot.Close())
	fmt.Printf("AOTCompilationResult saved to %s\n", fAot.Name())

	aotExec, err := xla.NewAOTExecutable(manager.Client(), aotCompilationResult)
	if err != nil {
		t.Fatalf("Failed to create NewAOTExecutable: %+v", err)
	}

	a := tensor.FromAnyValue([]float32{0.0, 0.0})
	b := tensor.FromAnyValue([]float32{3.0, 4.0})
	buf, err := aotExec.Run([]*xla.OnDeviceBuffer{
		a.Device(manager, manager.DefaultDeviceNum()).ShapedBuffer(),
		b.Device(manager, manager.DefaultDeviceNum()).ShapedBuffer(),
	})
	if err != nil {
		t.Fatalf("Failed to create NewAOTExecutable: %+v", err)
	}
	result := tensor.InternalNewDevice(buf)
	fmt.Printf("Final result: %s\n", result.Local().GoStr())
}
