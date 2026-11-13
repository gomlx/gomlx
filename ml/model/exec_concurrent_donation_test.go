// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package model_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/gomlx/compute/gobackend"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/core/tensors"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/stretchr/testify/require"
)

func TestExecColdConcurrentDonation(t *testing.T) {
	backend, err := gobackend.New("")
	require.NoError(t, err)
	defer backend.Finalize()
	store := model.NewStore()
	const width, layers, workers = 16, 32, 16
	for layer := range layers {
		values := make([]float32, width*width)
		for i := range width {
			values[i*width+i] = 1
		}
		_, err := store.CreateVariable(fmt.Sprintf("/weights/%d", layer), tensors.FromFlatDataAndDimensions(values, width, width))
		require.NoError(t, err)
	}
	exec, err := model.NewExec(backend, store, func(scope *model.Scope, x *Node) *Node {
		for layer := range layers {
			w := scope.Store().GetVariable(fmt.Sprintf("/weights/%d", layer)).NodeValue(x.Graph())
			x = MatMul(x, w)
		}
		return x
	})
	require.NoError(t, err)
	defer exec.Finalize()
	exec.SetMaxCache(-1)
	exec.DonateStore()

	start := make(chan struct{})
	var done sync.WaitGroup
	errors := make([]error, workers)
	correct := make([]bool, workers)
	for worker := range workers {
		done.Add(1)
		go func() {
			defer done.Done()
			values := make([]float32, (worker+1)*width)
			for i := range values {
				values[i] = 1
			}
			x := tensors.FromFlatDataAndDimensions(values, worker+1, width)
			defer x.FinalizeAll()
			<-start
			out, err := exec.Exec(x)
			errors[worker] = err
			if err != nil {
				return
			}
			defer out[0].FinalizeAll()
			correct[worker] = true
			for _, row := range out[0].Value().([][]float32) {
				for _, value := range row {
					correct[worker] = correct[worker] && value == 1
				}
			}
		}()
	}
	close(start)
	done.Wait()
	for worker := range workers {
		require.NoError(t, errors[worker], "worker %d", worker)
		require.True(t, correct[worker], "worker %d", worker)
	}
}
