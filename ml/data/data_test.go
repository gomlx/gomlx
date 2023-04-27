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

package data

import (
	"io"
	"sync/atomic"
	"testing"

	"github.com/gomlx/gomlx/types/tensor"
	"github.com/stretchr/testify/require"
)

type testDS struct {
	count atomic.Int64
}

var (
	testDSMaxValue = int64(10000)
)

func (ds *testDS) Name() string { return "testDS" }
func (ds *testDS) Reset()       { ds.count.Store(0) }
func (ds *testDS) Yield() (spec any, inputs []tensor.Tensor, labels []tensor.Tensor, err error) {
	value := ds.count.Add(1)
	if value > testDSMaxValue {
		err = io.EOF
		return
	}
	inputs = []tensor.Tensor{nil} // One nil element.
	return                        // As if a batch was returned.
}

// TestNewParallelDataset with and without cache.
func TestNewParallelDataset(t *testing.T) {
	for _, cacheSize := range []int{0, 10} {
		ds := &testDS{}
		pDS := NewParallelDataset(ds, 0, cacheSize)
		count := int64(0)
		for {
			_, inputs, _, err := pDS.Yield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err, "Test failed with unexpected error")
			require.Len(t, inputs, 1, "Expected Dataset to yield 1 input tensor")
			count++
		}
		require.Equalf(t, testDSMaxValue, count, "Number of yielded batches first loop, cacheSize=%d.", cacheSize)
		count = 0
		pDS.Reset()
		for {
			_, inputs, _, err := pDS.Yield()
			if err == io.EOF {
				break
			}
			require.NoError(t, err, "Test failed with unexpected error")
			require.Len(t, inputs, 1, "Expected Dataset to yield 1 input tensor")
			count++
		}
		require.Equal(t, testDSMaxValue, count, "Number of yielded batches at second loop, cacheSize=%d.", cacheSize)
	}
}
