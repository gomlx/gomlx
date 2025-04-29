/*
 *	Copyright 2025 Rener Castro
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

package mnist

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gopjrt/dtypes"

	_ "github.com/gomlx/gomlx/backends/default"
)

func TestDataset(t *testing.T) {
	dataDir := "~/tmp/mnist"
	dataDir = data.ReplaceTildeInDir(dataDir)
	if err := Download(dataDir); err != nil {
		t.Errorf("Download: %v", err)
		return
	}

	modes := []string{"train", "test"}
	backend := backends.New()
	batchSize := 100

	for _, m := range modes {
		ds, err := NewDataset(backend, "MNIST "+m, dataDir, m, dtypes.Float32)
		if err != nil {
			t.Errorf("NewDataset: %v", err)
			return
		}
		ds.BatchSize(batchSize, false)
		_, images, labels, err := ds.Yield()
		if err != nil {
			t.Errorf("Download: %v", err)
			return
		}
		images[0].Shape().AssertDims(batchSize, Width, Height, 3)
		labels[0].Shape().AssertDims(batchSize, 1)
		if ds.NumExamples() != mnistSamples[m] {
			t.Fatalf("size different ds.NumExamples(%d) != mnistSamples(%d) ", ds.NumExamples(), mnistSamples[m])
			return
		}
	}
}
