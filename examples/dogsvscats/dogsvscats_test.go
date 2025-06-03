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

package dogsvscats

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"io"
	"path"
	"runtime"
	"testing"
)

var (
	flagDataDir      = flag.String("data", "~/tmp/dogs_vs_cats", "Directory to cache downloaded and generated dataset files.")
	flagAngleStdDev  = flag.Float64("angle", 5.0, "Standard deviation of noise used to rotate the image. Disabled if --augment=false.")
	flagFlipRandomly = flag.Bool("flip", true, "Randomly flip the image horizontally. Disabled if --augment=false.")
	flagBatchSize    = flag.Int("batch", DefaultConfig.BatchSize, "Batch size for training")
)

func buildConfig() *PreprocessingConfiguration {
	*flagDataDir = data.ReplaceTildeInDir(*flagDataDir)
	config := &PreprocessingConfiguration{}
	*config = *DefaultConfig
	config.DataDir = *flagDataDir
	config.AngleStdDev = *flagAngleStdDev
	config.FlipRandomly = *flagFlipRandomly
	config.BatchSize = *flagBatchSize

	return config
}

func filesDownloaded(config *PreprocessingConfiguration) bool {
	if !data.FileExists(config.DataDir) {
		return false
	}
	imagesDir := path.Join(config.DataDir, LocalZipDir)
	if !data.FileExists(imagesDir) {
		return false
	}
	return true
}

func BenchmarkDataset(b *testing.B) {
	config := buildConfig()
	config.ForceOriginal = true
	config.UseParallelism = false
	if !filesDownloaded(config) {
		b.Skipf("Files not yet downloaded to %q. Use demo to download files.", config.DataDir)
	}
	trainDS, trainEvalDS, _ := CreateDatasets(config)
	datasets := []train.Dataset{trainDS, trainEvalDS}
	for dsIdx, dsType := range []string{"train-augmented", "eval"} {
		for _, useParallelism := range []bool{false, true} {
			name := dsType
			ds := datasets[dsIdx]
			ds.Reset()
			if useParallelism {
				name = fmt.Sprintf("%s-parallel(%d)", name, runtime.NumCPU())
			}
			b.Run(name, func(b *testing.B) {
				if useParallelism {
					ds = data.CustomParallel(ds).Buffer(10).Start()
				}
				for ii := 0; ii < b.N; ii++ {
					_, _, _, err := ds.Yield()
					if err != nil {
						if err == io.EOF {
							ds.Reset()
						} else {
							b.Fatalf("Failed reading dataset: %+v", err)
						}
					}
				}
			})
		}
	}
}
