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

// IMDB Movie Review library (imdb) demo: you can run this program in 4 different ways:
package main

import (
	"flag"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/examples/imdb"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

// DType used for the demo.
const DType = dtypes.Float32

var (
	flagDataDir    = flag.String("data", "~/tmp/imdb", "Directory to cache downloaded and generated dataset files.")
	flagEval       = flag.Bool("eval", true, "Whether to evaluate the model on the validation data in the end.")
	flagVerbosity  = flag.Int("verbosity", 1, "Level of verbosity, the higher the more verbose.")
	flagCheckpoint = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
)

func main() {
	ctx := imdb.CreateDefaultContext()
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *settings))
	err := exceptions.TryCatch[error](func() {
		imdb.TrainModel(ctx, *flagDataDir, *flagCheckpoint, paramsSet, *flagEval, *flagVerbosity)
	})
	if err != nil {
		klog.Fatalf("Failed with error: %+v", err)
	}
}
