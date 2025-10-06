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

// demo for Dogs vs Cats library: you can run this program in 3 different ways:
//
//  1. With `demo --download`: it will simply download and unpack Kaggle Cats and Dogs dataset.
//  2. With `demo --pre`: It will pre-generate augmented data for subsequent training: since it spends more time
//     augmenting data than training, this is handy and accelerates training. But uses up lots of space (~13Gb with
//     the default number of generated epochs).
//  3. With `demo --train`: trains a CNN (convolutional neural network) model for "Dogs vs Cats".
package main

import (
	"flag"
	"os"

	"github.com/gomlx/gomlx/examples/dogsvscats"
	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagDataDir    = flag.String("data", "~/tmp/dogs_vs_cats", "Directory to cache downloaded dataset and save checkpoints.")
	flagCheckpoint = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
	flagEval       = flag.Bool("eval", true, "Whether to evaluate trained model on test data in the end.")

	// Pre-Generation parameters:
	flagPreGenerate  = flag.Bool("pre", false, "Pre-generate preprocessed image data to speed up training.")
	flagPreGenEpochs = flag.Int("pregen_epochs", 40, "Number of epochs to pre-generate for the training data. Each epoch will take ~310Mb")
)

func main() {
	ctx := dogsvscats.CreateDefaultContext()
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *settings))

	// --force_original better set by
	err := exceptions.TryCatch[error](func() {
		if *flagPreGenerate {
			preGenerate(ctx, *flagDataDir)
		} else {
			dogsvscats.TrainModel(ctx, *flagDataDir, *flagCheckpoint, *flagEval, paramsSet)
		}
	})
	if err != nil {
		klog.Errorf("Error:\n%+v", err)
	}
}

func preGenerate(ctx *context.Context, dataDir string) {
	*flagDataDir = fsutil.MustReplaceTildeInDir(*flagDataDir)
	if !fsutil.MustFileExists(*flagDataDir) {
		must.M(os.MkdirAll(*flagDataDir, 0777))
	}
	must.M(dogsvscats.Download(*flagDataDir))
	must.M(dogsvscats.FilterValidImages(*flagDataDir))

	config := dogsvscats.NewPreprocessingConfigurationFromContext(ctx, *flagDataDir)
	dogsvscats.PreGenerate(config, *flagPreGenEpochs, true)
}
