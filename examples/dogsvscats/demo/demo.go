// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

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
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/exceptions"
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
	paramsSet := check1(commandline.ParseContextSettings(ctx, *settings))

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
		check(os.MkdirAll(*flagDataDir, 0777))
	}
	check(dogsvscats.Download(*flagDataDir))
	check(dogsvscats.FilterValidImages(*flagDataDir))

	config := dogsvscats.NewPreprocessingConfigurationFromContext(ctx, *flagDataDir)
	dogsvscats.PreGenerate(config, *flagPreGenEpochs, true)
}

// check reports and exits on error.
func check(err error) {
	if err == nil {
		return
	}
	klog.Fatalf("Fatal error: %+v", err)
}

// check1 reports and exits on error. Otherwise returns the value passed.
func check1[T any](v T, err error) T {
	check(err)
	return v
}
