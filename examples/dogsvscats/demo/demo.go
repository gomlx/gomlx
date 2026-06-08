// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// demo for Dogs vs Cats library: you can run this program in 3 different ways:
//
//  1. With `demo --download`: it will simply download and unpack Kaggle Cats and Dogs dataset.
//  2. With `demo --pre`: It will pre-generate augmented data for subsequent training: since it spends more time
//     augmenting data than training, this is handy and accelerates training. But uses up lots of space (~13Gb with
//     the default number of generated epochs).
//  3. With `demo --train`: trains a CNN (convolutional neural network) model for "Dogs vs Cats" (default).
package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/examples/dogsvscats"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagTrain      = flag.Bool("train", true, "Train the model.")
	flagDownload   = flag.Bool("download", false, "Download and pre-filter Cats & Dogs dataset.")
	flagDataDir    = flag.String("data", "~/tmp/dogs_vs_cats", "Directory to cache downloaded dataset and save checkpoint.")
	flagCheckpoint = flag.String("checkpoint", "", "Directory to save and load checkpoints from. If left empty, no checkpoints are created.")
	flagEval       = flag.Bool("eval", true, "Whether to evaluate trained model on test data in the end.")

	// Pre-Generation parameters:
	flagPreGenerate  = flag.Bool("pre", false, "Pre-generate preprocessed image data to speed up training.")
	flagPreGenEpochs = flag.Int("pregen_epochs", 40, "Number of epochs to pre-generate for the training data. Each epoch will take ~310Mb")
)

func main() {
	store := dogsvscats.CreateModelStore()
	settings := commandline.CreateSettingsFlag(store, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet, err := commandline.ParseSettings(store, *settings)
	if err != nil {
		klog.Fatalf("Fatal error parsing settings: %+v", err)
	}

	backend := compute.MustNew()
	fmt.Printf("Backend: %s\n\t%s\n", backend.Name(), backend.Description())
	fmt.Println(commandline.SprintSettings(store))

	*flagDataDir, err = fsutil.ReplaceTildeInDir(*flagDataDir)
	if err != nil {
		klog.Fatalf("Fatal error replacing tilde in data directory: %+v", err)
	}

	switch {
	case *flagDownload:
		if !fsutil.MustFileExists(*flagDataDir) {
			must(os.MkdirAll(*flagDataDir, 0777))
		}
		must(dogsvscats.Download(*flagDataDir))
		must(dogsvscats.FilterValidImages(*flagDataDir))
		klog.Infof("Data downloaded in %s", *flagDataDir)
	case *flagPreGenerate:
		must(preGenerate(store, *flagDataDir))
	case *flagTrain:
		must(dogsvscats.Train(store, *flagDataDir, *flagCheckpoint, *flagEval, paramsSet))
	default:
		klog.Info("exit: usage -download, -pre and/or -train, optional -data")
	}
}

func preGenerate(store *model.Store, dataDir string) error {
	exists, err := fsutil.FileExists(dataDir)
	if err != nil {
		return err
	}
	if !exists {
		if err := os.MkdirAll(dataDir, 0777); err != nil {
			return err
		}
	}
	if err := dogsvscats.Download(dataDir); err != nil {
		return err
	}
	if err := dogsvscats.FilterValidImages(dataDir); err != nil {
		return err
	}

	config := dogsvscats.NewPreprocessingConfiguration(store, dataDir)
	dogsvscats.PreGenerate(config, *flagPreGenEpochs, true)
	return nil
}

// must reports and exits on error.
func must(err error) {
	if err == nil {
		return
	}
	klog.Fatalf("Fatal error: %+v", err)
}

// must1 return the v value if err is nil, otherwise report the error and exit.
func must1[T any](v T, err error) T {
	if err != nil {
		must(err)
	}
	return v
}
