// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"os"

	"github.com/gomlx/compute"
	"github.com/gomlx/gomlx/examples/oxfordflowers102"
	"github.com/gomlx/gomlx/examples/oxfordflowers102/diffusion"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/fsutil"
	"github.com/gomlx/gomlx/ui/commandline"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagTrain      = flag.Bool("train", true, "Train the model.")
	flagDownload   = flag.Bool("download", false, "Download Oxford Flowers 102 dataset.")
	flagDataDir    = flag.String("data", "~/work/oxfordflowers102", "Directory to cache downloaded and generated dataset files.")
	flagEval       = flag.Bool("eval", true, "Whether to evaluate the model on the validation data in the end.")
	flagVerbosity  = flag.Int("verbosity", 1, "Level of verbosity, the higher the more verbose.")
	flagCheckpoint = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
)

func main() {
	store := diffusion.CreateModelStore()
	// Set default train_steps to 1000 for demo purposes (so it trains in a few seconds on GPU).
	store.SetParam("train_steps", 1000)

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

	err = exceptions.TryCatch[error](func() {
		if *flagDownload {
			*flagDataDir = fsutil.MustReplaceTildeInDir(*flagDataDir)
			if !fsutil.MustFileExists(*flagDataDir) {
				check(os.MkdirAll(*flagDataDir, 0777))
			}
			check(oxfordflowers102.DownloadAndParse(*flagDataDir))
			klog.Infof("Data downloaded in %s", *flagDataDir)
		}
		if *flagTrain {
			diffusion.TrainWithStore(store, *flagDataDir, *flagCheckpoint, paramsSet, *flagEval, *flagVerbosity)
		}
		if !*flagDownload && !*flagTrain {
			klog.Info("exit: usage -download and/or -train, optional -data")
		}
	})
	if err != nil {
		klog.Fatalf("Error:\n%+v", err)
	}
}

// check reports and exits on error.
func check(err error) {
	if err == nil {
		return
	}
	klog.Fatalf("Fatal error: %+v", err)
}
