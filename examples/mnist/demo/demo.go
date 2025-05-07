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

// demo for mnist library
//
//  1. With `demo --download`: it will simply download and unpack dataset.
//  2. With `demo --train`: trains a CNN (convolutional neural network) model for "MNIST".
package main

import (
	"flag"

	"github.com/gomlx/gomlx/examples/mnist"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

// Example usage:  go run demo.go -train -checkpoint=cnn_triplet_hard_01 -set="model=cnn;loss=triplet;triplet_loss_mining_strategy=hard;triplet_loss_margin=0.1"

var (
	flagTrain      = flag.Bool("train", true, "Flag to train")
	flagDownload   = flag.Bool("download", false, "Flag to download")
	flagDataDir    = flag.String("data", "~/work/mnist", "Directory to cache downloaded dataset.")
	flagCheckpoint = flag.String("checkpoint", "", "Checkpoint directory from/to where to load/save the trained model. Path is relative to --data.")
)

func main() {
	ctx := mnist.CreateDefaultContext()
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *settings))
	if *flagDownload {
		must.M(mnist.Download(*flagDataDir))
		klog.Infof("Data downloaded in %s", *flagDataDir)
	}
	if *flagTrain {
		must.M(mnist.TrainModel(ctx, *flagDataDir, *flagCheckpoint, paramsSet))
	}
	if !*flagDownload && !*flagTrain {
		klog.Info("exit: usage -download and/or -train, optional -data")
	}
}
