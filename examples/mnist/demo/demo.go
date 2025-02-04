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

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/examples/mnist"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/xla"
)

var (
	flagTrain    = flag.Bool("train", true, "Flag to train")
	flagDownload = flag.Bool("download", false, "Flag to download")
	flagModel    = flag.String("model", "logistic", "Model function")
	flagLoss     = flag.String("loss", "cross-entropy", "Loss function")
	flagDataDir  = flag.String("data", "~/tmp/mnist", "Directory to cache downloaded dataset.")
)

func main() {

	klog.InitFlags(nil)
	flag.Parse()

	ctx := context.New()
	err := exceptions.TryCatch[error](func() {

		if *flagDownload {
			must.M(mnist.Download(*flagDataDir))
			klog.Infof("Data downloaded in %s", *flagDataDir)
		}
		if *flagTrain {
			mnist.TrainModel(ctx, *flagDataDir, *flagModel, *flagLoss)
			klog.Infof("model trained in %s", *flagDataDir)
		}
		if !*flagDownload && !*flagTrain {
			klog.Info("exit: usage -download and/or -train, optional -data")
		}
	})
	if err != nil {
		klog.Errorf("Error:\n%+v", err)
	}
}
