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

// Package commandline contains convenience UI training tools for the command line. In includes a ProgressBar for
// train.Loop and ReportEval for the registered eval metrics in train.Trainer.
package commandline

import (
	"fmt"
	"github.com/gomlx/gomlx/examples/notebook"
	"github.com/gomlx/gomlx/ml/train"
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/schollz/progressbar/v3"
	"os"
	"strings"
)

// progressBar holds a progressbar being displayed.
type progressBar struct {
	numSteps         int
	lastStepReported int
	bar              *progressbar.ProgressBar
	suffix           string
	inNotebook       bool

	totalAmount int
}

// Write implements io.Writer, and appends the current suffix with metrics to each
// line. It is meant to be used as the default writer for the enclosed progressbar.ProgressBar.
// This ensures that the progress bar and its suffix are written in the same write operation,
// otherwise Jupyter Notebook may display things in different lines.
func (pBar *progressBar) Write(data []byte) (n int, err error) {
	newData := append(data, []byte(pBar.suffix)...)
	n, err = os.Stdout.Write(newData)
	if err == nil {
		n = len(data)
	}
	return
}

func (pBar *progressBar) onStart(loop *train.Loop, _ train.Dataset) error {
	pBar.lastStepReported = loop.LoopStep
	var stepsMsg string
	if loop.EndStep < 0 {
		pBar.numSteps = 1000 // Guess for now.
	} else {
		pBar.numSteps = loop.EndStep - loop.StartStep
		stepsMsg = fmt.Sprintf(" (%d steps)", pBar.numSteps)
	}
	pBar.bar = progressbar.NewOptions(pBar.numSteps,
		progressbar.OptionSetDescription(fmt.Sprintf("Training%s: ", stepsMsg)),
		progressbar.OptionUseANSICodes(true),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionShowIts(),
		progressbar.OptionSetItsString("steps"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[green]=[reset]",
			SaucerHead:    "[green]>[reset]",
			SaucerPadding: ".",
			BarStart:      "[",
			BarEnd:        "]",
		}),
		progressbar.OptionSetWriter(pBar), // Required to work with Jupyter notebook.
	)
	return nil
}

func (pBar *progressBar) onStep(loop *train.Loop, metrics []tensor.Tensor) error {
	if !pBar.bar.IsFinished() {
		// Set suffix -- it will be displayed along with the progressbar.
		trainMetrics := loop.Trainer.TrainMetrics()
		parts := make([]string, 0, len(trainMetrics)+1)
		for metricIdx, metricObj := range trainMetrics {
			parts = append(parts, fmt.Sprintf(" [%s=%s]", metricObj.ShortName(), metricObj.PrettyPrint(metrics[metricIdx])))
		}
		if pBar.inNotebook {
			// Erase to end-of-line escape sequence not supported:
			parts = append(parts, "        ")
		} else {
			// Erase to end-of-line:
			parts = append(parts, "\033[J")
		}
		pBar.suffix = strings.Join(parts, "")

		// Update progressbar.
		amount := loop.LoopStep + 1 - pBar.lastStepReported // +1 because the current LoopStep is finished.
		if amount > 0 {
			pBar.bar.Add(amount)
			pBar.totalAmount += amount
			pBar.lastStepReported = loop.LoopStep + 1
		}
	}
	return nil
}

func (pBar *progressBar) onEnd(loop *train.Loop, metrics []tensor.Tensor) error {
	fmt.Println()
	return nil
}

const ProgressBarName = "gomlx.ml.train.commandline.progressBar"

// AttachProgressBar creates a commandline progress bar and attaches it to the Loop, so that
// everytime Loop is run it will display a progress bar with progression and metrics.
//
// The associated data will be attached to the train.Loop, so nothing is returned.
func AttachProgressBar(loop *train.Loop) {
	pBar := &progressBar{}
	pBar.inNotebook = notebook.IsPresent()
	loop.OnStart(ProgressBarName, 0, pBar.onStart)
	train.NTimesDuringLoop(loop, 1000, ProgressBarName, 0, pBar.onStep)
	loop.OnEnd(ProgressBarName, 0, pBar.onEnd)
}

// ReportEval reports on the command line the results of evaluating the datasets using trainer.Eval.
func ReportEval(trainer *train.Trainer, datasets ...train.Dataset) error {
	return TryCatch[error](func() {
		for _, ds := range datasets {
			fmt.Printf("Results on %s:\n", ds.Name())
			metricsValues := trainer.Eval(ds)
			for metricIdx, metric := range trainer.EvalMetrics() {
				value := metricsValues[metricIdx]
				fmt.Printf("\t%s (%s): %s\n", metric.Name(), metric.ShortName(), metric.PrettyPrint(value))
			}
			ds.Reset()
		}
	})
}
