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

// Package commandline contains convenience UI training tools for the command line.
package commandline

import (
	"encoding/json"
	"flag"
	"fmt"
	"github.com/charmbracelet/lipgloss"
	lgtable "github.com/charmbracelet/lipgloss/table"
	"github.com/gomlx/gomlx/examples/notebook"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/train"
	. "github.com/gomlx/gomlx/types/exceptions"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/muesli/termenv"
	"github.com/pkg/errors"
	"github.com/schollz/progressbar/v3"
	"os"
	"strings"
	"time"
)

// progressBar holds a progressbar being displayed.
type progressBar struct {
	numSteps         int
	lastStepReported int
	bar              *progressbar.ProgressBar
	suffix           string
	inNotebook       bool
	totalAmount      int

	// lipgloss based rich display for the command-line.
	termenv       *termenv.Output
	statsStyle    lipgloss.Style
	statsTable    *lgtable.Table
	isFirstOutput bool
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
	// Check whether it is finished.
	if pBar.bar.IsFinished() {
		return nil
	}

	// Check whether there is something to update.
	amount := loop.LoopStep + 1 - pBar.lastStepReported // +1 because the current LoopStep is finished.
	if amount <= 0 {
		return nil
	}

	trainMetrics := loop.Trainer.TrainMetrics()
	if pBar.inNotebook {
		// For notebooks set a suffix that will be written along with the progressbar in [progressBar.Write].
		parts := make([]string, 0, len(trainMetrics)+1)
		parts = append(parts, fmt.Sprintf(" [step=%d]", loop.LoopStep))
		for metricIdx, metricObj := range trainMetrics {
			parts = append(parts, fmt.Sprintf(" [%s=%s]", metricObj.ShortName(), metricObj.PrettyPrint(metrics[metricIdx])))
		}
		// Erase to end-of-line escape sequence ("\033[J") not supported in notebook:
		parts = append(parts, "        ")
		pBar.suffix = strings.Join(parts, "")

	} else {
		// For command-line, we clear the previous lines that will be overwritten.
		if !pBar.isFirstOutput {
			pBar.termenv.ClearLines(len(trainMetrics) + 1 + 2 + 1)
		}
		pBar.isFirstOutput = false
	}

	// Add amount run since last time.
	_ = pBar.bar.Add(amount) // Triggers print, see [pBar.Write] method.
	pBar.totalAmount += amount
	pBar.lastStepReported = loop.LoopStep + 1

	// For non-notebook, report the metrics on the following lines -- already reserved with the `ClearLines` call above.
	if !pBar.inNotebook {
		fmt.Printf("\n") // End current progressbar line.

		pBar.statsTable.Data(lgtable.NewStringData())
		pBar.statsTable.Row("Global Step", fmt.Sprintf("%d of %d", loop.LoopStep, loop.EndStep))
		for metricIdx, metricObj := range trainMetrics {
			pBar.statsTable.Row(metricObj.Name(), metricObj.PrettyPrint(metrics[metricIdx]))
		}
		fmt.Println(pBar.statsStyle.Render(pBar.statsTable.String()))
	}
	return nil
}

func (pBar *progressBar) onEnd(loop *train.Loop, metrics []tensor.Tensor) error {
	fmt.Println()
	return nil
}

const ProgressBarName = "gomlx.ml.train.commandline.progressBar"

var (
	normalStyle       = lipgloss.NewStyle().Padding(0, 1)
	rightAlignedStyle = lipgloss.NewStyle().Align(lipgloss.Right).Padding(0, 1)
)

// AttachProgressBar creates a commandline progress bar and attaches it to the Loop, so that
// everytime Loop is run it will display a progress bar with progression and metrics.
//
// The associated data will be attached to the train.Loop, so nothing is returned.
func AttachProgressBar(loop *train.Loop) {
	pBar := &progressBar{
		inNotebook: notebook.IsPresent(),
	}
	if !pBar.inNotebook {
		pBar.isFirstOutput = true
		pBar.termenv = termenv.NewOutput(os.Stdout)
		pBar.statsStyle = lipgloss.NewStyle().PaddingLeft(8)
		pBar.statsTable = lgtable.New().
			Border(lipgloss.RoundedBorder()).
			StyleFunc(func(row, col int) lipgloss.Style {
				if col == 0 {
					return rightAlignedStyle
				}
				return normalStyle
			})
	}
	loop.OnStart(ProgressBarName, 0, pBar.onStart)
	// Run at least 1000 during loop or at least every 3 seconds.
	train.NTimesDuringLoop(loop, 1000, ProgressBarName, 0, pBar.onStep)
	train.PeriodicCallback(loop, 3*time.Second, false, ProgressBarName, 0, pBar.onStep)
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

// ParseContextSettings from for example a flag definition.
//
// The settings are a list separated by ";": e.g.: "param1=value1;param2=value2;...".
//
// All the parameters "param1", "param2", etc. must be already set with default values
// in the context `ctx`. The default values are also used to set the type to which the
// string values will be parsed to.
//
// It updates `ctx` parameters accordingly, and returns an error in case a parameter
// is unknown or the parsing failed.
//
// Note, one can also provide a scope for the parameters: "layer_1/l2_regularization=0.1"
// will work, as long as a default "l2_regularization" is defined in `ctx`.
//
// See example in [CreateContextSettingsFlag], which will create a flag for the settings.
func ParseContextSettings(ctx *context.Context, settings string) error {
	settingsList := strings.Split(settings, ";")
	for _, setting := range settingsList {
		parts := strings.Split(setting, "=")
		if len(parts) != 2 {
			return errors.Errorf("can't parse settings %q: each setting requires the format \"<param>=<value>\", got %q",
				settings, setting)
		}
		paramPath, valueStr := parts[0], parts[1]
		paramPathParts := strings.Split(paramPath, context.ScopeSeparator)
		key := paramPathParts[len(paramPathParts)-1]
		value, found := ctx.GetParam(key)
		if !found {
			return errors.Errorf("can't set parameter %q because the param %q is not known in the root context",
				paramPath, key)
		}

		// Set the new parameter in the selected scope.
		ctxInScope := ctx
		if len(paramPathParts) > 1 {
			for _, part := range paramPathParts[:len(paramPathParts)-1] {
				if part == "" {
					continue
				}
				ctxInScope = ctxInScope.In(part)
			}
		}

		// Parse value accordingly.
		// Is there a better way of doing this using reflection ?
		var err error
		switch v := value.(type) {
		case int:
			err = json.Unmarshal([]byte(valueStr), &v)
			value = v
		case int32:
			err = json.Unmarshal([]byte(valueStr), &v)
			value = v
		case int64:
			err = json.Unmarshal([]byte(valueStr), &v)
			value = v
		case float64:
			err = json.Unmarshal([]byte(valueStr), &v)
			value = v
		case float32:
			err = json.Unmarshal([]byte(valueStr), &v)
			value = v
		case bool:
			err = json.Unmarshal([]byte(valueStr), &v)
			value = v
		default:
			err = fmt.Errorf("don't know how to parse type %T for setting parameter %q",
				value, setting)
		}
		if err != nil {
			return errors.Wrapf(err, "failed to parse value %q for parameter %q (default value is %#v)", valueStr, paramPath, value)
		}
		ctxInScope.SetParam(key, value)
	}
	return nil
}

// CreateContextSettingsFlag create a string flag with the given name (if empty it will be named
// "set") and with a description of the current defined parameters in the context `ctx`.
//
// The flag should be created before the call to `flags.Parse()`.
//
// Example:
//
//	func main() {
//		ctx := createDefaultContext()
//		settings := CreateContextSettingsFlag(ctx, "")
//		flags.Parse()
//		err := ParseContextSettings(ctx, *settings)
//		if err != nil {...}
//		...
//	}
func CreateContextSettingsFlag(ctx *context.Context, name string) *string {
	if name == "" {
		name = "set"
	}
	var parts []string
	parts = append(parts, fmt.Sprintf(
		`Set context parameters defining the model. `+
			`It should be a list of elements "param=value" separated by ";". `+
			`Scoped settings are allowed, by using %q to separated scopes. `+
			`Current available parameters that can be set:`,
		context.ScopeSeparator))
	ctx.EnumerateParams(func(scope, key string, value any) {
		if scope != context.RootScope {
			return
		}
		parts = append(parts, fmt.Sprintf("%q: default value is %v", key, value))
	})
	usage := strings.Join(parts, "\n")
	var settings string
	flag.StringVar(&settings, name, "", usage)
	return &settings
}
