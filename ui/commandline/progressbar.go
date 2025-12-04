package commandline

import (
	"fmt"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/charmbracelet/lipgloss"
	lgtable "github.com/charmbracelet/lipgloss/table"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/ui/notebooks"
	"github.com/muesli/termenv"
	"github.com/schollz/progressbar/v3"
)

// ExtraMetricFn is any function that will give extra values to display along the progress bar.
// It is called at each time the progress bar is updated, and it should return a name and the current value when it is called.
type ExtraMetricFn func() (name, value string)

// RefreshPeriod is the time between terminal updates.
var RefreshPeriod = time.Second * 3

// progressBar holds a progressbar being displayed.
type progressBar struct {
	numSteps         int
	lastStepReported int
	bar              *progressbar.ProgressBar
	suffix           string
	inNotebook       bool
	totalAmount      int

	// lipgloss-based rich and asynchronous display for the command-line.
	termenv          *termenv.Output
	statsStyle       lipgloss.Style
	statsTable       *lgtable.Table
	isFirstOutput    bool
	updates          chan progressBarUpdate
	asyncUpdatesDone sync.WaitGroup

	extraMetricFns []ExtraMetricFn
}

// ProgressbarStyle to use. Defaults to the ASCII version.
// Consider "progressbar.ThemeUnicode" for a prettier version.
// But it requires some of the graphical symbols to be supported.
var ProgressbarStyle = progressbar.ThemeASCII

// Write implements io.Writer, and appends the current suffix with metrics to each
// line. It is meant to be used as the default writer for the enclosed progressbar.ProgressBar.
// This ensures that the progress bar and its suffix are written in the same write operation;
// otherwise Jupyter Notebook may display things in different lines.
func (pBar *progressBar) Write(data []byte) (n int, err error) {
	n, err = os.Stdout.Write(data)
	if err != nil {
		return n, err
	}
	_, err = os.Stdout.Write([]byte(pBar.suffix))
	if err != nil {
		return 0, err
	}
	return
}

func (pBar *progressBar) onStart(loop *train.Loop, _ train.Dataset) error {
	pBar.lastStepReported = loop.LoopStep
	//var stepsMsg string
	if loop.EndStep < 0 {
		pBar.numSteps = 1000 // Guess for now.
	} else {
		pBar.numSteps = loop.EndStep - loop.StartStep
		//stepsMsg = fmt.Sprintf(" (%d steps)", pBar.numSteps)
	}
	pBar.bar = progressbar.NewOptions(pBar.numSteps,
		//progressbar.OptionSetDescription(fmt.Sprintf("[dim]Training%s: [reset]", stepsMsg)),
		progressbar.OptionSetDescription("      [bold]"),
		progressbar.OptionUseANSICodes(true),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionShowIts(),
		progressbar.OptionSetItsString("steps"),
		progressbar.OptionSetTheme(ProgressbarStyle),
		progressbar.OptionSetWriter(pBar), // Required to work with Jupyter notebook.
	)
	return nil
}

func (pBar *progressBar) onStep(loop *train.Loop, metrics []*tensors.Tensor) error {
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
		// Erase to an end-of-line escape sequence ("\033[J") not supported in Jupyter notebooks:
		parts = append(parts, "        ")
		pBar.suffix = strings.Join(parts, "")
		_ = pBar.bar.Add(amount) // Triggers print, see [pBar.Write] method.

	} else {
		// Suffix to erase spurious characters from previous prints.
		pBar.suffix = "\033[J" // Using "\033[J" to erase to the end of the line causes flickering on terminals (gnome-terminal and alacritty).

		// For the command-line instead we create and enqueue an update to be asynchronously printed.
		update := progressBarUpdate{
			amount:  amount,
			metrics: make([]string, 0, len(trainMetrics)+1),
		}
		if loop.Trainer.NumAccumulatingSteps() > 1 {
			// GlobalStep and TrainingStep are different
			update.metrics = append(update.metrics, fmt.Sprintf("%s / %s of %s",
				humanizeInt(loop.Trainer.GlobalStep()), humanizeInt(loop.LoopStep), humanizeInt(loop.EndStep)))
		} else {
			// GlobalStep and TrainingStep are the same.
			update.metrics = append(update.metrics, fmt.Sprintf("%s of %s",
				humanizeInt(loop.LoopStep), humanizeInt(loop.EndStep)))
		}
		for metricIdx, metricObj := range trainMetrics {
			update.metrics = append(update.metrics, metricObj.PrettyPrint(metrics[metricIdx]))
		}
		pBar.updates <- update
	}

	// Add the number of steps run since last time.
	pBar.totalAmount += amount
	pBar.lastStepReported = loop.LoopStep + 1
	return nil
}

func (pBar *progressBar) onEnd(_ *train.Loop, _ []*tensors.Tensor) error {
	if pBar.updates != nil {
		close(pBar.updates)
	}
	pBar.asyncUpdatesDone.Wait()
	if pBar.termenv != nil {
		pBar.termenv.ShowCursor()
	}
	fmt.Println()
	return nil
}

const ProgressBarName = "gomlx.ml.train.commandline.progressBar"

var (
	normalStyle       = lipgloss.NewStyle().Padding(0, 1)
	rightAlignedStyle = lipgloss.NewStyle().Align(lipgloss.Right).Padding(0, 1)
	tableBorderColor  = "#705090"
)

type progressBarUpdate struct {
	amount  int
	metrics []string
}

// maxUpdateFrequency is the time between updates to the commandline display of stats.
const maxUpdateFrequency = time.Millisecond * 200

// AttachProgressBar creates a commandline progress bar and attaches it to the Loop, so that
// everytime Loop is run, it will display a progress bar with progression and metrics.
//
// The associated data will be attached to the train.Loop, so nothing is returned.
//
// Optionally, one can provide extraMetrics: functions that are called at every update of
// the progress bar and should return a name (title) and a value to be included in the
// updated print-out.
func AttachProgressBar(loop *train.Loop, extraMetrics ...ExtraMetricFn) {
	pBar := &progressBar{
		inNotebook:     notebooks.IsNotebook(),
		extraMetricFns: extraMetrics,
	}
	if !pBar.inNotebook {
		pBar.isFirstOutput = true
		pBar.termenv = termenv.NewOutput(os.Stdout)
		pBar.statsStyle = lipgloss.NewStyle().PaddingLeft(8)
		pBar.statsTable = lgtable.New().
			Border(lipgloss.RoundedBorder()).
			BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color(tableBorderColor))).
			StyleFunc(func(row, col int) lipgloss.Style {

				if col == 0 {
					return rightAlignedStyle
				}
				return normalStyle
			})
		pBar.updates = make(chan progressBarUpdate, 100) // Large buffer so things are not blocked.
		pBar.asyncUpdatesDone.Add(1)
		go func() {
			// Asynchronously draw updates: this is handy if the training is faster than the terminal, in particular
			// if running on cloud, with a relatively slow network connection.
			for update := range pBar.updates {
				// Exhaust the updates in the buffer:
				amount := update.amount
			exhaust:
				for {
					select {
					case newUpdate, ok := <-pBar.updates:
						if !ok {
							break exhaust
						}
						amount += newUpdate.amount
						update = newUpdate
					default:
						break exhaust
					}

				}

				// Create the table to be printed.
				pBar.statsTable.Data(lgtable.NewStringData())
				if loop.Trainer.NumAccumulatingSteps() > 1 {
					pBar.statsTable.Row("Global/Train Steps", update.metrics[0])
				} else {
					pBar.statsTable.Row("Global Step", update.metrics[0])
				}
				pBar.statsTable.Row("Median train step duration", FormatDuration(loop.MedianTrainStepDuration()))
				for metricIdx, metricObj := range loop.Trainer.TrainMetrics() {
					pBar.statsTable.Row(metricObj.Name(), update.metrics[1+metricIdx])
				}
				for _, extraMetric := range pBar.extraMetricFns {
					name, value := extraMetric()
					pBar.statsTable.Row(name, value)
				}

				// For command-line, we clear the previous lines that will be overwritten.
				pBar.termenv.HideCursor()
				if !pBar.isFirstOutput {
					numLinesToBackup := len(update.metrics) + 2 + 2 + len(pBar.extraMetricFns)
					pBar.termenv.CursorPrevLine(numLinesToBackup)
				}
				pBar.isFirstOutput = false

				// Print update.
				fmt.Println(pBar.statsStyle.Render(pBar.statsTable.String()))
				_ = pBar.bar.Add(amount) // Prints progress bar line.
				fmt.Println()
				pBar.termenv.ShowCursor()
				time.Sleep(maxUpdateFrequency)
			}
			pBar.asyncUpdatesDone.Done()
		}()
	}
	loop.OnStart(ProgressBarName, 0, pBar.onStart)
	// RunWithMap at least 1000 during the loop or at least every 3 seconds.
	train.NTimesDuringLoop(loop, 1000, ProgressBarName, 0, pBar.onStep)
	train.PeriodicCallback(loop, RefreshPeriod, false, ProgressBarName, 0, pBar.onStep)
	loop.OnEnd(ProgressBarName, 0, pBar.onEnd)
}

func humanizeInt[I interface {
	uint64 | uint32 | uint16 | uint8 | int64 | int32 | int16 | int8 | int
}](nI I) string {
	n := int(nI)
	str := fmt.Sprintf("%d", n)
	result := make([]byte, 0, len(str)+len(str)/3)
	strLen := len(str)
	for i := strLen - 1; i >= 0; i-- {
		if (strLen-i-1)%3 == 0 && i < strLen-1 {
			result = append([]byte{'_'}, result...)
		}
		result = append([]byte{str[i]}, result...)
	}
	return string(result)
}
