// Package fyneui implements a simple GUI app that displays how training progresses, as well
// as extra visualization for:
//
// - Training progress
// - Hyperparameters
// - Variables
// - Final metrics
//
// How to use this:
//
// 1. Write the following main function:
//
//	func main() {
//		fyneui.RunMain(mainContinue)
//	}
//
//	func mainContinue() {
//		// usual main() code.
//	}
//
// 2. After creating the `loop` object, do:
//
//	fyne.AttachGUI(loop)
//
// Or, if you don't want both a command-line and a GUI app, but dynamically decide based on the availability
// of a window system:
//
//	fyne.AttachGUIOrProgressBar(loop)
package fyneui

import (
	"fmt"
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/widget"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xsync"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"os"
	"os/signal"
	"sync"
	"time"
)

var (
	// App holds the current Fyne App singleton, created when the first NewWindow is called.
	//
	// It is here for someone who may want to customize the app. But otherwise consider using AttachGUI or AttachGUIOrProgressBar.
	App fyne.App

	numWindowsOpened   int
	muNumWindowsOpened sync.Mutex
	condNumWindowsOpen = sync.NewCond(&muNumWindowsOpened)
)

// RunMain is a wrapper that will execute your main function on a separate Goroutine, while reserving the current
// (presumably the main goroutine) to run Fyne loop.
//
// It should be called once at the beginning of your main function.
//
// Example:
//
//	func main() {
//		flag.Parse()
//		fyneui.RunMain(mainContinue)
//	}
//
//	func mainContinue() {
//			...
//	}
//
// It is here for someone who may want to customize the app. But otherwise consider using AttachGUI or AttachGUIOrProgressBar.
func RunMain(main func()) {
	var exception any
	if !HasWindows() {
		// No windows were created, just run the main function.
		// This is the common case.
		exception = exceptions.Try(main)
		return

	} else {
		// Coordinate with the new goroutine to run main() with fyne.App.Run().
		done := xsync.NewLatch()
		onInterrupt := make(chan os.Signal, 1)
		go func() {
			<-onInterrupt
			if exception != nil {
				exception = "Interrupt (control+C) signal received."
			}
			done.Trigger()
			App.Quit()
		}()

		go func() {
			// Override the behavior installed by Fyne.
			signal.Reset(os.Interrupt)
			signal.Notify(onInterrupt, os.Interrupt)
			exception = exceptions.Try(main)
			done.Trigger()
			if exception == nil {
				// Normal end, wait for all windows to close.
				muNumWindowsOpened.Lock()
				if numWindowsOpened > 0 {
					fmt.Println("Waiting for windows to close...")
				}
				muNumWindowsOpened.Unlock()
				WaitForWindows()
			} else {
				// An exception was thrown, force immediate quit.
				App.Quit()
			}
		}()
		App = app.New()
		App.Run()

		// Once App is returned, all windows are definitely closed.
		// We make sure we reset the counter to 0, in case the user calls WaitForWindows(), in case some window was not
		// cleanly closed.
		condNumWindowsOpen.L.Lock()
		numWindowsOpened = 0
		condNumWindowsOpen.Broadcast()
		condNumWindowsOpen.L.Unlock()

		// Wait for the main goroutine to finish and any exceptions to be reported.
		done.Wait()
	}

	if exception != nil {
		klog.Fatal("Panic:\n%+v", exception)
		os.Exit(1)
	}
}

// WaitForWindows waits for all GUI windows to be closed by the user.
//
// Usually RunMain will automatically call this function at the end of the program.
// But it's available if the user want to have some sync point.
func WaitForWindows() {
	condNumWindowsOpen.L.Lock()
	defer condNumWindowsOpen.L.Unlock()
	for numWindowsOpened > 0 {
		condNumWindowsOpen.Wait()
	}
}

// Window holds the Fyne window with the progress bar for a training loop. It can be created with NewWindow().
//
// The library assumes there can be multiple Window objects live at at any time.
//
// It is here for someone who may want to customize the app. But otherwise consider using AttachGUI or AttachGUIOrProgressBar.
type Window struct {
	Name string
	Loop *train.Loop

	Win                     fyne.Window
	ProgressBar             *widget.ProgressBar
	CancelButton            *widget.Button
	NumStepsText, SpeedText *widget.Label
	TrainingTable           *fyne.Container

	firstUpdate                bool
	LastUpdate                 time.Time
	UpdateFrequency            time.Duration
	numSteps, lastStepReported int

	speedFromTime time.Time
	speedFromStep int

	cancelled bool
}

// NewWindow creates and returns a new Window
//
// It is here for someone who may want to customize the app. But otherwise consider using AttachGUI or AttachGUIOrProgressBar.
func NewWindow(name string, loop *train.Loop) *Window {
	muNumWindowsOpened.Lock()
	defer muNumWindowsOpened.Unlock()

	win := &Window{
		Name:            name,
		Loop:            loop,
		UpdateFrequency: time.Second / 5,
		firstUpdate:     true,

		ProgressBar:  widget.NewProgressBar(),
		NumStepsText: widget.NewLabel("Training"),
		SpeedText:    widget.NewLabel("(compiling)"),
	}
	win.CancelButton = widget.NewButton("Cancel", func() {
		win.cancelled = true
		win.Close()
	})

	// bottomBar with progressBar.
	bottomBar := container.NewBorder(nil, nil, nil, win.SpeedText, win.ProgressBar)

	// TrainingTable.
	win.NumStepsText.Alignment = fyne.TextAlignCenter
	win.NumStepsText.TextStyle = fyne.TextStyle{Bold: true}
	win.NumStepsText.Importance = widget.HighImportance
	win.TrainingTable = container.NewHBox()

	// Action button(s).
	buttonStrip := container.NewHBox(layout.NewSpacer(), win.CancelButton)

	// Top part with training metrics.
	mainVBox := container.NewVBox(win.NumStepsText, win.TrainingTable, bottomBar, buttonStrip)

	w := App.NewWindow(win.Name)
	w.SetContent(mainVBox)
	w.Resize(fyne.NewSize(480, 20))
	w.Show()
	win.Win = w
	numWindowsOpened++
	return win
}

// Close closes the window, and if the last window is closed, it will wake up all goroutines waiting on WaitForWindows().
//
// This is called when the user clicks on "Done" button.
func (win *Window) Close() {
	condNumWindowsOpen.L.Lock()
	win.Win.Close()
	numWindowsOpened--
	if numWindowsOpened <= 0 {
		condNumWindowsOpen.Broadcast()
	}
	condNumWindowsOpen.L.Unlock()
}

// Update updates the progress bar's value
func (win *Window) Update(value float64) {
	win.ProgressBar.SetValue(value)
}

// OnStart is called when the training loop starts.
// It is called before the first OnStep() call.
func (win *Window) OnStart(loop *train.Loop, _ train.Dataset) error {
	if win.cancelled {
		return errors.New("Training cancelled by user request")
	}
	win.lastStepReported = loop.LoopStep
	if loop.EndStep < 0 {
		win.numSteps = 1000 // Guess for now.
	} else {
		win.numSteps = loop.EndStep - loop.StartStep
	}
	win.NumStepsText.SetText(fmt.Sprintf("Training (%d steps)", win.numSteps))
	return nil
}

// OnStep is called every time the training loop moves forward.
// It will also check if the training was cancelled by the user. If so, it returns an error.
func (win *Window) OnStep(loop *train.Loop, metrics []*tensors.Tensor) error {
	if win.cancelled {
		return errors.New("Training cancelled by user request")
	}

	// Throttle updates.
	if time.Since(win.LastUpdate) < win.UpdateFrequency {
		return nil
	}
	win.LastUpdate = time.Now()

	// Check how much it moved forward.
	amount := loop.LoopStep + 1 - win.lastStepReported // +1 because the current LoopStep is finished.
	if amount <= 0 {
		return nil
	}
	win.Update(float64(loop.LoopStep) / float64(win.numSteps))

	// First update is special.
	if win.firstUpdate {
		win.SpeedText.SetText("(? steps/s)")
		win.firstUpdate = false
		win.speedFromTime = time.Now()
		win.speedFromStep = loop.LoopStep
		return nil
	}

	// Get a new speed metric if either 100 steps or more than 10 seconds have passed.
	elapsed := time.Since(win.speedFromTime)
	speedAmount := loop.LoopStep - win.speedFromStep
	if elapsed > time.Second*10 || speedAmount >= 100 {
		speed := float64(speedAmount) / elapsed.Seconds()
		if speed > 1 {
			win.SpeedText.SetText(fmt.Sprintf("%.1f steps/s", speed))
		} else {
			win.SpeedText.SetText(fmt.Sprintf("%.1f steps/min", speed*60))
		}
		win.speedFromTime = time.Now()
		win.speedFromStep = loop.LoopStep
	}
	return nil
}

// OnEnd is called when the training loop ends.
//
// It will close the window, and if the last window is closed, it will wake up all goroutines waiting on WaitForWindows().
func (win *Window) OnEnd(_ *train.Loop, metrics []*tensors.Tensor) error {
	if win.cancelled {
		return errors.New("Training cancelled by user request")
	}
	win.Update(1.0)
	win.LastUpdate = time.Now()
	win.CancelButton.SetText("Done")
	win.Win.Show()
	return nil
}

// AttachGUI attaches a GUI app to the given loop.
//
// - loop: training loop to attach to.
// - name: is used for the window name.
//
// When the loop ends, the GUI is kept alive, until the user closes it.
// Call WaitForGUIs() to wait for all GUI apps to be closed by the user.
//
// It is a no-op if the GUI was already created.
func AttachGUI(loop *train.Loop, name string) {
	win := NewWindow(name, loop)
	loop.OnStart(name, 100, win.OnStart)
	loop.OnStep(name, 100, win.OnStep)
	loop.OnEnd(name, 100, win.OnEnd)
	//loop.OnEnd(ProgressBarName, 0, pBar.onEnd)
}

// AttachGUIOrProgressBar attaches a GUI or a progress bar to a training loop based on the availability of a graphical display.
// If a DISPLAY environment variable is present, it attaches a graphical user interface.
// Otherwise, it attaches a command-line progress bar.
func AttachGUIOrProgressBar(loop *train.Loop, name string) {
	if HasWindows() {
		AttachGUI(loop, name)
	} else {
		commandline.AttachProgressBar(loop)
	}
}

// HasWindows checks if the environment has a graphical display available by verifying the DISPLAY environment variable.
func HasWindows() bool {
	return os.Getenv("DISPLAY") != ""
}
