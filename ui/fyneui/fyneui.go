// Package fyneui implements a simple GUI app that displays how training progresses.
//
// It is EXPERIMENTAL, an incomplete.
//
// It aims at providing custom visualization for:
//
// - Training progress
// - Hyperparameters
// - Variables
// - Logged tensors (see Node.SetLogged method)
// - Final metrics
//
// How to use this:
//
// 1. Write the following main function (*)
//
//	func main() {
//		fyneui.RunMain(mainContinue)
//	}
//
//	func mainContinue() {
//		flag.Parse()
//		// usual main() code...
//	}
//
// 2. After creating the `loop` object, do:
//
//	fyne.AttachGUI(loop, "My Project Name")
//
// Or, if you don't want both a command-line and a GUI app, but dynamically decide based on the availability
// of a window system:
//
//	fyne.AttachGUIOrProgressBar(loop, "My Project Name")
//
// (*) Fyne requires one to hijack the main goroutine (and associated thread) to interact with the windowing system.
package fyneui

import (
	stderrors "errors"
	"fmt"
	"fyne.io/fyne/v2"
	"fyne.io/fyne/v2/app"
	"fyne.io/fyne/v2/container"
	"fyne.io/fyne/v2/layout"
	"fyne.io/fyne/v2/widget"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xsync"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/pkg/errors"
	"k8s.io/klog/v2"
	"os"
	"os/signal"
	"sync"
	"time"
)

var (
	// App holds the current Fyne App singleton, created when the first newWindow is called.
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
		klog.Fatalf("Panic:\n%+v", exception)
		os.Exit(1)
	}
}

// WaitForWindows waits for all GUI windows to be closed by the user.
//
// Usually RunMain will automatically call this function at the end of the program.
// But it's available if the user want to have a sync point after which all windows are closed.
func WaitForWindows() {
	condNumWindowsOpen.L.Lock()
	defer condNumWindowsOpen.L.Unlock()
	for numWindowsOpened > 0 {
		condNumWindowsOpen.Wait()
	}
}

// window holds the Fyne window with the progress bar for a training loop. It can be created with newWindow().
//
// The library assumes there can be multiple window objects live at at any time.
type window struct {
	Name string
	Loop *train.Loop

	// Fyne GUI components.
	Win                     fyne.Window
	bottomBar               *fyne.Container
	ProgressBar             *widget.ProgressBar
	CancelButton            *widget.Button
	NumStepsText, SpeedText *widget.Label
	TrainingForm            *widget.Form

	// Training parameters.
	startTime       time.Time
	UpdateFrequency time.Duration
	numTrainSteps   int

	// ProgressBar data.
	pollUpdatesDone   *xsync.Latch
	lastUpdateTime    time.Time
	lastUpdateStep    int
	lastUpdateMetrics []string
	muLastUpdate      sync.Mutex
	isFirstUpdate     bool
	speed             float64 // In steps/seconds.

	speedFromTime time.Time
	speedFromStep int

	cancelled bool
}

// newWindow creates and returns a new window
//
// It is here for someone who may want to customize the app. But otherwise consider using AttachGUI or AttachGUIOrProgressBar.
func newWindow(name string, loop *train.Loop) *window {
	muNumWindowsOpened.Lock()
	defer muNumWindowsOpened.Unlock()

	win := &window{
		Name: name,
		Loop: loop,

		startTime:       time.Now(),
		UpdateFrequency: time.Second / 2,
		isFirstUpdate:   true,
		pollUpdatesDone: xsync.NewLatch(),

		ProgressBar:  widget.NewProgressBar(),
		NumStepsText: widget.NewLabel("Training"),
		SpeedText:    widget.NewLabel("(compiling)"),
	}
	win.CancelButton = widget.NewButton("Cancel", func() {
		win.cancelled = true
		win.Close()
	})

	// bottomBar with progressBar.
	infinitePBar := widget.NewProgressBarInfinite()
	win.bottomBar = container.NewBorder(nil, nil, nil, win.SpeedText, infinitePBar)

	// TrainingForm.
	win.NumStepsText.Alignment = fyne.TextAlignCenter
	win.NumStepsText.TextStyle = fyne.TextStyle{Bold: true}
	win.NumStepsText.Importance = widget.HighImportance
	win.newTrainingForm()

	// Action button(s).
	buttonStrip := container.NewHBox(layout.NewSpacer(), win.CancelButton)

	// Top part with training metrics.
	mainLayout := container.NewBorder(
		win.NumStepsText,
		container.NewVBox(win.bottomBar, buttonStrip),
		nil, nil,
		win.TrainingForm)

	w := App.NewWindow(win.Name)
	w.SetContent(mainLayout)
	w.Resize(fyne.NewSize(480, 20))
	w.Show()
	infinitePBar.Start()
	infinitePBar.Refresh()
	fmt.Printf("inititeBar.running=%v\n", infinitePBar.Running())
	win.Win = w
	numWindowsOpened++
	return win
}

// Close closes the window, and if the last window is closed, it will wake up all goroutines waiting on WaitForWindows().
//
// This is called when the user clicks on "Done" button.
func (win *window) Close() {
	win.pollUpdatesDone.Trigger()
	condNumWindowsOpen.L.Lock()
	win.Win.Close()
	numWindowsOpened--
	if numWindowsOpened <= 0 {
		condNumWindowsOpen.Broadcast()
	}
	condNumWindowsOpen.L.Unlock()
}

// OnStart is called when the training loop starts.
// It is called before the first OnStep() call.
func (win *window) OnStart(loop *train.Loop, _ train.Dataset) error {
	if win.cancelled {
		return errors.New("Training cancelled by user request")
	}
	if loop.EndStep < 0 {
		win.numTrainSteps = -1 // Guess for now.
		win.NumStepsText.SetText("Training (?? steps)")
	} else {
		win.numTrainSteps = loop.EndStep - loop.StartStep
		win.NumStepsText.SetText(fmt.Sprintf("Training (%d steps)", win.numTrainSteps))
	}
	win.speedFromStep = loop.StartStep
	go win.updatesPolling() // Start polling for updates.

	return nil
}

// OnStep is called every time the training loop moves forward.
// It will also check if the training was cancelled by the user. If so, it returns an error.
func (win *window) OnStep(loop *train.Loop, metrics []*tensors.Tensor) error {
	if win.cancelled {
		return stderrors.New("training cancelled by user request")
	}
	win.refreshLastUpdate(loop, metrics, false)
	return nil
}

// refreshLastUpdate updates "lastUpdate" fields so they can be displayed asynchronously.
func (win *window) refreshLastUpdate(loop *train.Loop, metrics []*tensors.Tensor, force bool) {
	win.muLastUpdate.Lock()
	defer win.muLastUpdate.Unlock()
	if !force && time.Since(win.lastUpdateTime) < win.UpdateFrequency {
		return
	}
	if len(win.lastUpdateMetrics) != len(metrics) {
		win.lastUpdateMetrics = make([]string, len(metrics))
	}
	win.lastUpdateStep = loop.LoopStep
	win.lastUpdateTime = time.Now()
	for metricIdx, t := range metrics {
		win.lastUpdateMetrics[metricIdx] = loop.Trainer.TrainMetrics()[metricIdx].PrettyPrint(t)
	}
}

// OnEnd is called when the training loop ends.
//
// It will close the window, and if the last window is closed, it will wake up all goroutines waiting on WaitForWindows().
func (win *window) OnEnd(loop *train.Loop, metrics []*tensors.Tensor) error {
	if win.cancelled {
		return errors.New("Training cancelled by user request")
	}
	win.ProgressBar.SetValue(1.0)
	win.refreshLastUpdate(loop, metrics, true)
	win.pollUpdatesDone.Trigger()
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
	win := newWindow(name, loop)
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
