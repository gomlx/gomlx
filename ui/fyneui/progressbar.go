package fyneui

import (
	"fmt"
	"time"
)

// This file has the methods related to the progress bar.

// updatesPolling checks for the latest update every window.UpdateFrequency time is elapsed.
// It exits when window.pollUpdatesDone is triggered.
//
// It's started in a separate goroutine by OnStart.
func (win *window) updatesPolling() {
	var stop bool
	for {
		timer := time.NewTimer(win.UpdateFrequency)
		select {
		case <-win.pollUpdatesDone.WaitChan():
			// We are done, finish.
			timer.Stop()
			stop = true
		case <-timer.C:
			// Run update.
		}
		win.updateProgressBar()
		win.updateTrainingForm()
		if stop {
			return
		}
	}
}

// updateProgressBar updates the progress bar's value to the given globalStep, at the given time.
func (win *window) updateProgressBar() {
	win.muLastUpdate.Lock()
	step := win.lastUpdateStep
	when := win.lastUpdateTime
	win.muLastUpdate.Unlock()
	if step == win.speedFromStep {
		return
	}
	loop := win.Loop

	// Check how much it moved forward.
	if win.numTrainSteps > 0 {
		amount := step - loop.StartStep
		win.ProgressBar.SetValue(float64(amount) / float64(win.numTrainSteps))
	}

	// First update is special.
	if win.isFirstUpdate {
		win.bottomBar.Objects[0] = win.ProgressBar
		win.SpeedText.SetText("(? steps/s)")
		win.bottomBar.Refresh()
		win.isFirstUpdate = false
		win.speedFromTime = when
		win.speedFromStep = step

	} else {
		// Get a new speed metric if either 100 steps or more than 10 seconds have passed.
		elapsed := when.Sub(win.speedFromTime)
		amount := step - win.speedFromStep
		if elapsed > time.Second*10 || amount >= 100 {
			win.speed = float64(amount) / elapsed.Seconds()
			if win.speed > 10 {
				win.SpeedText.SetText(fmt.Sprintf("%.0f steps/s", win.speed))
			} else if win.speed > 1 {
				win.SpeedText.SetText(fmt.Sprintf("%.1f steps/s", win.speed))
			} else {
				win.SpeedText.SetText(fmt.Sprintf("%.1f steps/min", win.speed*60))
			}
			win.speedFromTime = time.Now()
			win.speedFromStep = loop.LoopStep
		}
	}
}
