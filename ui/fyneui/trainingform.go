package fyneui

import (
	"fmt"
	"fyne.io/fyne/v2/widget"
	"slices"
	"time"
)

// TrainingForm holds running information about the training: global step, elapsed times and various training metrics.

func (win *window) newTrainingForm() {
	win.TrainingForm = widget.NewForm(
		widget.NewFormItem("Global Step", widget.NewRichTextWithText("- / -")),
		widget.NewFormItem("Elapsed Time", widget.NewRichTextWithText(" - ")),
		widget.NewFormItem("ETA", widget.NewRichTextWithText(" - ")),
		widget.NewFormItem("Compiling", widget.NewRichTextWithText(" - ")),
	)
}

func (win *window) updateTrainingForm() {
	// Convenience method to access a specific row.
	rowAt := func(idx int) *widget.RichText {
		return win.TrainingForm.Items[idx].Widget.(*widget.RichText)
	}

	// Global Step.
	if win.Loop.EndStep <= 0 {
		rowAt(0).ParseMarkdown(fmt.Sprintf("%d", win.Loop.LoopStep))
	} else {
		rowAt(0).ParseMarkdown(fmt.Sprintf("%d / %d", win.Loop.LoopStep, win.Loop.EndStep))
	}

	// Elapsed/Compile/ETA times.
	elapsed := time.Since(win.startTime).Round(time.Second)
	rowAt(1).ParseMarkdown(elapsed.String())
	if win.isFirstUpdate {
		// Training hasn't started yet.
		compilationElapsed := time.Since(win.startTime).Round(time.Second)
		rowAt(3).ParseMarkdown(compilationElapsed.String())
	} else {
		var etaDesc string
		if win.speed == 0 {
			etaDesc = "∞"
		} else {
			etaSecs := float64(win.Loop.EndStep-win.speedFromStep) / win.speed
			etaDur := time.Duration(float64(time.Second) * etaSecs)
			etaDesc = etaDur.Round(time.Second).String()
		}
		rowAt(2).ParseMarkdown(etaDesc)
	}

	// Metrics: capture them.
	var metrics []string
	win.muLastUpdate.Lock()
	metrics = slices.Clone(win.lastUpdateMetrics)
	win.muLastUpdate.Unlock()

	baseIdx := 4
	for relIdx, value := range metrics {
		idx := baseIdx + relIdx
		if len(win.TrainingForm.Items) <= idx {
			metricHandler := win.Loop.Trainer.TrainMetrics()[relIdx]
			win.TrainingForm.Append(
				metricHandler.Name(),
				widget.NewRichTextWithText(value))
		} else {
			rowAt(idx).ParseMarkdown(value)
		}
	}
}
