package main

import (
	"testing"
)

func TestMainFunc(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	ctx := createDefaultContext()
	ctx.SetParam("train_steps", 10)
	mainWithContext(ctx)
}
