package main

import (
	"flag"
	"testing"
)

func TestMainFunc(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	flag.Parse()
	*flagNumSteps = 10
	main()
}
