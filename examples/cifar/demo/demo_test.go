package main

import (
	"flag"
	"testing"
)

func TestDemo(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	flag.Parse()
	*flagNumSteps = 10
	*flagPlatform = "Host"
	*flagEval = false
	main()
}
