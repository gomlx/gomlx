package main

import (
	"flag"
	"testing"
)

func TestMain(t *testing.T) {
	flag.Parse()
	*flagNumSteps = 10
	*flagPlatform = "Host"
	*flagEval = false
	main()
}