package main

import (
	"flag"

	"github.com/gomlx/gomlx/examples/oxfordflowers102/diffusion"
)

func main() {
	flag.Parse()
	diffusion.Init()
	diffusion.TrainModel()
}
