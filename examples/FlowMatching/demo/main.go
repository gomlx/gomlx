package main

import (
	"flag"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/examples/FlowMatching"
	"github.com/gomlx/gomlx/examples/oxfordflowers102/diffusion"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagDataDir    = flag.String("data", "~/work/oxfordflowers102", "Directory to cache downloaded and generated dataset files.")
	flagEval       = flag.Bool("eval", true, "Whether to evaluate the model on the validation data in the end.")
	flagVerbosity  = flag.Int("verbosity", 1, "Level of verbosity, the higher the more verbose.")
	flagCheckpoint = flag.String("checkpoint", "", "Directory save and load checkpoints from. If left empty, no checkpoints are created.")
)

var (
	backend = backends.MustNew()
)

func main() {
	ctx := fm.CreateDefaultContext()
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *settings))
	config := diffusion.NewConfig(backend, ctx, *flagDataDir, paramsSet)
	err := exceptions.TryCatch[error](func() {
		fm.TrainModel(config, *flagCheckpoint, *flagEval, *flagVerbosity)
	})
	if err != nil {
		klog.Fatalf("Failed with error: %+v", err)
	}
}
