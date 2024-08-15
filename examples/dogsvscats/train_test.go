package dogsvscats

import (
	"flag"
	"github.com/gomlx/gomlx/ml/train/commandline"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

// TestTrain train the default model for 50 steps.
func TestTrain(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	ctx := CreateDefaultContext()
	ctx.SetParam("train_steps", 50) // Only 50 steps.
	settings := commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	flag.Parse()
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *settings))
	TrainModel(ctx, *flagDataDir, "", paramsSet)
}
