package dogsvscats

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/ml/layers"
	"github.com/gomlx/gomlx/ui/commandline"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"os"
	"sync"
	"testing"

	_ "github.com/gomlx/gomlx/backends/xla"
)

var (
	flagSettings *string
	muTrain      sync.Mutex
)

func init() {
	ctx := CreateDefaultContext()
	flagSettings = commandline.CreateContextSettingsFlag(ctx, "")
	klog.InitFlags(nil)
	if _, found := os.LookupEnv(backends.ConfigEnvVar); !found {
		// For testing, we use the CPU backend (and avoid GPU if not explicitly requested).
		must.M(os.Setenv(backends.ConfigEnvVar, "xla:cpu"))
	}
}

// TestTrain train the default model for 50 steps.
func TestTrain(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping testing in short mode")
		return
	}
	ctx := CreateDefaultContext()
	ctx.SetParam("train_steps", 10)
	ctx.SetParam("plots", false)
	ctx.SetParam(layers.ParamNormalization, "layer")
	paramsSet := must.M1(commandline.ParseContextSettings(ctx, *flagSettings))
	TrainModel(ctx, *flagDataDir, "", false, paramsSet)
}
