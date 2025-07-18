// gomlx_checkpoints reports back on model size (and memory) usage (--summary), individual variables shapes and sizes (--vars),
// hyperparameters used with the model (--params) or metrics collected during model training (--metrics, --metrics_labels).
//
// See gomlx_checkpoint --help for details.
package main

import (
	"flag"
	"fmt"
	"os"
	"path"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/ui/plots"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	flagScope = flag.String("scope", "/", "The scope of the checkpoint to inspect. "+
		"Typically, a model will have several different support variables, that may not matter -- optimizers for instance. "+
		"This flag tells which scope are considered for the various reports.")

	flagAll     = flag.Bool("all", false, "Display all information. The same as -summary -params -vars -metrics -metrics_labels.")
	flagSummary = flag.Bool("summary", false, "Display a summary of the model sizes (for variables"+
		" under --scope and the global step.")
	flagParams  = flag.Bool("params", false, "Lists the hyperparameters.")
	flagMetrics = flag.Bool("metrics", false,
		fmt.Sprintf("Lists the metrics collected for plotting in file %q", plots.TrainingPlotFileName))
	flagMetricsLabels = flag.Bool("metrics_labels", false,
		fmt.Sprintf("Lists the metrics labels (short names) with their full description from file %q", plots.TrainingPlotFileName))

	flagBackup   = flag.Bool("backup", false, "Set to true to make a backup of the most recent checkpoint, under the 'backup' subdirectory.")
	flagGlossary = flag.Bool("glossary", true, "Whether to list glossary of abbreviation on the bottom of tables.")

	flagLoop = flag.Duration("loop", 0, "Sets looping with the given period. "+
		"This is used to monitor the training of a program, usually used in conjunction with --metrics. "+
		"It will also clear the terminal in between printing out the metrics")
)

func main() {
	flag.Usage = func() {
		pf := func(format string, args ...any) {
			_ = must.M1(fmt.Fprintf(flag.CommandLine.Output(), format, args...))
		}
		pf("Usage of gomlx_checkpoints (%q):\n", os.Args[0])
		pf("\n\t$ gomlx_checkpoints [flags...] <checkpoint_path> [<checkpoint_path2> ...]\n" +
			"\ngomlx_checkpoints reports back on model size (and memory) usage (--summary), individual variables shapes and sizes (--vars), " +
			"hyperparameters used with the model (--params) or metrics collected during model training (--metrics, --metrics_labels).\n" +
			"\n\t<checkpoint_path> is the path of a checkpoint directory used to save a GoMLX model (see package github.com/gomlx/gomlx/ml/context/checkpoints)\n" +
			"\tSome flags support more than one checkpoint, which can be used to compare models.\n\n" +
			"Flags:\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		klog.Errorf("Missing checkpoint directory to read from. See 'gomlx_checkpoint -help'")
		os.Exit(1)
	}

	if *flagBackup {
		Backup(args[0])
	}

	if *flagDeleteVars != "" {
		DeleteVars(args[0], strings.Split(*flagDeleteVars, ",")...)
	}

	if *flagPerturbVars > 0 {
		PerturbVars(args[0], *flagPerturbVars)
	}

	if *flagAll {
		*flagSummary = true
		*flagParams = true
		*flagVars = true
		*flagMetrics = true
		*flagMetricsLabels = true
	}

	if *flagLoop > 0 {
		for {
			ClearScreen()
			Reports(args)
			fmt.Println(italicStyle.Render(fmt.Sprintf("(... refreshing every %s ...)", *flagLoop)))
			time.Sleep(*flagLoop)
		}
	}
	Reports(args)
}

func ClearScreen() {
	fmt.Printf("\033c")
}

var (
	titleStyle    = lipgloss.NewStyle().Bold(true).Padding(1, 4, 1, 4)
	italicStyle   = lipgloss.NewStyle().Italic(true).Faint(true)
	emphasisStyle = lipgloss.NewStyle().Bold(true).Faint(false)
	sectionStyle  = lipgloss.NewStyle().Underline(true).Faint(false)
)

func Reports(checkpointPaths []string) {
	numCheckpoints := len(checkpointPaths)
	ctxs := make([]*context.Context, 0, len(checkpointPaths))
	scopedCtxs := make([]*context.Context, 0, len(checkpointPaths))
	for _, checkpointPath := range checkpointPaths {
		ctx := context.New()
		if *flagSummary || *flagParams || *flagVars {
			_ = must.M1(checkpoints.Build(ctx).
				Dir(checkpointPath).Immediate().Done())
		}
		scopedCtx := ctx
		if *flagScope != "" {
			scopedCtx = ctx.InAbsPath(*flagScope)
		}
		ctxs = append(ctxs, ctx)
		scopedCtxs = append(scopedCtxs, scopedCtx)
	}
	var names []string
	if numCheckpoints == 1 {
		names = []string{checkpointPaths[0]}
	} else {
		names = MinimalUniquePaths(checkpointPaths...)
	}

	if *flagSummary {
		Summary(ctxs, scopedCtxs, names)
	}
	if *flagParams {
		Params(ctxs, scopedCtxs, names)
	}
	if *flagMetrics || *flagMetricsLabels {
		metrics(checkpointPaths, names)
	}
	if *flagVars {
		if numCheckpoints > 1 {
			klog.Fatalf("More than one checkpoint not supported for --vars")
		}
		ListVariables(scopedCtxs[0])
	}

}

func Backup(checkpointPath string) {
	ctx := context.New()
	checkpoint := must.M1(checkpoints.Build(ctx).
		Dir(checkpointPath).Keep(-1).Immediate().Done())
	must.M(checkpoint.Backup())
	globalStep := optimizers.GetGlobalStep(ctx)
	fmt.Printf("Backup of latest checkpoint (global step %d) successful, see %q.\n", globalStep, path.Join(checkpoint.Dir(), checkpoints.BackupDir))
}
