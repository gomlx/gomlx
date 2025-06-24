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
	"slices"
	"strings"
	"time"

	"github.com/charmbracelet/lipgloss"
	"github.com/dustin/go-humanize"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/ui/plots"
	"github.com/gomlx/gopjrt/dtypes"
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
	flagVars    = flag.Bool("vars", false, "Lists the variables under --scope.")
	flagMetrics = flag.Bool("metrics", false,
		fmt.Sprintf("Lists the metrics collected for plotting in file %q", plots.TrainingPlotFileName))
	flagMetricsLabels = flag.Bool("metrics_labels", false,
		fmt.Sprintf("Lists the metrics labels (short names) with their full description from file %q", plots.TrainingPlotFileName))

	flagBackup     = flag.Bool("backup", false, "Set to true to make a backup of the most recent checkpoint, under the 'backup' subdirectory.")
	flagDeleteVars = flag.String("delete_vars", "", "Delete variables under the given scope(s). Useful for instance to remove training temporary data.")
	flagGlossary   = flag.Bool("glossary", true, "Whether to list glossary of abbreviation on the bottom of tables.")

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

// ListVariables list the variables of a model, with their shape and MAV (max absolute value), RMS (root mean square) and MaxAV (max absolute value) values.
func ListVariables(ctx *context.Context) {
	fmt.Println(titleStyle.Render(fmt.Sprintf("Variables in scope %q", ctx.Scope())))
	metricsFn := NewExec(backends.MustNew(), func(x *Node) (mav, rms, maxAV *Node) {
		x = ConvertDType(x, dtypes.Float64)
		mav = ReduceAllMean(Abs(x))
		rms = Sqrt(ReduceAllMean(Square(x)))
		maxAV = ReduceAllMax(Abs(x))
		return
	}).SetMaxCache(-1)
	table := newPlainTable(true)
	table.Headers("Scope", "Name", "Shape", "Size", "Bytes", "Scalar/MAV", "RMS", "MaxAV")
	var rows [][]string
	ctx.EnumerateVariablesInScope(func(v *context.Variable) {
		if !v.IsValid() {
			rows = append(rows, []string{v.Scope(), v.Name(), "<invalid>", "", "", "", "", ""})
			return
		}
		shape := v.Shape()
		var mav, rms, maxAV string
		if shape.Size() == 1 {
			mav = fmt.Sprintf("%8v", v.Value().Value())
		} else if shape.DType.IsFloat() {
			metrics := metricsFn.Call(v.Value())
			mav = fmt.Sprintf("%.3g", metrics[0].Value().(float64))
			rms = fmt.Sprintf("%.3g", metrics[1].Value().(float64))
			maxAV = fmt.Sprintf("%.3g", metrics[2].Value().(float64))
		}
		rows = append(rows, []string{
			v.Scope(), v.Name(), shape.String(),
			humanize.Comma(int64(shape.Size())),
			humanize.Bytes(uint64(shape.Memory())),
			mav, rms, maxAV,
		})
	})
	slices.SortFunc(rows, func(a, b []string) int {
		cmp := strings.Compare(a[0], b[0])
		if cmp != 0 {
			return cmp
		}
		return strings.Compare(a[1], b[1])
	})
	for _, row := range rows {
		table.Row(row...)
	}
	fmt.Println(table.Render())
	if *flagGlossary {
		fmt.Printf("  %s:\n", sectionStyle.Render("Glossary"))
		fmt.Printf("   ◦ %s: %s\n", emphasisStyle.Render("Scalar/MAV"), italicStyle.Render("If variable is a scalar then the value itself, else the Mean Absolute Value"))
		fmt.Printf("   ◦ %s: %s\n", emphasisStyle.Render("RMS"), italicStyle.Render("Root Mean Square"))
		fmt.Printf("   ◦ %s: %s\n", emphasisStyle.Render("MaxAV"), italicStyle.Render("Max Absolute Value"))
	}
}

// DeleteVars on the given scopes.
func DeleteVars(checkpointPath string, scopes ...string) {
	ctx := context.New()
	checkpoint := must.M1(checkpoints.Build(ctx).
		Dir(checkpointPath).Keep(-1).Immediate().Done())
	var varsToDelete []*context.Variable
	for _, scope := range scopes {
		if scope == "" {
			continue
		}
		scopePrefix := scope + context.ScopeSeparator
		ctx.EnumerateVariables(func(v *context.Variable) {
			if v.Scope() == scope || strings.HasPrefix(v.Scope(), scopePrefix) {
				varsToDelete = append(varsToDelete, v)
			}
		})
	}
	if len(varsToDelete) == 0 {
		// No changes needed.
		return
	}
	for _, v := range varsToDelete {
		ctx.DeleteVariable(v.Scope(), v.Name())
	}
	must.M(checkpoint.Save())
	fmt.Printf("%d deleted vars under scopes %v, new checkpoint saved.\n", len(varsToDelete), scopes)
}

func Backup(checkpointPath string) {
	ctx := context.New()
	checkpoint := must.M1(checkpoints.Build(ctx).
		Dir(checkpointPath).Keep(-1).Immediate().Done())
	must.M(checkpoint.Backup())
	globalStep := optimizers.GetGlobalStep(ctx)
	fmt.Printf("Backup of latest checkpoint (global step %d) successful, see %q.\n", globalStep, path.Join(checkpoint.Dir(), checkpoints.BackupDir))
}
