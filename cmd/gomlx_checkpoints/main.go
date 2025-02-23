// gomlx_checkpoints reports back on model size (and memory) usage (--summary), individual variables shapes and sizes (--vars),
// hyperparameters used with the model (--params) or metrics collected during model training (--metrics, --metrics_labels).
//
// See gomlx_checkpoint --help for details.
package main

import (
	"flag"
	"fmt"
	"github.com/charmbracelet/lipgloss"
	lgtable "github.com/charmbracelet/lipgloss/table"
	"github.com/dustin/go-humanize"
	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/graph"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/tensors"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gomlx/ui/plots"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"os"
	"path"
	"slices"
	"strings"
	"time"

	_ "github.com/gomlx/gomlx/backends/xla"
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

	flagMetricsNames = flag.String("metrics_names", "", "Comma-separate list of metric names to include in metrics Reports.")
	flagMetricsTypes = flag.String("metrics_types", "", "Comma-separate list of metric types to include in metrics Reports. ")

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
		pf("\n\t$ gomlx_checkpoints [flags...] <checkpoint_path>\n" +
			"\ngomlx_checkpoints reports back on model size (and memory) usage (--summary), individual variables shapes and sizes (--vars), " +
			"hyperparameters used with the model (--params) or metrics collected during model training (--metrics, --metrics_labels).\n" +
			"\n\t<checkpoint_path> is the path of a checkpoint directory used to save a GoMLX model (see package github.com/gomlx/gomlx/ml/context/checkpoints)\n\n" +
			"Flags:\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	args := flag.Args()
	if len(args) == 0 {
		klog.Errorf("Missing checkpoint directory to read from. See 'gomlx_checkpoint -help'")
		os.Exit(1)
	}
	if len(args) > 1 {
		klog.Errorf("Too many arguments. See 'gomlx_checkpoint -help'.")
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
			Reports(args[0])
			fmt.Println(italicStyle.Render(fmt.Sprintf("(... refreshing every %s ...)", *flagLoop)))
			time.Sleep(*flagLoop)
		}
	}
	Reports(args[0])
}

func ClearScreen() {
	fmt.Printf("\033c")
}

var (
	headerRowStyle = lipgloss.NewStyle().Reverse(true).
			Padding(0, 2, 0, 2).Align(lipgloss.Center)

	oddRowStyle = lipgloss.NewStyle().Faint(false).
			PaddingLeft(1).PaddingRight(1)
	evenRowStyle = lipgloss.NewStyle().Faint(true).
			PaddingLeft(1).PaddingRight(1)

	titleStyle    = lipgloss.NewStyle().Bold(true).Padding(1, 4, 1, 4)
	italicStyle   = lipgloss.NewStyle().Italic(true).Faint(true)
	emphasisStyle = lipgloss.NewStyle().Bold(true).Faint(false)
	sectionStyle  = lipgloss.NewStyle().Underline(true).Faint(false)
)

func newPlainTable(withHeader bool, alignments ...lipgloss.Position) *lgtable.Table {
	return lgtable.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("99"))).
		StyleFunc(func(row, col int) (s lipgloss.Style) {
			if row < 0 {
				s = headerRowStyle
				return
			}
			switch {
			case row%2 == 0:
				// Even row style.
				s = oddRowStyle
			default:
				// Odd row style
				s = evenRowStyle
			}
			alignment := lipgloss.Left
			if col < len(alignments) {
				alignment = alignments[col]
			} else if len(alignments) > 0 {
				alignment = alignments[len(alignments)-1]
			}
			s = s.Align(alignment)
			return
		})
}

func Reports(checkpointPath string) {
	ctx := context.New()
	if *flagSummary || *flagParams || *flagVars {
		_ = must.M1(checkpoints.Build(ctx).
			Dir(checkpointPath).Immediate().Done())
	}
	scopedCtx := ctx
	if *flagScope != "" {
		scopedCtx = ctx.InAbsPath(*flagScope)
	}

	// Summary table.
	if *flagSummary {
		fmt.Println(titleStyle.Render("Summary"))
		table := newPlainTable(false, lipgloss.Right, lipgloss.Left)
		table.Row("checkpoint", checkpointPath)
		table.Row("scope", *flagScope)
		globalStepVar := ctx.GetVariable(optimizers.GlobalStepVariableName)
		if globalStepVar != nil {
			globalStep := tensors.ToScalar[int64](globalStepVar.Value())
			table.Row(fmt.Sprintf("global_step(%s)", ctx.Scope()), humanize.Comma(globalStep))
		}

		var numVars, totalSize int
		var totalMemory uintptr
		scopedCtx.EnumerateVariablesInScope(func(v *context.Variable) {
			numVars++
			totalSize += v.Shape().Size()
			totalMemory += v.Shape().Memory()
		})
		table.Row("# variables", humanize.Comma(int64(numVars)))
		table.Row("# parameters", humanize.Comma(int64(totalSize)))
		table.Row("# bytes", humanize.Bytes(uint64(totalMemory)))
		fmt.Println(table.Render())
	}

	if *flagParams {
		fmt.Println(titleStyle.Render("Hyperparameters"))
		table := newPlainTable(true)
		table.Headers("Scope", "Name", "Type", "Value")
		ctx.EnumerateParams(func(scope, key string, value any) {
			table.Row(scope, key, fmt.Sprintf("%T", value), fmt.Sprintf("%v", value))
		})
		fmt.Println(table.Render())
	}

	if *flagVars {
		ListVariables(scopedCtx)
	}

	if *flagMetrics || *flagMetricsLabels {
		metrics(checkpointPath)
	}
}

func metrics(checkpointPath string) {
	trainingMetricsPath := path.Join(checkpointPath, plots.TrainingPlotFileName)
	points := must.M1(plots.LoadPoints(trainingMetricsPath))
	if len(points) == 0 {
		klog.Errorf("No metrics found in %q", trainingMetricsPath)
	}

	var metricsNames, metricsTypes types.Set[string]
	if *flagMetricsNames != "" {
		metricsNames = types.MakeSet[string]()
		for _, name := range strings.Split(*flagMetricsNames, ",") {
			metricsNames.Insert(name)
		}
	}
	if *flagMetricsTypes != "" {
		metricsTypes = types.MakeSet[string]()
		for _, name := range strings.Split(*flagMetricsTypes, ",") {
			metricsTypes.Insert(name)
		}
	}
	metricsUsed := types.MakeSet[string]()
	nameToShort := make(map[string]string)
	shortToName := make(map[string]string)
	for _, point := range points {
		nameToShort[point.MetricName] = point.Short
		shortToName[point.Short] = point.MetricName
		if metricsNames != nil || metricsTypes != nil {
			foundName := metricsNames != nil && (metricsNames.Has(point.MetricName) || metricsNames.Has(point.Short))
			foundType := metricsTypes != nil && metricsTypes.Has(point.MetricType)
			if !foundName && !foundType {
				continue
			}
		}
		metricsUsed.Insert(point.Short)
	}

	// map metric name to position in row, starting from 1 (position 0 is for the global set)
	metricsOrder := make(map[string]int)
	nextPos := 1
	if *flagMetricsNames != "" {
		// Start with the order given by the user:
		for _, name := range strings.Split(*flagMetricsNames, ",") {
			if short, found := nameToShort[name]; found {
				name = short
			}
			if metricsUsed.Has(name) {
				metricsOrder[name] = nextPos
				nextPos++
			}
		}
	}
	// Append all other metric names, in alphabetical order.
	for _, short := range xslices.SortedKeys(metricsUsed) {
		if _, found := metricsOrder[short]; found {
			continue // Already listed.
		}
		metricsOrder[short] = nextPos
		nextPos++
	}

	if *flagMetricsLabels {
		ReportMetricsLabels(shortToName, nameToShort, metricsOrder)
	}

	if *flagMetrics {
		ReportMetrics(checkpointPath, metricsUsed, metricsOrder, points)
	}
}

// ReportMetricsLabels list all metrics short and long names.
func ReportMetricsLabels(shortToName map[string]string, nameToShort map[string]string, metricsOrder map[string]int) {
	fmt.Println(titleStyle.Render("Metrics Labels"))
	table := newPlainTable(true, lipgloss.Center, lipgloss.Left)
	table.Headers("Short", "MetricName")
	rows := make([][]string, len(metricsOrder))
	for short, idx := range metricsOrder {
		name, found := shortToName[short]
		if !found {
			// metric manually selected by name:
			name = short
			short = nameToShort[name]
		}
		rows[idx-1] = []string{short, name}
	}
	for _, row := range rows {
		table.Row(row...)
	}
	fmt.Println(table.Render())
}

// ReportMetrics of the model.
func ReportMetrics(checkpointPath string, metricsUsed types.Set[string], metricsOrder map[string]int, points []plots.Point) {
	fmt.Println(titleStyle.Render(fmt.Sprintf("Metrics %q", checkpointPath)))
	table := newPlainTable(true, lipgloss.Right)
	header := make([]string, 1+len(metricsUsed))
	header[0] = "Global Step"
	for name, idx := range metricsOrder {
		header[idx] = name
	}
	table.Headers(header...)

	currentStep := int64(-1)
	var currentRow []string
	for _, point := range points {
		step := int64(point.Step)
		if step != currentStep {
			if currentStep != -1 {
				table.Row(currentRow...)
			}
			currentStep = step
			currentRow = make([]string, 1+len(metricsUsed))
			currentRow[0] = humanize.Comma(step)
		}
		idx, found := metricsOrder[point.Short]
		if found {
			var value string
			switch point.MetricType {
			case "accuracy":
				value = fmt.Sprintf("%.2f%%", 100.0*point.Value)
			default:
				value = fmt.Sprintf("%.3f", point.Value)
			}
			currentRow[idx] = value
		}
	}
	if currentStep != -1 {
		table.Row(currentRow...)
	}
	fmt.Println(table.Render())
}

// ListVariables list the variables of a model, with their shape and MAV (max absolute value), RMS (root mean square) and MaxAV (max absolute value) values.
func ListVariables(ctx *context.Context) {
	fmt.Println(titleStyle.Render(fmt.Sprintf("Variables in scope %q", ctx.Scope())))
	metricsFn := NewExec(backends.New(), func(x *Node) (mav, rms, maxAV *Node) {
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
