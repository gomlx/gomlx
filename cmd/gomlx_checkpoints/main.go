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
	"github.com/gomlx/gomlx/examples/notebook/gonb/plots"
	"github.com/gomlx/gomlx/ml/context"
	"github.com/gomlx/gomlx/ml/context/checkpoints"
	"github.com/gomlx/gomlx/ml/train/optimizers"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
	"os"
	"path"
	"slices"
	"strings"
	"time"
)

var (
	flagScope = flag.String("scope", "/model", "The scope of the checkpoint to inspect. "+
		"Typically, a model will have several different support variables, that may not matter -- optimizers for instance. "+
		"This flag tells which scope are considered for the various reports.")

	flagSummary = flag.Bool("summary", false, "Display a summary of the model sizes (for variables"+
		" under --scope and the global step.")
	flagParams  = flag.Bool("params", false, "Lists the hyperparameters.")
	flagVars    = flag.Bool("vars", false, "Lists the variables under --scope.")
	flagMetrics = flag.Bool("metrics", false,
		fmt.Sprintf("Lists the metrics collected for plotting in file %q", plots.TrainingPlotFileName))
	flagMetricsLabels = flag.Bool("metrics_labels", false,
		fmt.Sprintf("Lists the metrics labels (short names) with their full description from file %q", plots.TrainingPlotFileName))

	flagMetricsNames = flag.String("metrics_names", "", "Comma-separate list of metric names to include in metrics report.")
	flagMetricsTypes = flag.String("metrics_types", "", "Comma-separate list of metric types to include in metrics report. ")

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

	if *flagLoop > 0 {
		for {
			ClearScreen()
			report(args[0])
			fmt.Println(italicStyle.Render(fmt.Sprintf("(... refreshing every %s ...)", *flagLoop)))
			time.Sleep(*flagLoop)
		}
	}
	report(args[0])
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

	titleStyle = lipgloss.NewStyle().Bold(true).Padding(1, 4, 1, 4)

	italicStyle = lipgloss.NewStyle().PaddingTop(1).Italic(true).Faint(true)
)

func newPlainTable(withHeader bool, alignments ...lipgloss.Position) *lgtable.Table {
	return lgtable.New().
		Border(lipgloss.NormalBorder()).
		BorderStyle(lipgloss.NewStyle().Foreground(lipgloss.Color("99"))).
		StyleFunc(func(row, col int) (s lipgloss.Style) {
			if withHeader && row == 1 {
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

func report(checkpointPath string) {
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
		globalStep := int64(optimizers.GetGlobalStep(ctx))
		table.Row("global_step", humanize.Comma(globalStep))

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
		table.Row("Scope", "Name", "Type", "Value")
		ctx.EnumerateParams(func(scope, key string, value any) {
			table.Row(scope, key, fmt.Sprintf("%T", value), fmt.Sprintf("%v", value))
		})
		fmt.Println(table.Render())
	}

	if *flagVars {
		fmt.Println(titleStyle.Render("Variables"))
		table := newPlainTable(true)
		table.Row("Scope", "Name", "Shape", "Size", "Bytes")
		var rows [][]string
		scopedCtx.EnumerateVariablesInScope(func(v *context.Variable) {
			shape := v.Shape()
			rows = append(rows, []string{
				v.Scope(), v.Name(), shape.String(),
				humanize.Comma(int64(shape.Size())),
				humanize.Bytes(uint64(shape.Memory())),
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
		fmt.Println(titleStyle.Render("Metrics Labels"))
		table := newPlainTable(true, lipgloss.Center, lipgloss.Left)
		table.Row("Short", "MetricName")
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

	if *flagMetrics {
		fmt.Println(titleStyle.Render("Metrics"))
		table := newPlainTable(true, lipgloss.Right)
		header := make([]string, 1+len(metricsUsed))
		header[0] = "Global Step"
		for name, idx := range metricsOrder {
			header[idx] = name
		}
		table.Row(header...)

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
}