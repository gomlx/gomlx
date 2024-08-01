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
	flagMetricsNames = flag.String("metrics_names", "", "Comma-separate list of metric names to include in metrics report.")
	flagMetricsTypes = flag.String("metrics_types", "", "Comma-separate list of metric types to include in metrics report. ")
)

func main() {
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
	report(args[0])
}

var (
	headerRowStyle = lipgloss.NewStyle().Reverse(true).
			Padding(0, 2, 0, 2).Align(lipgloss.Center)

	oddRowStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#FFF")).
			PaddingLeft(1).PaddingRight(1)
	evenRowStyle = lipgloss.NewStyle().Foreground(lipgloss.Color("#999")).
			PaddingLeft(1).PaddingRight(1)

	titleStyle = lipgloss.NewStyle().Bold(true).Padding(1, 4, 1, 4)
)

func newPlainTable(withHeader bool) *lgtable.Table {
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
			if col == 0 {
				s = s.Align(lipgloss.Right)
			} else {
				s = s.Align(lipgloss.Left)
			}
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
		table := newPlainTable(false)
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

	if *flagMetrics {
		metrics(checkpointPath)
	}
}

func metrics(checkpointPath string) {
	trainingMetricsPath := path.Join(checkpointPath, plots.TrainingPlotFileName)
	points := must.M1(plots.LoadPoints(trainingMetricsPath))
	if len(points) == 0 {
		klog.Errorf("No metrics found in %q", trainingMetricsPath)
	}
	fmt.Println(titleStyle.Render("Metrics"))

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
	for _, point := range points {
		nameToShort[point.MetricName] = point.Short
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

	table := newPlainTable(true)
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
				value = fmt.Sprintf("%f", point.Value)
			}
			currentRow[idx] = value
		}
	}
	if currentStep != -1 {
		table.Row(currentRow...)
	}
	fmt.Println(table.Render())
}
