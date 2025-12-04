package main

import (
	"flag"
	"fmt"
	"slices"
	"strings"

	"github.com/dustin/go-humanize"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/internal/must"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/context/checkpoints"
	"github.com/gomlx/gopjrt/dtypes"
)

var (
	flagVars        = flag.Bool("vars", false, "Lists the variables under --scope.")
	flagDeleteVars  = flag.String("delete_vars", "", "Delete variables under the given scope(s). Useful for instance to remove training temporary data.")
	flagPerturbVars = flag.Float64("perturb", 0,
		"Perturbs trainable variables by <x>: it multiplies the weights by 1.0+(RandomUniform(-1, 1)*x). "+
			"If using Adam optimizer (or other optimizers) remember to clear their running moving averages with -delete_vars. "+
			"Only variables that are both trainable and float are modified.")
)

// ListVariables list the variables of a model, with their shape and MAV (max absolute value), RMS (root-mean-square) and MaxAV (max absolute value) values.
func ListVariables(ctx *context.Context) {
	fmt.Println(titleStyle.Render(fmt.Sprintf("Variables in scope %q", ctx.Scope())))
	metricsFn := MustNewExec(backends.MustNew(), func(x *Node) (mav, rms, maxAV *Node) {
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
			mav = fmt.Sprintf("%8v", must.M1(v.Value()).Value())
		} else if shape.DType.IsFloat() {
			metrics := metricsFn.MustExec(must.M1(v.Value()))
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
		for v := range ctx.IterVariables() {
			if v.Scope() == scope || strings.HasPrefix(v.Scope(), scopePrefix) {
				varsToDelete = append(varsToDelete, v)
			}
		}
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

func PerturbVars(checkpointPath string, x float64) {
	backend := must.M1(simplego.New(""))
	ctx := context.New()
	checkpoint := must.M1(checkpoints.Build(ctx).
		Dir(checkpointPath).Keep(-1).Immediate().Done())
	var numUpdates int
	for v := range ctx.IterVariables() {
		if !v.Trainable || !(v.DType().IsFloat() || v.DType().IsComplex()) {
			continue
		}
		newValue := context.MustExecOnce(backend, ctx, func(ctx *context.Context, g *Graph) *Node {
			value := v.ValueGraph(g)
			// Perturbation from -1 to 1
			perturbation := OneMinus(MulScalar(ctx.RandomUniform(g, value.Shape()), 2))
			perturbation = MulScalar(perturbation, x) // [-x, +x], -perturb=x
			perturbation = OnePlus(perturbation)      // [1-x, 1+x]
			return Mul(value, perturbation)
		})
		v.SetValue(newValue)
		numUpdates++
	}
	must.M(checkpoint.Save())
	fmt.Printf("%d variables updated new checkpoint saved.\n", numUpdates)
}
