// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/gobackend"
	"github.com/gomlx/compute/support/humanize"
	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/ml/model/checkpoint"
)

var (
	flagVars        = flag.Bool("vars", false, "Lists the variables under --scope.")
	flagDeleteVars  = flag.String("delete_vars", "", "Delete variables under the given scope(s). Useful for instance to remove training temporary data.")
	flagPerturbVars = flag.Float64("perturb", 0,
		"Perturbs trainable variables by <x>: it multiplies the weights by 1.0+(RandomUniform(-1, 1)*x). "+
			"If using Adam optimizer (or other optimizers) remember to clear their running moving averages with -delete_vars. "+
			"Only variables that are both trainable and float are modified.")
)

// ListVariables list the variables of a model under the given scope, with their shape and MAV (max absolute value), RMS (root-mean-square) and MaxAV (max absolute value) values.
func ListVariables(scope *model.Scope) {
	fmt.Println(titleStyle.Render(fmt.Sprintf("Variables in scope %q", scope.Scope())))
	metricsFn := MustNewExec(compute.MustNew(), func(x *Node) (mav, rms, maxAV *Node) {
		x = ConvertDType(x, dtypes.Float64)
		mav = ReduceAllMean(Abs(x))
		rms = Sqrt(ReduceAllMean(Square(x)))
		maxAV = ReduceAllMax(Abs(x))
		return
	}).SetMaxCache(-1)
	var rows [][]string
	for v := range scope.IterVariables() {
		if !v.IsValid() {
			rows = append(rows, []string{v.Scope(), v.Name(), "<invalid>", "", "", "", "", ""})
			continue
		}
		shape := v.Shape()
		var mav, rms, maxAV string
		if shape.Size() == 1 {
			mav = fmt.Sprintf("%8v", must.M1(v.Value()).Value())
		} else if shape.DType.IsFloat() {
			metrics := metricsFn.MustCall(must.M1(v.Value()))
			mav = fmt.Sprintf("%.3g", metrics[0].Value().(float64))
			rms = fmt.Sprintf("%.3g", metrics[1].Value().(float64))
			maxAV = fmt.Sprintf("%.3g", metrics[2].Value().(float64))
		}
		rows = append(rows, []string{
			v.Scope(), v.Name(), shape.String(),
			humanize.Underscores(int64(shape.Size())),
			humanize.Bytes(shape.ByteSize()),
			mav, rms, maxAV,
		})
	}
	slices.SortFunc(rows, func(a, b []string) int {
		cmp := strings.Compare(a[0], b[0])
		if cmp != 0 {
			return cmp
		}
		return strings.Compare(a[1], b[1])
	})
	rowAlt := make([]bool, len(rows))
	var lastScope string
	var isEven bool
	for i, row := range rows {
		scope := row[0]
		if i == 0 {
			lastScope = scope
			isEven = false
		} else if scope != lastScope {
			lastScope = scope
			isEven = !isEven
		}
		rowAlt[i] = isEven
	}
	table := newPlainTableWithRowAlt(true, rowAlt)
	table.Headers("Scope", "Name", "Shape", "Size", "Bytes", "Scalar/MAV", "RMS", "MaxAV")
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
	store := model.NewStore()
	checkpointHandler := must.M1(checkpoint.Build(store).
		Dir(checkpointPath).Keep(-1).Immediate().Done())
	var varsToDelete []*model.Variable
	for _, scopePath := range scopes {
		if scopePath == "" {
			continue
		}
		scope := store.Scope(scopePath)
		for v := range scope.IterVariables() {
			varsToDelete = append(varsToDelete, v)
		}
	}
	if len(varsToDelete) == 0 {
		// No changes needed.
		return
	}
	for _, v := range varsToDelete {
		store.DeleteVariable(v.Path())
	}
	must.M(checkpointHandler.Save())
	fmt.Printf("%d deleted vars under scopes %v, new checkpoint saved.\n", len(varsToDelete), scopes)
}

func PerturbVars(checkpointPath string, x float64) {
	backend := must.M1(gobackend.New(""))
	scope := model.NewStore()
	checkpointHandler := must.M1(checkpoint.Build(scope).
		Dir(checkpointPath).Keep(-1).Immediate().Done())
	var numUpdates int
	for v := range scope.IterVariables() {
		if !v.Trainable || !(v.DType().IsFloat() || v.DType().IsComplex()) {
			continue
		}
		newValue := model.MustCallOnce(backend, scope, func(scope *model.Scope, g *Graph) *Node {
			value := v.NodeValue(g)
			// Perturbation from -1 to 1
			perturbation := OneMinus(MulScalar(scope.RandomUniform(g, value.Shape()), 2))
			perturbation = MulScalar(perturbation, x) // [-x, +x], -perturb=x
			perturbation = OnePlus(perturbation)      // [1-x, 1+x]
			return Mul(value, perturbation)
		})
		v.SetValue(newValue)
		numUpdates++
	}
	must.M(checkpointHandler.Save())
	fmt.Printf("%d variables updated new checkpoint saved.\n", numUpdates)
}
