package main

import (
	"fmt"

	"github.com/charmbracelet/lipgloss"
	"github.com/dustin/go-humanize"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/ml/train/optimizers"
)

func Summary(ctxs, scopedCtxs []*context.Context, names []string) {
	numCheckpoints := len(names)

	// Header
	fmt.Println(titleStyle.Render("Summary"))
	table := newPlainTable(false, lipgloss.Right, lipgloss.Left)
	table.Row(append([]string{"checkpoint"}, names...)...)

	// Scope
	scopeRow := make([]string, numCheckpoints+1)
	scopeRow[0] = "scope"
	for ii := 1; ii < numCheckpoints; ii++ {
		scopeRow[ii] = *flagScope
	}
	table.Row(scopeRow...)

	// Global step:
	globalStepRow := make([]string, numCheckpoints+1)
	globalStepRow[0] = "global_step"
	haveGlobalStep := false
	for ii, ctx := range ctxs {
		globalStepVar := ctx.GetVariable(optimizers.GlobalStepVariableName)
		if globalStepVar != nil {
			haveGlobalStep = true
			globalStepT := must.M1(globalStepVar.Value())
			globalStepRow[ii+1] = humanize.Comma(tensors.ToScalar[int64](globalStepT))
		}
	}
	if haveGlobalStep {
		table.Row(globalStepRow...)
	}

	// Variables, parameters and memory.
	variablesRow := make([]string, numCheckpoints+1)
	parametersRow := make([]string, numCheckpoints+1)
	memoryRow := make([]string, numCheckpoints+1)
	variablesRow[0] = "# variables"
	parametersRow[0] = "# parameters"
	memoryRow[0] = "# bytes"
	for ii, scopedCtx := range scopedCtxs {
		var numVars, totalSize int
		var totalMemory uintptr
		scopedCtx.EnumerateVariablesInScope(func(v *context.Variable) {
			numVars++
			totalSize += v.Shape().Size()
			totalMemory += v.Shape().Memory()
		})
		variablesRow[ii+1] = humanize.Comma(int64(numVars))
		parametersRow[ii+1] = humanize.Comma(int64(totalSize))
		memoryRow[ii+1] = humanize.Bytes(uint64(totalMemory))
	}
	table.Row(variablesRow...)
	table.Row(parametersRow...)
	table.Row(memoryRow...)
	fmt.Println(table.Render())
}
