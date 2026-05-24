// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// constraint_generator prints out various lists of constraints used by generics,
// which can then be copy&pasted into the code.
// It is an internal tool, meant to be used by developers of GoMLX.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path"
	"strings"

	"github.com/gomlx/gomlx/internal/must"
	"k8s.io/klog/v2"
)

var baseTypes = []string{
	"bool", "float32", "float64", "int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64", "complex64", "complex128",
	"bfloat16.BFloat16", "float16.Float16"}

const SliceLevels = 5

const outputFile = "gen_constraints.go"

func TensorSlicesConstraints(w io.Writer) {
	count := 0
	for slices := range SliceLevels {
		fmt.Fprintf(w, "\t")
		for idxType, t := range baseTypes {
			if idxType > 0 {
				fmt.Fprint(w, " | ")
			}
			for ii := 0; ii < slices; ii++ {
				fmt.Fprint(w, "[]")
			}
			fmt.Fprint(w, t)
			count += 1
		}
		if slices < SliceLevels-1 {
			fmt.Fprintf(w, " |")
		}
		fmt.Fprintln(w)
	}
}

func GraphExecFnConstraints(w io.Writer, withScope bool) {
	p := ""
	if withScope {
		p = "graph."
	}
	fmt.Fprintf(w, "type CanonicalExecGraphFn func(%s*%sGraph, []*%sNode) []*%sNode\n\n", func() string {
		if withScope {
			return "*Scope, "
		}
		return ""
	}(), p, p, p)

	type inputConfig struct {
		params       string
		numInputs    int
		inputIsGraph bool
		inputAsSlice bool
		callArgs     string
	}

	possibleInputs := []inputConfig{
		{"*" + p + "Graph", 0, true, false, "g"},
		{"[]*" + p + "Node", -1, false, true, "inputs"},
		{"*" + p + "Graph, []*" + p + "Node", -1, true, true, "g, inputs"},
	}
	for i := 1; i <= 6; i++ {
		params := ""
		callArgs := ""
		for j := 0; j < i; j++ {
			if j > 0 {
				params += ", "
				callArgs += ", "
			}
			params += "*" + p + "Node"
			callArgs += fmt.Sprintf("inputs[%d]", j)
		}
		possibleInputs = append(possibleInputs, inputConfig{params, i, false, false, callArgs})
	}

	type outputConfig struct {
		name          string
		output        string
		numOutputs    int
		outputAsSlice bool
		wrapOutputs   func(call string) string
	}
	possibleOutputs := []outputConfig{}
	if withScope {
		possibleOutputs = append(possibleOutputs, outputConfig{"ZeroOutputs", "", 0, false, func(call string) string {
			return fmt.Sprintf("%s; return nil", call)
		}})
	}
	possibleOutputs = append(possibleOutputs,
		outputConfig{"OneOutput", "*" + p + "Node", 1, false, func(call string) string {
			return fmt.Sprintf("return []*Node{%s}", call)
		}},
		outputConfig{"TwoOutputs", "(*" + p + "Node, *" + p + "Node)", 2, false, func(call string) string {
			return fmt.Sprintf("r0, r1 := %s; return []*Node{r0, r1}", call)
		}},
		outputConfig{"ThreeOutputs", "(*" + p + "Node, *" + p + "Node, *" + p + "Node)", 3, false, func(call string) string {
			return fmt.Sprintf("r0, r1, r2 := %s; return []*Node{r0, r1, r2}", call)
		}},
		outputConfig{"SliceOutputs", "[]*" + p + "Node", -1, true, func(call string) string {
			return fmt.Sprintf("return %s", call)
		}},
	)

	var contextInput string
	if withScope {
		contextInput = "*Scope, "
	}

	var interfaceNames []string
	for _, outConfig := range possibleOutputs {
		interfaceName := fmt.Sprintf("ExecGraphFn%s", outConfig.name)
		interfaceNames = append(interfaceNames, interfaceName)
		fmt.Fprintf(w, "type %s interface {\n", interfaceName)
		var parts []string
		outputs := outConfig.output
		if outputs != "" {
			outputs = " " + outputs
		}
		for _, inConfig := range possibleInputs {
			parts = append(parts, fmt.Sprintf("\tfunc (%s%s)%s", contextInput, inConfig.params, outputs))
		}
		fmt.Fprintf(w, "%s\n}\n\n", strings.Join(parts, " |\n"))
	}
	fmt.Fprintf(w, "type ExecGraphFn interface {\n\t%s\n}\n\n", strings.Join(interfaceNames, " | "))

	// Generate convertExecFn
	fmt.Fprintf(w, "func convertExecFn[F ExecGraphFn](graphFn F) (canonicalFn CanonicalExecGraphFn, numInputs, numOutputs int, inputIsGraph, inputAsSlice, outputAsSlice bool) {\n")
	fmt.Fprintf(w, "\tswitch f := any(graphFn).(type) {\n")
	for _, outConfig := range possibleOutputs {
		outputs := outConfig.output
		if outputs != "" {
			outputs = " " + outputs
		}
		for _, inConfig := range possibleInputs {
			fmt.Fprintf(w, "\tcase func(%s%s)%s:\n", contextInput, inConfig.params, outputs)
			fmt.Fprintf(w, "\t\tnumInputs, numOutputs = %d, %d\n", inConfig.numInputs, outConfig.numOutputs)
			fmt.Fprintf(w, "\t\tinputIsGraph, inputAsSlice, outputAsSlice = %v, %v, %v\n",
				inConfig.inputIsGraph, inConfig.inputAsSlice, outConfig.outputAsSlice)
			if withScope {
				fmt.Fprintf(w, "\t\tcanonicalFn = func(s *Scope, g *Graph, inputs []*Node) []*Node { %s }\n",
					outConfig.wrapOutputs(fmt.Sprintf("f(s, %s)", inConfig.callArgs)))
			} else {
				fmt.Fprintf(w, "\t\tcanonicalFn = func(g *Graph, inputs []*Node) []*Node { %s }\n",
					outConfig.wrapOutputs(fmt.Sprintf("f(%s)", inConfig.callArgs)))
			}
		}
	}
	fmt.Fprintf(w, "\tdefault:\n\t\tpanic(fmt.Sprintf(\"invalid graphFn type %%T\", graphFn))\n")
	fmt.Fprintf(w, "\t}\n")
	fmt.Fprintf(w, "\treturn\n")
	fmt.Fprintf(w, "}\n")
}

var (
	flagModel   = flag.Bool("model", false, "Generate constraints for the model package.")
	flagGraph   = flag.Bool("graph", false, "Generate constraints for the graph package.")
	flagTensors = flag.Bool("tensors", false, "Generate constraints for the tensors package.")
)

func main() {
	flag.Parse()

	numFlags := 0
	if *flagModel {
		numFlags++
	}
	if *flagGraph {
		numFlags++
	}
	if *flagTensors {
		numFlags++
	}
	if numFlags != 1 {
		klog.Exitf("Please specify exactly one of -context, -graph, or -tensors flag")
	}

	outputPath := path.Join(must.M1(os.Getwd()), outputFile)
	f := must.M1(os.Create(outputPath))

	fmt.Fprintln(f, "// ***** This file is generated by internal/cmd/constraints_generator. DO NOT EDIT DIRECTLY. *****")
	fmt.Fprintln(f)

	if *flagTensors {
		fmt.Fprintln(f, "package tensors")
		fmt.Fprintln(f)
		fmt.Fprintln(f, "import (")
		fmt.Fprintln(f, "\t\"github.com/gomlx/compute/dtypes/bfloat16\"")
		fmt.Fprintln(f, "\t\"github.com/gomlx/compute/dtypes/float16\"")
		fmt.Fprintln(f, ")")
		fmt.Fprintln(f)
		fmt.Fprintln(f, "// MultiDimensionSlice lists the Go types a Tensor can be converted to/from.")
		fmt.Fprintln(f, "// There are no recursions in generics' constraint definitions,")
		fmt.Fprintln(f, "// so we list up to 5 levels of slices. Feel free to add")
		fmt.Fprintln(f, "// more if needed, as the implementation will work with any arbitrary number.")
		fmt.Fprintln(f)
		fmt.Fprintln(f, "// Generated by `github.com/gomlx/gomlx/internal/cmd/constraints_generator`.")
		fmt.Fprintln(f, "type MultiDimensionSlice interface {")
		TensorSlicesConstraints(f)
		fmt.Fprintln(f, "}")
		fmt.Fprintln(f)
	} else if *flagGraph {
		fmt.Fprintln(f, "package graph")
		fmt.Fprintln(f)
		fmt.Fprintln(f, "import \"fmt\"")
		fmt.Fprintln(f)
		fmt.Fprintf(f, "// For graph/exec.go:\n\n")
		GraphExecFnConstraints(f, false)
		fmt.Fprintln(f)
	} else if *flagModel {
		fmt.Fprintln(f, "package model")
		fmt.Fprintln(f)
		fmt.Fprintln(f, "import (")
		fmt.Fprintln(f, "\t\"fmt\"")
		fmt.Fprintln(f, "\t\"github.com/gomlx/gomlx/core/graph\"")
		fmt.Fprintln(f, ")")
		fmt.Fprintln(f)
		fmt.Fprintf(f, "// For ml/model/exec.go:\n\n")
		GraphExecFnConstraints(f, true)
	}

	must.M(f.Close())
	cmd := exec.Command("go", "fmt", outputPath)
	klog.V(1).Infof("\t%s\n", cmd)
	cmd.Stderr = os.Stderr
	must.M(cmd.Run())

	fmt.Printf("✅ internal/cmd/constraints_generator: \tsuccessfully generated %s\n", outputPath)
}
