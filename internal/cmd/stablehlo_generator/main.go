// stablehlo_generator generates stablehlo.Backend implementations based on backend.Builder API.
package main

import (
	"bufio"
	"flag"
	"fmt"
	"os"
	"strings"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/internal/backendparser"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
)

var flagOp = flag.String("op", "", "Generate only the specified operation in the standard out: "+
	"This is meant to generate a skeleton that will copy&pasted and then manually completed.")

func main() {
	klog.InitFlags(nil)
	flag.Parse()
	klog.V(1).Info("stablehlo_generator:")
	methods := must.M1(backendparser.ParseBuilder())

	if *flagOp == "" {
		// Generate standard files for stablehlo backend.
		GenerateBinaryOps(methods)
		GenerateUnaryOps(methods)
		return
	}

	// Select the method corresponding to the selected op.
	var method backendparser.Method
	for _, m := range methods {
		if m.Name == *flagOp {
			method = m
			break
		}
	}
	if method.Name == "" {
		klog.Fatalf("No such op %q in the list of Backend methods", *flagOp)
	}
	GenerateSingleOp(method)
}

func GenerateSingleOp(method backendparser.Method) {
	// Prepare writing.
	writer := bufio.NewWriter(os.Stdout)
	defer func() {
		must.M(writer.Flush())
	}()
	w := func(format string, args ...interface{}) {
		_, err := fmt.Fprintf(writer, format, args...)
		if err != nil {
			exceptions.Panicf("Failed to write to stdout: %v", err)
		}
	}

	// Comments:
	for _, comment := range method.Comments {
		w("%s\n", comment)
	}

	// Method signature:
	w("func (b *Builder) %s(", method.Name)
	for i, param := range method.Parameters {
		if i > 0 {
			w(", ")
		}
		w("%s", param.Name)
		if i < len(method.Parameters)-2 && method.Parameters[i+1].Type == param.Type {
			// This parameter will use the same type as the next one, no need to repeat it here.
			continue
		}
		if param.Type == "Op" {
			w(" backends.Op")
		} else {
			w(" %s", param.Type)
		}
	}
	w(")")
	if len(method.Outputs) > 0 {
		if len(method.Outputs) == 1 && method.Outputs[0].Name == "" {
			w(" %s", method.Outputs[0].Type)
		} else {
			w(" (")
			for i, output := range method.Outputs {
				if i > 0 {
					w(", ")
				}
				if output.Name == "" {
					w("%s", output.Type)
				} else {
					w("%s %s", output.Name, output.Type)
				}
			}
			w(")")
		}
	}
	w(" {\n")

	// Write parsing of the "backend.Op" parameters.
	var opsParams []string
	for _, param := range method.Parameters {
		if param.Type == "Op" {
			opsParams = append(opsParams, param.Name)
		}
	}
	w("\tnodes, err := b.verifyAndCastValues(\"Dot\", %s)\n\tif err != nil {\n\t\treturn nil, err\n\t}\n",
		strings.Join(opsParams, ", "))
	for i, opsParam := range opsParams {
		w("\t%sNode := nodes[%d]\n", opsParam, i)
	}

	// Call the method from the stablehlo backend.
	w("\tvalue, err := b.fn.%s(", method.Name)
	for i, param := range method.Parameters {
		if i > 0 {
			w(", ")
		}
		if param.Type == "Op" {
			w("%sNode.value", opsParams[i])
		} else {
			w("%s", param.Name)
		}
	}
	w(")\n")
	w("\tif err != nil {\n\t\treturn nil, err\n\t}\n")

	// Return the node.
	w("\treturn b.newNode(value)\n")
	w("}\n")
}
