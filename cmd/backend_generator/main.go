// backend_generator generates the graphs.Backend interface based on the github.com/gomlx/gopjrt/xlabuilder implementation.
//
// Although GoMLX can support more than one backend, the XlaBuilder is the reference implementation for now.
//
// If the environment variable GOPJRT_SRC is set, it parses the ops from there.
// Otherwise it
package main

import (
	"fmt"
	"github.com/gomlx/gomlx/cmd/backend_generator/parsexlabuilder"
	"go/ast"
)

func main() {
	opsInfo := parsexlabuilder.ReadOpsInfo()
	_ = opsInfo
	fileSet, xlaBuilderAst := parsexlabuilder.Parse()
	_, _ = fileSet, xlaBuilderAst
	for fileName, fileAst := range xlaBuilderAst.Files {
		for _, decl := range fileAst.Decls {
			funcDecl, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			fmt.Printf("Function %q declared in %s\n", funcDecl.Name.Name, fileName)
		}
	}
}
