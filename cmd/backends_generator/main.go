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
	"github.com/gomlx/gomlx/types/xslices"
	"go/ast"
	"strings"
)

func main() {
	opsInfo := parsexlabuilder.ReadOpsInfo()
	_ = opsInfo
	fmt.Println(parsexlabuilder.GopjrtSourcePath)
	extractor, xlaBuilderAst := parsexlabuilder.Parse()
	parsexlabuilder.EnumerateStandardOpsFunctions(extractor, xlaBuilderAst, func(funcDecl *ast.FuncDecl) {
		parts := xslices.Map(funcDecl.Type.Results.List, func(field *ast.Field) string {
			return extractor.Get(field.Type)
		})
		fmt.Printf("\t%s -> %s\n", funcDecl.Name.Name, strings.Join(parts, ", "))
	})
}
