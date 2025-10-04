// main.go
package main

import (
	"bytes"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"strings"

	"k8s.io/klog/v2"
)

func main() {
	if len(os.Args) != 4 {
		klog.Fatalf("Usage: %s <old_pkg_path> <new_pkg_import_path> <output_file>", os.Args[0])
	}
	oldPkgPath := os.Args[1]
	newPkgImportPath := os.Args[2]
	outputFile := os.Args[3]

	newPkgAlias := filepath.Base(newPkgImportPath)

	// 1. Find all .go files in the new package directory.
	goPath := os.Getenv("GOPATH")
	if goPath == "" {
		klog.Fatal("GOPATH not set")
	}
	newPkgPath := filepath.Join(goPath, "src", newPkgImportPath)

	files, err := filepath.Glob(filepath.Join(newPkgPath, "*.go"))
	if err != nil {
		klog.Fatal(err)
	}

	var out bytes.Buffer
	fmt.Fprintf(&out, "package %s\n\n", filepath.Base(oldPkgPath))
	fmt.Fprintf(&out, "import %s \"%s\"\n\n", newPkgAlias, newPkgImportPath)

	// 2. Parse each file and generate shims.
	fset := token.NewFileSet()
	for _, file := range files {
		if strings.HasSuffix(file, "_test.go") {
			continue // Skip test files
		}
		node, err := parser.ParseFile(fset, file, nil, parser.ParseComments)
		if err != nil {
			klog.Fatal(err)
		}

		for _, decl := range node.Decls {
			switch d := decl.(type) {
			case *ast.GenDecl:
				for _, spec := range d.Specs {
					switch s := spec.(type) {
					case *ast.TypeSpec:
						if s.Name.IsExported() {
							fmt.Fprintf(&out, "// Deprecated: Use %s.%s instead.\n", newPkgImportPath, s.Name.Name)
							fmt.Fprintf(&out, "type %s = %s.%s\n", s.Name.Name, newPkgAlias, s.Name.Name)
						}
					case *ast.ValueSpec:
						for _, name := range s.Names {
							if name.IsExported() {
								fmt.Fprintf(&out, "// Deprecated: Use %s.%s instead.\n", newPkgImportPath, name.Name)
								fmt.Fprintf(&out, "var %s = %s.%s\n", name.Name, newPkgAlias, name.Name)
							}
						}
					}
				}
			case *ast.FuncDecl:
				if d.Name.IsExported() {
					fmt.Fprintf(&out, "// Deprecated: Use %s.%s instead.\n", newPkgImportPath, d.Name.Name)
					fmt.Fprintf(&out, "var %s = %s.%s\n", d.Name.Name, newPkgAlias, d.Name.Name)
				}
			}
		}
	}

	// 3. Write the output to the specified file.
	if err := os.WriteFile(outputFile, out.Bytes(), 0644); err != nil {
		klog.Fatal(err)
	}
	fmt.Printf("Generated deprecation shim at %s\n", outputFile)
}
