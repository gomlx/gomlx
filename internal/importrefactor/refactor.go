package importrefactor

import (
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"

	"golang.org/x/tools/go/ast/astutil"
)

// RewriteRules defines the mappings for import paths and package names.
type RewriteRules struct {
	// ImportPathMap maps old import paths to new import paths.
	ImportPathMap map[string]string
	// PackageNameMap maps old package names to new package names.
	PackageNameMap map[string]string
}

// RefactorFile applies the rewrite rules to the given Go file.
// It returns true if the file was modified.
func RefactorFile(filename string, rules RewriteRules) (bool, error) {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, filename, nil, parser.ParseComments)
	if err != nil {
		return false, err
	}

	modified := false

	// 1. Rewrite import paths
	for oldPath, newPath := range rules.ImportPathMap {
		if astutil.RewriteImport(fset, f, oldPath, newPath) {
			modified = true
		}
	}

	// 2. Rename package references if needed
	if len(rules.PackageNameMap) > 0 {
		ast.Inspect(f, func(n ast.Node) bool {
			if sel, ok := n.(*ast.SelectorExpr); ok {
				if x, ok := sel.X.(*ast.Ident); ok {
					if newName, found := rules.PackageNameMap[x.Name]; found {
						// Simple check: only rename if it looks like a package reference.
						// In a more robust implementation, we'd check if x refers to an imported package.
						// For this conversion tool, we follow the pattern of the original shell script.
						x.Name = newName
						modified = true
					}
				}
			}
			return true
		})
	}

	if modified {
		out, err := os.Create(filename)
		if err != nil {
			return false, err
		}
		defer out.Close()

		err = format.Node(out, fset, f)
		if err != nil {
			return false, err
		}
	}

	return modified, nil
}
