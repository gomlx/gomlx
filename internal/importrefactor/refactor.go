package importrefactor

import (
	"go/ast"
	"go/format"
	"go/parser"
	"go/token"
	"os"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
)

// RewriteRules defines the mappings for import paths, package names, types, and variables.
type RewriteRules struct {
	// ImportPathMap maps old import paths to new import paths.
	ImportPathMap map[string]string
	// PackageNameMap maps old package names to new package names.
	PackageNameMap map[string]string
	// TypeNameMap maps old type names to new type names (e.g., "model.Context" -> "Scope").
	// The key is the full identifier or just the name if it's in the same package.
	TypeNameMap map[string]string
	// FunctionNameMap maps old function names to new function names.
	// The key is the full identifier or just the name if it's in the same package.
	FunctionNameMap map[string]string
	// VariableNameMap maps old variable names to new variable names,
	// optionally filtered by their type name.
	VariableNameMap map[string]VariableRename
	// MethodNameMap maps old method names to new method names.
	// The key is the type-qualified method name (e.g., "model.Scope.VariableWithValueGraph").
	MethodNameMap map[string]string
}

type VariableRename struct {
	NewName  string
	TypeName string // Optional: only rename if the type matches.
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

	// 2. Walk AST for other renames
	ast.Inspect(f, func(n ast.Node) bool {
		switch node := n.(type) {
		case *ast.SelectorExpr:
			// Method rename
			// We use a heuristic: match the method name part since we don't have full type info.
			for oldMethod, newMethodName := range rules.MethodNameMap {
				parts := strings.Split(oldMethod, ".")
				methodName := parts[len(parts)-1]
				if node.Sel.Name == methodName {
					node.Sel.Name = newMethodName
					modified = true
				}
			}

			// Handle package name renames and type name renames (e.g., model.Context)
			if x, ok := node.X.(*ast.Ident); ok {
				// Package rename
				if newPkgName, found := rules.PackageNameMap[x.Name]; found {
					x.Name = newPkgName
					modified = true
				}

				// Type rename (e.g., model.Context -> model.Scope)
				fullID := x.Name + "." + node.Sel.Name
				if newTypeName, found := rules.TypeNameMap[fullID]; found {
					node.Sel.Name = newTypeName
					modified = true
				}

				// Function rename
				if newFuncName, found := rules.FunctionNameMap[fullID]; found {
					node.Sel.Name = newFuncName
					modified = true
				}
			}

		case *ast.Field: // Function parameters, struct fields
			for _, ident := range node.Names {
				if rename, found := rules.VariableNameMap[ident.Name]; found {
					if rename.TypeName == "" || isMatchingType(node.Type, rename.TypeName) {
						ident.Name = rename.NewName
						modified = true
					}
				}
			}

		case *ast.ValueSpec: // Variable declarations
			for _, ident := range node.Names {
				if rename, found := rules.VariableNameMap[ident.Name]; found {
					if rename.TypeName == "" || isMatchingType(node.Type, rename.TypeName) {
						ident.Name = rename.NewName
						modified = true
					}
				}
			}

		case *ast.AssignStmt: // Short variable declarations (x := ...)
			if node.Tok == token.DEFINE {
				for _, lhs := range node.Lhs {
					if ident, ok := lhs.(*ast.Ident); ok {
						if rename, found := rules.VariableNameMap[ident.Name]; found {
							// For short declarations, we don't have explicit type info in AST easily.
							// But the request is to rename 'ctx' to 'scope' if it's being used as *model.Context.
							// Without full type checking, we'll rename 'ctx' if it's in the VariableNameMap
							// and optionally check if the right hand side is a call to something returning model.Context.
							// For simplicity and following "worth the risk", we'll rename it if TypeName matches "" or 
							// we can heuristically detect it.
							if rename.TypeName == "" {
								ident.Name = rename.NewName
								modified = true
							}
						}
					}
				}
			}

		case *ast.Ident:
			// Rename usage of variables renamed above.
			// This is tricky without a scope analyzer. 
			// However, for this specific task, we can rename 'ctx' to 'scope' globally in the file
			// if it's not a field/selector.
			if rename, found := rules.VariableNameMap[node.Name]; found {
				// Simple heuristic: if we are renaming ctx to scope, we do it for all idents named ctx.
				if rename.TypeName == "" || rename.TypeName == "model.Context" || rename.TypeName == "model.Scope" {
					node.Name = rename.NewName
					modified = true
				}
			}

			// Rename bare function calls / references if they are in the same package
			if newFuncName, found := rules.FunctionNameMap[node.Name]; found {
				node.Name = newFuncName
				modified = true
			}
		}
		return true
	})

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

// isMatchingType checks if the type node matches the given type name string.
func isMatchingType(expr ast.Expr, typeName string) bool {
	if expr == nil {
		return false
	}
	
	// Handle pointers: *model.Context
	if star, ok := expr.(*ast.StarExpr); ok {
		return isMatchingType(star.X, typeName)
	}

	// Handle selectors: model.Context
	if sel, ok := expr.(*ast.SelectorExpr); ok {
		if x, ok := sel.X.(*ast.Ident); ok {
			return x.Name+"."+sel.Sel.Name == typeName
		}
	}
	
	// Handle simple idents: Context (if already in package model)
	if ident, ok := expr.(*ast.Ident); ok {
		return ident.Name == typeName || strings.HasSuffix(typeName, "."+ident.Name)
	}

	return false
}
