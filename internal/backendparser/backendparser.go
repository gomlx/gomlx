package backendparser

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"slices"

	"github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/internal/must"
)

// Method represents a single method from the backends.Builder interface
// with all its signature information as strings.
type Method struct {
	// Name is the method name
	Name string
	// Comment is the method documentation comment
	Comments []string
	// Parameters of the method.
	Parameters []NameAndType
	// Outputs of the method.
	// Outputs names may contain all empty strings if they are not defined.
	Outputs []NameAndType
}

type NameAndType struct {
	Name, Type string
}

// ParseBuilder returns all methods defined in the backends.Builder interface,
// including those from embedded interfaces like backends.StandardOps.
func ParseBuilder() ([]Method, error) {
	fileSet := token.NewFileSet()
	var methods []Method

	root, err := findModuleRoot()
	if err != nil {
		return nil, err
	}

	// Parse both files
	builderFile, err := parser.ParseFile(fileSet, filepath.Join(root, "backends", "builder.go"),
		nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	standardOpsFile, err := parser.ParseFile(fileSet, filepath.Join(root, "backends", "standard_ops.go"),
		nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}
	collectiveOpsFile, err := parser.ParseFile(fileSet, filepath.Join(root, "backends", "collectiveops.go"),
		nil, parser.ParseComments)
	if err != nil {
		return nil, err
	}

	// File contents cache
	fileCache := make(map[string][]byte)
	getFileContent := func(fileName string) []byte {
		fileContent, ok := fileCache[fileName]
		if !ok {
			// File not in cache, read it
			fileContent = must.M1(os.ReadFile(fileName))
			fileCache[fileName] = fileContent
		}
		return fileContent
	}

	// Extract the text from a node
	getText := func(node ast.Node) string {
		pos := fileSet.Position(node.Pos())
		fileName := pos.Filename
		fileContent := getFileContent(fileName)

		// Extract text from the cached file content
		endOffset := fileSet.Position(node.End()).Offset
		if endOffset > len(fileContent) {
			exceptions.Panicf("end offset out of bounds for file %s", fileName)
		}
		return string(fileContent[pos.Offset:endOffset])
	}

	// Helper to extract methods from interface declarations
	includeInterfaces := []string{"Builder", "StandardOps", "CollectiveOps"}
	extractMethods := func(file *ast.File) {
		ast.Inspect(file, func(n ast.Node) bool {
			if typeSpec, ok := n.(*ast.TypeSpec); ok {
				if interfaceType, ok := typeSpec.Type.(*ast.InterfaceType); ok {
					if slices.Index(includeInterfaces, typeSpec.Name.Name) == -1 {
						return true
					}
					for _, method := range interfaceType.Methods.List {
						// Extract method information
						funcType, ok := method.Type.(*ast.FuncType)
						if !ok {
							continue
						}

						m := Method{
							Name: method.Names[0].Name,
						}

						// Get method comment if any
						if method.Doc != nil {
							m.Comments = make([]string, 0, len(method.Doc.List))
							for _, comment := range method.Doc.List {
								m.Comments = append(m.Comments, comment.Text)
							}
						}

						// Get parameters
						if funcType.Params != nil {
							for _, param := range funcType.Params.List {
								paramType := getText(param.Type)
								for _, name := range param.Names {
									param := NameAndType{Name: name.Name, Type: paramType}
									m.Parameters = append(m.Parameters, param)
								}
							}
						}

						// Get outputs
						if funcType.Results != nil {
							for _, result := range funcType.Results.List {
								resultType := getText(result.Type)
								if len(result.Names) == 0 {
									m.Outputs = append(m.Outputs, NameAndType{Type: resultType})
								} else {
									for _, name := range result.Names {
										param := NameAndType{Name: name.Name, Type: resultType}
										m.Outputs = append(m.Outputs, param)
									}
								}
							}
						}

						methods = append(methods, m)
					}
				}
			}
			return true
		})
	}

	extractMethods(builderFile)
	extractMethods(standardOpsFile)
	extractMethods(collectiveOpsFile)

	return methods, nil
}

// findModuleRoot returns the absolute path to the module root directory
// by walking up the directory tree looking for the go.mod file.
func findModuleRoot() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir, nil
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("could not find module root (no go.mod file found)")
		}
		dir = parent
	}
}
