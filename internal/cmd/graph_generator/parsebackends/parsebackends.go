// Package parsebackends parses the backends.Builder API to enumerate graph building methods.
package parsebackends

import (
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path"
	"sync"

	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/janpfeifer/must"
)

// FindRepositoryRoot returns the first directory, from the current, with the file `go.mod`, which is assumed to be repository root.
func FindRepositoryRoot() string {
	pwd := must.M1(os.Getwd())
	for {
		_, err := os.Stat(path.Join(pwd, "go.mod"))
		if err == nil {
			// Found it.
			return pwd
		}
		if !os.IsNotExist(err) {
			panic(err)
		}
		// Move up.
		pwd = path.Dir(pwd)
	}
}

// BackendsDir returns the path to the "backends" package.
func BackendsDir() string {
	return path.Join(FindRepositoryRoot(), "backends")
}

// Parse returns the parse tree of the github.com/gomlx/gomlx/backends pacakge.
//
// Notice ast.Package is deprecated, but the go/types package it suggests as a replacement doesn't seem to do the same thing.
func Parse() (*NodeTextExtractor, *ast.Package) {
	fSet := token.NewFileSet()
	pkgs := must.M1(parser.ParseDir(fSet, BackendsDir(), nil, parser.ParseComments|parser.AllErrors))
	extractor := NewNodeTextExtractor(fSet)
	return extractor, pkgs["backends"]
}

var MethodsBlackList = types.SetWith("Compile", "OpShape", "Name")

type FuncInfo struct {
	Type     *ast.FuncType
	Comments []string
}

func ParseBuilder() (extractor *NodeTextExtractor, funcs map[string]FuncInfo) {
	var pkg *ast.Package
	extractor, pkg = Parse()
	funcs = extractInterfaceMethods(pkg, "Builder")
	for name := range MethodsBlackList {
		delete(funcs, name)
	}
	return
}

func extractInterfaceMethods(pkg *ast.Package, interfaceName string) map[string]FuncInfo {
	methods := make(map[string]FuncInfo)
	for _, file := range pkg.Files {
		for _, decl := range file.Decls {
			// Check if the declaration is a GenDecl (e.g., type, const, var)
			genDecl, ok := decl.(*ast.GenDecl)
			if !ok {
				continue
			}

			// Look for a type specification within the GenDecl
			for _, spec := range genDecl.Specs {
				typeSpec, ok := spec.(*ast.TypeSpec)
				if !ok {
					continue
				}

				// Check if the type is an interface with the target name
				interfaceType, ok := typeSpec.Type.(*ast.InterfaceType)
				if !ok {
					continue
				}
				if typeSpec.Name.Name != interfaceName {
					continue
				}

				// Extract method names from the interface
				for _, field := range interfaceType.Methods.List {
					comments := xslices.Map(field.Doc.List, func(c *ast.Comment) string { return c.Text })
					switch fieldType := field.Type.(type) {
					case *ast.FuncType:
						for _, name := range field.Names {
							methods[name.Name] = FuncInfo{Type: fieldType, Comments: comments}
						}
					case *ast.Ident:
						if len(field.Names) == 0 {
							// Assume this is another interface that we collect as well.
							// TODO: check for infinite loops among interfaces.
							importedMethods := extractInterfaceMethods(pkg, fieldType.Name)
							for name, funcType := range importedMethods {
								methods[name] = funcType
							}
						}
					default:
						fmt.Printf("Unknown field %q -> %T\n", field.Names, field.Type)
					}
				}
			}
		}
	}
	return methods
}

type NodeTextExtractor struct {
	fSet    *token.FileSet
	cache   map[string][]byte
	cacheMu sync.Mutex
}

func NewNodeTextExtractor(fset *token.FileSet) *NodeTextExtractor {
	return &NodeTextExtractor{
		fSet:  fset,
		cache: make(map[string][]byte),
	}
}

func (e *NodeTextExtractor) Get(node ast.Node) string {
	pos := e.fSet.Position(node.Pos())
	fileName := pos.Filename

	// Check cache
	e.cacheMu.Lock()
	fileContent, ok := e.cache[fileName]
	e.cacheMu.Unlock()

	if !ok {
		// File not in cache, read it
		fileContent = must.M1(os.ReadFile(fileName))

		// Store in cache
		e.cacheMu.Lock()
		e.cache[fileName] = fileContent
		e.cacheMu.Unlock()
	}

	// Extract text from the cached file content
	endOffset := e.fSet.Position(node.End()).Offset
	if endOffset > len(fileContent) {
		exceptions.Panicf("end offset out of bounds for file %s", fileName)
	}
	return string(fileContent[pos.Offset:endOffset])
}
