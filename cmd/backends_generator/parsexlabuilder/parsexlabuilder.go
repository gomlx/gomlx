// Package parsexlabuilder parses the xlabuilder API to enumerate graph building functions, and the `op_types.txt`
// file to get a list of the supported ops.
//
// It will clone a temporary copy of gopjrt (github.com/gomlx/gopjrt) repository by default, or use the one under
// GOPJRT_SRC if it is set.
package parsexlabuilder

import (
	"bufio"
	"fmt"
	"github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types"
	"github.com/janpfeifer/must"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"os/exec"
	"path"
	"strings"
	"sync"
)

const GopjrtEnv = "GOPJRT_SRC"

var (
	// DefaultGopjrtSource is the directory where to clone the gopjrt (github.com/gomlx/gopjrt) source code.
	// If not an absolute directory, it will be prefixed by os.TempDir().
	DefaultGopjrtSource = "parsexlabuilder"

	// GopjrtSourcePath is the cached path used for the source code.
	GopjrtSourcePath = ""
)

// GetGopjrt returns GopjrtSourcePath if it is set, otherwise sets it to:
//
// 1. $GOPJRT_SRC, if it is set.
// 2. Download and return the source code into DefaultGopjrtSource.
func GetGopjrt() string {
	if GopjrtSourcePath != "" {
		return GopjrtSourcePath
	}

	var found bool
	GopjrtSourcePath, found = os.LookupEnv(GopjrtEnv)
	if found {
		return GopjrtSourcePath
	}

	// Create/find temporary repository directory.
	basePath := DefaultGopjrtSource
	if !path.IsAbs(basePath) {
		basePath = path.Join(os.TempDir(), basePath)
	}
	GopjrtSourcePath = path.Join(basePath, "gopjrt")
	fi, err := os.Stat(GopjrtSourcePath)
	if err != nil && !os.IsNotExist(err) {
		// Can't stat GopjrtSourcePath for some other reason.
		panic(err)
	}
	if err == nil && !fi.IsDir() {
		exceptions.Panicf("Gopjrt source path %q is not a directory!?", GopjrtSourcePath)
	}
	if err != nil {
		// Repository not downloaded yet, clone it:
		must.M(os.MkdirAll(basePath, 0755))
		cmd := exec.Command("git", "clone", "https://github.com/gomlx/gopjrt.git")
		cmd.Dir = basePath
		fmt.Printf("Downloading gopjrt under %s:\n\t%s\n", cmd.Dir, cmd)
		must.M(cmd.Run())

	} else {
		// Repository already cloned, just in case sync repository for latest updates.
		cmd := exec.Command("git", "pull")
		cmd.Dir = GopjrtSourcePath
		fmt.Printf("Sync'ing gopjrt in %s:\n\t%s\n", GopjrtSourcePath, cmd)
		must.M(cmd.Run())
	}
	return GopjrtSourcePath
}

const OpTypesFileName = "xlabuilder/op_types.txt"

// OpInfo is the information collected from the `op_types.txt` file.
type OpInfo struct {
	Name, Type string
}

// ReadOpsInfo reads Gopjrt op_types.txt file.
func ReadOpsInfo() []OpInfo {
	opInfoPath := path.Join(GetGopjrt(), OpTypesFileName)
	opsInfo := make([]OpInfo, 0, 200)
	f := must.M1(os.OpenFile(opInfoPath, os.O_RDONLY, os.ModePerm))
	scanner := bufio.NewScanner(f)
	lineNum := 0
	for scanner.Scan() {
		line := strings.TrimSpace(scanner.Text())
		lineNum++
		if line == "" {
			// Skip empty lines
			continue
		}
		if strings.HasPrefix(line, "//") || strings.HasPrefix(line, "#") {
			// Skip comments.
			continue
		}
		parts := strings.Split(line, ":")
		if len(parts) != 2 {
			exceptions.Panicf("Invalid op definition in %q:%d : %q", OpTypesFileName, lineNum, line)
		}
		opsInfo = append(opsInfo, OpInfo{Name: parts[0], Type: parts[1]})
	}
	must.M(scanner.Err())
	return opsInfo
}

// Parse returns the parse tree of the gopjrt/xlabuilder pacakge.
//
// Notice ast.Package is deprecated, but the go/types package it suggests as a replacement doesn't seem to do the same thing.
func Parse() (*NodeTextExtractor, *ast.Package) {
	xlaBuilderPath := path.Join(GetGopjrt(), "xlabuilder")
	fSet := token.NewFileSet()
	pkgs := must.M1(parser.ParseDir(fSet, xlaBuilderPath, nil, parser.ParseComments|parser.AllErrors))
	extractor := NewNodeTextExtractor(fSet)
	return extractor, pkgs["xlabuilder"]
}

// EnumerateFunctions calls callback for every function declaration in the package.
// Presumably to be used with the return value of Parse.
func EnumerateFunctions(pkg *ast.Package, callback func(funcDecl *ast.FuncDecl)) {
	for _, fileAst := range pkg.Files {
		for _, decl := range fileAst.Decls {
			funcDecl, ok := decl.(*ast.FuncDecl)
			if !ok {
				continue
			}
			callback(funcDecl)
		}
	}
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

var FunctionsBlackList = types.SetWith("Parameter", "ScalarZero", "ScalarOne", "PopulationCount")

// EnumerateStandardOpsFunctions calls callback for every "standard" op declaring function of the xlaBuilder package AST,
// that can be automatically converted to a backends.Backend API, and implemented in the xla.Backend.
func EnumerateStandardOpsFunctions(extractor *NodeTextExtractor, xlaBuilderPkg *ast.Package, callback func(funcDecl *ast.FuncDecl)) {
	EnumerateFunctions(xlaBuilderPkg, func(funcDecl *ast.FuncDecl) {
		if funcDecl.Recv != nil {
			// Skip methods.
			return
		}

		// We are looking for methods that have 2 outputs: (*Op, error)
		if funcDecl.Type.Results.NumFields() != 2 {
			return
		}
		if extractor.Get(funcDecl.Type.Results.List[0].Type) != "*Op" ||
			extractor.Get(funcDecl.Type.Results.List[1].Type) != "error" {
			return
		}
		if !funcDecl.Name.IsExported() {
			return
		}

		// Skip functions that take a sub-computation as a parameter.
		for _, param := range funcDecl.Type.Params.List {
			typeName := extractor.Get(param.Type)
			if typeName == "*XlaComputation" || typeName == "*Literal" {
				//fmt.Printf("*** dropping %q because it takes a computation as input\n", funcDecl.Name.Name)
				return
			}
		}

		// Skip tuple-functions and black-listed functions.
		if strings.Index(funcDecl.Name.Name, "Tuple") != -1 || FunctionsBlackList.Has(funcDecl.Name.Name) {
			return
		}

		callback(funcDecl)
	})
}
