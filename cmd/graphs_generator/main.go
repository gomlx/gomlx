package main

import (
	"flag"
	"fmt"
	"github.com/gomlx/gomlx/cmd/graphs_generator/parsebackends"
	"github.com/gomlx/gomlx/types"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/janpfeifer/must"
	"go/ast"
	"os"
	"os/exec"
	"slices"
	"strings"
	"text/template"
)

func main() {
	flag.Parse()
	fmt.Println("graphs_generator:")
	methods := buildMethodInfo()
	GenerateBackendOps(methods)
}

var (
	// methodsNotExported list methods that will have a non-exported "backend<Method>" function written, that can
	// be used by the public graphs implementation.
	methodsNotExported = types.SetWith(
		"ArgMinMax", "Broadcast", "BroadcastInDim",
		"BatchNormForInference", "BatchNormForTraining", "BatchNormGradient",
		"Concatenate", "ConvGeneralDilated", "DotGeneral", "FFT", "Gather", "Iota",
		"ReduceMax", "ReduceMin", "ReduceProduct", "ReduceSum", "ReduceWindow",
		"Reshape", "Reverse", "RngBitGenerator",
		"ScatterAdd", "SelectAndScatterSum", "SelectAndScatterMax", "SelectAndScatterMin",
		"Sign", "Slice",
		"Transpose", "Where")

	// methodsToExclude from writing, but the corresponding will be written and maintained manually.
	methodsToExclude = types.SetWith("Constant", "Parameter")
)

func buildMethodInfo() (methods []*MethodInfo) {
	extractor, funcs := parsebackends.ParseBuilder()
	for name, funcInfo := range funcs {
		mi := &MethodInfo{
			BackendName: name,
			GraphName:   name,
			Exported:    !methodsNotExported.Has(name),
			Excluded:    methodsToExclude.Has(name),
			Comments:    funcInfo.Comments,
		}
		methods = append(methods, mi)
		if !mi.Exported {
			mi.GraphName = "backend" + name
		}
		for _, param := range funcInfo.Type.Params.List {
			paramNames := xslices.Map(param.Names, func(ident *ast.Ident) string { return ident.Name })
			for _, paramName := range paramNames {
				pi := &ParameterInfo{
					Name:        paramName,
					BackendType: extractor.Get(param.Type),
				}
				mi.Inputs = append(mi.Inputs, pi)
				switch pi.BackendType {
				case "Op":
					pi.BackendType = "backends.Op"
					pi.GraphType = "*Node"
					pi.ConvertStatement = fmt.Sprintf("%s.outputOps[0]", paramName)
					mi.OpInputs = append(mi.OpInputs, paramName)
					pi.Format = "[#%d]"
					pi.FormatValue = fmt.Sprintf("ni.%s.Id()", paramName)
				case "...Op":
					pi.BackendType = "...backends.Op"
					pi.GraphType = "...*Node"
					mi.OpInputsList = paramName
					pi.NodeInputType = "[]*Node"
					pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", paramName)
					pi.ConvertStatement = fmt.Sprintf("xslices.Map(%s, func(node *Node) backends.Op { return node.outputOps[0] })...", paramName)
					pi.Format = "[#%s]"
					pi.FormatValue = fmt.Sprintf(
						`strings.Join(xslices.Map(ni.%s, func (node *Node) string { return fmt.Sprintf("#%%d", node.Id()) }), ", ")`,
						paramName)
				case "ConvolveAxesConfig":
					pi.BackendType = "backends." + pi.BackendType
					pi.CopyStatement = fmt.Sprintf("%s.Clone()", paramName)
					pi.Format = "%+v"
				case "PadAxis":
					pi.BackendType = "backends." + pi.BackendType
					pi.Format = "%+v"
				case "...PadAxis":
					pi.BackendType = "...backends." + pi.BackendType[3:]
					pi.NodeInputType = "[]" + pi.BackendType[3:]
					pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", paramName)
					pi.ConvertStatement = fmt.Sprintf("inputs.%s...", paramName)
					pi.Format = "%+v"
				case "FFTType":
					pi.BackendType = "backends." + pi.BackendType
					pi.Format = "%s"
				default:
					if strings.HasPrefix(pi.BackendType, "...") {
						pi.NodeInputType = "[]" + pi.BackendType[3:]
						pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", paramName)
						pi.ConvertStatement = fmt.Sprintf("inputs.%s...", paramName)
					} else if strings.HasPrefix(pi.GraphType, "[]") {
						pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", paramName)
					}
				}
				if pi.GraphType == "" {
					pi.GraphType = pi.BackendType
				}
				if pi.NodeInputType == "" {
					pi.NodeInputType = pi.GraphType
				}
				if pi.CopyStatement == "" {
					pi.CopyStatement = pi.Name
				}
				if pi.ConvertStatement == "" {
					pi.ConvertStatement = "inputs." + pi.Name
				}
				if pi.Format == "" {
					pi.Format = "%v"
				}
				if pi.FormatValue == "" {
					pi.FormatValue = "ni." + pi.Name
				}
			}
			mi.HasGraph = mi.OpInputsList == "" && len(mi.OpInputs) == 0

		}
		for _, field := range funcInfo.Type.Results.List {
			for _, nameIdent := range field.Names {
				// Save the names of the outputs: we assume all outputs are of type Op (to be converted to *Node in graphs package).
				mi.OutputNames = append(mi.OutputNames, nameIdent.Name)
			}
		}
		if len(mi.OutputNames) > 1 {
			mi.HasMultipleOutputs = true
		}
	}
	return methods
}

type MethodInfo struct {
	BackendName, GraphName string
	HasGraph               bool
	OpInputs               []string
	OpInputsList           string
	Inputs                 []*ParameterInfo
	Exported, Excluded     bool
	Comments               []string

	HasMultipleOutputs bool
	OutputNames        []string
}

// ParameterInfo represents one parameter only.
type ParameterInfo struct {
	Name                                  string
	BackendType, GraphType, NodeInputType string
	CopyStatement, ConvertStatement       string
	Format, FormatValue                   string
}

const (
	backendsOpsFile = "gen_backend_ops.go"
)

var (
	backendOpsTemplate = template.Must(template.New(backendsOpsFile).Parse(`
/***** File generated by ./cmd/graphs_codegen, based on backends.Builder interface. Don't edit it directly. *****/

package graph

import (
	"fmt"
	"slices"
	"strings"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/xslices"
	"github.com/gomlx/gopjrt/dtypes"
)

type NodeType int

const (
	NodeTypeInvalid NodeType = iota
	NodeTypeSplitNode
{{range .}}	NodeType{{.BackendName}}
{{end}})

{{range .}}{{if not .Excluded}}// nodeInputs{{.BackendName}} holds the inputs used for the call to backends.{{.BackendName}}.
type nodeInputs{{.BackendName}} struct {
{{range .Inputs}}	{{.Name}} {{.NodeInputType}}
{{end}}}

// Type implements the interface NodeInputs.
func (ni *nodeInputs{{.BackendName}}) Type() NodeType {
	return NodeType{{.BackendName}}
}

// String implements the interface NodeInputs.
func (ni *nodeInputs{{.BackendName}}) String() string {
	return fmt.Sprintf("%s({{range $index, $input := .Inputs}}{{if $index}}, {{end}}{{$input.Name}}={{$input.Format}}{{end}})", 
		ni.Type(),
{{range .Inputs}}		{{.FormatValue}},
{{end}}	)
}

{{if not .Exported}}// {{.GraphName}} is a Graph wrapper for the backend.Builder.{{.BackendName}} method.
{{else}}{{range .Comments}}{{.}}
{{end}}{{end}}func {{/*

Inputs:  */}}{{.GraphName}}({{if .HasGraph}}g *Graph, {{end}}{{range .Inputs}}{{.Name}} {{.GraphType}}, {{end}}) ({{/*

Outputs: */}}{{if not .HasMultipleOutputs}}node *Node{{/*
*/}} {{else}} {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}{{$name}}{{end}} *Node{{/*
*/}}{{end}}) {{/*

Body: */}}{
{{if .HasGraph}}	g.AssertBuilding()
{{else}}{{if ne .OpInputsList ""}}	g := validateBuildingGraphFromInputs({{.OpInputsList}}...)
{{else}}	g := validateBuildingGraphFromInputs({{range .OpInputs}}{{.}}, {{end}})
{{end}}{{end}}	inputs := &nodeInputs{{.BackendName}}{
{{range .Inputs}}		{{.Name}}: {{.CopyStatement}},		
{{end}}	}
{{if not .HasGraph}}	{{if eq .OpInputsList ""}}inputNodes := []*Node{ {{range .OpInputs}}{{.}}, {{end}} } 
{{else}}	inputNodes := {{.OpInputsList}}
{{end}}{{end}}{{/*

Convert result(s) to node(s):

*/}}{{if not .HasMultipleOutputs}}	result := g.builder.{{.BackendName}}({{range .Inputs}}{{.ConvertStatement}}, {{end}})
	node = &Node{
		outputOps: []backends.Op{result},
		outputShapes: []shapes.Shape{g.builder.OpShape(result)},
{{else}}{{/*

Version with multiple outputs:

*/}}{{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}v{{$ii}}{{end}} := g.builder.{{.BackendName}}({{range .Inputs}}{{.ConvertStatement}}, {{end}})
	node := &Node{
		outputOps: []backends.Op{ {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}v{{$ii}}{{end}} },
		outputShapes: []shapes.Shape{ {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}g.builder.OpShape(v{{$ii}}){{end}} },
{{end}}		graph: g,
		inputs: inputs,
{{if not .HasGraph}}		inputNodes: inputNodes,
{{end}}	}
	g.registerNode(node)
{{/*

If multiple-outputs, split node into each separate one:

*/}}{{if .HasMultipleOutputs}}	splitNodes := splitNode(node)
	{{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}{{$name}}{{end}} = {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}splitNodes[{{$ii}}]{{end}}
{{end}}	return
}
{{end}}{{end}}
`))
)

// GenerateBackendOps generates the list of NodeType and the default implementation of ops to all backends.Builder
// interface methods and corresponding NodeInputs* struct.
func GenerateBackendOps(methods []*MethodInfo) {
	// Sort by backend method name:
	slices.SortFunc(methods, func(a, b *MethodInfo) int { return strings.Compare(a.BackendName, b.BackendName) })

	fileName := backendsOpsFile
	f := must.M1(os.Create(fileName))
	must.M(backendOpsTemplate.Execute(f, methods))
	cmd := exec.Command("gofmt", "-w", fileName)
	fmt.Printf("\t%s\n", cmd)
	must.M(cmd.Run())
	fmt.Printf("\tGenerated %q based on backends.Builder interface\n", fileName)

	cmd = exec.Command("stringer", "-type", "NodeType", "-trimprefix", "NodeType", fileName)
	fmt.Printf("\t%s\n", cmd)
	must.M(cmd.Run())

}
