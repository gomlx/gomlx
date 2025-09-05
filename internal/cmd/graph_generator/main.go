package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path"
	"slices"
	"strings"
	"text/template"

	"github.com/gomlx/gomlx/internal/backendparser"
	"github.com/gomlx/gomlx/types"
	"github.com/janpfeifer/must"
)

func main() {
	flag.Parse()
	fmt.Println("graph_generator:")
	methods := buildMethodInfo()
	GenerateBackendOps(methods)
}

var (
	// methodsNotExported list methods that will have a non-exported "backend<Method>" function written, that can
	// be used by the public graphs implementation.
	methodsNotExported = types.SetWith(
		"ArgMinMax", "Broadcast", "BroadcastInDim",
		"BatchNormForInference", "BatchNormForTraining", "BatchNormGradient",
		"Concatenate", "ConvertDType", "ConvGeneral", "DotGeneral", "FFT", "Gather", "Iota",
		"ReduceMax", "ReduceMin", "ReduceProduct", "ReduceSum", "ReduceWindow",

		// Reduce of logical/bitwise operators:
		"ReduceLogicalAnd", "ReduceLogicalOr", "ReduceLogicalXor",
		"ReduceBitwiseAnd", "ReduceBitwiseOr", "ReduceBitwiseXor",

		"Reshape", "Reverse", "RngBitGenerator",
		"ScatterSum", "ScatterMax", "ScatterMin",
		"ScatterAdd", // Deprecated
		"SelectAndScatterSum", "SelectAndScatterMax", "SelectAndScatterMin",
		"Sign",
		"ShiftLeft", "ShiftRightArithmetic", "ShiftRightLogical",
		"Slice",
		"Transpose", "Where")

	// methodsNotGenerated but for which there is still a NodeType.
	methodsNotGenerated = types.SetWith(
		"Constant", "Parameter")

	// methodsExcluded from generating and even from having a NodeType.
	// These are utility methods, not part of building a graph.
	methodsExcluded = types.SetWith(
		"Name", "Compile", "OpShape")

	// methodsNoGradient will add a stop gradient to the node.
	methodsNoGradient = types.SetWith(
		"And", "Or", "Xor", "LogicalNot",
		"Equal", "NotEqual", "GreaterOrEqual", "GreaterThan", "LessOrEqual", "LessThan",
		"EqualTotalOrder", "NotEqualTotalOrder", "GreaterOrEqualTotalOrder", "GreaterThanTotalOrder", "LessOrEqualTotalOrder", "LessThanTotalOrder")
)

func buildMethodInfo() (methods []*MethodInfo) {
	rawMethods := must.M1(backendparser.ParseBuilder())
	for _, raw := range rawMethods {
		name := raw.Name
		if methodsExcluded.Has(name) {
			continue
		}
		mi := &MethodInfo{
			BackendName:  name,
			GraphName:    name,
			Exported:     !methodsNotExported.Has(name),
			Excluded:     methodsNotGenerated.Has(name),
			Comments:     raw.Comments,
			StopGradient: methodsNoGradient.Has(name),
		}
		methods = append(methods, mi)
		if !mi.Exported {
			mi.GraphName = "backend" + name
		}
		for _, param := range raw.Parameters {
			pi := &ParameterInfo{
				Name:        param.Name,
				BackendType: param.Type,
			}
			mi.Inputs = append(mi.Inputs, pi)
			switch pi.BackendType {
			case "Op":
				pi.BackendType = "backends.Op"
				pi.GraphType = "*Node"
				pi.ConvertStatement = fmt.Sprintf("%s.outputOps[0]", param.Name)
				mi.OpInputs = append(mi.OpInputs, param.Name)
				pi.Format = "[#%d]"
				pi.FormatValue = fmt.Sprintf("ni.%s.Id()", param.Name)
			case "...Op":
				pi.BackendType = "...backends.Op"
				pi.GraphType = "...*Node"
				mi.OpInputSlices = append(mi.OpInputSlices, param.Name)
				pi.NodeInputType = "[]*Node"
				pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
				pi.ConvertStatement = fmt.Sprintf("xslices.Map(%s, func(node *Node) backends.Op { return node.outputOps[0] })...", param.Name)
				pi.Format = "[#%s]"
				pi.FormatValue = fmt.Sprintf(
					`strings.Join(xslices.Map(ni.%s, func (node *Node) string { return fmt.Sprintf("#%%d", node.Id()) }), ", ")`,
					param.Name)
			case "[]Op":
				pi.BackendType = "[]backends.Op"
				pi.GraphType = "[]*Node"
				mi.OpInputSlices = append(mi.OpInputSlices, param.Name)
				pi.NodeInputType = "[]*Node"
				pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
				pi.ConvertStatement = fmt.Sprintf("xslices.Map(%s, func(node *Node) backends.Op { return node.outputOps[0] })", param.Name)
				pi.Format = "[#%s]"
				pi.FormatValue = fmt.Sprintf(
					`strings.Join(xslices.Map(ni.%s, func (node *Node) string { return fmt.Sprintf("#%%d", node.Id()) }), ", ")`,
					param.Name)
			case "ConvolveAxesConfig":
				pi.BackendType = "backends." + pi.BackendType
				pi.CopyStatement = fmt.Sprintf("%s.Clone()", param.Name)
				pi.Format = "%+v"
			case "PadAxis":
				pi.BackendType = "backends." + pi.BackendType
				pi.Format = "%+v"
			case "...PadAxis":
				pi.BackendType = "...backends." + pi.BackendType[3:]
				pi.NodeInputType = "[]" + pi.BackendType[3:]
				pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
				pi.ConvertStatement = fmt.Sprintf("inputs.%s...", param.Name)
				pi.Format = "%+v"
			case "FFTType":
				pi.BackendType = "backends." + pi.BackendType
				pi.Format = "%s"
			default:
				if strings.HasPrefix(pi.BackendType, "...") {
					pi.NodeInputType = "[]" + pi.BackendType[3:]
					pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
					pi.ConvertStatement = fmt.Sprintf("inputs.%s...", param.Name)
				} else if strings.HasPrefix(pi.GraphType, "[]") {
					pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
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
			mi.HasGraph = len(mi.OpInputSlices) == 0 && len(mi.OpInputs) == 0
		}
		for _, output := range raw.Outputs[:len(raw.Outputs)-1] { // Skip the error.
			mi.OutputNames = append(mi.OutputNames, output.Name)
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
	OpInputSlices          []string
	Inputs                 []*ParameterInfo
	Exported, Excluded     bool
	Comments               []string
	StopGradient           bool

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
{{- range .}}
	NodeType{{.BackendName}}
{{- end}}
)

{{- range .}}

{{- if not .Excluded}}

// nodeInputs{{.BackendName}} holds the inputs used for the call to backends.{{.BackendName}}.
type nodeInputs{{.BackendName}} struct {
	{{- range .Inputs}}
	{{.Name}} {{.NodeInputType}}
	{{- end}}
}

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

{{- if not .Exported}}
// {{.GraphName}} is a Graph wrapper for the backend.Builder.{{.BackendName}} method.
{{- else}}
{{- range .Comments}}
{{.}}
{{- end}}
{{- end}}
func {{/*

Inputs:  */}}{{.GraphName}}({{if .HasGraph}}g *Graph, {{end}}{{range .Inputs}}{{.Name}} {{.GraphType}}, {{end}}) ({{/*

Outputs: */}}{{if not .HasMultipleOutputs}}node *Node{{/*
*/}} {{else}} {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}{{$name}}{{end}} *Node{{/*
*/}}{{end}}) {{/*

Body: */}}{
{{- if .HasGraph}}
	g.AssertBuilding()
{{- else}}
	inputNodes := []*Node{ {{range .OpInputs}}{{.}}, {{end}} }
{{- range .OpInputSlices}}
	inputNodes = append(inputNodes, {{.}}...)
{{- end}}
	g := validateBuildingGraphFromInputs(inputNodes...)
{{- end}}
	inputs := &nodeInputs{{.BackendName}}{
{{- range .Inputs}}
		{{.Name}}: {{.CopyStatement}},		
{{- end}}
	}
{{- /*

Convert result(s) to node(s):

*/}}
{{- if not .HasMultipleOutputs}}
	result, err := g.builder.{{.BackendName}}({{range .Inputs}}{{.ConvertStatement}}, {{end}})
	if err != nil {
		panic(err)
	}
	node = &Node{
		outputOps: []backends.Op{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
{{- else}}
{{- /*

Version with multiple outputs:

*/}}
{{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}v{{$ii}}{{end}}, err := g.builder.{{.BackendName}}({{range .Inputs}}{{.ConvertStatement}}, {{end}})
	if err != nil {
		panic(err)
	}
	node := &Node{
		outputOps: []backends.Op{ {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}v{{$ii}}{{end}} },
		outputShapes: []shapes.Shape{ {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}mustNoError(g.builder.OpShape(v{{$ii}})){{end}} },
{{- end}}
		graph: g,
		inputs: inputs,
{{- if not .HasGraph}}
		inputNodes: inputNodes,
{{- end}}{{/*
*/}}
{{- if .StopGradient}}
		stopGradient: true,
{{- end}}
	}
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
	cmd := exec.Command("go", "fmt", fileName)
	fmt.Printf("\t%s\n", cmd)
	must.M(cmd.Run())
	cmd = exec.Command("enumer", "-type=NodeType", "-trimprefix=NodeType", "-yaml", "-json", "-text", "-values", fileName)
	fmt.Printf("\t%s\n", cmd)
	must.M(cmd.Run())
	fmt.Printf("✅ Successfully generated %s\n", path.Join(must.M1(os.Getwd()), fileName))
}
