// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

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

	"github.com/gomlx/compute/support/backendparser"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/support/exceptions"
	"github.com/gomlx/gomlx/support/sets"
	"k8s.io/klog/v2"
)

var flagPublic = flag.Bool("public", false, "Generate thin public wrappers in core/graph instead of internal/graphimpl.")

func main() {
	klog.InitFlags(nil)
	flag.Parse()
	klog.V(1).Info("graph_generator:")
	methods := buildMethodInfo()
	if *flagPublic {
		GeneratePublicOps(methods)
		return
	}
	GenerateBackendOps(methods)
}

var (
	// methodsNotExported list methods that will have a non-exported "backend<Method>" function written, that can
	// be used by the public graphs implementation.
	methodsNotExported = sets.MakeWith(
		"AllReduce", "ArgMinMax", "Broadcast", "BroadcastInDim",
		"BatchNormForInference", "BatchNormForTraining", "BatchNormGradient",
		"Ceil", "Concatenate", "ConvertDType", "ConvGeneral", "CumMax", "CumSum", "DotGeneral",
		"FFT", "Floor", "Gather", "Iota", "OptimizationBarrier",
		"ReduceMax", "ReduceMin", "ReduceProduct", "ReduceSum", "ReduceWindow",

		// Reduce of logical/bitwise operators:
		"ReduceLogicalAnd", "ReduceLogicalOr", "ReduceLogicalXor",
		"ReduceBitwiseAnd", "ReduceBitwiseOr", "ReduceBitwiseXor",

		"Reshape", "Reverse", "RNGBitGenerator",
		"ScatterSum", "ScatterMax", "ScatterMin",
		"ScatterAdd", // Deprecated
		"SelectAndScatterSum", "SelectAndScatterMax", "SelectAndScatterMin",
		"Sign",
		"ShiftLeft", "ShiftRightArithmetic", "ShiftRightLogical",
		"Slice",
		"Transpose", "Where",

		// Fused ops: exported wrappers with "Internal:" comments are hand-written in ops_fused.go.
		"FusedActivation", "FusedActivationVJP", "FusedDense", "FusedDenseVJP",
		"FusedLayerNorm", "FusedLayerNormVJP", "FusedSoftmax",
		"FusedScaledDotProductAttention", "FusedScaledDotProductAttentionVJP",
		"FusedAttentionQKVProjection", "FusedTopK",
		"FusedQuantizedDense", "FusedQuantizedDenseVJP", "QuantizedGather",

		// Dynamic shape methods (see ops_dynamic.go):
		"DynamicDimensionSize",
		"DynamicReshape",
		"DynamicBroadcastInDim",
		"DynamicIota",
		"DynamicPad",
		"NonZero",
	)

	// methodsNotGenerated get a NodeType but no auto-generated wrapper
	// (hand-written implementations).
	methodsNotGenerated = sets.MakeWith(
		"Constant", "Parameter", "FusedQuantizedDense", "FusedQuantizedDenseVJP", "QuantizedGather", "FusedDenseVJP", "FusedLayerNormVJP")

	// nillableParams lists Value parameters that can be nil (passed as *Node).
	// Key format: "MethodName.paramName"
	nillableParams = sets.MakeWith(
		"FusedLayerNorm.gamma", "FusedLayerNorm.beta",
		"FusedLayerNormVJP.gamma", "FusedLayerNormVJP.beta",
		"FusedDense.bias",
		"FusedDenseVJP.bias",
		"FusedActivationVJP.x",
		"FusedAttentionQKVProjection.biasQ", "FusedAttentionQKVProjection.biasK", "FusedAttentionQKVProjection.biasV",
	)

	// methodsExcluded from generating and even from having a NodeType.
	// These are utility methods, not part of building a graph.
	methodsExcluded = sets.MakeWith(
		"Name", "Builder", "Compile", "OpShape",
		"DeviceAssignment", "DistributedSPMD", "DistributedAutoSharding",
		"Main", "NewFunction", "Return", "Closure", "Parent",
		"Call", "If", "Sort", "While", "Shape", "ConstantFromBuffer")

	// methodsNoGradient will add a stop gradient to the node.
	methodsNoGradient = sets.MakeWith(
		"And", "Or", "Xor", "LogicalNot", "Equal", "NotEqual", "GreaterOrEqual", "GreaterThan",
		"LessOrEqual", "LessThan", "EqualTotalOrder", "NotEqualTotalOrder", "GreaterOrEqualTotalOrder",
		"GreaterThanTotalOrder", "LessOrEqualTotalOrder", "LessThanTotalOrder")
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
			mi.GraphName = "Backend" + name
		}
		for _, param := range raw.Parameters {
			fieldName := strings.ToUpper(param.Name[:1]) + param.Name[1:]
			pi := &ParameterInfo{
				Name:        param.Name,
				FieldName:   fieldName,
				BackendType: param.Type,
				Printable:   true,
				IsVariadic:  strings.HasPrefix(param.Type, "..."),
			}
			mi.Inputs = append(mi.Inputs, pi)
			switch pi.BackendType {
			case "Value":
				pi.BackendType = "compute.Value"
				pi.GraphType = "*NodeImpl"
				pi.PublicType = "*Node"
				pi.PublicArg = fmt.Sprintf("unwrapNode(%s)", param.Name)
				if nillableParams.Has(name + "." + param.Name) {
					pi.IsNillable = true
					mi.NillableInputs = append(mi.NillableInputs, param.Name)
					pi.ConvertStatement = fmt.Sprintf("%sVal", param.Name)
					pi.Format = "%s"
					pi.FormatValue = fmt.Sprintf("strNillableNode(ni.%s)", fieldName)
				} else {
					pi.ConvertStatement = fmt.Sprintf("%s.outputOps[0]", param.Name)
					mi.OpInputs = append(mi.OpInputs, param.Name)
					pi.Format = "[#%d]"
					pi.FormatValue = fmt.Sprintf("ni.%s.Id()", fieldName)
				}
			case "...Value":
				pi.BackendType = "...compute.Value"
				pi.GraphType = "...*NodeImpl"
				pi.PublicType = "...*Node"
				pi.PublicArg = fmt.Sprintf("unwrapNodes(%s)...", param.Name)
				mi.OpInputSlices = append(mi.OpInputSlices, param.Name)
				pi.NodeInputType = "[]*NodeImpl"
				pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
				pi.ConvertStatement = fmt.Sprintf(
					"xslices.Map(%s, func(node *NodeImpl) compute.Value { return node.outputOps[0] })...", param.Name)
				pi.Format = "[#%s]"
				pi.FormatValue = fmt.Sprintf(
					`strings.Join(xslices.Map(ni.%s, func (node *NodeImpl) string { return fmt.Sprintf("#%%d", node.Id()) }), ", ")`,
					fieldName,
				)
			case "[]Value":
				pi.BackendType = "[]compute.Value"
				pi.GraphType = "[]*NodeImpl"
				pi.PublicType = "[]*Node"
				pi.PublicArg = fmt.Sprintf("unwrapNodes(%s)", param.Name)
				mi.OpInputSlices = append(mi.OpInputSlices, param.Name)
				pi.NodeInputType = "[]*NodeImpl"
				pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
				pi.ConvertStatement = fmt.Sprintf(
					"xslices.Map(%s, func(node *NodeImpl) compute.Value { return node.outputOps[0] })", param.Name)
				pi.Format = "[#%s]"
				pi.FormatValue = fmt.Sprintf(
					`strings.Join(xslices.Map(ni.%s, func (node *NodeImpl) string { return fmt.Sprintf("#%%d", node.Id()) }), ", ")`,
					fieldName,
				)
			case "ConvolveAxesConfig":
				pi.BackendType = "compute." + pi.BackendType
				pi.CopyStatement = fmt.Sprintf("%s.Clone()", param.Name)
				pi.Format = "%+v"
			case "PadAxis":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%+v"
			case "...PadAxis":
				pi.BackendType = "...compute." + pi.BackendType[3:]
				pi.NodeInputType = "[]" + pi.BackendType[3:]
				pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
				pi.ConvertStatement = fmt.Sprintf("inputs.%s...", fieldName)
				pi.Format = "%+v"
			case "FFTType":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%s"
			case "ActivationType":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%s"
			case "ActivationConfig":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%+v"
			case "DenseConfig":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%+v"
			case "DenseLayout":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%s"
			case "AttentionAxesLayout":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%s"
			case "AxesLayout":
				pi.BackendType = "compute.AttentionAxesLayout"
				pi.Format = "%s"
			case "QuantizationScheme":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%s"
			case "DotGeneralConfig":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%+v"
			case "CumSumOptions":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%+v"
			case "RNGAlgorithm":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%s"
			case "FusedTopKConfig":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%+v"
			case "NonZeroConfig":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%+v"
			case "*Quantization":
				pi.BackendType = "*compute.Quantization"
				pi.Format = "%+v"
			case "*ScaledDotProductAttentionConfig":
				pi.BackendType = "*compute.ScaledDotProductAttentionConfig"
				pi.Format = "%+v"
			case "DynamicDimensionSpec":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%+v"
			case "...DynamicDimensionSpec":
				pi.BackendType = "...compute." + pi.BackendType[3:]
				pi.NodeInputType = "[]" + pi.BackendType[3:]
				pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
				pi.ConvertStatement = fmt.Sprintf("inputs.%s...", fieldName)
				pi.Format = "%+v"
			case "DynamicPadAxis":
				pi.BackendType = "compute." + pi.BackendType
				pi.Format = "%+v"
			case "...DynamicPadAxis":
				pi.BackendType = "...compute." + pi.BackendType[3:]
				pi.NodeInputType = "[]" + pi.BackendType[3:]
				pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
				pi.ConvertStatement = fmt.Sprintf("inputs.%s...", fieldName)
				pi.Format = "%+v"
			default:
				switch {
				case strings.HasPrefix(pi.BackendType, "..."):
					pi.NodeInputType = "[]" + pi.BackendType[3:]
					pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
					pi.ConvertStatement = fmt.Sprintf("inputs.%s...", fieldName)
				case strings.HasPrefix(pi.BackendType, "[]"):
					pi.CopyStatement = fmt.Sprintf("slices.Clone(%s)", param.Name)
				case strings.HasPrefix(pi.BackendType, "func"):
					pi.Printable = false
				default:
					// Nothing to add.
				}
			}
			if pi.GraphType == "" {
				pi.GraphType = pi.BackendType
			}
			if pi.PublicType == "" {
				pi.PublicType = pi.GraphType
			}
			if pi.PublicArg == "" {
				pi.PublicArg = pi.Name
				if pi.IsVariadic {
					pi.PublicArg += "..."
				}
			}
			if pi.NodeInputType == "" {
				pi.NodeInputType = pi.GraphType
			}
			if pi.CopyStatement == "" {
				pi.CopyStatement = pi.Name
			}
			if pi.ConvertStatement == "" {
				pi.ConvertStatement = "inputs." + fieldName
			}
			if pi.Format == "" {
				pi.Format = "%v"
			}
			if pi.FormatValue == "" {
				pi.FormatValue = "ni." + fieldName
			}
			mi.HasGraph = len(mi.OpInputSlices) == 0 && len(mi.OpInputs) == 0 && len(mi.NillableInputs) == 0
		}
		sliceOutputCount := 0
		for outputIdx, output := range raw.Outputs[:len(raw.Outputs)-1] { // Skip the error.
			switch output.Type {
			case "Value":
				mi.OutputNames = append(mi.OutputNames, output.Name)
			case "[]Value":
				mi.OutputNames = append(mi.OutputNames, output.Name)
				mi.IsOutputSlice = true
				sliceOutputCount++
			default:
				exceptions.Panicf("unexpected output type %q for output #%d of %q", output.Type, outputIdx, name)
			}
		}
		switch {
		case mi.IsOutputSlice && len(mi.OutputNames) > 1:
			// Mixed shape: scalar output(s) then exactly one trailing []Value (the last output).
			if sliceOutputCount != 1 || raw.Outputs[len(raw.Outputs)-2].Type != "[]Value" {
				exceptions.Panicf("method %q: mixed outputs only supported as scalar output(s) followed by exactly one trailing []Value", name)
			}
			mi.HasTrailingSlice = true
			mi.IsOutputSlice = false // not the pure-slice case
			mi.ScalarOutputNames = mi.OutputNames[:len(mi.OutputNames)-1]
			mi.SliceOutputName = mi.OutputNames[len(mi.OutputNames)-1]
		case len(mi.OutputNames) > 1:
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
	NillableInputs         []string
	Inputs                 []*ParameterInfo
	Exported, Excluded     bool
	Comments               []string
	StopGradient           bool

	HasMultipleOutputs bool
	IsOutputSlice      bool
	OutputNames        []string

	// HasTrailingSlice marks the mixed output shape: scalar output(s) then one trailing []Value.
	HasTrailingSlice  bool
	ScalarOutputNames []string
	SliceOutputName   string
}

// ParameterInfo represents one parameter only.
type ParameterInfo struct {
	Name, FieldName                       string
	BackendType, GraphType, NodeInputType string
	PublicType, PublicArg                 string
	CopyStatement, ConvertStatement       string
	Format, FormatValue                   string
	Printable                             bool
	IsNillable                            bool
	IsVariadic                            bool
}

const (
	computeOpsFile = "gen_compute_ops.go"
	publicOpsFile  = "gen_public_ops.go"
)

var (
	backendOpsTemplate = template.Must(template.New(computeOpsFile).Parse(`
/***** File generated by ./internal/cmd/graph_generator, based on compute.Builder interface. Don't edit it directly. *****/

package graphimpl

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/compute"
	"github.com/gomlx/compute/dtypes"
	"github.com/gomlx/compute/shapes"
	"github.com/gomlx/compute/support/xslices"
)

// strNillableNode formats a nillable *NodeImpl for display.
func strNillableNode(n *NodeImpl) string {
	if n == nil {
		return "nil"
	}
	return fmt.Sprintf("[#%d]", n.Id())
}

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

// NodeInputs{{.BackendName}} holds the inputs used for the call to compute.{{.BackendName}}.
type NodeInputs{{.BackendName}} struct {
	{{- range .Inputs}}
	{{.FieldName}} {{.NodeInputType}}
	{{- end}}
}

// Type implements the interface NodeInputs.
func (ni *NodeInputs{{.BackendName}}) Type() NodeType {
	return NodeType{{.BackendName}}
}

// String implements the interface NodeInputs.
func (ni *NodeInputs{{.BackendName}}) String() string {
	return fmt.Sprintf("%s(
{{- range $index, $input := .Inputs -}}
	{{- if $input.Printable -}}
		{{- if $index -}}, {{end}}{{$input.Name}}={{$input.Format}}
	{{- end -}}
{{- end -}}
)",
		ni.Type(),
{{- range .Inputs -}}
	{{- if .Printable }}
		{{.FormatValue}},
	{{- end -}}
{{- end }}
	)
}

// CloneWithInputs implements the interface NodeInputs.
func (ni *NodeInputs{{.BackendName}}) CloneWithInputs(originalNode *NodeImpl, newInputs ...*NodeImpl) *NodeImpl {
	{{if .HasGraph -}}
	return {{.GraphName}}(originalNode.Graph(), {{range .Inputs}}ni.{{.FieldName}}{{if .IsVariadic}}...{{end}}, {{end}})
	{{- else -}}
	// Reconstruct inputs from newInputs
	idx := 0
	{{- range .Inputs}}
	{{- if eq .GraphType "*NodeImpl"}}
		{{- if .IsNillable}}
		var new_{{.Name}} *NodeImpl
		if ni.{{.FieldName}} != nil {
			new_{{.Name}} = newInputs[idx]
			idx++
		}
		{{- else}}
		new_{{.Name}} := newInputs[idx]
		idx++
		{{- end}}
	{{- else if or (eq .GraphType "[]*NodeImpl") (eq .GraphType "...*NodeImpl") (eq .NodeInputType "[]*NodeImpl")}}
		new_{{.Name}} := newInputs[idx : idx+len(ni.{{.FieldName}})]
		idx += len(ni.{{.FieldName}})
	{{- end}}
	{{- end}}
	{{- if .HasMultipleOutputs}}
	{{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}{{if eq $ii 0}}r0{{else}}_{{end}}{{end}} := {{.GraphName}}({{range .Inputs}}{{if or (eq .GraphType "*NodeImpl") (eq .GraphType "[]*NodeImpl") (eq .GraphType "...*NodeImpl") (eq .NodeInputType "[]*NodeImpl")}}new_{{.Name}}{{if .IsVariadic}}...{{end}}{{else}}ni.{{.FieldName}}{{if .IsVariadic}}...{{end}}{{end}}, {{end}})
	return r0.inputNodes[0]
	{{- else if .HasTrailingSlice}}
	r0{{range $ii, $name := .ScalarOutputNames}}{{if $ii}}, _{{end}}{{end}}, _ := {{.GraphName}}({{range .Inputs}}{{if or (eq .GraphType "*NodeImpl") (eq .GraphType "[]*NodeImpl") (eq .GraphType "...*NodeImpl") (eq .NodeInputType "[]*NodeImpl")}}new_{{.Name}}{{if .IsVariadic}}...{{end}}{{else}}ni.{{.FieldName}}{{if .IsVariadic}}...{{end}}{{end}}, {{end}})
	return r0.inputNodes[0]
	{{- else if .IsOutputSlice}}
	r := {{.GraphName}}({{range .Inputs}}{{if or (eq .GraphType "*NodeImpl") (eq .GraphType "[]*NodeImpl") (eq .GraphType "...*NodeImpl") (eq .NodeInputType "[]*NodeImpl")}}new_{{.Name}}{{if .IsVariadic}}...{{end}}{{else}}ni.{{.FieldName}}{{if .IsVariadic}}...{{end}}{{end}}, {{end}})
	return r[0].inputNodes[0]
	{{- else}}
	return {{.GraphName}}({{range .Inputs}}{{if or (eq .GraphType "*NodeImpl") (eq .GraphType "[]*NodeImpl") (eq .GraphType "...*NodeImpl") (eq .NodeInputType "[]*NodeImpl")}}new_{{.Name}}{{if .IsVariadic}}...{{end}}{{else}}ni.{{.FieldName}}{{if .IsVariadic}}...{{end}}{{end}}, {{end}})
	{{- end}}
	{{- end}}
}


{{- if not .Exported}}
// {{.GraphName}} is an internal GraphImpl wrapper for the compute.Function.{{.BackendName}} method.
{{- else}}
{{- range .Comments}}
{{.}}
{{- end}}
{{- end}}
func {{/*

Inputs:  */}}{{.GraphName}}({{if .HasGraph}}g *GraphImpl, {{end}}{{range .Inputs}}{{.Name}} {{.GraphType}}, {{end}}) (
{{- /* Outputs: */}}
{{- if .HasMultipleOutputs}}
{{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}{{$name}}{{end}} *NodeImpl
{{- else if .HasTrailingSlice}}
{{range .ScalarOutputNames}}{{.}} *NodeImpl, {{end}}{{.SliceOutputName}} []*NodeImpl
{{- else if .IsOutputSlice}}
[]*NodeImpl
{{- else}}
node *NodeImpl
{{- end}})
{{- /*

Body: */}}{
{{- if .HasGraph}}
	g.AssertBuilding()
{{- else}}
	inputNodes := []*NodeImpl{ {{range .OpInputs}}{{.}}, {{end}} }
{{- range .OpInputSlices}}
	inputNodes = append(inputNodes, {{.}}...)
{{- end}}
{{- range .NillableInputs}}
	if {{.}} != nil {
		inputNodes = append(inputNodes, {{.}})
	}
{{- end}}
	g := ValidateBuildingGraphFromInputs(inputNodes...)
{{- end}}
	inputs := &NodeInputs{{.BackendName}}{
{{- range .Inputs}}
		{{.FieldName}}: {{.CopyStatement}},
{{- end}}
	}
{{- range .Inputs}}{{- if .IsNillable}}
	var {{.Name}}Val compute.Value
	if {{.Name}} != nil {
		{{.Name}}Val = {{.Name}}.outputOps[0]
	}
{{- end}}{{- end}}
{{- /*

Convert result(s) to node(s):

*/}}
{{- if .HasMultipleOutputs}}
{{- /* Version with multiple outputs: */}}
{{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}v{{$ii}}{{end}}, err := g.currentFunc.backendFunc.{{.BackendName}}({{range .Inputs}}{{.ConvertStatement}}, {{end}})
	if err != nil {
		panic(err)
	}
	node := &NodeImpl{
		outputOps: []compute.Value{ {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}v{{$ii}}{{end}} },
		outputShapes: []shapes.Shape{ {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}mustNoError(g.builder.OpShape(v{{$ii}})){{end}} },
{{- else if .HasTrailingSlice}}
{{- /* Version with scalar output(s) followed by a trailing []Value: */}}
	{{range $ii, $name := .ScalarOutputNames}}v{{$ii}}, {{end}}vTrailing, err := g.currentFunc.backendFunc.{{.BackendName}}({{range .Inputs}}{{.ConvertStatement}}, {{end}})
	if err != nil {
		panic(err)
	}
	outputOps := append([]compute.Value{ {{range $ii, $name := .ScalarOutputNames}}{{if $ii}}, {{end}}v{{$ii}}{{end}} }, vTrailing...)
	node := &NodeImpl{
		outputOps: outputOps,
		outputShapes: xslices.Map(outputOps,
			func (op compute.Value) shapes.Shape { return mustNoError(g.builder.OpShape(op)) }),
{{- else if .IsOutputSlice}}
{{- /* Version with output slice: */}}
	results, err := g.currentFunc.backendFunc.{{.BackendName}}({{range .Inputs}}{{.ConvertStatement}}, {{end}})
	if err != nil {
		panic(err)
	}
	node := &NodeImpl{
		outputOps: results,
		outputShapes: xslices.Map(results,
			func (op compute.Value) shapes.Shape { return mustNoError(g.builder.OpShape(op)) }),
{{- else}}
{{- /* Version with single output: - node already defined. */}}
	result, err := g.currentFunc.backendFunc.{{.BackendName}}({{range .Inputs}}{{.ConvertStatement}}, {{end}})
	if err != nil {
		panic(err)
	}
	node = &NodeImpl{
		outputOps: []compute.Value{result},
		outputShapes: []shapes.Shape{mustNoError(g.builder.OpShape(result))},
{{- end}}
{{- /* Rest of node definition */}}
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
{{- /*

If multiple-outputs, split resulting node into its separate parts:
*/}}
{{- if .HasMultipleOutputs}}
	splitNodes := SplitNode(node)
	{{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}{{$name}}{{end}} = {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}splitNodes[{{$ii}}]{{end}}
	return
{{- else if .HasTrailingSlice}}
	splitNodes := SplitNode(node)
	{{range $ii, $name := .ScalarOutputNames}}{{$name}} = splitNodes[{{$ii}}]
	{{end}}{{.SliceOutputName}} = splitNodes[{{len .ScalarOutputNames}}:]
	return
{{- else if .IsOutputSlice}}
	return SplitNode(node)
{{- else}}
	return
{{- end}}
}
{{end}}{{end}}
`))

	publicOpsTemplate = template.Must(template.New(publicOpsFile).Parse(`
/***** File generated by ./internal/cmd/graph_generator -public, based on compute.Builder interface. Don't edit it directly. *****/

package graph

import (
{{- if .UseCompute}}
	"github.com/gomlx/compute"
{{- end}}
{{- if .UseDTypes}}
	"github.com/gomlx/compute/dtypes"
{{- end}}
{{- if .UseShapes}}
	"github.com/gomlx/compute/shapes"
{{- end}}
	"github.com/gomlx/gomlx/internal/graphimpl"
)

// NodeType identifies the operation performed by a Node.
type NodeType = graphimpl.NodeType

const (
	NodeTypeInvalid   NodeType = graphimpl.NodeTypeInvalid
	NodeTypeSplitNode NodeType = graphimpl.NodeTypeSplitNode
{{- range .AllMethods}}
	NodeType{{.BackendName}} NodeType = graphimpl.NodeType{{.BackendName}}
{{- end}}
	NodeTypeCall  NodeType = graphimpl.NodeTypeCall
	NodeTypeSort  NodeType = graphimpl.NodeTypeSort
	NodeTypeWhile NodeType = graphimpl.NodeTypeWhile
	NodeTypeIf    NodeType = graphimpl.NodeTypeIf
)

// NodeTypeString retrieves an enum value from its string representation.
func NodeTypeString(s string) (NodeType, error) {
	return graphimpl.NodeTypeString(s)
}

// NodeTypeValues returns all values of NodeType.
func NodeTypeValues() []NodeType {
	return graphimpl.NodeTypeValues()
}

// NodeTypeStrings returns all string representations of NodeType.
func NodeTypeStrings() []string {
	return graphimpl.NodeTypeStrings()
}
{{range .ExportedMethods}}
{{- range .Comments}}
{{.}}
{{- end}}
func {{.GraphName}}({{if .HasGraph}}g *Graph, {{end}}{{range .Inputs}}{{.Name}} {{.PublicType}}, {{end}}) (
{{- if .HasMultipleOutputs}}
{{- range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}{{$name}}{{end}} *Node
{{- else if .HasTrailingSlice}}
{{- range .ScalarOutputNames}}{{.}} *Node, {{end}}{{.SliceOutputName}} []*Node
{{- else if .IsOutputSlice}}
[]*Node
{{- else}}
*Node
{{- end}}) {
{{- $callArgs := ""}}
{{- if .HasMultipleOutputs}}
	{{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}r{{$ii}}{{end}} := graphimpl.{{.GraphName}}({{if .HasGraph}}unwrapGraph(g), {{end}}{{range .Inputs}}{{.PublicArg}}, {{end}})
	return {{range $ii, $name := .OutputNames}}{{if $ii}}, {{end}}wrapNode(r{{$ii}}){{end}}
{{- else if .HasTrailingSlice}}
	{{range $ii, $name := .ScalarOutputNames}}r{{$ii}}, {{end}}rTrailing := graphimpl.{{.GraphName}}({{if .HasGraph}}unwrapGraph(g), {{end}}{{range .Inputs}}{{.PublicArg}}, {{end}})
	return {{range $ii, $name := .ScalarOutputNames}}wrapNode(r{{$ii}}), {{end}}wrapNodes(rTrailing)
{{- else if .IsOutputSlice}}
	return wrapNodes(graphimpl.{{.GraphName}}({{if .HasGraph}}unwrapGraph(g), {{end}}{{range .Inputs}}{{.PublicArg}}, {{end}}))
{{- else}}
	return wrapNode(graphimpl.{{.GraphName}}({{if .HasGraph}}unwrapGraph(g), {{end}}{{range .Inputs}}{{.PublicArg}}, {{end}}))
{{- end}}
}
{{end}}
`))
)

// GeneratePublicOps generates core/graph/gen_public_ops.go containing NodeType re-exports and
// thin wrappers for only the exported backend ops.
func GeneratePublicOps(methods []*MethodInfo) {
	slices.SortFunc(methods, func(a, b *MethodInfo) int { return strings.Compare(a.BackendName, b.BackendName) })
	var exported []*MethodInfo
	var useCompute, useDTypes, useShapes bool
	for _, m := range methods {
		if m.Exported && !m.Excluded {
			exported = append(exported, m)
			for _, p := range m.Inputs {
				if strings.Contains(p.PublicType, "compute.") {
					useCompute = true
				}
				if strings.Contains(p.PublicType, "dtypes.") {
					useDTypes = true
				}
				if strings.Contains(p.PublicType, "shapes.") {
					useShapes = true
				}
			}
		}
	}
	data := struct {
		AllMethods      []*MethodInfo
		ExportedMethods []*MethodInfo
		UseCompute      bool
		UseDTypes       bool
		UseShapes       bool
	}{
		AllMethods:      methods,
		ExportedMethods: exported,
		UseCompute:      useCompute,
		UseDTypes:       useDTypes,
		UseShapes:       useShapes,
	}

	curDir := must.M1(os.Getwd())
	fileName := path.Join(curDir, publicOpsFile)
	f := must.M1(os.Create(fileName))
	must.M(publicOpsTemplate.Execute(f, data))
	must.M(f.Close())
	cmd := exec.Command("go", "fmt", fileName)
	klog.V(1).Infof("\t%s\n", cmd)
	must.M(cmd.Run())
	fmt.Printf("✅ graph_generator -public:\tsuccessfully generated %s\n", fileName)
}

// GenerateBackendOps generates the list of NodeType and the default implementation of ops to all compute.Builder
// interface methods and corresponding NodeInputs* struct.
func GenerateBackendOps(methods []*MethodInfo) {
	// Sort by backend method name:
	slices.SortFunc(methods, func(a, b *MethodInfo) int { return strings.Compare(a.BackendName, b.BackendName) })

	curDir := must.M1(os.Getwd())
	fileName := path.Join(curDir, computeOpsFile)
	f := must.M1(os.Create(fileName))
	must.M(backendOpsTemplate.Execute(f, methods))
	must.M(f.Close())
	cmd := exec.Command("go", "fmt", fileName)
	klog.V(1).Infof("\t%s\n", cmd)
	must.M(cmd.Run())
	fmt.Printf("✅ graph_generator:       \tsuccessfully generated %s\n", fileName)

	// Generate enumer for NodeType:
	enumerOutput := path.Join(curDir, "gen_nodetype_enumer.go")
	cmd = exec.Command(
		"go",
		"tool",
		"enumer",
		"-type=NodeType",
		"-trimprefix=NodeType",
		"-yaml",
		"-json",
		"-text",
		"-values",
		"-output="+enumerOutput,
		fileName,
	)
	klog.V(1).Infof("\t%s\n", cmd)
	must.M(cmd.Run())
	fmt.Printf("✅ graph_generator:       \tsuccessfully generated %s\n", enumerOutput)
}
