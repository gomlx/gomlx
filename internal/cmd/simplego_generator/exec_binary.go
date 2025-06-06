package main

import (
	"fmt"
	"os"
	"os/exec"
	"text/template"

	"github.com/gomlx/gomlx/types"
	"github.com/janpfeifer/must"
)

const (
	execBinaryFile = "gen_exec_binary.go"
)

// methodsToExclude from generating the API, they are maintained manually,
// or simply excluded (deprecated methods).
var methodsToExclude = types.SetWith(
	"BatchNormForInference", "BatchNormForTraining", "BatchNormGradient",
	"And", "Or", "Xor", "Not", "ReduceAnd", "ReduceOr", "ReduceXor", "ScatterAdd")

var (
	execBinaryTemplate = template.Must(
		template.
			New(execBinaryFile).
			Funcs(execBinaryFuncMap).
			Parse(
				`/***** File generated by ./internal/cmd/simplego_generator. Don't edit it directly. *****/

package simplego

import (
	"math"
	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/pkg/errors"
)

func init() { 
{{- range .BinaryOps}}
	nodeExecutors[backends.OpType{{.Name}}] = exec{{.Name}}
{{- end}}
}

{{- range .BinaryOps}}
{{- $name := .Name }}
{{- $is_comparison := .IsComparison }}

// exec{{.Name}} executes the binary op {{.Name}}.
func exec{{.Name}}(backend *Backend, node *Node, inputs []*Buffer, inputsOwned []bool) (*Buffer, error) {

{{- if .IsComparison }}
	lhs, rhs := inputs[0], inputs[1]
	lhsIsScalarOr1, rhsIsScalarOr1 := lhs.shape.Size() == 1, rhs.shape.Size() == 1
	output := backend.getBuffer(node.shape.DType, node.shape.Size())
	output.shape = node.shape
{{- else }}
	lhs, rhs, output, lhsIsScalarOr1, rhsIsScalarOr1 := binaryOperandsAndOutput(backend, inputs, inputsOwned, node.shape)
{{- end }}

{{- if .IsCommutative}}// Add is commutative, so if any of the two is scalar, make the rhs the scalar one.
	if lhsIsScalarOr1 && !rhsIsScalarOr1 {
		lhs, rhs = rhs, lhs
		lhsIsScalarOr1, rhsIsScalarOr1 = rhsIsScalarOr1, lhsIsScalarOr1
	}
{{- else }}
	_, _ = lhsIsScalarOr1, rhsIsScalarOr1
{{- end }}

	switch lhs.shape.DType {
{{- range .Versions}}
{{- $version := .Name }}

{{- if or .Numeric .Integer }}

{{- range $.IntegerTypes}}	

	case dtypes.{{.DType}}:
		exec{{$name}}{{$version}}Generic[{{.GoType}}](lhs.flat.([]{{.GoType}}), rhs.flat.([]{{.GoType}}), output.flat.([]
	{{- if $is_comparison }} bool {{- else }} {{.GoType}} {{- end }} ), lhs.shape, rhs.shape, output.shape)
{{- end}}
{{- end}}

{{- if or .Numeric .Float }} 

{{- range $.FloatTypes}}

	case dtypes.{{.DType}}:
		exec{{$name}}{{$version}}Generic[{{.GoType}}](lhs.flat.([]{{.GoType}}), rhs.flat.([]{{.GoType}}), output.flat.([]
	{{- if $is_comparison }} bool {{- else }} {{.GoType}} {{- end }} ), lhs.shape, rhs.shape, output.shape)
{{- end}}
{{- end}}

{{- if or .Numeric .BFloat16 }}
{{- range $.BFloat16Types}}

	case dtypes.{{.DType}}:
		exec{{$name}}{{$version}}BFloat16(lhs.flat.([]{{.GoType}}), rhs.flat.([]{{.GoType}}), output.flat.([]
	{{- if $is_comparison }} bool {{- else }} {{.GoType}} {{- end }} ), lhs.shape, rhs.shape, output.shape)
{{- end}}
{{- end}}

{{- if .Boolean }}
	// Boolean:
{{- range $.BooleanTypes}}
	case dtypes.{{.DType}}:
		exec{{$name}}{{$version}}Generic[{{.GoType}}](lhs.flat.([]{{.GoType}}), rhs.flat.([]{{.GoType}}), output.flat.([]{{.GoType}}),
			lhs.shape, rhs.shape, output.shape)
{{- end}}
{{- end}}

{{- end}}
	default:
		return nil, errors.Errorf("unsupported data type %s for %s", output.shape.DType, node.opType)
	}
	return output, nil
}

{{- $is_commutative := .IsCommutative }}
{{- range .Versions}}
{{- $version := .Name }}

{{- if or .Numeric .Integer .Float .Boolean }}

func exec{{$name}}{{$version}}Generic[T POD{{$version}}Constraints](lhs, rhs []T, output []{{if $is_comparison}}bool{{else}}T{{end}},
	lhsShape, rhsShape, outputShape shapes.Shape) {
	if len(rhs) == 1 {
		// Case 1: One side (rhs) is a scalar: only iterate over the lhs.
		c := rhs[0]
		for ii, input := range lhs {
			output[ii] = {{ CallOp .Format "input" "c" }}
		}
		return
{{- if not $is_commutative }}
	} else if len(lhs) == 1 {
		// Case 1b: One side (lhs) is a scalar: only iterate over the rhs.
		c := lhs[0]
		for ii, input := range rhs {
			output[ii] = {{ CallOp .Format "c" "input" }}
		}
		return
{{- end}}

	} else if lhsShape.Equal(rhsShape) {
		// Case 2: Exact same shapes, no broadcasting.
		for ii, input := range lhs {
			output[ii] = {{ CallOp .Format "input" "rhs[ii]" }} 
		}
		return

	} else {
		// Case 3: with broadcasting non-scalar tensors:
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for outputIdx := range output {
			lhsIdx := lhsIter.Next()
			rhsIdx := rhsIter.Next()
			output[outputIdx] = {{ CallOp .Format "lhs[lhsIdx]" "rhs[rhsIdx]" }}
		}
	}
	return
}
{{- end}}

{{- if or .Numeric .BFloat16 }}

func exec{{$name}}{{$version}}BFloat16(lhs, rhs []bfloat16.BFloat16, output []{{if $is_comparison}}bool{{else}}bfloat16.BFloat16{{end}},
	lhsShape, rhsShape, outputShape shapes.Shape) {
	if len(rhs) == 1 {
		// One side (rhs) is a scalar: only iterate over the lhs.
		c := rhs[0].Float32()
		for ii, input := range lhs {
			a := input.Float32()
		{{- if $is_comparison }}
			output[ii] = {{CallOp .Format "a" "c"}}
		{{- else }}
			output[ii] = bfloat16.FromFloat32({{CallOp .Format "a" "c"}})
		{{- end }}
		}
		return
{{- if not $is_commutative }}
	} else if len(lhs) == 1 {
		// Case 1b: One side (lhs) is a scalar: only iterate over the rhs.
		c := lhs[0].Float32()
		for ii, input := range rhs {
			a := input.Float32()	
		{{- if $is_comparison }}
			output[ii] = {{CallOp .Format "c" "a"}}
		{{- else }}
			output[ii] = bfloat16.FromFloat32({{ CallOp .Format "c" "a" }})
		{{- end }}
		}
		return
{{- end}}

	} else if lhsShape.Equal(rhsShape) {
		// Case 2: Exact same shapes, no broadcasting.
		for outputIdx := range output {
			a := lhs[outputIdx].Float32()
			b := rhs[outputIdx].Float32()
		{{- if $is_comparison }}
			output[outputIdx] = {{CallOp .Format "a" "b"}}
		{{- else }}
			output[outputIdx] = bfloat16.FromFloat32({{CallOp .Format "a" "b"}})
		{{- end }}
		}
		return

	} else {
		// Case 3: with broadcasting non-scalar tensors:
		lhsIter := newBroadcastIterator(lhsShape, outputShape)
		rhsIter := newBroadcastIterator(rhsShape, outputShape)
		for outputIdx := range output {
			lhsIdx := lhsIter.Next()
			rhsIdx := rhsIter.Next()
			a := lhs[lhsIdx].Float32()
			b := rhs[rhsIdx].Float32()
		{{- if $is_comparison }}
			output[outputIdx] = {{CallOp .Format "a" "b"}}
		{{- else }}
			output[outputIdx] = bfloat16.FromFloat32({{CallOp .Format "a" "b"}})
		{{- end }}
		}
	}
	return
}
{{- end}}

{{- end}}
{{- end}}
`))
)

type DataTypes struct {
	DType, GoType string
}

var (
	IntegerDataTypes = []DataTypes{
		{"Uint8", "uint8"},
		{"Uint16", "uint16"},
		{"Uint32", "uint32"},
		{"Uint64", "uint64"},
		{"Int8", "int8"},
		{"Int16", "int16"},
		{"Int32", "int32"},
		{"Int64", "int64"},
	}

	FloatDataTypes = []DataTypes{
		{"Float32", "float32"},
		{"Float64", "float64"},
	}

	BFloat16DataTypes = []DataTypes{
		{"BFloat16", "bfloat16.BFloat16"},
	}

	BooleanDataTypes = []DataTypes{
		{"Bool", "bool"},
	}
)

func callBinaryOp(format, s1, s2 string) string {
	return fmt.Sprintf(format, s1, s2)
}

var (
	execBinaryFuncMap = template.FuncMap{
		"CallOp": callBinaryOp,
	}
)

type BinaryOpVersion struct {
	Name                                       string
	Numeric, Integer, Float, BFloat16, Boolean bool
	Format                                     string
}

type BinaryOp struct {
	Name          string
	Versions      []BinaryOpVersion
	IsCommutative bool
	IsComparison  bool
}

var (
	binaryOps []BinaryOp = []BinaryOp{
		{Name: "Add", IsCommutative: true, Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "%s + %s"}}},
		{Name: "Mul", IsCommutative: true, Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "%s * %s"}}},
		{Name: "Sub", Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "%s - %s"}}},
		{Name: "Div", Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "%s / %s"}}},
		{Name: "Rem", Versions: []BinaryOpVersion{
			{Integer: true, Name: "Integer", Format: "%s %% %s"},
			{Float: true, Name: "Float", Format: "T(math.Mod(float64(%s), float64(%s)))"},
			{BFloat16: true, Name: "Float", Format: "float32(math.Mod(float64(%s), float64(%s)))"},
		}},
		{Name: "Pow", Versions: []BinaryOpVersion{
			{Integer: true, Name: "Integer", Format: "execScalarPowIntGeneric(%s, %s)"},
			{Float: true, Name: "Float", Format: "T(math.Pow(float64(%s), float64(%s)))"},
			{BFloat16: true, Name: "Float", Format: "float32(math.Pow(float64(%s), float64(%s)))"},
		}},
		{Name: "Max", IsCommutative: true, Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "max(%s, %s)"}}},
		{Name: "Min", IsCommutative: true, Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "min(%s, %s)"}}},
		{Name: "BitwiseAnd", Versions: []BinaryOpVersion{
			{Integer: true, Name: "Integer", Format: "%s & %s"},
		}},
		{Name: "BitwiseOr", Versions: []BinaryOpVersion{
			{Integer: true, Name: "Integer", Format: "%s | %s"},
		}},
		{Name: "BitwiseXor", Versions: []BinaryOpVersion{
			{Integer: true, Name: "Integer", Format: "%s ^ %s"},
		}},
		{Name: "LogicalAnd", Versions: []BinaryOpVersion{
			{Boolean: true, Name: "Boolean", Format: "%s && %s"},
		}},
		{Name: "LogicalOr", Versions: []BinaryOpVersion{
			{Boolean: true, Name: "Boolean", Format: "%s || %s"},
		}},
		{Name: "LogicalXor", Versions: []BinaryOpVersion{
			{Boolean: true, Name: "Boolean", Format: "%s != %s"},
		}},

		{Name: "Equal", IsComparison: true, IsCommutative: true, Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "%s == %s"}}},
		{Name: "NotEqual", IsComparison: true, IsCommutative: true, Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "%s != %s"}}},
		{Name: "GreaterOrEqual", IsComparison: true, Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "%s >= %s"}}},
		{Name: "GreaterThan", IsComparison: true, Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "%s > %s"}}},
		{Name: "LessOrEqual", IsComparison: true, Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "%s <= %s"}}},
		{Name: "LessThan", IsComparison: true, Versions: []BinaryOpVersion{{Numeric: true, Name: "Numeric", Format: "%s < %s"}}},
	}
)

type ExecBinaryData struct {
	BinaryOps []BinaryOp

	IntegerTypes  []DataTypes
	FloatTypes    []DataTypes
	BFloat16Types []DataTypes
	BooleanTypes  []DataTypes
}

func GenerateExecBinary() {
	data := ExecBinaryData{
		BinaryOps:     binaryOps,
		IntegerTypes:  IntegerDataTypes,
		FloatTypes:    FloatDataTypes,
		BFloat16Types: BFloat16DataTypes,
		BooleanTypes:  BooleanDataTypes,
	}

	fileName := execBinaryFile
	f := must.M1(os.Create(fileName))
	must.M(execBinaryTemplate.Execute(f, data))
	must.M(f.Close())

	cmd := exec.Command("gofmt", "-w", fileName)
	fmt.Printf("\t%s\n", cmd)
	must.M(cmd.Run())
	fmt.Printf("\t\tgenerated %q\n", fileName)
}
