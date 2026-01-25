// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path"
	"text/template"

	"github.com/gomlx/gomlx/internal/must"
	"k8s.io/klog/v2"
)

type DTypeInfo struct {
	DType, GoType string
}

type DispatcherInfo struct {
	Dispatcher, Generic string
	DTypes              []DTypeInfo
}

type MapInfo struct {
	MapName, Generic string
	DTypes           []DTypeInfo
}

type MapPairInfo struct {
	MapName, Generic string
	DTypes1, DTypes2 []DTypeInfo
}

type Data struct {
	Dispatchers []DispatcherInfo
	Maps        []MapInfo
	PairMaps    []MapPairInfo
}

var (
	// data lists the dispatchers to include, their generic function and with which set of dtypes to support.
	data = Data{
		Dispatchers: []DispatcherInfo{
			{"dispatchBroadcast", "execBroadcastGeneric", makeDTypes(true, true, true, true, true)},
			{"dispatchBroadcastInDim", "execBroadcastInDimGeneric", makeDTypes(true, true, true, true, true)},
			{"dispatchIota", "execIotaGeneric", makeDTypes(true, true, true, false, false)},
			{"dispatchGather", "execGatherGeneric", makeDTypes(true, true, false, false, false)},
		},
		Maps: []MapInfo{
			{"dotGeneralFlatToBlockDTypeMap", "dgCopyFlatToBlockShape", makeDTypes(true, true, true, true, false)},
			{"dotGeneralOutputBlockToFlatDTypeMap", "dgCopyOutputBlockToFlat", makeDTypes(true, true, true, false, false)},
			{"dotGeneralKernelDTypeMap", "buildDotGeneralKernel", makeDTypes(true, true, true, false, false)},
			{"dotGeneralNormalizeShapeDTypeMap", "dgNormalizeShape", makeDTypes(true, true, true, true, false)},
			{"dotGeneralNormalizedDTypeMap", "execNormalizedDotGeneralGeneric", makeDTypes(true, true, true, false, false)},
			{"mutableBytesDTypeMap", "mutableBytesGeneric", makeDTypes(true, true, true, true, true)},
			{"fillBufferDTypeMap", "fillBufferGeneric", makeDTypes(true, true, true, true, true)},
			{"reduceMaxDTypeMap", "execReduceMaxGeneric", makeDTypes(true, true, true, false, false)},
			{"reduceMinDTypeMap", "execReduceMinGeneric", makeDTypes(true, true, true, false, false)},
			{"reduceSumDTypeMap", "execReduceSumGeneric", makeDTypes(true, true, true, false, false)},
			{"reduceProductDTypeMap", "execReduceProductGeneric", makeDTypes(true, true, true, false, false)},
			{"reduceBitwiseAndDTypeMap", "execReduceBitwiseAndGeneric", makeDTypes(true, true, false, false, false)},
			{"reduceBitwiseOrDTypeMap", "execReduceBitwiseOrGeneric", makeDTypes(true, true, false, false, false)},
			{"reduceBitwiseXorDTypeMap", "execReduceBitwiseXorGeneric", makeDTypes(true, true, false, false, false)},
			{"transposeDTypeMap", "execTransposeGeneric", makeDTypes(true, true, true, true, true)},
			{"whereDTypeMap", "execWhereGeneric", makeDTypes(true, true, true, true, true)},
			{"combineMaxDTypeMap", "combineForScatterMaxGeneric", makeDTypes(true, true, true, false, false)},
			{"combineMinDTypeMap", "combineForScatterMinGeneric", makeDTypes(true, true, true, false, false)},
			{"combineSumDTypeMap", "combineForScatterSumGeneric", makeDTypes(true, true, true, false, false)},
			{"scatterDTypeMap", "execScatterGeneric", makeDTypes(true, true, true, true, false)},
			{"dereferenceIntsDTypeMap", "dereferenceIntsGeneric", makeDTypes(true, true, false, false, false)},
			{"sliceDTypeMap", "execSliceGeneric", makeDTypes(true, true, true, true, true)},
			{"argMinMaxDTypeMap", "execArgMinMaxGeneric", makeDTypes(true, true, true, false, false)},
			{"argMinMaxCopyIntsDTypeMap", "buildArgMinMaxCopyIntsFn", makeDTypes(true, true, false, false, false)},
			{"reduceWindowMaxDTypeMap", "reduceWindowMaxBuildUpdateFn", makeDTypes(true, true, true, false, false)},
			{"reduceWindowMinDTypeMap", "reduceWindowMinBuildUpdateFn", makeDTypes(true, true, true, false, false)},
			{"reduceWindowSumDTypeMap", "reduceWindowSumBuildUpdateFn", makeDTypes(true, true, true, false, false)},
			{"reduceWindowProductDTypeMap", "reduceWindowProductBuildUpdateFn", makeDTypes(true, true, true, false, false)},
			{"convNoDilationDTypeMap", "execConvNoDilationGeneric", makeDTypes(true, true, true, false, false)},
			{"convDTypeMap", "execConvGeneric", makeDTypes(true, true, true, false, false)},
			{"dotGeneralSmallMatMulDTypeMap", "execDotGeneralSmallMatMulGeneric", makeDTypes(true, true, true, false, false)},
		},
		PairMaps: []MapPairInfo{
			// Various ConvertDType instantiations.
			{
				MapName: "convertDTypePairMap", Generic: "execConvertDTypeGeneric",
				DTypes1: makeDTypes(true, true, true, false, false),
				DTypes2: makeDTypes(true, true, true, false, false),
			},
			{
				MapName: "convertDTypePairMap", Generic: "execConvertDTypeToBFloat16",
				DTypes1: makeDTypes(true, true, true, false, false),
				DTypes2: dtypesBFloat16,
			},
			{
				MapName: "convertDTypePairMap", Generic: "execConvertDTypeFromBFloat16",
				DTypes1: dtypesBFloat16,
				DTypes2: makeDTypes(true, true, true, false, false),
			},
			{
				MapName: "convertDTypePairMap", Generic: "execConvertDTypeToFloat16",
				DTypes1: makeDTypes(true, true, true, false, false),
				DTypes2: dtypesFloat16,
			},
			{
				MapName: "convertDTypePairMap", Generic: "execConvertDTypeFromFloat16",
				DTypes1: dtypesFloat16,
				DTypes2: makeDTypes(true, true, true, false, false),
			},
			{
				MapName: "convertDTypePairMap", Generic: "execConvertDTypeToBool",
				DTypes1: makeDTypes(true, true, true, false, false),
				DTypes2: makeDTypes(false, false, false, false, true),
			},
			{
				MapName: "convertDTypePairMap", Generic: "execConvertDTypeFromBool",
				DTypes1: makeDTypes(false, false, false, false, true),
				DTypes2: makeDTypes(true, true, true, false, false),
			},
			//{
			//	MapName: "scatterDTypeMap", Generic: "execScatterGeneric",
			//	// Indices DTypes:
			//	DTypes1: makeDTypes(true, true, false, false, false),
			//	// Operand DTypes:
			//	DTypes2: makeDTypes(true, true, true, true, false),
			//},
		},
	}
	fileName = "gen_register_dtypes.go"
)

var (
	dtypesBFloat16 = []DTypeInfo{DTypeInfo{"BFloat16", "bfloat16.BFloat16"}}
	dtypesFloat16  = []DTypeInfo{DTypeInfo{"Float16", "float16.Float16"}}
)

func makeDTypes(ints, uints, floats, floats16, boolean bool) []DTypeInfo {
	dtypes := make([]DTypeInfo, 0, 32)
	if ints {
		dtypes = append(dtypes,
			DTypeInfo{"Int8", "int8"},
			DTypeInfo{"Int16", "int16"},
			DTypeInfo{"Int32", "int32"},
			DTypeInfo{"Int64", "int64"},
		)
	}
	if uints {
		dtypes = append(dtypes,
			DTypeInfo{"Uint8", "uint8"},
			DTypeInfo{"Uint16", "uint16"},
			DTypeInfo{"Uint32", "uint32"},
			DTypeInfo{"Uint64", "uint64"},
		)
	}
	if floats {
		dtypes = append(dtypes,
			DTypeInfo{"Float32", "float32"},
			DTypeInfo{"Float64", "float64"},
		)
	}
	if floats16 {
		dtypes = append(dtypes,
			DTypeInfo{"BFloat16", "bfloat16.BFloat16"},
			DTypeInfo{"Float16", "float16.Float16"},
		)
	}
	if boolean {
		dtypes = append(dtypes,
			DTypeInfo{"Bool", "bool"},
		)
	}
	return dtypes
}

func main() {
	klog.InitFlags(nil)
	flag.Parse()

	registerTemplate := template.Must(
		template.
			New(fileName).
			Parse(

				`/***** File generated by ./internal/cmd/simplego_dispatcher. Don't edit it directly. *****/

package simplego

import (
	"github.com/gomlx/gomlx/pkg/core/dtypes"
	"github.com/gomlx/gomlx/pkg/core/dtypes/bfloat16"
	"github.com/x448/float16"
)


func init() {
{{- range .Dispatchers}}

	// DTypeDispatcher: {{.Dispatcher}}
{{- $dispatcher := .Dispatcher }}
{{- $generic := .Generic }}
{{- range .DTypes }}
	{{$dispatcher}}.Register(dtypes.{{.DType}}, priorityGeneric, {{$generic}}[{{.GoType}}])
{{- end }}
{{- end }}

{{- range .Maps}}

	// DTypeMap: {{.MapName}}
{{- $mapName := .MapName }}
{{- $generic := .Generic }}
{{- range .DTypes }}
	{{$mapName}}.Register(dtypes.{{.DType}}, priorityGeneric, {{$generic}}[{{.GoType}}])
{{- end }}
{{- end }}

{{- range .PairMaps}}

	// DTypePairMap: {{.MapName}}
{{- $mapName := .MapName }}
{{- $generic := .Generic }}
{{- $dtypes2 := .DTypes2 }}
{{- range .DTypes1 }}
{{- $dtype1 := .DType }}
{{- $goType1 := .GoType }}
{{- range $dtypes2 }}
	{{$mapName}}.Register(dtypes.{{$dtype1}}, dtypes.{{.DType}}, priorityGeneric, {{$generic}}[{{$goType1}}, {{.GoType}}])
{{- end }}
{{- end }}
{{- end }}

}
`))
	fullPath := path.Join(must.M1(os.Getwd()), fileName)
	f := must.M1(os.Create(fullPath))
	must.M(registerTemplate.Execute(f, data))
	must.M(f.Close())

	cmd := exec.Command("gofmt", "-w", fullPath)
	klog.V(1).Infof("\t%s\n", cmd)
	must.M(cmd.Run())
	fmt.Printf("✅ simplego_dispatcher:  \tsuccessfully generated %s\n", fullPath)
}
