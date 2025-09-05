// backends_generator generates parts of the backends.Builder interface based on the github.com/gomlx/gopjrt/xlabuilder implementation.
//
// Although GoMLX can support more than one backend, the XlaBuilder is the reference implementation for now.
//
// If the environment variable GOPJRT_SRC is set, it parses the ops from there.
// Otherwise it clones the gopjrt repository to a temporary sub-directory.
package main

import (
	"flag"
	"fmt"

	"github.com/gomlx/gomlx/internal/cmd/backends_generator/parsexlabuilder"
)

var flagSure = flag.Bool("sure", false, "Confirm that's what you want")

func main() {
	fmt.Println("backends_generator:")
	if !*flagSure {
		fmt.Println("This generator is deprecated, use -sure to confirm")
		return
	}
	opsInfo := parsexlabuilder.ReadOpsInfo()
	_ = opsInfo
	extractor, xlaBuilderAst := parsexlabuilder.Parse()
	GenerateOpTypesEnum(extractor, xlaBuilderAst)
	GenerateStandardOpsInterface(extractor, xlaBuilderAst)
}
