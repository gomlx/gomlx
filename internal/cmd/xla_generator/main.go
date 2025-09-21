// xla_generator generates the xla.Backend implementation based on the github.com/gomlx/gopjrt/xlabuilder implementation.
//
// Although GoMLX can support more than one backend, the XlaBuilder is the reference implementation for now.
//
// If the environment variable GOPJRT_SRC is set, it parses the ops from there.
// Otherwise it clones the gopjrt repository to a temporary sub-directory.
package main

import (
	"flag"

	"github.com/gomlx/gomlx/internal/xlabuilderparser"
	"k8s.io/klog/v2"
)

func main() {
	klog.InitFlags(nil)
	flag.Parse()
	klog.V(1).Info("xla_generator:")
	opsInfo := xlabuilderparser.ReadOpsInfo()
	_ = opsInfo
	extractor, xlaBuilderAst := xlabuilderparser.Parse()
	GenerateStandardOpsImplementation(extractor, xlaBuilderAst)
}
