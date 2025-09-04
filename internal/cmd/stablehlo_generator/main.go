// stablehlo_generator generates stablehlo.Backend implementations based on backend.Builder API.
package main

import (
	"github.com/gomlx/gomlx/internal/backendparser"
	"github.com/janpfeifer/must"
	"k8s.io/klog/v2"
)

func main() {
	klog.V(1).Info("stablehlo_generator:")
	methods := must.M1(backendparser.ParseBuilder())
	GenerateBinaryOps(methods)
}
