package graph_test

import (
	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

func init() {
	klog.InitFlags(nil)
}
