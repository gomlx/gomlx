package graph_test

import (
	"flag"
	"fmt"
	"os"
	"testing"

	"k8s.io/klog/v2"

	_ "github.com/gomlx/gomlx/backends/default"
)

func init() {
	klog.InitFlags(nil)
}

func must(err error) {
	if err != nil {
		klog.Errorf("Failed with error: %+v", err)
		panic(err)
	}
}

func must1[T any](value T, err error) T {
	must(err)
	return value
}

func TestMain(m *testing.M) {
	fmt.Println(">> TestMain():")
	flag.Parse()
	exitCode := m.Run()
	fmt.Println(">> TestMain(): finished")
	os.Exit(exitCode)
}
