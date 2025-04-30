package simplego

import (
	"fmt"
	"github.com/gomlx/gomlx/backends"
	"github.com/janpfeifer/must"
	"os"
	"testing"
)

var backend backends.Backend

func setup() {
	fmt.Printf("Available backends: %q\n", backends.List())
	// Perform your setup logic here
	if os.Getenv(backends.ConfigEnvVar) == "" {
		must.M(os.Setenv(backends.ConfigEnvVar, "go"))
	}
	backend = backends.New()
	fmt.Printf("Backend: %s, %s\n", backend.Name(), backend.Description())
}

func teardown() {
	backend.Finalize()
}

func TestMain(m *testing.M) {
	setup()
	code := m.Run() // Run all tests in the file
	teardown()
	os.Exit(code)
}
