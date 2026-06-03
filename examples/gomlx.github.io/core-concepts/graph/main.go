package main

//md_start:backend
import (
	"fmt"

	"github.com/gomlx/compute"
	_ "github.com/gomlx/gomlx/backends/default" // Includes default backends: "go", "xla:cpu", "xla:cuda", etc.
)

//md_end:

func main() {
	backend := compute.MustNew() // auto-selects best available backend //md:backend

	fmt.Printf("Backend: %s\n", backend.Description()) //md:backend
}
