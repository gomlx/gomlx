package main

import (
	"fmt"

	"github.com/gomlx/compute"
	. "github.com/gomlx/gomlx/core/graph"

	// Import default backends.
	_ "github.com/gomlx/gomlx/backends/default"
)

//md:simple

// MyComputation builds the symbolic graph (Column 2 in the diagram).
// It runs ONLY when GoMLX encounters a new input shape.
// //md_start:simple
func MyComputation(x, y *Node) *Node {
	return Sqrt(ReduceAllSum(Square(Sub(x, y))))
}

//md_end:simple

func main() {
	fmt.Println("md:simple") // Select output for "simple" cell.

	//md:simple(-1)

	// Step 1: Create the backend (Column 3).
	backend, err := compute.New() //md:simple
	if err != nil {
		panic(err)
	}

	// Step 2: Create the executor.
	// MyComputation is passed but NOT executed yet.
	exec, err := NewExec(backend, MyComputation) //md:simple
	if err != nil {
		panic(err)
	}

	// Step 3: First execution (triggers build & compile).
	// GoMLX sees a new shape ([]float32), calls MyComputation, and compiles via XLA.
	results, err := exec.Call([]float32{1.0, 2.0}, []float32{4.0, 6.0}) //md:simple
	if err != nil {
		panic(err)
	}
	fmt.Printf("Distance (1,2)->(4,6):  %v\n", results[0]) //md:simple

	// Step 4: Second execution with the SAME shape.
	// Bypasses MyComputation entirely. XLA uses the cached Executable.
	results, err = exec.Call([]float32{0, 0}, []float32{5, 12}) //md:simple
	if err != nil {
		panic(err)
	}
	fmt.Printf("Distance (0,0)->(5,12): %v\n", results[0]) //md:simple
}
