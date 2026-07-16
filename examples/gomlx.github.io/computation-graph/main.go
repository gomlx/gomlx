package main

import ( //md:main_exec
	"fmt" //md:main_exec

	"github.com/gomlx/compute"                  //md:main_exec
	_ "github.com/gomlx/gomlx/backends/default" //md:main_exec
	. "github.com/gomlx/gomlx/core/graph"       //md:main_exec
) //md:main_exec

//md_start:distance_fn,main_exec

// EuclideanDistance between two values.
func EuclideanDistance(a, b *Node) *Node {
	return Sqrt(ReduceAllSum(Square(Sub(a, b))))
}

//md_end:distance_fn,main_exec

func main() {
	backend := compute.MustNew() // auto-selects best available backend

	// Output to main_exec
	fmt.Println("md:main_exec")

	//md_start:main_exec
	// 1. Create the executor (that expects 1 output)
	exec, err := NewExec1(backend, EuclideanDistance)
	if err != nil {
		panic(err)
	}

	// 2. Call the executor with inputs (automatically JIT-compiled for Float64[] slices)
	resultTensor, err := exec.Call([]float64{1.0, 2.0}, []float64{4.0, 6.0})
	if err != nil {
		panic(err)
	}

	// 3. Print the result: float64(5)
	fmt.Printf("Distance: %s\n", resultTensor)
	//md_end:main_exec
}
