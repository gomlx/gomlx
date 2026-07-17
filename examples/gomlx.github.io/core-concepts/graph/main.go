package main

import ( //md:cell1,cell2
	"fmt"
	"log"
	"strings"

	"github.com/gomlx/compute"                  //md:cell1
	_ "github.com/gomlx/gomlx/backends/default" // Includes default backends.  //md:cell1
	. "github.com/gomlx/gomlx/core/graph"       //md:cell2
) //md:cell1,cell2

//md:cell1,cell2

func main() {
	// Output to cell1
	fmt.Println("md:cell1")

	//md_start:backend
	backend, err := compute.New() // auto-selects best available backend //md:cell1
	if err != nil {
		log.Fatalf("error creating backend: %+v\n", err)
	}
	fmt.Printf("Backend: %s\n", backend.Description())
	//md_end:backend

	// Output to cell2
	fmt.Println("md:cell2")

	//md_start:cell2
	addFn := func(a, b *Node) *Node {
		fmt.Printf("* building addFn computation graph: a.shape=%s, b.shape=%s\n", a.Shape(), b.Shape())
		return Add(a, b)
	}
	addExec, err := NewExec1(backend, addFn)
	if err != nil {
		log.Fatalf("error creating computation graph: %+v\n", err)
	}
	v1, err := addExec.Call(1.0, 1.0)
	if err != nil {
		log.Fatalf("Failed to compile/execute: %+v", err)
	}
	fmt.Printf("\t- 1+1=%s\n", v1)
	fmt.Printf("\t- 2+2=%s\n", addExec.MustCall(2.0, 2.0))
	//md_end:cell2

	// Output to cell3
	fmt.Println("md:cell3")

	_, err = addExec.Call(int32(1), float32(1.0)) //md:cell3
	if err != nil {                               //md:cell3
		//... //md:cell3
		// Only output the relevant part of the error for the snippet:
		lines := strings.Split(fmt.Sprintf("%+v", err), "\n")
		for i, line := range lines {
			if i == 0 {
				fmt.Printf("Error: %s\n", line)
			} else if idx := strings.Index(line, "main.go"); idx >= 0 {
				line = line[strings.Index(line, "gomlx.github.io"):]
				fmt.Printf("\t.../%s\n", line)
			}
		}
	} //md:cell3
}
