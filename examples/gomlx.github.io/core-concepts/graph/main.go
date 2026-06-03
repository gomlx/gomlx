package main

import ( //md:cell1,cell2
	"fmt"
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
	backend := compute.MustNew() // auto-selects best available backend
	fmt.Printf("Backend: %s\n", backend.Description())
	//md_end:backend

	// Output to cell2
	fmt.Println("md:cell2")

	//md_start:cell2
	addFn := func(a, b *Node) *Node {
		fmt.Println("* building addFn computation graph")
		return Add(a, b)
	}
	addExec := MustNewExec(backend, addFn)
	fmt.Printf("\t- 1+1=%s\n", addExec.MustCall1(1.0, 1.0))
	fmt.Printf("\t- 2+2=%s\n", addExec.MustCall1(2.0, 2.0))
	//md_end:cell2

	// Output to cell2
	fmt.Println("md:cell3")

	//md_start:cell3
	_, err := addExec.Call1(int32(1), float32(1.0))
	if err != nil {
		//md_end:cell3
		//... //md:cell3
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
