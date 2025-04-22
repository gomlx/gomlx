// simplego_generator auto-generates parts of the SimpleGo backend:
//
// - exec_binary.go: binary ops execution, e.g.: Add, Mul, Div, Sub, Pow, etc.
package main

import (
	"fmt"
)

func main() {
	fmt.Println("simplego_generator:")
	GenerateExecBinary()
}
