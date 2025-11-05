/*
 *	Copyright 2025 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

//go:build js && wasm

// WASM demo for GoMLX using the pure Go backend (simplego).
// Compiles to WebAssembly and runs tensor computations in the browser.
package main

import (
	"fmt"
	"strconv"
	"strings"
	"syscall/js"

	"github.com/gomlx/gomlx/backends"
	_ "github.com/gomlx/gomlx/backends/simplego"
	"github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/tensors"
)

// Element IDs in the HTML page.
const (
	idInputA  = "a"
	idInputB  = "b"
	idOp      = "op"
	idRun     = "run"
	idOutput  = "out"
	opAdd     = "add"
	opMultiply = "mul"
)

// Global event handlers to prevent GC.
var (
	runHandler    js.Func
	unloadHandler js.Func
)

func writeToDOM(id, msg string) {
	el := js.Global().Get("document").Call("getElementById", id)
	if el.Truthy() {
		el.Set("textContent", msg)
	}
}

func getValue(id string) (float32, error) {
	el := js.Global().Get("document").Call("getElementById", id)
	if !el.Truthy() {
		return 0, fmt.Errorf("element #%s not found", id)
	}
	v := strings.TrimSpace(el.Get("value").String())
	f, err := strconv.ParseFloat(v, 32)
	if err != nil {
		return 0, fmt.Errorf("invalid number in #%s: %q: %w", id, v, err)
	}
	return float32(f), nil
}

func main() {
	defer func() {
		if r := recover(); r != nil {
			writeToDOM(idOutput, fmt.Sprintf("panic: %v", r))
		}
	}()

	// With simplego imported, MustNew picks the pure-Go backend under WASM.
	be := backends.MustNew()

	// Prepare cached executors for fast, repeated runs.
	addExec := graph.MustNewExec(be, func(x, y *graph.Node) *graph.Node { return graph.Add(x, y) })
	mulExec := graph.MustNewExec(be, func(x, y *graph.Node) *graph.Node { return graph.Mul(x, y) })

	compute := func() {
		doc := js.Global().Get("document")
		
		// Get operation type.
		opEl := doc.Call("getElementById", idOp)
		if !opEl.Truthy() {
			writeToDOM(idOutput, "element #"+idOp+" not found")
			return
		}
		op := opEl.Get("value").String()

		// Get input values.
		a, errA := getValue(idInputA)
		b, errB := getValue(idInputB)
		if errA != nil || errB != nil {
			msg := fmt.Sprintf("Backend: %s\n", be.Name())
			if errA != nil {
				msg += fmt.Sprintf("Input A: %v\n", errA)
			}
			if errB != nil {
				msg += fmt.Sprintf("Input B: %v\n", errB)
			}
			writeToDOM(idOutput, msg)
			return
		}

		// Execute the computation.
		var t *tensors.Tensor
		switch op {
		case opMultiply:
			t = mulExec.MustExec(a, b)[0]
		default: // opAdd
			t = addExec.MustExec(a, b)[0]
		}
		res := tensors.ToScalar[float32](t)
		writeToDOM(idOutput, fmt.Sprintf("Backend: %s\nResult: %g", be.Name(), res))
	}

	// Attach event listeners.
	doc := js.Global().Get("document")
	btn := doc.Call("getElementById", idRun)
	if btn.Truthy() {
		runHandler = js.FuncOf(func(this js.Value, args []js.Value) any {
			compute()
			return nil
		})
		btn.Call("addEventListener", "click", runHandler)
	}

	// Set up cleanup on page unload.
	unloadHandler = js.FuncOf(func(this js.Value, args []js.Value) any {
		if btn.Truthy() {
			btn.Call("removeEventListener", "click", runHandler)
		}
		runHandler.Release()
		unloadHandler.Release()
		return nil
	})
	js.Global().Get("window").Call("addEventListener", "beforeunload", unloadHandler)

	// Run once to show an initial result.
	compute()

	// Keep WASM alive for interactions.
	<-make(chan struct{})
}
