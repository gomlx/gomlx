# WASM Example

A minimal example demonstrating GoMLX running in the browser via WebAssembly.

This example shows how to:

- Use GoMLX's pure Go backend (`simplego`) which works under WASM (no cgo required)
- Compile Go code to WebAssembly
- Execute tensor computations directly in the browser
- Interact with the DOM from Go/WASM

## Building and Running

### 1. Copy the WASM runtime helper from your Go installation

```bash
cp "$(go env GOROOT)/lib/wasm/wasm_exec.js" .
```

### 2. Build the WASM binary

```bash
GOOS=js GOARCH=wasm go build -o main.wasm main.go
```

### Building with TinyGo

TinyGo needs its own `wasm_exec.js`. Use `-no-debug` to drop debug info and make the binary much smaller.

```bash
cp "$(tinygo env TINYGOROOT)/targets/wasm_exec.js" .
tinygo build -target wasm -no-debug -o main.wasm .
```

### 3. Open in browser

Use any simple server and visit <http://localhost:8080>

You should see an interactive calculator that performs tensor operations (add/multiply) using GoMLX in the browser.

## Files

- `main.go` - WASM application code
- `index.html` - Web interface
- `README.md` - This file

## Notes

- The `simplego` backend is portable but slower than XLA backends. This example focuses on portability and browser compatibility.
- The WASM binary is about 24.6MB with Go, 21.5MB with TinyGo, and 7.6MB with TinyGo and `-no-debug`.
