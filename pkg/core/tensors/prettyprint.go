package tensors

import (
	"bytes"
	"fmt"
	"reflect"
	"strings"

	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/gomlx/gopjrt/dtypes/bfloat16"
	"github.com/x448/float16"
)

var (
	typeFloat16  = reflect.TypeOf(float16.Float16(0))
	typeBFloat16 = reflect.TypeOf(bfloat16.BFloat16(0))
)

// Summary returns a multi-line summary of the Tensor's content.
// Inspired by numpy output.
func (t *Tensor) Summary(precision int) string {
	if t.Shape().IsZeroSize() {
		return t.Shape().String()
	}

	// Easy string building.
	var buf bytes.Buffer
	w := func(format string, args ...any) { _, _ = fmt.Fprintf(&buf, format, args...) }

	// Print value with appropriate formatting:
	wValue := func(v reflect.Value) {
		if v.Type() == typeFloat16 {
			w("%.*g", precision, v.Interface().(float16.Float16).Float32())
			return
		} else if v.Type() == typeBFloat16 {
			w("%.*g", precision, v.Interface().(bfloat16.BFloat16).Float32())
			return
		}
		switch v.Kind() {
		case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
			w("%d", v.Int())
		case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
			w("%d", v.Uint())
		case reflect.Complex64, reflect.Complex128:
			c := v.Complex()
			w("(%.*g+%.*gi)", precision, real(c), precision, imag(c))
		case reflect.Bool:
			w("%v", v.Bool())
		default:
			w("%.*g", precision, v.Interface())
		}
	}

	// Access the contents of the tensor without copy:
	dims := t.Shape().Dimensions
	t.MustConstFlatData(func(flat any) {
		values := reflect.ValueOf(flat)

		// Print Go type equivalent
		for _, dim := range dims {
			w("[%d]", dim)
		}
		w("%s", values.Type().Elem())
		if len(dims) == 0 {
			// Scalar value.
			w("(")
			wValue(values.Index(0))
			w(")")
			return
		}

		// Recursive function to print elements
		var printElements func(int, int, []int)
		printElements = func(index, indent int, currentShape []int) {
			if len(currentShape) == 1 {
				// One row of data:
				w("{")
				if currentShape[0] > 6 {
					// Apply ellipsis for large arrays
					for i := 0; i < 3; i++ {
						if i > 0 {
							w(", ")
						}
						wValue(values.Index(index + i))
					}
					w(", ..., ")
					for i := currentShape[0] - 3; i < currentShape[0]; i++ {
						if i > currentShape[0]-3 {
							w(", ")
						}
						wValue(values.Index(index + i))
					}

				} else {
					// Print full row:
					for i := 0; i < currentShape[0]; i++ {
						if i > 0 {
							w(", ")
						}
						wValue(values.Index(index + i))
					}
				}
				w("}")
				return
			}

			// Outer axes:
			numRows := 1
			for _, dim := range currentShape[:len(currentShape)-1] {
				numRows *= dim
			}
			stride := 1
			for _, dim := range currentShape[1:] {
				stride *= dim
			}

			w("{")
			if indent == -1 {
				if numRows > 1 {
					// Break the line before outputting data if we are using more than one row.
					w("\n ")
				}
				indent = 1
			}
			indentStr := strings.Repeat(" ", indent)

			if numRows > 6 {
				if len(currentShape) > 2 {
					// Only print first and last element of this outer dimension.
					printElements(index, indent+1, currentShape[1:])
					if currentShape[0] > 1 {
						if currentShape[0] > 2 {
							w(",\n%s...,\n%s", indentStr, indentStr)
						} else {
							w(",\n%s", indentStr)
						}
						printElements(index+(currentShape[0]-1)*stride, indent+1, currentShape[1:])
					}
					w("}")
					return
				}

				// This is the one-before last dimension, first 3 and last 3 rows for this outer dimension.
				firstNRows := min(3, currentShape[0])
				var lastNRows int
				if currentShape[0] <= 6 {
					firstNRows = currentShape[0]
				} else {
					lastNRows = 3
				}
				for ii := 0; ii < firstNRows; ii++ {
					if ii > 0 {
						w(",\n%s", indentStr)
					}
					printElements(index+ii*stride, indent+1, currentShape[1:])
				}
				if lastNRows > 0 {
					w(",\n%s...", indentStr)
					for ii := currentShape[0] - lastNRows; ii < currentShape[0]; ii++ {
						w(",\n%s", indentStr)
						printElements(index+ii*stride, indent+1, currentShape[1:])
					}
				}
				w("}")
				return
			}

			// Print all rows of the outer edge:
			for ii := range currentShape[0] {
				if ii > 0 {
					w(",\n%s", indentStr)
				}
				printElements(index, indent+1, currentShape[1:])
				w("}")
				index += stride
			}
		}
		printElements(0, -1, dims)
	})
	return buf.String()
}

// GoStr converts to string, using a Go-syntax representation that can be copied&pasted back to code.
func (t *Tensor) GoStr() string {
	t.AssertValid()
	if t.Shape().IsZeroSize() {
		// For zero-dimensioned tensors (for some axis), we simply return the shape.
		return t.shape.String()
	}
	value := t.Value()
	if t.IsScalar() {
		return fmt.Sprintf("%s(%v)", t.shape.DType.GoStr(), value)
	}
	return fmt.Sprintf("%s: %s", t.shape, xslices.SliceToGoStr(value))
}
