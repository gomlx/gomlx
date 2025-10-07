// Package encoding defines some of the types used during the encoding and decoding process of saved models.
//
// This is only used for tools handling the saved models themselves.
package encoding

import (
	"github.com/gomlx/gomlx/pkg/core/shapes"
)

const (
	// Version1 encoding format. The only one for now.
	Version1 = "model.v1"
)

type Header struct {
	Version   string
	Variables []EncodedVariable
}

type EncodedVariable struct {
	Name, Path string
	Shape      shapes.Shape
	Trainable  bool

	// HasValue is set to true if the variable's value was encoded during the saving process.
	HasValue bool
}

// VarPath is how the variable is encoded in the .jsonl file, with just the path (the key) to the variable.
type VarPath struct {
	Path string
}
