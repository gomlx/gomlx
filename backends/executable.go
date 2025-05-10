package backends

import (
	"github.com/gomlx/gomlx/types/shapes"
)

// Executable is the API for compiled programs ready to execute.
type Executable interface {
	// Finalize immediately frees resources associated to the executable.
	Finalize()

	// Inputs returns the parameters' names and shapes, in order created by the Builder.Parameter calls.
	Inputs() (names []string, inputShapes []shapes.Shape)

	// Outputs returns the computation's output shapes, in the order given to the Builder.Compile call.
	Outputs() (outputShapes []shapes.Shape)

	// Execute the executable on the default device (0).
	// The number and shapes of the inputs must match those returned by Inputs.
	//
	// The inputs marked in donate will become invalid after use.
	// This is useful if the input buffer is no longer needed or if updating a variable
	// so its Buffer space can be reused as an output Buffer.
	//
	// Donated buffers are no longer valid after the call.
	// If donate is nil, it is assumed to be false for all buffers, and no buffer is donated.
	Execute(inputs []Buffer, donate []bool) ([]Buffer, error)
}
