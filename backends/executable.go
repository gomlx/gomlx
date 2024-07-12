package backends

import (
	"github.com/gomlx/gomlx/types/shapes"
)

// Executable is the API for compiled programs ready to execute.
type Executable interface {
	// Finalize immediately frees resources associated to the executable.
	Finalize()

	// Inputs returns the list of parameters names and shapes, in order created by the Builder.Parameter calls.
	Inputs() (names []string, inputShapes []shapes.Shape)

	// Outputs returns the list of the shapes of the outputs of the computation, in order given to the Builder.Compile call.
	Outputs() (outputShapes []shapes.Shape)

	// Execute the executable on the default device (0). The number and shapes of the inputs must match those returned by Inputs.
	Execute(inputs ...Buffer) []Buffer
}
