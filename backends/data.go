package backends

import "github.com/gomlx/gomlx/types/shapes"

// Buffer represents actual data (a tensor) stored in the accelerator that is actually going to execute the graph.
// It's used as input/output of computation execution.
// A Buffer is always associated to a DeviceNum, even if there is only one.
//
// It is opaque from GoMLX perspective, but one of the backend methods take this value as input, and needs
type Buffer any

// DataInterface is the sub-interface defines the API to transfer Buffer to/from accelerators for the backend.
type DataInterface interface {
	// BufferFinalize allows client to inform backend that buffer is no longer needed and associated resources can be
	// freed immediately.
	BufferFinalize(buffer Buffer)

	// BufferShape returns the shape for the buffer.
	BufferShape(buffer Buffer) shapes.Shape

	// BufferDeviceNum returns the deviceNum for the buffer.
	BufferDeviceNum(buffer Buffer) DeviceNum

	// BufferToFlatData transfers the flat values of buffer to the Go flat array.
	// The slice flat must have the exact number of elements required to store the Buffer shape.
	//
	// See also FlatDataToBuffer, BufferShape, and shapes.Shape.Size.
	BufferToFlatData(buffer Buffer, flat any)

	// BufferFromFlatData transfers data from Go given as a flat slice (of the type corresponding to the shape DType)
	// to the deviceNum, and returns the corresponding Buffer.
	BufferFromFlatData(deviceNum DeviceNum, flat any, shape shapes.Shape) Buffer
}
