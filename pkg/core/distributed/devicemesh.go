package distributed

import (
	"fmt"
	"slices"
	"strings"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

// DeviceMesh defines the logical topology of a set of devices on a backend.
//
// For the initial "SimpleSPMD" implementation, we only support a 1D mesh,
// which represents data parallelism (replicas).
type DeviceMesh struct {
	backend backends.Backend

	// axisNames are the names of the mesh axes.
	axisNames []string

	// shape defines the number of devices along each mesh axis.
	shape []int

	// nameToAxis maps axis names to their index.
	nameToAxis map[string]int

	// numDevices is the total number of devices in the mesh.
	numDevices int

	// devicesInMesh is the list of devices in the mesh, in the order they appear in the mesh.
	devicesInMesh []backends.DeviceNum

	// physicalDeviceMapping is the mapping of concrete devices to the flat index in the mesh.
	physicalDeviceMapping map[backends.DeviceNum]int
}

// NewDeviceMesh creates a new logical topology of a set of devices.
//
// - shape: defines the number of devices along each mesh axis, one value per axis.
// - axisNames: the names of the mesh axes. One value per axis.
//
// For the "SimpleSPMD" Strategy the shape should be 1D, e.g., NewDeviceMesh([]int{8}, []string{"replica"}).
//
// The default mapping of concrete devices to the mesh is sequential, starting from 0.
// For non-symmetric devices, where connection speed among the devices matter, a custom mapping can be provided
// with the DeviceMesh.WithDeviceMapping() method.
func NewDeviceMesh(backend backends.Backend, shape []int, axisNames []string) (*DeviceMesh, error) {
	if len(shape) != len(axisNames) {
		return nil, errors.Errorf("shape and axisNames must have the same length, got %d and %d", len(shape), len(axisNames))
	}
	if len(shape) == 0 {
		return nil, errors.New("DeviceMesh shape cannot be empty")
	}

	numDevices := 1
	nameToAxis := make(map[string]int, len(shape))
	for i, name := range axisNames {
		if name == "" {
			return nil, errors.Errorf("DeviceMesh axis name at index %d cannot be empty", i)
		}
		if _, found := nameToAxis[name]; found {
			return nil, errors.Errorf("DeviceMesh axis name %q is duplicated", name)
		}
		nameToAxis[name] = i
		numDevices *= shape[i]
	}
	if numDevices > backend.NumDevices() {
		return nil, errors.Errorf("DeviceMesh has %d devices, but the backend only has %d devices", numDevices, backend.NumDevices())
	}

	m := &DeviceMesh{
		backend:       backend,
		axisNames:     axisNames,
		shape:         shape,
		nameToAxis:    nameToAxis,
		numDevices:    numDevices,
		devicesInMesh: xslices.Iota(backends.DeviceNum(0), numDevices),
	}
	m.buildPhysicalDeviceMapping()
	return m, nil
}

func (m *DeviceMesh) buildPhysicalDeviceMapping() {
	m.physicalDeviceMapping = make(map[backends.DeviceNum]int, m.numDevices)
	for i, device := range m.devicesInMesh {
		m.physicalDeviceMapping[device] = i
	}
}

// NumDevices returns the total number of devices in the mesh.
func (m *DeviceMesh) NumDevices() int {
	return m.numDevices
}

// Rank returns the number of axes in the mesh.
func (m *DeviceMesh) Rank() int {
	return len(m.shape)
}

// AxisNames returns a copy of the mesh's axis names.
func (m *DeviceMesh) AxisNames() []string {
	return slices.Clone(m.axisNames)
}

// Shape returns a copy of the mesh's shape.
func (m *DeviceMesh) Shape() []int {
	shape := make([]int, len(m.shape))
	copy(shape, m.shape)
	return shape
}

// AxisSize returns the number of devices along the given mesh axis.
func (m *DeviceMesh) AxisSize(axisName string) (int, error) {
	idx, found := m.nameToAxis[axisName]
	if !found {
		return 0, errors.Errorf("mesh axis %q not found", axisName)
	}
	return m.shape[idx], nil
}

// String implements the fmt.Stringer interface.
func (m *DeviceMesh) String() string {
	var sb strings.Builder
	sb.WriteString("DeviceMesh(shape={")
	for i, name := range m.axisNames {
		if i > 0 {
			sb.WriteString(", ")
		}
		_, _ = fmt.Fprintf(&sb, "%s: %d", name, m.shape[i])
	}
	sb.WriteString("})")
	return sb.String()
}

// SetDeviceMapping sets the mapping of concrete devices to the mesh.
//
// It returns an error if devicesInMesh has invalid device numbers or len(devicesInMessh) != NumDevices().
func (m *DeviceMesh) SetDeviceMapping(devicesInMesh ...backends.DeviceNum) error {
	if len(devicesInMesh) != m.numDevices {
		return errors.Errorf("devicesInMesh must have %d elements, got %d", m.numDevices, len(devicesInMesh))
	}
	numPhysicalDevices := m.backend.NumDevices()
	seen := sets.Make[backends.DeviceNum](m.numDevices)
	for _, device := range devicesInMesh {
		if seen.Has(device) {
			return errors.Errorf("physical device #%d is duplicated in mapping", device)
		}
		seen.Insert(device)
		if device < 0 || int(device) >= numPhysicalDevices {
			return errors.Errorf("device %d is out of range, backend only has %d devices", device, numPhysicalDevices)
		}
	}
	copy(m.devicesInMesh, devicesInMesh)
	m.buildPhysicalDeviceMapping()
	if len(m.physicalDeviceMapping) != m.numDevices {
		return errors.Errorf("provided devicesIn: physicalDeviceMapping has %d elements, expected %d", len(m.physicalDeviceMapping), m.numDevices)
	}
	return nil
}

// DeviceToMesh return the indices (flat and per-axis) assigned to the given physicalDevice.
func (m *DeviceMesh) DeviceToMesh(physicalDevice backends.DeviceNum) (flatIdx int, axisIndices []int, err error) {
	var ok bool
	flatIdx, ok = m.physicalDeviceMapping[physicalDevice]
	if !ok {
		return 0, nil, errors.Errorf("physical device %d is not part of the mesh", physicalDevice)
	}

	// Convert flat index to per-axis indices
	axisIndices = make([]int, len(m.shape))
	remaining := flatIdx
	for i := len(m.shape) - 1; i >= 0; i-- {
		axisIndices[i] = remaining % m.shape[i]
		remaining /= m.shape[i]
	}
	return flatIdx, axisIndices, nil
}
