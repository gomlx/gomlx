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
// For the initial "SPMD" implementation, we only support a 1D mesh,
// which represents data parallelism (replicas).
type DeviceMesh struct {
	backend backends.Backend

	// axisNames are the names of the mesh axes.
	axisNames []string

	// axisSizes defines the number of devices along each mesh axis.
	axisSizes []int

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
// - axisSizes: defines the number of devices along each mesh axis, one value per axis.
// - axisNames: the names of the mesh axes. One value per axis.
//
// For the "SPMD" Strategy the axisSizes should be 1D, e.g., NewDeviceMesh([]int{8}, []string{"replica"}).
//
// The default mapping of concrete devices to the mesh is sequential, starting from 0.
// For non-symmetric devices, where connection speed among the devices matter, a custom mapping can be provided
// with the DeviceMesh.WithDeviceMapping() method.
func NewDeviceMesh(backend backends.Backend, axisSizes []int, axisNames []string) (*DeviceMesh, error) {
	if len(axisSizes) != len(axisNames) {
		return nil, errors.Errorf(
			"axisSizes and axisNames must have the same length, got %d and %d",
			len(axisSizes), len(axisNames))
	}
	if len(axisSizes) == 0 {
		return nil, errors.New("DeviceMesh axisSizes cannot be empty")
	}

	numDevices := 1
	nameToAxis := make(map[string]int, len(axisSizes))
	for i, name := range axisNames {
		if name == "" {
			return nil, errors.Errorf("DeviceMesh axis name at index %d cannot be empty", i)
		}
		if _, found := nameToAxis[name]; found {
			return nil, errors.Errorf("DeviceMesh axis name %q is duplicated", name)
		}
		nameToAxis[name] = i
		numDevices *= axisSizes[i]
	}
	if numDevices > backend.NumDevices() {
		return nil, errors.Errorf("DeviceMesh has %d devices, but the backend only has %d devices", numDevices, backend.NumDevices())
	}

	m := &DeviceMesh{
		backend:       backend,
		axisNames:     axisNames,
		axisSizes:     axisSizes,
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
	return len(m.axisSizes)
}

// AxisNames returns a copy of the mesh's axis names.
func (m *DeviceMesh) AxisNames() []string {
	return slices.Clone(m.axisNames)
}

// Shape returns a copy of the mesh's axisSizes.
func (m *DeviceMesh) Shape() []int {
	shape := make([]int, len(m.axisSizes))
	copy(shape, m.axisSizes)
	return shape
}

// AxisSize returns the number of devices along the given mesh axis.
func (m *DeviceMesh) AxisSize(axisName string) (int, error) {
	idx, found := m.nameToAxis[axisName]
	if !found {
		return 0, errors.Errorf("mesh axis %q not found", axisName)
	}
	return m.axisSizes[idx], nil
}

// String implements the fmt.Stringer interface.
func (m *DeviceMesh) String() string {
	var sb strings.Builder
	sb.WriteString("DeviceMesh(axisSizes={")
	for i, name := range m.axisNames {
		if i > 0 {
			sb.WriteString(", ")
		}
		_, _ = fmt.Fprintf(&sb, "%s: %d", name, m.axisSizes[i])
	}
	sb.WriteString("})")
	return sb.String()
}

// SetDeviceAssignment sets the assignment of concrete devices to the mesh.
//
// It returns an error if devicesInMesh has invalid device numbers or len(devices) != NumDevices().
func (m *DeviceMesh) SetDeviceAssignment(devices ...backends.DeviceNum) error {
	if len(devices) != m.numDevices {
		return errors.Errorf("devices must have %d elements, got %d", m.numDevices, len(devices))
	}
	numPhysicalDevices := m.backend.NumDevices()
	seen := sets.Make[backends.DeviceNum](m.numDevices)
	for _, device := range devices {
		if seen.Has(device) {
			return errors.Errorf("physical device #%d is duplicated in mapping", device)
		}
		seen.Insert(device)
		if device < 0 || int(device) >= numPhysicalDevices {
			return errors.Errorf("device %d is out of range, backend only has %d devices", device, numPhysicalDevices)
		}
	}
	copy(m.devicesInMesh, devices)
	m.buildPhysicalDeviceMapping()
	if len(m.physicalDeviceMapping) != m.numDevices {
		return errors.Errorf("provided devicesIn: physicalDeviceMapping has %d elements, expected %d", len(m.physicalDeviceMapping), m.numDevices)
	}
	return nil
}

// DeviceAssignment returns the list of devices in the mesh, in the order they appear in the mesh.
func (m *DeviceMesh) DeviceAssignment() []backends.DeviceNum {
	return slices.Clone(m.devicesInMesh)
}

// DeviceToMesh return the indices (flat and per-axis) assigned to the given physicalDevice.
func (m *DeviceMesh) DeviceToMesh(physicalDevice backends.DeviceNum) (flatIdx int, axisIndices []int, err error) {
	var ok bool
	flatIdx, ok = m.physicalDeviceMapping[physicalDevice]
	if !ok {
		return 0, nil, errors.Errorf("physical device %d is not part of the mesh", physicalDevice)
	}

	// Convert flat index to per-axis indices
	axisIndices = make([]int, len(m.axisSizes))
	remaining := flatIdx
	for i := len(m.axisSizes) - 1; i >= 0; i-- {
		axisIndices[i] = remaining % m.axisSizes[i]
		remaining /= m.axisSizes[i]
	}
	return flatIdx, axisIndices, nil
}

// ComputeReplicaGroups returns the replica groups participating in some collective (distributed) operation given the
// axes along which the operation is performed.
//
// Each replica group (a []int) includes the device indices (from the DeviceAssignment) for the axes specified.
// The other axes will be split into different replica groups.
//
// Example:
//
//		m := NewDeviceMesh([]int{2, 2}, []string{"batch", "data"})
//		batchGroups, _ := m.ComputeReplicaGroups([]string{"batch"})  // -> [][]int{{0, 2}, {1, 3}}
//		dataGroups, _ := m.ComputeReplicaGroups([]string{"data"})    // -> [][]int{{0, 1}, {2, 3}}
//	 globalGroups, _ := m.ComputeReplicaGroups([]string{"batch", "data"})  // -> [][]int{{0, 1, 2, 3}}
func (m *DeviceMesh) ComputeReplicaGroups(axes []string) ([][]int, error) {
	// Find indices of the specified axes
	axisIndices := make([]int, 0, len(axes))
	axisSet := sets.Make[int](len(axes))
	for _, axis := range axes {
		if idx, found := m.nameToAxis[axis]; found {
			if axisSet.Has(idx) {
				return nil, errors.Errorf("axis %q is duplicated: each axis can only appear once", axis)
			}
			axisIndices = append(axisIndices, idx)
			axisSet.Insert(idx)
		} else {
			return nil, errors.Errorf("axis %q not found in mesh", axis)
		}
	}

	// Create indices for each axis dimension
	nonAxisIndices := make([]int, 0, len(m.axisSizes)-len(axisIndices))
	for i := range m.axisSizes {
		if !slices.Contains(axisIndices, i) {
			nonAxisIndices = append(nonAxisIndices, i)
		}
	}

	// Calculate the size of each group and number of groups
	groupSize := 1
	for _, idx := range axisIndices {
		groupSize *= m.axisSizes[idx]
	}
	numGroups := m.numDevices / groupSize

	// Initialize the result
	groups := make([][]int, numGroups)
	for i := range groups {
		groups[i] = make([]int, groupSize)
	}

	// Fill in the groups
	for flatIdx := 0; flatIdx < m.numDevices; flatIdx++ {
		// Convert flat index to per-axis indices
		indices := make([]int, len(m.axisSizes))
		remaining := flatIdx
		for i := len(m.axisSizes) - 1; i >= 0; i-- {
			indices[i] = remaining % m.axisSizes[i]
			remaining /= m.axisSizes[i]
		}

		// Calculate group index from non-axis indices
		groupIdx := 0
		multiplier := 1
		for i := len(nonAxisIndices) - 1; i >= 0; i-- {
			axisIdx := nonAxisIndices[i]
			groupIdx += indices[axisIdx] * multiplier
			multiplier *= m.axisSizes[axisIdx]
		}

		// Calculate position within group from axis indices
		posInGroup := 0
		multiplier = 1
		for i := len(axisIndices) - 1; i >= 0; i-- {
			axisIdx := axisIndices[i]
			posInGroup += indices[axisIdx] * multiplier
			multiplier *= m.axisSizes[axisIdx]
		}

		groups[groupIdx][posInGroup] = flatIdx
	}

	return groups, nil
}
