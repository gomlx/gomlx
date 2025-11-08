package distributed

import (
	"fmt"
	"strings"

	"github.com/pkg/errors"
)

// DeviceMesh defines the logical topology of a set of devices.
//
// For the initial "SimpleSPMD" implementation, we only support a 1D mesh,
// which represents data parallelism (replicas).
type DeviceMesh struct {
	// axisNames are the names of the mesh axes.
	axisNames []string

	// shape defines the number of devices along each mesh axis.
	shape []int

	// nameToAxis maps axis names to their index.
	nameToAxis map[string]int

	// numDevices is the total number of devices in the mesh.
	numDevices int
}

// NewDeviceMesh creates a new DeviceMesh.
//
// For "SimpleSPMD" shape should be 1D, e.g., NewDeviceMesh([]int{8}, []string{"replica"}).
func NewDeviceMesh(shape []int, axisNames []string) (*DeviceMesh, error) {
	if len(shape) != len(axisNames) {
		return nil, errors.Errorf("shape and axisNames must have the same length, got %d and %d", len(shape), len(axisNames))
	}
	if len(shape) == 0 {
		return nil, errors.New("DeviceMesh shape cannot be empty")
	}

	numDevices := 1
	axisIndices := make(map[string]int, len(shape))
	for i, name := range axisNames {
		if name == "" {
			return nil, errors.Errorf("DeviceMesh axis name at index %d cannot be empty", i)
		}
		if _, found := axisIndices[name]; found {
			return nil, errors.Errorf("DeviceMesh axis name %q is duplicated", name)
		}
		axisIndices[name] = i
		numDevices *= shape[i]
	}

	return &DeviceMesh{
		axisNames:  axisNames,
		shape:      shape,
		nameToAxis: axisIndices,
		numDevices: numDevices,
	}, nil
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
	names := make([]string, len(m.axisNames))
	copy(names, m.axisNames)
	return names
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
		fmt.Fprintf(&sb, "%s: %d", name, m.shape[i])
	}
	sb.WriteString("})")
	return sb.String()
}
