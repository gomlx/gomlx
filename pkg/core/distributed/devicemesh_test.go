package distributed_test

import (
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/backends/notimplemented"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockBackend is a simple mock backend for testing.
type mockBackend struct {
	notimplemented.Backend
	numDevices int
}

func (m *mockBackend) NumDevices() int {
	return m.numDevices
}

func newMockBackend(numDevices int) backends.Backend {
	return &mockBackend{numDevices: numDevices}
}

func TestDeviceMesh(t *testing.T) {
	t.Run("NewDeviceMesh_Valid", func(t *testing.T) {
		backend := newMockBackend(8)

		tests := []struct {
			name      string
			shape     []int
			axisNames []string
			wantRank  int
			wantNum   int
		}{
			{
				name:      "1D mesh",
				shape:     []int{8},
				axisNames: []string{"replica"},
				wantRank:  1,
				wantNum:   8,
			},
			{
				name:      "2D mesh",
				shape:     []int{2, 4},
				axisNames: []string{"x", "y"},
				wantRank:  2,
				wantNum:   8,
			},
			{
				name:      "3D mesh",
				shape:     []int{2, 2, 2},
				axisNames: []string{"x", "y", "z"},
				wantRank:  3,
				wantNum:   8,
			},
			{
				name:      "single device",
				shape:     []int{1},
				axisNames: []string{"replica"},
				wantRank:  1,
				wantNum:   1,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				mesh, err := distributed.NewDeviceMesh(backend, tt.shape, tt.axisNames)
				require.NoError(t, err)
				assert.NotNil(t, mesh)
				assert.Equal(t, tt.wantRank, mesh.Rank())
				assert.Equal(t, tt.wantNum, mesh.NumDevices())
			})
		}
	})

	t.Run("NewDeviceMesh_Errors", func(t *testing.T) {
		backend := newMockBackend(8)

		tests := []struct {
			name      string
			shape     []int
			axisNames []string
			wantErr   string
		}{
			{
				name:      "mismatched lengths",
				shape:     []int{2, 4},
				axisNames: []string{"x"},
				wantErr:   "shape and axisNames must have the same length",
			},
			{
				name:      "empty shape",
				shape:     []int{},
				axisNames: []string{},
				wantErr:   "DeviceMesh shape cannot be empty",
			},
			{
				name:      "empty axis name",
				shape:     []int{4},
				axisNames: []string{""},
				wantErr:   "axis name at index 0 cannot be empty",
			},
			{
				name:      "duplicate axis names",
				shape:     []int{2, 4},
				axisNames: []string{"x", "x"},
				wantErr:   "axis name \"x\" is duplicated",
			},
			{
				name:      "too many devices",
				shape:     []int{16},
				axisNames: []string{"replica"},
				wantErr:   "DeviceMesh has 16 devices, but the backend only has 8 devices",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				mesh, err := distributed.NewDeviceMesh(backend, tt.shape, tt.axisNames)
				require.Error(t, err)
				assert.Nil(t, mesh)
				assert.Contains(t, err.Error(), tt.wantErr)
			})
		}
	})

	t.Run("AxisNames", func(t *testing.T) {
		backend := newMockBackend(8)
		mesh, err := distributed.NewDeviceMesh(backend, []int{2, 4}, []string{"x", "y"})
		require.NoError(t, err)

		axisNames := mesh.AxisNames()
		assert.Equal(t, []string{"x", "y"}, axisNames)

		// Verify it returns a copy
		axisNames[0] = "modified"
		assert.Equal(t, []string{"x", "y"}, mesh.AxisNames())
	})

	t.Run("Shape", func(t *testing.T) {
		backend := newMockBackend(8)
		mesh, err := distributed.NewDeviceMesh(backend, []int{2, 4}, []string{"x", "y"})
		require.NoError(t, err)

		shape := mesh.Shape()
		assert.Equal(t, []int{2, 4}, shape)

		// Verify it returns a copy
		shape[0] = 99
		assert.Equal(t, []int{2, 4}, mesh.Shape())
	})

	t.Run("AxisSize", func(t *testing.T) {
		backend := newMockBackend(8)
		mesh, err := distributed.NewDeviceMesh(backend, []int{2, 4}, []string{"x", "y"})
		require.NoError(t, err)

		tests := []struct {
			name     string
			axisName string
			wantSize int
			wantErr  bool
		}{
			{
				name:     "valid axis x",
				axisName: "x",
				wantSize: 2,
				wantErr:  false,
			},
			{
				name:     "valid axis y",
				axisName: "y",
				wantSize: 4,
				wantErr:  false,
			},
			{
				name:     "non-existent axis",
				axisName: "z",
				wantSize: 0,
				wantErr:  true,
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				size, err := mesh.AxisSize(tt.axisName)
				if tt.wantErr {
					require.Error(t, err)
					assert.Contains(t, err.Error(), "not found")
				} else {
					require.NoError(t, err)
					assert.Equal(t, tt.wantSize, size)
				}
			})
		}
	})

	t.Run("String", func(t *testing.T) {
		backend := newMockBackend(8)

		tests := []struct {
			name      string
			shape     []int
			axisNames []string
			want      string
		}{
			{
				name:      "1D mesh",
				shape:     []int{8},
				axisNames: []string{"replica"},
				want:      "DeviceMesh(shape={replica: 8})",
			},
			{
				name:      "2D mesh",
				shape:     []int{2, 4},
				axisNames: []string{"x", "y"},
				want:      "DeviceMesh(shape={x: 2, y: 4})",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				mesh, err := distributed.NewDeviceMesh(backend, tt.shape, tt.axisNames)
				require.NoError(t, err)
				assert.Equal(t, tt.want, mesh.String())
			})
		}
	})

	t.Run("SetDeviceAssignment_Valid", func(t *testing.T) {
		backend := newMockBackend(8)
		mesh, err := distributed.NewDeviceMesh(backend, []int{4}, []string{"replica"})
		require.NoError(t, err)

		tests := []struct {
			name    string
			devices []backends.DeviceNum
		}{
			{
				name:    "sequential mapping",
				devices: []backends.DeviceNum{0, 1, 2, 3},
			},
			{
				name:    "reverse mapping",
				devices: []backends.DeviceNum{3, 2, 1, 0},
			},
			{
				name:    "custom mapping",
				devices: []backends.DeviceNum{2, 5, 1, 7},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := mesh.SetDeviceAssignment(tt.devices...)
				require.NoError(t, err)

				// Verify mapping is applied correctly
				for i, device := range tt.devices {
					flatIdx, axisIndices, err := mesh.DeviceToMesh(device)
					require.NoError(t, err)
					assert.Equal(t, i, flatIdx)
					assert.Equal(t, []int{i}, axisIndices)
				}
			})
		}
	})

	t.Run("SetDeviceAssignment_Errors", func(t *testing.T) {
		backend := newMockBackend(8)
		mesh, err := distributed.NewDeviceMesh(backend, []int{4}, []string{"replica"})
		require.NoError(t, err)

		tests := []struct {
			name    string
			devices []backends.DeviceNum
			wantErr string
		}{
			{
				name:    "wrong number of devices",
				devices: []backends.DeviceNum{0, 1, 2},
				wantErr: "devices must have 4 elements",
			},
			{
				name:    "duplicate device",
				devices: []backends.DeviceNum{0, 1, 1, 3},
				wantErr: "physical device #1 is duplicated",
			},
			{
				name:    "device out of range (negative)",
				devices: []backends.DeviceNum{0, 1, -1, 3},
				wantErr: "device -1 is out of range",
			},
			{
				name:    "device out of range (too large)",
				devices: []backends.DeviceNum{0, 1, 2, 8},
				wantErr: "device 8 is out of range",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := mesh.SetDeviceAssignment(tt.devices...)
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
			})
		}
	})

	t.Run("DeviceToMesh_1D", func(t *testing.T) {
		backend := newMockBackend(8)
		mesh, err := distributed.NewDeviceMesh(backend, []int{4}, []string{"replica"})
		require.NoError(t, err)

		for i := 0; i < 4; i++ {
			flatIdx, axisIndices, err := mesh.DeviceToMesh(backends.DeviceNum(i))
			require.NoError(t, err)
			assert.Equal(t, i, flatIdx)
			assert.Equal(t, []int{i}, axisIndices)
		}
	})

	t.Run("DeviceToMesh_2D", func(t *testing.T) {
		backend := newMockBackend(8)
		mesh, err := distributed.NewDeviceMesh(backend, []int{2, 4}, []string{"x", "y"})
		require.NoError(t, err)

		tests := []struct {
			device      backends.DeviceNum
			wantFlat    int
			wantIndices []int
		}{
			{device: 0, wantFlat: 0, wantIndices: []int{0, 0}},
			{device: 1, wantFlat: 1, wantIndices: []int{0, 1}},
			{device: 2, wantFlat: 2, wantIndices: []int{0, 2}},
			{device: 3, wantFlat: 3, wantIndices: []int{0, 3}},
			{device: 4, wantFlat: 4, wantIndices: []int{1, 0}},
			{device: 5, wantFlat: 5, wantIndices: []int{1, 1}},
			{device: 6, wantFlat: 6, wantIndices: []int{1, 2}},
			{device: 7, wantFlat: 7, wantIndices: []int{1, 3}},
		}

		for _, tt := range tests {
			t.Run(string(rune(tt.device)), func(t *testing.T) {
				flatIdx, axisIndices, err := mesh.DeviceToMesh(tt.device)
				require.NoError(t, err)
				assert.Equal(t, tt.wantFlat, flatIdx)
				assert.Equal(t, tt.wantIndices, axisIndices)
			})
		}
	})

	t.Run("DeviceToMesh_3D", func(t *testing.T) {
		backend := newMockBackend(8)
		mesh, err := distributed.NewDeviceMesh(backend, []int{2, 2, 2}, []string{"x", "y", "z"})
		require.NoError(t, err)

		tests := []struct {
			device      backends.DeviceNum
			wantFlat    int
			wantIndices []int
		}{
			{device: 0, wantFlat: 0, wantIndices: []int{0, 0, 0}},
			{device: 1, wantFlat: 1, wantIndices: []int{0, 0, 1}},
			{device: 2, wantFlat: 2, wantIndices: []int{0, 1, 0}},
			{device: 3, wantFlat: 3, wantIndices: []int{0, 1, 1}},
			{device: 4, wantFlat: 4, wantIndices: []int{1, 0, 0}},
			{device: 5, wantFlat: 5, wantIndices: []int{1, 0, 1}},
			{device: 6, wantFlat: 6, wantIndices: []int{1, 1, 0}},
			{device: 7, wantFlat: 7, wantIndices: []int{1, 1, 1}},
		}

		for _, tt := range tests {
			t.Run(string(rune(tt.device)), func(t *testing.T) {
				flatIdx, axisIndices, err := mesh.DeviceToMesh(tt.device)
				require.NoError(t, err)
				assert.Equal(t, tt.wantFlat, flatIdx)
				assert.Equal(t, tt.wantIndices, axisIndices)
			})
		}
	})

	t.Run("DeviceToMesh_WithCustomMapping", func(t *testing.T) {
		backend := newMockBackend(8)
		mesh, err := distributed.NewDeviceMesh(backend, []int{4}, []string{"replica"})
		require.NoError(t, err)

		// Set custom mapping: devices [7, 5, 3, 1]
		err = mesh.SetDeviceAssignment(7, 5, 3, 1)
		require.NoError(t, err)

		tests := []struct {
			device      backends.DeviceNum
			wantFlat    int
			wantIndices []int
		}{
			{device: 7, wantFlat: 0, wantIndices: []int{0}},
			{device: 5, wantFlat: 1, wantIndices: []int{1}},
			{device: 3, wantFlat: 2, wantIndices: []int{2}},
			{device: 1, wantFlat: 3, wantIndices: []int{3}},
		}

		for _, tt := range tests {
			t.Run(string(rune(tt.device)), func(t *testing.T) {
				flatIdx, axisIndices, err := mesh.DeviceToMesh(tt.device)
				require.NoError(t, err)
				assert.Equal(t, tt.wantFlat, flatIdx)
				assert.Equal(t, tt.wantIndices, axisIndices)
			})
		}

		// Devices not in the mesh should error
		_, _, err = mesh.DeviceToMesh(0)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "not part of the mesh")
	})

	t.Run("DeviceToMesh_NotInMesh", func(t *testing.T) {
		backend := newMockBackend(8)
		mesh, err := distributed.NewDeviceMesh(backend, []int{4}, []string{"replica"})
		require.NoError(t, err)

		// Device 5 is not in the mesh (only 0-3 are used)
		_, _, err = mesh.DeviceToMesh(5)
		require.Error(t, err)
		assert.Contains(t, err.Error(), "physical device 5 is not part of the mesh")
	})

	t.Run("ComputeReplicaGroups", func(t *testing.T) {
		t.Run("2D mesh batch groups", func(t *testing.T) {
			backend := newMockBackend(8)
			mesh, err := distributed.NewDeviceMesh(backend, []int{2, 2}, []string{"batch", "data"})
			require.NoError(t, err)

			// Example from comments: m.ComputeReplicaGroups([]string{"batch"}) -> [][]int{{0, 2}, {1, 3}}
			groups, err := mesh.ComputeReplicaGroups([]string{"batch"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 2}, {1, 3}}, groups)
		})

		t.Run("2D mesh data groups", func(t *testing.T) {
			backend := newMockBackend(8)
			mesh, err := distributed.NewDeviceMesh(backend, []int{2, 2}, []string{"batch", "data"})
			require.NoError(t, err)

			// Example from comments: m.ComputeReplicaGroups([]string{"data"}) -> [][]int{{0, 1}, {2, 3}}
			groups, err := mesh.ComputeReplicaGroups([]string{"data"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 1}, {2, 3}}, groups)
		})

		t.Run("2D mesh global groups", func(t *testing.T) {
			backend := newMockBackend(8)
			mesh, err := distributed.NewDeviceMesh(backend, []int{2, 2}, []string{"batch", "data"})
			require.NoError(t, err)

			// Example from comments: m.ComputeReplicaGroups([]string{"batch", "data"}) -> [][]int{{0, 1, 2, 3}}
			groups, err := mesh.ComputeReplicaGroups([]string{"batch", "data"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 1, 2, 3}}, groups)
		})

		t.Run("1D mesh", func(t *testing.T) {
			backend := newMockBackend(8)
			mesh, err := distributed.NewDeviceMesh(backend, []int{4}, []string{"replica"})
			require.NoError(t, err)

			groups, err := mesh.ComputeReplicaGroups([]string{"replica"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 1, 2, 3}}, groups)
		})

		t.Run("3D mesh single axis", func(t *testing.T) {
			backend := newMockBackend(8)
			mesh, err := distributed.NewDeviceMesh(backend, []int{2, 2, 2}, []string{"x", "y", "z"})
			require.NoError(t, err)

			// Groups along x axis: should split by y and z
			groups, err := mesh.ComputeReplicaGroups([]string{"x"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 4}, {1, 5}, {2, 6}, {3, 7}}, groups)
		})

		t.Run("3D mesh two axes", func(t *testing.T) {
			backend := newMockBackend(8)
			mesh, err := distributed.NewDeviceMesh(backend, []int{2, 2, 2}, []string{"x", "y", "z"})
			require.NoError(t, err)

			// Groups along x and y axes: should split by z
			groups, err := mesh.ComputeReplicaGroups([]string{"x", "y"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 2, 4, 6}, {1, 3, 5, 7}}, groups)
		})

		t.Run("empty axes list", func(t *testing.T) {
			backend := newMockBackend(8)
			mesh, err := distributed.NewDeviceMesh(backend, []int{2, 2}, []string{"batch", "data"})
			require.NoError(t, err)

			// Empty axes list: each device is its own group
			groups, err := mesh.ComputeReplicaGroups([]string{})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0}, {1}, {2}, {3}}, groups)
		})

		t.Run("non-existent axis", func(t *testing.T) {
			backend := newMockBackend(8)
			mesh, err := distributed.NewDeviceMesh(backend, []int{2, 2}, []string{"batch", "data"})
			require.NoError(t, err)

			// Non-existent axis should return an error.
			_, err = mesh.ComputeReplicaGroups([]string{"nonexistent"})
			require.Error(t, err)
		})
	})
}
