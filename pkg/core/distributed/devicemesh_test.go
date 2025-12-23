package distributed_test

import (
	"testing"

	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestDeviceMesh(t *testing.T) {
	t.Run("NewDeviceMesh_Valid", func(t *testing.T) {
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
				mesh, err := distributed.NewDeviceMesh(tt.shape, tt.axisNames)
				require.NoError(t, err)
				assert.NotNil(t, mesh)
				assert.Equal(t, tt.wantRank, mesh.Rank())
				assert.Equal(t, tt.wantNum, mesh.NumDevices())
			})
		}
	})

	t.Run("NewDeviceMesh_Errors", func(t *testing.T) {
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
				wantErr:   "axesSizes and axesNames must have the same length",
			},
			{
				name:      "empty axesSizes",
				shape:     []int{},
				axisNames: []string{},
				wantErr:   "DeviceMesh axesSizes cannot be empty",
			},
			{
				name:      "empty axis name",
				shape:     []int{4},
				axisNames: []string{""},
				wantErr:   "is not a valid identifier",
			},
			{
				name:      "duplicate axis names",
				shape:     []int{2, 4},
				axisNames: []string{"x", "x"},
				wantErr:   "axis name \"x\" is duplicated",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				mesh, err := distributed.NewDeviceMesh(tt.shape, tt.axisNames)
				require.Error(t, err)
				assert.Nil(t, mesh)
				assert.Contains(t, err.Error(), tt.wantErr)
			})
		}
	})

	t.Run("AxesNames", func(t *testing.T) {
		mesh, err := distributed.NewDeviceMesh([]int{2, 4}, []string{"x", "y"})
		require.NoError(t, err)

		axisNames := mesh.AxesNames()
		assert.Equal(t, []string{"x", "y"}, axisNames)

		// Verify it returns a copy
		axisNames[0] = "modified"
		assert.Equal(t, []string{"x", "y"}, mesh.AxesNames())
	})

	t.Run("Shape", func(t *testing.T) {
		mesh, err := distributed.NewDeviceMesh([]int{2, 4}, []string{"x", "y"})
		require.NoError(t, err)

		axesSizes := mesh.AxesSizes()
		assert.Equal(t, []int{2, 4}, axesSizes)

		// Verify it returns a copy
		axesSizes[0] = 99
		assert.Equal(t, []int{2, 4}, mesh.AxesSizes())
	})

	t.Run("AxisSize", func(t *testing.T) {
		mesh, err := distributed.NewDeviceMesh([]int{2, 4}, []string{"x", "y"})
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
				want:      "DeviceMesh(axesSizes={replica: 8})",
			},
			{
				name:      "2D mesh",
				shape:     []int{2, 4},
				axisNames: []string{"x", "y"},
				want:      "DeviceMesh(axesSizes={x: 2, y: 4})",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				mesh, err := distributed.NewDeviceMesh(tt.shape, tt.axisNames)
				require.NoError(t, err)
				assert.Equal(t, tt.want, mesh.String())
			})
		}
	})

	t.Run("SetDeviceAssignment_Valid", func(t *testing.T) {
		mesh, err := distributed.NewDeviceMesh([]int{4}, []string{"replica"})
		require.NoError(t, err)

		tests := []struct {
			name    string
			devices []int
		}{
			{
				name:    "sequential mapping",
				devices: []int{0, 1, 2, 3},
			},
			{
				name:    "reverse mapping",
				devices: []int{3, 2, 1, 0},
			},
			{
				name:    "custom mapping",
				devices: []int{2, 1, 3, 0},
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := mesh.SetLogicalDeviceAssignment(tt.devices...)
				require.NoErrorf(t, err, "failed test %q", tt.name)
			})
		}
	})

	t.Run("SetDeviceAssignment_Errors", func(t *testing.T) {
		mesh, err := distributed.NewDeviceMesh([]int{4}, []string{"replica"})
		require.NoError(t, err)

		tests := []struct {
			name    string
			devices []int
			wantErr string
		}{
			{
				name:    "wrong number of devices",
				devices: []int{0, 1, 2},
				wantErr: "devices must have 4 elements",
			},
			{
				name:    "duplicate device",
				devices: []int{0, 1, 1, 3},
				wantErr: "physical device #1 is duplicated",
			},
			{
				name:    "device out of range (negative)",
				devices: []int{0, 1, -1, 3},
				wantErr: "devices must be between 0 and 3",
			},
		}

		for _, tt := range tests {
			t.Run(tt.name, func(t *testing.T) {
				err := mesh.SetLogicalDeviceAssignment(tt.devices...)
				require.Error(t, err)
				assert.Contains(t, err.Error(), tt.wantErr)
			})
		}
	})

	t.Run("DeviceToMesh_2D", func(t *testing.T) {
		mesh, err := distributed.NewDeviceMesh([]int{2, 4}, []string{"x", "y"})
		require.NoError(t, err)
		require.Equal(t, 8, mesh.NumDevices())
	})

	t.Run("DeviceToMesh_3D", func(t *testing.T) {
		mesh, err := distributed.NewDeviceMesh([]int{2, 2, 2}, []string{"x", "y", "z"})
		require.NoError(t, err)
		require.Equal(t, 8, mesh.NumDevices())
	})

	t.Run("DeviceToMesh_WithCustomMapping", func(t *testing.T) {
		mesh, err := distributed.NewDeviceMesh([]int{4}, []string{"replica"})
		require.NoError(t, err)
		err = mesh.SetLogicalDeviceAssignment(3, 2, 1, 0)
		require.NoError(t, err)
		require.Equal(t, 4, mesh.NumDevices())
		err = mesh.SetLogicalDeviceAssignment(4, 2, 1, 0)
		require.Error(t, err)
	})

	t.Run("ComputeReplicaGroups", func(t *testing.T) {
		t.Run("2D mesh batch groups", func(t *testing.T) {
			mesh, err := distributed.NewDeviceMesh([]int{2, 2}, []string{"batch", "data"})
			require.NoError(t, err)

			// Example from comments: m.ComputeReplicaGroups([]string{"batch"}) -> [][]int{{0, 2}, {1, 3}}
			groups, err := mesh.ComputeReplicaGroups([]string{"batch"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 2}, {1, 3}}, groups)
		})

		t.Run("2D mesh data groups", func(t *testing.T) {
			mesh, err := distributed.NewDeviceMesh([]int{2, 2}, []string{"batch", "data"})
			require.NoError(t, err)

			// Example from comments: m.ComputeReplicaGroups([]string{"data"}) -> [][]int{{0, 1}, {2, 3}}
			groups, err := mesh.ComputeReplicaGroups([]string{"data"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 1}, {2, 3}}, groups)
		})

		t.Run("2D mesh global groups", func(t *testing.T) {
			mesh, err := distributed.NewDeviceMesh([]int{2, 2}, []string{"batch", "data"})
			require.NoError(t, err)

			// Example from comments: m.ComputeReplicaGroups([]string{"batch", "data"}) -> [][]int{{0, 1, 2, 3}}
			groups, err := mesh.ComputeReplicaGroups([]string{"batch", "data"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 1, 2, 3}}, groups)
		})

		t.Run("1D mesh", func(t *testing.T) {
			mesh, err := distributed.NewDeviceMesh([]int{4}, []string{"replica"})
			require.NoError(t, err)

			groups, err := mesh.ComputeReplicaGroups([]string{"replica"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 1, 2, 3}}, groups)
		})

		t.Run("3D mesh single axis", func(t *testing.T) {
			mesh, err := distributed.NewDeviceMesh([]int{2, 2, 2}, []string{"x", "y", "z"})
			require.NoError(t, err)

			// Groups along x axis: should split by y and z
			groups, err := mesh.ComputeReplicaGroups([]string{"x"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 4}, {1, 5}, {2, 6}, {3, 7}}, groups)
		})

		t.Run("3D mesh two axes", func(t *testing.T) {
			mesh, err := distributed.NewDeviceMesh([]int{2, 2, 2}, []string{"x", "y", "z"})
			require.NoError(t, err)

			// Groups along x and y axes: should split by z
			groups, err := mesh.ComputeReplicaGroups([]string{"x", "y"})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0, 2, 4, 6}, {1, 3, 5, 7}}, groups)
		})

		t.Run("empty axes list", func(t *testing.T) {
			mesh, err := distributed.NewDeviceMesh([]int{2, 2}, []string{"batch", "data"})
			require.NoError(t, err)

			// Empty axes list: each device is its own group
			groups, err := mesh.ComputeReplicaGroups([]string{})
			require.NoError(t, err)
			assert.Equal(t, [][]int{{0}, {1}, {2}, {3}}, groups)
		})

		t.Run("non-existent axis", func(t *testing.T) {
			mesh, err := distributed.NewDeviceMesh([]int{2, 2}, []string{"batch", "data"})
			require.NoError(t, err)

			// A non-existent axis should return an error.
			_, err = mesh.ComputeReplicaGroups([]string{"nonexistent"})
			require.Error(t, err)
		})
	})
}
