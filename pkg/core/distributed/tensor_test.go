package distributed_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/internal/must"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	_ "github.com/gomlx/gomlx/backends/default"
)

var (
	backendOnce   sync.Once
	cachedBackend backends.Backend
)

// BuildTestBackend and sets backends.DefaultConfig to "xla:cpu" -- it can be overwritten by GOMLX_BACKEND environment variable.
func BuildTestBackend() backends.Backend {
	backends.DefaultConfig = "xla:cpu"
	backendOnce.Do(func() {
		cachedBackend = backends.MustNew()
		fmt.Printf("Backend: %s\n", cachedBackend.Description())
	})
	return cachedBackend
}

func TestTensor(t *testing.T) {
	backend := BuildTestBackend()
	if backend.NumDevices() < 2 {
		t.Skipf("Skipping distributed tests: backend only has %d device.", backend.NumDevices())
	}

	// ShardTensor() tests.
	t.Run("ShardTensor()", func(t *testing.T) {
		t.Run("sharded", func(t *testing.T) {
			// Create a new device mesh.
			mesh, err := distributed.NewDeviceMesh([]int{2}, []string{"shards"})
			require.NoError(t, err)

			// Create a new tensor.
			tensor := tensors.FromValue([][]int32{{1, 2, 3, 4}, {5, 6, 7, 8}})

			// Shard the tensor.
			spec, err := distributed.NewShardingSpec(mesh, distributed.AxisSpec{"shards"}, distributed.ReplicatedAxis)
			require.NoError(t, err)
			distTensor, err := distributed.ShardTensor(spec, tensor)
			require.NoError(t, err)

			// Check the logical shape.
			assert.Equal(t, shapes.Make(dtypes.Int32, 2, 4), distTensor.Shape())

			// Check the shard shape.
			assert.Equal(t, shapes.Make(dtypes.Int32, 1, 4), distTensor.ShardShape())

			// Check the number of shards.
			assert.Len(t, distTensor.Shards(), 2)

			// Check the values of the shards.
			assert.Equal(t, [][]int32{{1, 2, 3, 4}}, distTensor.Shards()[0].Value())
			assert.Equal(t, [][]int32{{5, 6, 7, 8}}, distTensor.Shards()[1].Value())
		})

		// Replicate is the same as sharding, with no sharding axes.
		t.Run("replicated", func(t *testing.T) {
			mesh, err := distributed.NewDeviceMesh([]int{2}, []string{"replicas"})
			require.NoError(t, err)
			tensor := tensors.FromValue([][]float32{{1, 2, 3, 4}})
			spec := distributed.NewReplicatedShardingSpec(mesh)
			distTensor := must.M1(distributed.ShardTensor(spec, tensor))

			// For replicated values, logical shape = shard shape.
			assert.Equal(t, shapes.Make(dtypes.Float32, 1, 4), distTensor.Shape())
			assert.Equal(t, shapes.Make(dtypes.Float32, 1, 4), distTensor.ShardShape())

			// Check the number of shards.
			assert.Len(t, distTensor.Shards(), 2)

			// Check the values of the shards.
			assert.Equal(t, [][]float32{{1, 2, 3, 4}}, distTensor.Shards()[0].Value())
			assert.Equal(t, [][]float32{{1, 2, 3, 4}}, distTensor.Shards()[1].Value())
		})

		// Sharded and replicated: tensor axis 0 is sharded by "shards", and mesh axis "replicas" causes replication.
		t.Run("sharded and replicated", func(t *testing.T) {
			if backend.NumDevices() < 4 {
				t.Skipf("Skipping %s: backend only has %d device, this test requires 4.",
					t.Name(), backend.NumDevices())
			}
			mesh, err := distributed.NewDeviceMesh([]int{2, 2}, []string{"replicas", "shards"})
			require.NoError(t, err)
			tensor := tensors.FromValue([][]float32{{1, 2, 3, 4}, {5, 6, 7, 8}})
			// Spec: tensor axis 0 is sharded by mesh axis "shards" (size 2), tensor axis 1 is replicated.
			// Mesh axis "replicas" is not used for any tensor axis, so it causes replication.
			spec := must.M1(distributed.BuildSpec(mesh).S("shards").R().Done())
			distTensor := must.M1(distributed.ShardTensor(spec, tensor))

			// Logical shape is the original tensor shape.
			assert.Equal(t, shapes.Make(dtypes.Float32, 2, 4), distTensor.Shape())
			// Shard shape: tensor axis 0 (size 2) divided by mesh "shards" (size 2) = 1.
			assert.Equal(t, shapes.Make(dtypes.Float32, 1, 4), distTensor.ShardShape())

			// Check the number of shards.
			assert.Len(t, distTensor.Shards(), 4)

			// Mesh layout for [2, 2] with axes ["replicas", "shards"]:
			// Device 0: (replicas=0, shards=0) -> gets row 0
			// Device 1: (replicas=0, shards=1) -> gets row 1
			// Device 2: (replicas=1, shards=0) -> gets row 0 (replica)
			// Device 3: (replicas=1, shards=1) -> gets row 1 (replica)
			assert.Equal(t, [][]float32{{1, 2, 3, 4}}, distTensor.Shards()[0].Value())
			assert.Equal(t, [][]float32{{5, 6, 7, 8}}, distTensor.Shards()[1].Value())
			assert.Equal(t, [][]float32{{1, 2, 3, 4}}, distTensor.Shards()[2].Value())
			assert.Equal(t, [][]float32{{5, 6, 7, 8}}, distTensor.Shards()[3].Value())
		})
	})

	t.Run("Merge", func(t *testing.T) {
		t.Run("replicated and sharded", func(t *testing.T) {
			// Create a new device mesh.
			mesh, err := distributed.NewDeviceMesh([]int{2, 2}, []string{"replicas", "shards"})
			require.NoError(t, err)

			// Create a new distributed tensor.
			shards := []*tensors.Tensor{
				tensors.FromFlatDataAndDimensions([]int32{1, 2, 3, 4, 5, 6}, 3, 1, 2),
				tensors.FromFlatDataAndDimensions([]int32{10, 20, 30, 40, 50, 60}, 3, 1, 2),
				tensors.FromFlatDataAndDimensions([]int32{1, 2, 3, 4, 5, 6}, 3, 1, 2),
				tensors.FromFlatDataAndDimensions([]int32{10, 20, 30, 40, 50, 60}, 3, 1, 2),
			}
			spec, err := distributed.NewShardingSpec(mesh,
				distributed.ReplicatedAxis, distributed.ReplicatedAxis, distributed.AxisSpec{"shards"})
			require.NoError(t, err)
			distTensor, err := distributed.NewTensor(spec, shards)
			require.NoError(t, err)

			// Merge the tensor.
			tensor, err := distTensor.Merge()
			require.NoError(t, err)

			// Check the shape.
			assert.Equal(t, shapes.Make(dtypes.Int32, 3, 1, 4), tensor.Shape())

			// Check the values.
			assert.Equal(t, [][][]int32{{{1, 2, 10, 20}}, {{3, 4, 30, 40}}, {{5, 6, 50, 60}}}, tensor.Value())
		})

		t.Run("Merge-2", func(t *testing.T) {
			// Create a new device mesh.
			mesh, err := distributed.NewDeviceMesh([]int{2}, []string{"shards"})
			require.NoError(t, err)

			// Create a new distributed tensor.
			shards := []*tensors.Tensor{
				tensors.FromFlatDataAndDimensions([]int32{1, 2, 3, 4, 5, 6}, 2, 3, 1),
				tensors.FromFlatDataAndDimensions([]int32{10, 20, 30, 40, 50, 60}, 2, 3, 1),
			}
			spec, err := distributed.BuildSpec(mesh).S("shards").R().R().Done()
			require.NoError(t, err)
			distTensor, err := distributed.NewTensor(spec, shards)
			require.NoError(t, err)

			// Merge the tensor.
			merged, err := distTensor.Merge()
			require.NoError(t, err)

			// Check the shape.
			assert.Equal(t, shapes.Make(dtypes.Int32, 4, 3, 1), merged.Shape())

			// Check the values.
			assert.Equal(t, []int32{1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50, 60}, tensors.MustCopyFlatData[int32](merged))
		})
	})

	t.Run("Clone", func(t *testing.T) {
		// Create a new device mesh.
		mesh, err := distributed.NewDeviceMesh([]int{2}, []string{"shards"})
		require.NoError(t, err)

		// Create a new tensor.
		tensor := tensors.FromValue([][]int32{{1, 2, 3, 4}, {5, 6, 7, 8}})

		// Shard the tensor.
		spec, err := distributed.NewShardingSpec(mesh, distributed.AxisSpec{"shards"}, distributed.ReplicatedAxis)
		require.NoError(t, err)
		distTensor, err := distributed.ShardTensor(spec, tensor)
		require.NoError(t, err)

		// Clone the distributed tensor.
		distClone, err := distTensor.Clone()
		require.NoError(t, err)

		// Verify metadata.
		assert.Equal(t, distTensor.Shape(), distClone.Shape())
		assert.Equal(t, distTensor.ShardShape(), distClone.ShardShape())
		require.Len(t, distClone.Shards(), len(distTensor.Shards()))

		// Verify content.
		for i, shard := range distTensor.Shards() {
			cloneShard := distClone.Shards()[i]
			assert.Equal(t, shard.Value(), cloneShard.Value())
		}

		// Verify independence.
		// Modify the first element of the first shard of the original tensor.
		shard0 := distTensor.Shards()[0]
		err = tensors.MutableFlatData(shard0, func(flat []int32) {
			flat[0] = 100
		})
		require.NoError(t, err)

		// Check that the clone's first shard is unchanged (original value was 1).
		cloneShard0 := distClone.Shards()[0]
		cloneShard0Val := cloneShard0.Value().([][]int32)
		assert.Equal(t, int32(1), cloneShard0Val[0][0])
	})

}
