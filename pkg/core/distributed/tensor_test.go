package distributed_test

import (
	"fmt"
	"sync"
	"testing"

	"github.com/gomlx/gomlx/backends"
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

func TestShardTensor(t *testing.T) {
	backend := BuildTestBackend()
	if backend.NumDevices() < 2 {
		t.Skipf("Skipping distributed tests: backend only has %d device.", backend.NumDevices())
	}

	// Create a new device mesh.
	mesh, err := distributed.NewDeviceMesh(backend, []int{2}, []string{"replica"})
	require.NoError(t, err)

	// Create a new tensor.
	tensor := tensors.FromValue([][]int32{{1, 2, 3, 4}, {5, 6, 7, 8}})

	// Shard the tensor.
	spec := distributed.NewShardSpec("replica", "")
	distTensor, err := distributed.ShardTensor(tensor, mesh, spec)
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
}

func TestMergeTensor(t *testing.T) {
	backend := BuildTestBackend()
	if backend.NumDevices() < 2 {
		t.Skipf("Skipping distributed tests: backend only has %d device.", backend.NumDevices())
	}

	// Create a new device mesh.
	mesh, err := distributed.NewDeviceMesh(backend, []int{2}, []string{"replica"})
	require.NoError(t, err)

	// Create a new distributed tensor.
	shards := []*tensors.Tensor{
		tensors.FromValue([][]int32{{1, 2, 3, 4}}),
		tensors.FromValue([][]int32{{5, 6, 7, 8}}),
	}
	spec := distributed.NewShardSpec("replica", "")
	distTensor, err := distributed.NewTensor(mesh, spec, shards)
	require.NoError(t, err)

	// Merge the tensor.
	tensor, err := distTensor.Merge()
	require.NoError(t, err)

	// Check the shape.
	assert.Equal(t, shapes.Make(dtypes.Int32, 2, 4), tensor.Shape())

	// Check the values.
	assert.Equal(t, [][]int32{{1, 2, 3, 4}, {5, 6, 7, 8}}, tensor.Value())
}
