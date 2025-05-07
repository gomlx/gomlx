package gnn

import "github.com/gomlx/gomlx/examples/ogbnmag/sampler"

type PoolingStrategy int

const (
	PoolingMax PoolingStrategy = iota
	PoolingSum
	PoolingMean
)

// AggregateFeatures generates direct aggregation of the features along the paths described in the sampler.
//
// Notice there is no sampler: we aggregate across all examples, one edge set at a time.
//
// Since the generate tensors are large, the function immediately save them in the file
// `downloadDir + "/feature_aggregation_<seed_node>.bin"`.
//
// The parameter `pooling` has to select at least one strategy. Careful, because the size of the final embedding grow
// with `O(p^d)`, where `p` is the number of pooling strategies and `d` is the depth of the sampler tree.
func AggregateFeatures(downloadDir string, sampler sampler.Sampler, pooling []PoolingStrategy) error {

	return nil
}
