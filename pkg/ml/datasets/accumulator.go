package datasets

import (
	"fmt"
	"io"
	"strings"
	"sync"

	"github.com/gomlx/gomlx/backends"
	"github.com/gomlx/gomlx/pkg/core/distributed"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/pkg/errors"
)

// bucketKey uniquely identifies a bucket by the shape and number of inputs/labels.
type bucketKey struct {
	numInputs, numLabels int
	inputShapes          []shapes.Shape
	labelShapes          []shapes.Shape
}

// bucket holds accumulated shards for a specific shape/number combination.
// No synchronization needed - only one goroutine modifies buckets.
type bucket struct {
	key bucketKey

	// Accumulated shards from source dataset
	// Pre-allocated with numInputShards capacity
	shards []shardBatch
}

// shardBatch represents a single batch from the source dataset.
type shardBatch struct {
	spec   any
	inputs []*tensors.Tensor
	labels []*tensors.Tensor
}

// distributedBatch represents a ready distributed batch.
type distributedBatch struct {
	spec   any
	inputs []*distributed.Tensor
	labels []*distributed.Tensor
	err    error // io.EOF or other error
}

// DistributedAccumulator is a dataset that is distributed across multiple devices, ready for
// distributed execution.
//
// It accumulates the shards from the source dataset and yields them as a single distributed.Tensor.
type DistributedAccumulator struct {
	backend backends.Backend
	source  train.Dataset

	// Distribution parameters
	strategy                               distributed.Strategy
	numDevices                             int
	numInputShards                         int // Number of shards to read from source (can be < numDevices if replicated)
	inputShardingSpecs, labelShardingSpecs []*distributed.ShardingSpec
	deviceAssignment                       []backends.DeviceNum

	// Bucketing system - no mutex needed, only one goroutine modifies buckets
	buckets map[string]*bucket

	// Statistics - protected by sync.Map for concurrent access
	stats sync.Map // key (string) -> count (int) of source inputs seen

	// Channel for next prepared batch (size 1)
	nextBatch chan *distributedBatch

	// Inherited from source dataset.
	isOwnershipTransferred bool
}

// Compile time check that DistributedAccumulator implements both train.Dataset and train.DistributedDataset.
var _ train.DistributedDataset = (*DistributedAccumulator)(nil)
var _ train.Dataset = (*DistributedAccumulator)(nil)

// NewDistributedAccumulator creates a distributed dataset from the given source dataset, by accumulating the shards
// and yielding them as a single distributed.Tensor.
//
// It uses the given strategy to distribute the data across the devices.
// The input and label sharding specs are used to specify how the data should be distributed.
//
// deviceAssignment can be nil, in which case a default sequential assignment (starting from 0) is used.
//
// The last element of inputShardingSpec is used if there are more inputs than specs defined.
// The same for labels.
func NewDistributedAccumulator(backend backends.Backend, source train.Dataset, strategy distributed.Strategy,
	inputShardingSpecs, labelShardingSpecs []*distributed.ShardingSpec,
	deviceAssignment []backends.DeviceNum) (
	*DistributedAccumulator, error) {
	if backend == nil {
		return nil, errors.New("backend cannot be nil")
	}
	if source == nil {
		return nil, errors.New("source dataset cannot be nil")
	}
	if len(inputShardingSpecs) == 0 {
		return nil, errors.New("input sharding specs cannot be empty")
	}

	// Infer numDevices from meshes in sharding specs
	numDevices := 0
	seenMeshes := make(map[*distributed.DeviceMesh]bool)

	allSpecs := append(inputShardingSpecs, labelShardingSpecs...)
	for _, spec := range allSpecs {
		if spec == nil || spec.Mesh == nil {
			continue
		}
		if !seenMeshes[spec.Mesh] {
			seenMeshes[spec.Mesh] = true
			meshNumDevices := spec.Mesh.NumDevices()
			if numDevices == 0 {
				numDevices = meshNumDevices
			} else if numDevices != meshNumDevices {
				return nil, errors.Errorf(
					"inconsistent number of devices across meshes: found %d and %d",
					numDevices, meshNumDevices)
			}
		}
	}

	if numDevices == 0 {
		return nil, errors.New("could not infer numDevices from sharding specs: no valid meshes found")
	}

	// Create default device assignment if nil (sequential, starting from 0)
	if deviceAssignment == nil {
		deviceAssignment = make([]backends.DeviceNum, numDevices)
		for i := range numDevices {
			deviceAssignment[i] = backends.DeviceNum(i)
		}
	} else if len(deviceAssignment) != numDevices {
		return nil, errors.Errorf("device assignment length (%d) must match numDevices (%d)",
			len(deviceAssignment), numDevices)
	}

	// Calculate numInputShards: for each spec, calculate how many shards it needs independently.
	// All non-replicated specs must have the same specNumInputShards, otherwise behavior is undefined.
	// Replicated specs (specNumInputShards == 1) are allowed to differ as they only use the first shard.
	numInputShards := -1
	for _, spec := range allSpecs {
		if spec == nil {
			continue
		}
		specNumShards := numShardsForSpec(spec)
		// Skip fully replicated specs (they can have different specNumInputShards)
		if specNumShards == 1 {
			continue
		}
		if numInputShards == -1 {
			numInputShards = specNumShards
		} else if specNumShards != numInputShards {
			return nil, errors.Errorf(
				"inconsistent specNumInputShards across non-replicated specs: found %d and %d (behavior undefined when specs require different numbers of input shards)",
				numInputShards, specNumShards)
		}
	}

	// If no non-replicated specs found, default to 1 (fully replicated)
	if numInputShards == -1 {
		numInputShards = 1
	}

	ds := &DistributedAccumulator{
		backend:                backend,
		source:                 source,
		strategy:               strategy,
		numDevices:             numDevices,
		numInputShards:         numInputShards,
		inputShardingSpecs:     inputShardingSpecs,
		labelShardingSpecs:     labelShardingSpecs,
		deviceAssignment:       deviceAssignment,
		buckets:                make(map[string]*bucket),
		stats:                  sync.Map{},
		nextBatch:              make(chan *distributedBatch, 1),
		isOwnershipTransferred: true,
	}

	// Inherit ownership transfer from source dataset.
	if isOwnershipTransferred, ok := source.(train.DatasetCustomOwnership); ok {
		ds.isOwnershipTransferred = isOwnershipTransferred.IsOwnershipTransferred()
	}

	// Start the first reader
	ds.startReader()

	return ds, nil
}

// Name implements train.Dataset.
func (ds *DistributedAccumulator) Name() string {
	return "Distributed(" + ds.source.Name() + ")"
}

// Reset implements train.Dataset.
func (ds *DistributedAccumulator) Reset() {
	// Read and discard any prepared batch in the channel
	select {
	case batch := <-ds.nextBatch:
		if batch != nil {
			// Finalize any distributed tensors
			for _, dt := range batch.inputs {
				dt.Finalize()
			}
			for _, dt := range batch.labels {
				dt.Finalize()
			}
		}
	default:
		// Channel was empty
	}

	// Clear buckets and discard leftovers
	for _, b := range ds.buckets {
		// Finalize accumulated shards
		for _, sb := range b.shards {
			for _, t := range sb.inputs {
				t.FinalizeAll()
			}
			for _, t := range sb.labels {
				t.FinalizeAll()
			}
		}
	}
	ds.buckets = make(map[string]*bucket)

	// Clear statistics
	ds.stats = sync.Map{}

	// Reset source dataset
	ds.source.Reset()

	// Start the next reader
	ds.startReader()
}

// Strategy implements train.DistributedDataset.
func (ds *DistributedAccumulator) Strategy() distributed.Strategy {
	return ds.strategy
}

// DeviceAssignment implements train.DistributedDataset.
func (ds *DistributedAccumulator) DeviceAssignment() []backends.DeviceNum {
	return ds.deviceAssignment
}

// bucketKeyString creates a string key for the bucket.
// Uses strings.Builder for performance (called in inner loop).
func bucketKeyString(key bucketKey) string {
	var sb strings.Builder
	sb.WriteString("inputs:")
	sb.WriteString(fmt.Sprintf("%d[", key.numInputs))
	for i, shape := range key.inputShapes {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(shape.String())
	}
	sb.WriteString("]labels:")
	sb.WriteString(fmt.Sprintf("%d[", key.numLabels))
	for i, shape := range key.labelShapes {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(shape.String())
	}
	sb.WriteString("]")
	return sb.String()
}

// getOrCreateBucket gets or creates a bucket for the given key.
// No synchronization needed - only called from single reader goroutine.
func (ds *DistributedAccumulator) getOrCreateBucket(key bucketKey) *bucket {
	keyStr := bucketKeyString(key)

	b, exists := ds.buckets[keyStr]
	if !exists {
		b = &bucket{
			key:    key,
			shards: make([]shardBatch, 0, ds.numInputShards),
		}
		ds.buckets[keyStr] = b
	}
	return b
}

// numShardsForSpec calculates the number of shards needed for a given ShardingSpec.
// This is the product of sizes of mesh axes that are actually used (sharded) in the spec.
// Returns 1 if the spec is fully replicated.
func numShardsForSpec(spec *distributed.ShardingSpec) int {
	if spec == nil || spec.Mesh == nil || spec.IsReplicated() {
		return 1
	}
	meshAxesUsed := make(map[string]bool)
	for _, axisSpec := range spec.Axes {
		for _, meshAxis := range axisSpec {
			meshAxesUsed[meshAxis] = true
		}
	}
	numShards := 1
	for meshAxis := range meshAxesUsed {
		axisSize, err := spec.Mesh.AxisSize(meshAxis)
		if err == nil {
			numShards *= axisSize
		}
	}
	return numShards
}

// aggregateShards aggregates shards into distributed tensors and moves them to devices.
// Returns a distributedBatch ready to be sent to the channel.
func (ds *DistributedAccumulator) aggregateShards(shardsToUse []shardBatch, spec any) (*distributedBatch, error) {
	if len(shardsToUse) != ds.numInputShards {
		return nil, errors.Errorf("expected %d input shards, got %d", ds.numInputShards, len(shardsToUse))
	}

	if len(shardsToUse) == 0 {
		return nil, errors.New("no shards to aggregate")
	}

	// Determine shapes from first shard
	firstShard := shardsToUse[0]
	numInputs := len(firstShard.inputs)
	numLabels := len(firstShard.labels)

	// Aggregate inputs
	distributedInputs := make([]*distributed.Tensor, numInputs)
	for inputIdx := range numInputs {
		// Use the last spec if inputIdx is beyond the number of specs provided
		specIdx := inputIdx
		if specIdx >= len(ds.inputShardingSpecs) {
			specIdx = len(ds.inputShardingSpecs) - 1
		}
		spec := ds.inputShardingSpecs[specIdx]
		if spec == nil {
			return nil, errors.Errorf("input sharding spec #%d is nil", specIdx)
		}

		// Collect shards for this input from source
		sourceShards := make([]*tensors.Tensor, ds.numInputShards)
		for i, sb := range shardsToUse {
			if inputIdx >= len(sb.inputs) {
				return nil, errors.Errorf("shard batch #%d has only %d inputs, need input #%d",
					i, len(sb.inputs), inputIdx)
			}
			sourceShards[i] = sb.inputs[inputIdx]
		}

		// Calculate how many shards this spec needs
		numShardsNeeded := numShardsForSpec(spec)

		// Handle replication: we always need numDevices shards for distributed.NewTensor
		shards := make([]*tensors.Tensor, ds.numDevices)
		if spec.IsReplicated() {
			// Fully replicated: clone first shard to all devices
			firstShard := sourceShards[0]
			shards[0] = firstShard // Keep the first one
			for i := 1; i < ds.numDevices; i++ {
				var err error
				shards[i], err = firstShard.Clone()
				if err != nil {
					return nil, errors.WithMessagef(err, "failed to clone shard for replication for input #%d", inputIdx)
				}
			}
			// Finalize the extra source shards we don't need
			for i := 1; i < len(sourceShards); i++ {
				sourceShards[i].FinalizeAll()
			}
		} else {
			// Not fully replicated: replicate sourceShards across replicated mesh axes
			// numShardsNeeded equals numInputShards (validated during construction)
			// specNumDevices is the total number of devices in this spec's mesh
			specNumDevices := spec.Mesh.NumDevices()
			replicationFactor := specNumDevices / numShardsNeeded
			if replicationFactor*numShardsNeeded != specNumDevices {
				return nil, errors.Errorf("input #%d: spec mesh has %d devices which must be divisible by numShardsNeeded (%d)",
					inputIdx, specNumDevices, numShardsNeeded)
			}

			// Replicate each source shard replicationFactor times
			shardIdx := 0
			for i := 0; i < numShardsNeeded; i++ {
				for j := 0; j < replicationFactor; j++ {
					if j == 0 {
						// First replication: use the original shard
						shards[shardIdx] = sourceShards[i]
					} else {
						// Subsequent replications: clone the shard
						var err error
						shards[shardIdx], err = sourceShards[i].Clone()
						if err != nil {
							return nil, errors.WithMessagef(err, "failed to clone shard %d for replication for input #%d", i, inputIdx)
						}
					}
					shardIdx++
				}
			}

			// Verify we filled exactly specNumDevices shards
			if shardIdx != specNumDevices {
				return nil, errors.Errorf("input #%d: expected to fill %d shards, but filled %d",
					inputIdx, specNumDevices, shardIdx)
			}
		}

		// Move shards to devices in parallel
		err := ds.moveShardsToDevices(shards)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to move input #%d shards to devices", inputIdx)
		}

		// Create distributed tensor
		dt, err := distributed.NewTensor(spec, shards)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create distributed tensor for input #%d", inputIdx)
		}
		distributedInputs[inputIdx] = dt
	}

	// Aggregate labels
	if numLabels > 0 && len(ds.labelShardingSpecs) == 0 {
		return nil, errors.New("label sharding specs cannot be empty when there are labels")
	}
	distributedLabels := make([]*distributed.Tensor, numLabels)
	for labelIdx := range numLabels {
		// Use the last spec if labelIdx is beyond the number of specs provided
		specIdx := labelIdx
		if specIdx >= len(ds.labelShardingSpecs) {
			specIdx = len(ds.labelShardingSpecs) - 1
		}
		spec := ds.labelShardingSpecs[specIdx]
		if spec == nil {
			return nil, errors.Errorf("label sharding spec #%d is nil", specIdx)
		}

		// Collect shards for this label from source
		sourceShards := make([]*tensors.Tensor, ds.numInputShards)
		for i, sb := range shardsToUse {
			if labelIdx >= len(sb.labels) {
				return nil, errors.Errorf("shard batch #%d has only %d labels, need label #%d",
					i, len(sb.labels), labelIdx)
			}
			sourceShards[i] = sb.labels[labelIdx]
		}

		// Calculate how many shards this spec needs
		numShardsNeeded := numShardsForSpec(spec)

		// Handle replication: we always need numDevices shards for distributed.NewTensor
		shards := make([]*tensors.Tensor, ds.numDevices)
		if spec.IsReplicated() {
			// Fully replicated: clone first shard to all devices
			firstShard := sourceShards[0]
			shards[0] = firstShard // Keep the first one
			for i := 1; i < ds.numDevices; i++ {
				var err error
				shards[i], err = firstShard.Clone()
				if err != nil {
					return nil, errors.WithMessagef(err, "failed to clone shard for replication for label #%d", labelIdx)
				}
			}
			// Finalize the extra source shards we don't need
			for i := 1; i < len(sourceShards); i++ {
				sourceShards[i].FinalizeAll()
			}
		} else {
			// Not fully replicated: replicate sourceShards across replicated mesh axes
			// numShardsNeeded equals numInputShards (validated during construction)
			// specNumDevices is the total number of devices in this spec's mesh
			specNumDevices := spec.Mesh.NumDevices()
			replicationFactor := specNumDevices / numShardsNeeded
			if replicationFactor*numShardsNeeded != specNumDevices {
				return nil, errors.Errorf("label #%d: spec mesh has %d devices which must be divisible by numShardsNeeded (%d)",
					labelIdx, specNumDevices, numShardsNeeded)
			}

			// Replicate each source shard replicationFactor times
			shardIdx := 0
			for i := 0; i < numShardsNeeded; i++ {
				for j := 0; j < replicationFactor; j++ {
					if j == 0 {
						// First replication: use the original shard
						shards[shardIdx] = sourceShards[i]
					} else {
						// Subsequent replications: clone the shard
						var err error
						shards[shardIdx], err = sourceShards[i].Clone()
						if err != nil {
							return nil, errors.WithMessagef(err, "failed to clone shard %d for replication for label #%d", i, labelIdx)
						}
					}
					shardIdx++
				}
			}

			// Verify we filled exactly specNumDevices shards
			if shardIdx != specNumDevices {
				return nil, errors.Errorf("label #%d: expected to fill %d shards, but filled %d",
					labelIdx, specNumDevices, shardIdx)
			}
		}

		// Move shards to devices in parallel
		err := ds.moveShardsToDevices(shards)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to move label #%d shards to devices", labelIdx)
		}

		// Create distributed tensor
		dt, err := distributed.NewTensor(spec, shards)
		if err != nil {
			return nil, errors.WithMessagef(err, "failed to create distributed tensor for label #%d", labelIdx)
		}
		distributedLabels[labelIdx] = dt
	}

	// Finalize local copies of source tensors
	for _, sb := range shardsToUse {
		for _, t := range sb.inputs {
			t.FinalizeLocal()
		}
		for _, t := range sb.labels {
			t.FinalizeLocal()
		}
	}

	return &distributedBatch{
		spec:   spec,
		inputs: distributedInputs,
		labels: distributedLabels,
	}, nil
}

// moveShardsToDevices moves shards to their corresponding devices in parallel.
func (ds *DistributedAccumulator) moveShardsToDevices(shards []*tensors.Tensor) error {
	if len(shards) != len(ds.deviceAssignment) {
		return errors.Errorf("number of shards (%d) doesn't match device assignment (%d)",
			len(shards), len(ds.deviceAssignment))
	}

	var wg sync.WaitGroup
	errCh := make(chan error, len(shards))

	for i, shard := range shards {
		wg.Add(1)
		go func(idx int, s *tensors.Tensor) {
			defer wg.Done()
			deviceNum := ds.deviceAssignment[idx]
			err := s.MaterializeOnDevice(ds.backend, false, deviceNum)
			if err != nil {
				errCh <- errors.WithMessagef(err, "failed to materialize shard #%d on device %d", idx, deviceNum)
			}
		}(i, shard)
	}

	wg.Wait()
	close(errCh)

	// Collect any errors
	var firstErr error
	for err := range errCh {
		if firstErr == nil {
			firstErr = err
		}
	}

	return firstErr
}

// startReader starts a background reader goroutine.
// No synchronization needed - the channel ensures only one reader is active at a time.
func (ds *DistributedAccumulator) startReader() {
	go ds.reader()
}

// reader is the background goroutine that reads from source dataset, accumulates shards,
// aggregates them, and sends ready batches to the channel.
// It reads until it has enough shards for one batch, sends it, then exits.
// Yield() will start the next reader.
func (ds *DistributedAccumulator) reader() {
	// Read from source dataset and accumulate shards until we have enough for one batch
	for {
		// Read from source dataset
		sourceSpec, sourceInputs, sourceLabels, err := ds.source.Yield()
		if err == io.EOF {
			// End of source dataset - check if we have any ready buckets to send
			ds.sendReadyBatch()
			// Send EOF batch
			ds.nextBatch <- &distributedBatch{err: io.EOF}
			return
		}
		if err != nil {
			// Send error batch
			ds.nextBatch <- &distributedBatch{err: err}
			return
		}

		// Create bucket key
		key := bucketKey{
			numInputs:   len(sourceInputs),
			numLabels:   len(sourceLabels),
			inputShapes: make([]shapes.Shape, len(sourceInputs)),
			labelShapes: make([]shapes.Shape, len(sourceLabels)),
		}
		for i, t := range sourceInputs {
			key.inputShapes[i] = t.Shape()
		}
		for i, t := range sourceLabels {
			key.labelShapes[i] = t.Shape()
		}

		// Update statistics
		keyStr := bucketKeyString(key)
		// Load current value, increment, and store back (thread-safe)
		if val, ok := ds.stats.Load(keyStr); ok {
			ds.stats.Store(keyStr, val.(int)+1)
		} else {
			ds.stats.Store(keyStr, 1)
		}

		// Get or create bucket
		b := ds.getOrCreateBucket(key)

		// Add shard to bucket
		b.shards = append(b.shards, shardBatch{
			spec:   sourceSpec,
			inputs: sourceInputs,
			labels: sourceLabels,
		})

		// Check if we have enough shards to aggregate
		if len(b.shards) >= ds.numInputShards {
			// Extract shards
			shardsToUse := make([]shardBatch, ds.numInputShards)
			copy(shardsToUse, b.shards[:ds.numInputShards])
			b.shards = b.shards[ds.numInputShards:]

			// Aggregate shards
			batch, err := ds.aggregateShards(shardsToUse, sourceSpec)
			if err != nil {
				ds.nextBatch <- &distributedBatch{err: err}
				return
			}

			// Send to channel (blocking if channel is full)
			ds.nextBatch <- batch
			return // Reader done, will be restarted by DistributedYield()
		}
	}
}

// sendReadyBatch checks all buckets and sends the first ready batch if available.
func (ds *DistributedAccumulator) sendReadyBatch() {
	// Check all buckets for ready data
	for _, b := range ds.buckets {
		if len(b.shards) >= ds.numInputShards {
			// Extract shards
			shardsToUse := make([]shardBatch, ds.numInputShards)
			copy(shardsToUse, b.shards[:ds.numInputShards])
			b.shards = b.shards[ds.numInputShards:]

			// Aggregate shards
			batch, err := ds.aggregateShards(shardsToUse, shardsToUse[0].spec)
			if err != nil {
				ds.nextBatch <- &distributedBatch{err: err}
				return
			}

			// Send to channel
			ds.nextBatch <- batch
			return
		}
	}
}

// Yield implements train.Dataset, by simply returning an error to indicate one should use DistributedYield instead.
func (ds *DistributedAccumulator) Yield() (spec any, inputs, labels []*tensors.Tensor, err error) {
	return nil, nil, nil, errors.New("the DistributedAccumulator dataset was meant to be used in a distributed manner; " +
		"either add support for DistributedDatasets or use a normal Dataset instead")
}

// DistributedYield implements train.DistributedDataset.
func (ds *DistributedAccumulator) DistributedYield() (spec any, inputs, labels []*distributed.Tensor, err error) {
	// Read the next prepared batch from channel
	batch := <-ds.nextBatch

	if batch.err != nil {
		if batch.err == io.EOF {
			return nil, nil, nil, io.EOF
		}
		return nil, nil, nil, batch.err
	}

	// Start the next reader in parallel
	ds.startReader()

	return batch.spec, batch.inputs, batch.labels, nil
}

// Stats returns information on the number of source inputs of each shape/number seen so far.
func (ds *DistributedAccumulator) Stats() map[string]int {
	result := make(map[string]int)
	ds.stats.Range(func(key, value any) bool {
		result[key.(string)] = value.(int)
		return true
	})
	return result
}

// StatsString returns a pretty-print version of the statistics for debugging.
func (ds *DistributedAccumulator) StatsString() string {
	stats := ds.Stats()
	if len(stats) == 0 {
		return "No statistics collected yet"
	}

	var s string
	s += fmt.Sprintf("Distributed Dataset Statistics (%d bucket types):\n", len(stats))
	for key, count := range stats {
		s += fmt.Sprintf("  %s: %d batches\n", key, count)
	}
	return s
}
