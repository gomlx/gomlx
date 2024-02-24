package sampler

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"io"
	"math/rand/v2"
	"sync"
)

// Dataset is created by a configured [Strategy].
// Before using it -- by calling [Dataset.Yield] -- it can be configured to
// shuffle and number of epochs, or to loop indefinitely.
// But batch size is not configurable in the Dataset, it is defined as part of the
// Strategy rules configuration (see [Strategy.Nodes] to define the seeds).
//
// The Dataset is created to be re-entrant, so it can be used with [data.Parallel].
type Dataset struct {
	name                    string
	sampler                 *Sampler
	strategy                *Strategy
	numEpochs, currentEpoch int
	shuffle                 bool

	muSample                sync.Mutex
	frozen                  bool
	startOfEpoch, exhausted bool

	// Position of the seeds -- pointers either into the seeds indices or
	// into seedsShuffle, if dataset is shuffled.
	seedsPosition []int32

	// seedsShuffle provides the sampling of the seeds, if shuffling was used.
	// These are reshuffled at the start of every epoch.
	seedsShuffle [][]int32
}

// NewDataset creates a new [Dataset] from the configured [Strategy].
// One can create multiple datasets from the same [Strategy], but once a [Dataset] is created,
// the [Strategy] is considered frozen and can no longer be modified.
func (strategy *Strategy) NewDataset(name string) *Dataset {
	if len(strategy.seeds) == 0 {
		Panicf("cannot create a new Dataset from a strategy with no seeds defined -- see Strategy.Nodes and Strategy.NodesFromSets")
	}
	strategy.frozen = true
	return &Dataset{
		name:          name,
		sampler:       strategy.sampler,
		strategy:      strategy,
		frozen:        false,
		numEpochs:     1,
		shuffle:       false,
		startOfEpoch:  true,
		exhausted:     false,
		seedsPosition: make([]int32, len(strategy.seeds)),
	}
}

// Epochs configures the dataset to yield those many epochs. Default is 1.
//
// Notice if there are more than one seed node type, an epoch is considered finished
// whenever the first of the seed types is exhaused.
//
// It returns itself to allow cascading configuration calls.
func (ds *Dataset) Epochs(n int) *Dataset {
	if ds.frozen {
		Panicf("cannot change a Dataset that has already started yielding results")
	}
	if n <= 0 {
		Panicf("for Dataset.Epochs(n), n > 0, but got n=%d instead", n)
	}
	ds.numEpochs = n
	return ds
}

// Infinite configures the dataset to yield looping over epochs indefinitely.
// Default is 1 epoch.
func (ds *Dataset) Infinite() *Dataset {
	if ds.frozen {
		Panicf("cannot change a Dataset that has already started yielding results")
	}
	ds.numEpochs = -1
	return ds
}

// Shuffle configures the dataset to shuffle the seed nodes before sampling it.
// It is reshuffled at every new epoch, resulting and random samples without
// replacement.
func (ds *Dataset) Shuffle() *Dataset {
	if ds.frozen {
		Panicf("cannot change a Dataset that has already started yielding results")
	}
	ds.shuffle = true
	return ds
}

var _ train.Dataset = &Dataset{}

// Name implements train.Dataset.
func (ds *Dataset) Name() string {
	return ds.name
}

// Reset implements train.Dataset: it restarts a Dataset after it has been exhausted.
func (ds *Dataset) Reset() {
	ds.muSample.Lock()
	defer ds.muSample.Unlock()
	ds.frozen = true
	ds.startOfEpoch = true
	ds.exhausted = false
	ds.currentEpoch = 0
}

// Yield implements train.Dataset.
// The returned spec is a pointer to the Strategy, and can be used to build a map of the names to the sampled
// tensors.
func (ds *Dataset) Yield() (spec any, inputs, labels []tensor.Tensor, err error) {
	spec = ds.strategy
	if ds.exhausted {
		err = io.EOF
		return
	}

	inputs = make([]tensor.Tensor, 0, 2*len(ds.strategy.rules))
	ds.muSample.Lock()
	ds.frozen = true
	if ds.startOfEpoch {
		ds.startEpoch()
	}

	// Sample seeds: requires a lock for the sampling.
	numSeeds := len(ds.strategy.seeds)
	seedsTensors := make([]*tensor.Local, 0, 2*numSeeds)
	for ii, seedsRule := range ds.strategy.seeds {
		seeds, mask := ds.sampleSeeds(ii, seedsRule)
		seedsTensors = append(seedsTensors, seeds, mask)
	}
	ds.muSample.Unlock()

	// Sampling edges: doesn't require lock.
	for seedIdx, seedsRule := range ds.strategy.seeds {
		seeds, mask := seedsTensors[2*seedIdx], seedsTensors[2*seedIdx+1]
		inputs = recursivelySampleEdges(seedsRule, seeds, mask, inputs)
	}
	return
}

// sampleSeeds returns the sampled seeds and their masks.
// For sampling seeds, ds.muSample must be locked.
func (ds *Dataset) sampleSeeds(seedIdx int, rule *Rule) (seeds, mask *tensor.Local) {
	seeds = tensor.FromScalarAndDimensions(int32(0), rule.count)
	seedsRef := seeds.AcquireData()
	defer seedsRef.Release()
	seedsData := seedsRef.Flat().([]int32)

	mask = tensor.FromScalarAndDimensions(false, rule.count)
	maskRef := mask.AcquireData()
	defer maskRef.Release()
	maskData := maskRef.Flat().([]bool)

	if ds.shuffle {
		// Sample from shuffles of the candidate seed nodes.
		shuffle := ds.seedsShuffle[seedIdx]
		pos := ds.seedsPosition[seedIdx]
		numToSample := int32(min(len(shuffle)-int(pos), rule.count))
		ds.seedsPosition[seedIdx] += numToSample
		if int(ds.seedsPosition[seedIdx]) >= len(shuffle) {
			ds.epochFinished()
		}
		copy(seedsData, shuffle[pos:pos+numToSample])
		for ii := range numToSample {
			maskData[ii] = true
		}

	} else {
		// Sample without changing the original order.
		pos := ds.seedsPosition[seedIdx]
		var numToSample int32
		if len(rule.nodeSet) > 0 {
			// Sample for given set.
			numToSample = int32(min(len(rule.nodeSet)-int(pos), rule.count))
			ds.seedsPosition[seedIdx] += numToSample
			if int(ds.seedsPosition[seedIdx]) >= len(rule.nodeSet) {
				ds.epochFinished()
			}
			for ii := range numToSample {
				seedsData[ii] = rule.nodeSet[pos+ii]
				maskData[ii] = true
			}

		} else {
			// Sample for all node indices, from 0 to `numNodes - 1` sequentially.
			numToSample = min(rule.numNodes-pos, int32(rule.count))
			ds.seedsPosition[seedIdx] += numToSample
			if ds.seedsPosition[seedIdx] >= rule.numNodes {
				ds.epochFinished()
			}
			for ii := range numToSample {
				seedsData[ii] = pos + ii
				maskData[ii] = true
			}
		}
	}
	return
}

// recursivelySampleEdges in the dependency tree of rules, storing the results that will become the yielded values
// by the Dataset.
func recursivelySampleEdges(rule *Rule, nodes, mask *tensor.Local, store []tensor.Tensor) []tensor.Tensor {
	store = append(store, nodes, mask)
	for _, subRule := range rule.dependents {
		subNodes, subMask := sampleEdges(subRule, nodes, mask)
		store = recursivelySampleEdges(subRule, subNodes, subMask, store)
	}
	return store
}

// sampleEdges based on a edge sampling rule `rule`, and the source nodes from which to sample.
func sampleEdges(rule *Rule, srcNodes, srcMask *tensor.Local) (nodes, mask *tensor.Local) {
	nodes = tensor.FromScalarAndDimensions(int32(0), rule.shape.Dimensions...)
	mask = tensor.FromScalarAndDimensions(false, rule.shape.Dimensions...)

	nodesRef := nodes.AcquireData()
	maskRef := mask.AcquireData()
	srcNodesRef := srcNodes.AcquireData()
	srcMaskRef := srcMask.AcquireData()
	defer func() {
		nodesRef.Release()
		maskRef.Release()
		srcNodesRef.Release()
		srcMaskRef.Release()
	}()

	tgtNodesData := nodesRef.Flat().([]int32)
	tgtMaskData := maskRef.Flat().([]bool)
	srcNodesData := srcNodesRef.Flat().([]int32)
	srcMaskData := srcMaskRef.Flat().([]bool)
	edgeDef := rule.edgeType
	sampledEdges := make([]int32, rule.count) // reserve space for sampling edges (reused over all iterations).

	// Iterator over source nodes, sampling edges for each.
	for fromIdx, fromValid := range srcMaskData {
		// Source node we are sampling from.
		if !fromValid {
			continue
		}
		srcNode := srcNodesData[fromIdx]

		// Find all edges from the source node.
		start := int32(0)
		if srcNode > 0 {
			start = edgeDef.Starts[srcNode-1]
		}

		end := edgeDef.Starts[srcNode]
		edges := edgeDef.EdgeTargets[start:end]
		if len(edges) == 0 {
			continue // No edges to sample from.
		}

		// If we don't have enough edges to sample from, take what we got.
		baseIdx := fromIdx * rule.count
		if len(edges) <= rule.count {
			// Take all edges, since we want to sample more than there are available.
			for ii, tgtNode := range edges {
				tgtNodesData[baseIdx+ii] = tgtNode
				tgtMaskData[baseIdx+ii] = true
			}
			continue
		}

		// Otherwise sample randomly without replacement from edges.
		randKOfN(sampledEdges, len(edges))
		for ii, edgeIdx := range sampledEdges {
			tgtNodesData[baseIdx+ii] = edges[edgeIdx]
			tgtMaskData[baseIdx+ii] = true
		}
	}
	return
}

// randKOfN return k random values without replacement out of `0..n-1`, and stores them in `values`.
// Note: `k = len(values)`.
func randKOfN(values []int32, n int) {
	k := len(values)
	if k*k < n {
		// Random sampling, checking for previous choices: this is O(k^2), but since usually we are working
		// with small values of K, it's faster than creating a map.
		//
		// FutureWork: for larger values of K, create a map/set.
		for ii := range values {
			// Take a unique number.
			var x int32
		takeANumber:
			for {
				x = int32(rand.IntN(n))
				for jj := range ii {
					if values[jj] == x {
						continue takeANumber
					}
				}
				break
			}
			values[ii] = x
		}

	} else {
		// Reservoir sampling: go over all n values and check whether it replaces a previous value.
		for ii := range k {
			values[ii] = int32(ii)
		}
		for ii := k; ii < n; ii++ {
			if ii < k {
				// Start by populating
				continue
			}
			pos := rand.IntN(int(ii))
			if pos < k {
				values[pos] = int32(ii)
			}
		}
	}
}

// startEpoch resets position counter and creates shuffle where required.
func (ds *Dataset) startEpoch() {
	ds.startOfEpoch = false

	// Restart the positions.
	for ii := range ds.seedsPosition {
		ds.seedsPosition[ii] = 0
	}
	if !ds.shuffle {
		return
	}

	// If the very first time, reserve the space for the shuffles for each rule type.
	strategy := ds.strategy
	if ds.seedsShuffle == nil {
		ds.seedsShuffle = make([][]int32, len(ds.seedsPosition))
		for ii, rule := range strategy.seeds {
			if rule.nodeSet != nil {
				ds.seedsShuffle[ii] = slices.Copy(rule.nodeSet)
			} else {
				ds.seedsShuffle[ii] = slices.Iota[int32](int32(0), int(rule.numNodes))
			}
		}
	}

	// Shuffle rules for each seeds set.
	for _, shuffle := range ds.seedsShuffle {
		shuffleLen := len(shuffle)
		for ii := range shuffle {
			jj := rand.IntN(shuffleLen)
			shuffle[ii], shuffle[jj] = shuffle[jj], shuffle[ii]
		}
	}
}

func (ds *Dataset) epochFinished() {
	ds.startOfEpoch = true
	ds.currentEpoch++
	if ds.numEpochs > 0 && ds.currentEpoch >= ds.numEpochs {
		ds.exhausted = true
	}
}
