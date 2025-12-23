package sampler

import (
	"io"
	"math/rand/v2"
	"sync"

	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/xslices"
)

// Dataset is created by a configured [Strategy].
// Before using it -- by calling [Dataset.Yield] -- it can be configured to
// shuffle and number of epochs, or to loop indefinitely.
// But batch size is not configurable in the Dataset, it is defined as part of the
// Strategy Rules configuration (see [Strategy.Nodes] to define the Seeds).
//
// The Dataset is created to be re-entrant, so it can be used with [data.Parallel].
//
// No labels are generated, and [Dataset.Yield] will return nil labels.
type Dataset struct {
	name                     string
	sampler                  *Sampler
	strategy                 *Strategy
	numEpochs                int
	shuffle, withReplacement bool
	degree                   bool

	muSample                sync.Mutex
	currentEpoch            int
	frozen                  bool
	startOfEpoch, exhausted bool

	// Position of the Seeds -- pointers either into the Seeds indices or
	// into seedsShuffle, if dataset is shuffled.
	seedsPosition []int32

	// seedsShuffle provides the sampling of the Seeds, if shuffling was used.
	// These are reshuffled at the start of every epoch.
	seedsShuffle [][]int32
}

// NewDataset creates a new [Dataset] from the configured [Strategy].
// One can create multiple datasets from the same [Strategy], but once a [Dataset] is created,
// the [Strategy] is considered frozen and can no longer be modified.
func (strategy *Strategy) NewDataset(name string) *Dataset {
	if len(strategy.Seeds) == 0 {
		Panicf(
			"cannot create a new Dataset from a strategy with no Seeds defined -- see Strategy.Nodes and Strategy.NodesFromSets",
		)
	}
	strategy.frozen = true
	return &Dataset{
		name:            name,
		sampler:         strategy.Sampler,
		strategy:        strategy,
		frozen:          false,
		numEpochs:       1,
		shuffle:         false,
		withReplacement: false,

		startOfEpoch:  true,
		exhausted:     false,
		seedsPosition: make([]int32, len(strategy.Seeds)),
	}
}

// Epochs configures the dataset to yield those many epochs. Default is 1.
//
// Notice if there are more than one seed node type, an epoch is considered finished
// whenever the first of the seed types is exhausted.
//
// It returns itself to allow cascading configuration calls.
func (ds *Dataset) Epochs(n int) *Dataset {
	if ds.frozen {
		Panicf("cannot change a Dataset that has already started yielding results")
	}
	if ds.withReplacement {
		Panicf("cannot configure Epochs for a dataset configured WithReplacement()")
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

// WithReplacement configures the dataset to yield with replacement.
// This automatically implies `Shuffle` and `Infinite`.
func (ds *Dataset) WithReplacement() *Dataset {
	if ds.frozen {
		Panicf("cannot change a Dataset that has already started yielding results")
	}
	ds.withReplacement = true
	return ds.Infinite().Shuffle()
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
func (ds *Dataset) Yield() (spec any, inputs, labels []*tensors.Tensor, err error) {
	ds.muSample.Lock()
	var unlocked bool
	defer func() {
		if !unlocked {
			ds.muSample.Unlock()
		}
	}()

	spec = ds.strategy
	if ds.exhausted {
		err = io.EOF
		return
	}

	if ds.strategy.KeepDegrees {
		// 2 tensors per node (value and mask), plus one tensor per edge (degree).
		numEdges := len(ds.strategy.Rules) - len(ds.strategy.Seeds)
		inputs = make([]*tensors.Tensor, 0, 2*len(ds.strategy.Rules)+numEdges)
	} else {
		// 2 tensors per node: value and mask.
		inputs = make([]*tensors.Tensor, 0, 2*len(ds.strategy.Rules))
	}
	ds.frozen = true
	if ds.startOfEpoch {
		ds.startEpoch()
	}

	// Sample Seeds: requires a lock for the sampling.
	numSeeds := len(ds.strategy.Seeds)
	seedsTensors := make([]*tensors.Tensor, 0, 2*numSeeds)
	for ii, seedsRule := range ds.strategy.Seeds {
		seeds, mask := ds.sampleSeeds(ii, seedsRule)
		seedsTensors = append(seedsTensors, seeds, mask)
	}

	// Sampling edges: doesn't require lock.
	ds.muSample.Unlock()
	unlocked = true
	for seedIdx, seedsRule := range ds.strategy.Seeds {
		seeds, mask := seedsTensors[2*seedIdx], seedsTensors[2*seedIdx+1]
		inputs = append(inputs, seeds, mask)
		inputs = recursivelySampleEdges(seedsRule, seeds, mask, inputs)
	}
	return
}

// sampleSeeds returns the sampled Seeds and their masks.
// For sampling Seeds, ds.muSample must be locked.
func (ds *Dataset) sampleSeeds(seedIdx int, rule *Rule) (seeds, mask *tensors.Tensor) {
	seeds = tensors.FromScalarAndDimensions(int32(0), rule.Count)
	mask = tensors.FromScalarAndDimensions(false, rule.Count)

	tensors.MustMutableFlatData[int32](seeds, func(seedsData []int32) {
		tensors.MustMutableFlatData[bool](mask, func(maskData []bool) {
			if ds.withReplacement {
				for ii := range rule.Count {
					maskData[ii] = true
				}
				if len(rule.NodeSet) > 0 {
					for ii := range rule.Count {
						seedsData[ii] = rule.NodeSet[rand.IntN(len(rule.NodeSet))]
					}
				} else {
					for ii := range rule.Count {
						seedsData[ii] = int32(rand.IntN(int(rule.NumNodes)))
					}
				}
			} else if ds.shuffle {
				// Sample from shuffles of the candidate seed nodes.
				shuffle := ds.seedsShuffle[seedIdx]
				pos := ds.seedsPosition[seedIdx]
				numToSample := int32(min(len(shuffle)-int(pos), rule.Count))
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
				if len(rule.NodeSet) > 0 {
					// Sample for given set.
					numToSample = int32(min(len(rule.NodeSet)-int(pos), rule.Count))
					ds.seedsPosition[seedIdx] += numToSample
					if int(ds.seedsPosition[seedIdx]) >= len(rule.NodeSet) {
						ds.epochFinished()
					}
					for ii := range numToSample {
						seedsData[ii] = rule.NodeSet[pos+ii]
						maskData[ii] = true
					}

				} else {
					// Sample for all node indices, from 0 to `NumNodes - 1` sequentially.
					numToSample = min(rule.NumNodes-pos, int32(rule.Count))
					ds.seedsPosition[seedIdx] += numToSample
					if ds.seedsPosition[seedIdx] >= rule.NumNodes {
						ds.epochFinished()
					}
					for ii := range numToSample {
						seedsData[ii] = pos + ii
						maskData[ii] = true
					}
				}
			}
		})
	})
	return
}

// recursivelySampleEdges in the dependency tree of Rules, storing the results that will become the yielded values
// by the Dataset.
func recursivelySampleEdges(rule *Rule, nodes, mask *tensors.Tensor, store []*tensors.Tensor) []*tensors.Tensor {
	for _, subRule := range rule.Dependents {
		subNodes, subMask, degrees := sampleEdges(subRule, nodes, mask)
		store = append(store, subNodes, subMask)
		if degrees != nil {
			store = append(store, degrees)
		}
		store = recursivelySampleEdges(subRule, subNodes, subMask, store)
	}
	return store
}

// accessMutableDataOfTensors is a short-cut to calling ConstFlatData or MutableFlatData on several tensors at once.
// It will call accessFn with the flatData of all tensors in the same order.
// The values in mutable defines if it calls Tensor.ConstFlatData or Tensor.MutableFlatData on the corresponding tensor.
func accessTensorsData(tensorsList []*tensors.Tensor, mutableList []bool, accessFn func(flatData []any)) {
	if len(tensorsList) != len(mutableList) {
		Panicf(
			"accessTensorsData got %d tensorsList and %d mutableList, they must be the same",
			len(tensorsList),
			len(mutableList),
		)
	}
	tensorsIdx := 0
	var allFlat []any
	var recursion func(flat any)
	recursion = func(flat any) {
		if flat != nil {
			allFlat = append(allFlat, flat)
			tensorsIdx++
		}
		if tensorsIdx == len(tensorsList) {
			// End recursion.
			accessFn(allFlat)
			return
		}
		tensor := tensorsList[tensorsIdx]
		if mutableList[tensorsIdx] {
			tensor.MustMutableFlatData(recursion)
		} else {
			tensor.MustConstFlatData(recursion)
		}
	}
	recursion(nil)
}

// sampleEdges based on an edge sampling rule `rule`, and the source nodes from which to sample.
func sampleEdges(rule *Rule, srcNodes, srcMask *tensors.Tensor) (nodes, mask, degrees *tensors.Tensor) {
	nodes = tensors.FromScalarAndDimensions(int32(0), rule.Shape.Dimensions...)
	mask = tensors.FromScalarAndDimensions(false, rule.Shape.Dimensions...)

	tensorsList := []*tensors.Tensor{srcNodes, srcMask, nodes, mask}
	mutableList := []bool{false, false, true, true}

	if rule.Strategy.KeepDegrees {
		degreesShape := srcNodes.Shape().Clone()
		degreesShape.Dimensions = append(degreesShape.Dimensions, 1)
		degrees = tensors.FromScalarAndDimensions(int32(0), degreesShape.Dimensions...)
		tensorsList = append(tensorsList, degrees)
		mutableList = append(mutableList, true)
	}

	accessTensorsData(tensorsList, mutableList, func(flatData []any) {
		srcNodesData := flatData[0].([]int32)
		srcMaskData := flatData[1].([]bool)
		tgtNodesData := flatData[2].([]int32)
		tgtMaskData := flatData[3].([]bool)
		var degreesData []int32
		if rule.Strategy.KeepDegrees {
			degreesData = flatData[4].([]int32)
		}

		if rule.IsIdentitySubRule() {
			// Identity Sub-Rule has the exact same data, just different shapes (the sub-rule has an extra axis of dimension 1).
			copy(tgtNodesData, srcNodesData)
			copy(tgtMaskData, srcMaskData)
			if len(degreesData) != 0 {
				xslices.FillSlice(degreesData, int32(1))
			}
			return
		}

		edgeDef := rule.EdgeType
		sampledEdges := make([]int32, rule.Count) // reserve space for sampling edges (reused over all iterations).

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
			if degreesData != nil {
				degreesData[fromIdx] = int32(len(edges))
			}

			// If we don't have enough edges to sample from, take what we got.
			baseIdx := fromIdx * rule.Count
			if len(edges) <= rule.Count {
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
	})
	return
}

// randKOfN return k random values without replacement out of `0..n-1`, and stores them in `values`.
// Note: `k = len(values)`.
func randKOfN(values []int32, n int) {
	k := len(values)
	if k*k < n {
		randKOfNLinear(values, n)
	} else {
		randKOfNReservoir(values, n)
	}
}

// randKOfNLinear is the linear implementation of rankKOfN that works well when k is small.
func randKOfNLinear(values []int32, n int) {
	// Random sampling, checking for previous choices: this is O(k^2), but since usually we are working
	// with small values of K, it's faster than creating a hash.
	//
	// FutureWork: for larger values of K, create some form of hash.
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
}

func randKOfNReservoir(values []int32, n int) {
	k := len(values)
	// Reservoir sampling: go over all n values and check whether it replaces a previous value.
	for ii := range k {
		values[ii] = int32(ii)
	}
	for ii := k; ii < n; ii++ {
		pos := rand.IntN(ii + 1)
		if pos < k {
			values[pos] = int32(ii)
		}
	}
}

// startEpoch resets position counter and creates shuffle where required.
func (ds *Dataset) startEpoch() {
	ds.startOfEpoch = false
	if ds.withReplacement {
		return
	}

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
		for ii, rule := range strategy.Seeds {
			if rule.NodeSet != nil {
				ds.seedsShuffle[ii] = xslices.Copy(rule.NodeSet)
			} else {
				ds.seedsShuffle[ii] = xslices.Iota[int32](int32(0), int(rule.NumNodes))
			}
		}
	}

	// Shuffle Rules for each Seeds set.
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
