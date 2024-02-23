package sampler

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
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
	inputs = make([]tensor.Tensor, 0, 2*len(ds.strategy.rules))

	ds.muSample.Lock()
	ds.frozen = true
	if ds.startOfEpoch {
		ds.startEpoch()
	}

	// Sample seeds: requires a lock for the sampling.
	for ii, seedsRule := range ds.strategy.seeds {
		seeds, mask := ds.sampleSeeds(ii, seedsRule)
		inputs = append(inputs, seeds, mask)
	}
	ds.muSample.Unlock()

	// Sampling edges: doesn't require lock.

	return
}

// sampleSeeds returns the sampled seeds and their masks.
// For sampling seeds, ds.muSample must be locked.
func (ds *Dataset) sampleSeeds(seedIdx int, rule *Rule) (seeds, mask *tensor.Local) {
	seeds = tensor.FromShape(shapes.Make(shapes.Int32, rule.count))
	seedsRef := seeds.AcquireData()
	defer seedsRef.Release()
	seedsData := seedsRef.Flat().([]int32)
	if ds.shuffle {

	} else {

	}

	mask = tensor.FromScalarAndDimensions(true, rule.count)
	return
}

// startEpoch resets position counter and creates shuffle where required.
func (ds *Dataset) startEpoch() {
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
				numNodes := ds.sampler.d.NodeTypesToCount[rule.nodeTypeName]
				ds.seedsShuffle[ii] = slices.Iota[int32](int32(0), int(numNodes))
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
