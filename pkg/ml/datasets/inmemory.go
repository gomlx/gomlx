/*
 *	Copyright 2023 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package datasets

import (
	"encoding/gob"
	"fmt"
	"io"
	"log"
	"math/rand"
	"sync"
	"time"

	"github.com/gomlx/gomlx/backends"
	. "github.com/gomlx/gomlx/internal/exceptions"
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/core/shapes"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/xslices"
	"github.com/pkg/errors"
)

// InMemoryDataset represents a Dataset that has been completely read into the memory of the device
// it was created with -- the platform of the associated `graph.Backend`.
//
// It supports batching, shuffling (with and without replacement) and can be duplicated (only one copy
// of the underlying data is used).
//
// Finally, it supports serialization and deserialization, to accelerate loading of the data -- in case
// generating the original dataset is expensive (e.g: image transformations).
type InMemoryDataset struct {
	backend backends.Backend

	// name of the original dataset used to populate the InMemoryDataset.
	name      string
	shortName string

	// spec is saved from the original dataset.
	spec any

	// inputsAndLabelsData contains the full dataset for each of the inputs and labels.
	inputsAndLabelsData []*tensors.Tensor

	// numInputsTensors indicate how many in inputsAndLabelsData are inputs, the remainder are labels.
	numInputsTensors int

	// numExamples indicates the total number of examples cached.
	numExamples int

	// gatherExec gather the slices of the inputs/labels for a particular batch being yielded.
	gatherExec *Exec // Batch tensors.

	// muSampling serializes the sampling information, all the member variables below.
	muSampling sync.Mutex

	// batchSize to yield. If set to 0 yields only one result at a time.
	batchSize int

	// dropIncompleteBatch, when there are not enough remaining examples in the epoch.
	dropIncompleteBatch bool

	// next record to be sampled. If shuffle is given, this is an index in shuffle. If randomWithReplacement,
	// this is a count only.
	//
	// If it is set to -1, it means the dataset has been exhausted already.
	next int

	// randomWithReplacement indicates that one should simply take a random entry every time.
	randomWithReplacement bool

	// shuffle holds the current shuffle if RandomSampling was selected.
	shuffle []int

	// infinite sets whether to loop indefinitely.
	infinite bool

	// randomNumberGenerator used when random sampling, allows for deterministic random datasets.
	randomNumberGenerator *rand.Rand

	// takeN is the maximum number of examples to take, before forcing an end of epoch.
	// If <= 0, take as many as available (or continuously if InMemoryDataset.infinite=true)
	takeN int
}

// InMemory creates dataset that reads the whole contents of `ds` into memory.
//
// It uses GoMLX to batch the tensors themselves, so it takes a graph.Backend as its first
// parameter. Flat will be cached in the platform (device) the Backend was configured with.
//
// Args:
//   - `backend`: will be used to create the graph that does the caching and the ca actually does the batching.
//   - `ds`: dataset to be cached. It is read in full once, concatenating the results in the cache.
//   - `dsIsBatched`: whether the input `ds` is batched, and it's leading (first) axis is a batch size. If true,
//     count of examples is adjusted accordingly. Notice if true, the batch size must be the same for all elements
//     of the inputs and labels yielded by `ds`.
//
// Returns a `InMemoryDataset`, that is initially not shuffled and not batched. You can configure how you want to
// use it with the other configuration methods.
func InMemory(backend backends.Backend, ds train.Dataset, dsIsBatched bool) (mds *InMemoryDataset, err error) {
	mds = &InMemoryDataset{
		backend:               backend,
		randomNumberGenerator: rand.New(rand.NewSource(time.Now().UnixNano())),
		gatherExec:            MustNewExec(backend, gatherFromDataTensorsGraph),
		name:                  ds.Name(),
	}
	if sn, ok := ds.(train.HasShortName); ok {
		mds.shortName = sn.ShortName()
	} else {
		mds.shortName = mds.name[:3]
	}
	err = mds.readDataset(ds, dsIsBatched)
	if err != nil {
		return
	}
	return
}

// InMemoryFromData creates an InMemoryDataset from the static data given -- it is immediately converted to
// a tensor, if not a tensor already.
// The first dimension of each element of inputs and labels must be the batch size, and the same for every
// element.
//
// This is useful to writing unit tests, with small datasets provided inline.
//
// Example: A dataset with one input tensor and one label tensor. Each with two examples.
//
//	mds, err := InMemoryFromData(backend, "test",
//		[]any{[][]float32{{1, 2}, {3, 4}}},
//		[]any{[][]float32{{3}, {7}}})
func InMemoryFromData(
	backend backends.Backend,
	name string,
	inputs []any,
	labels []any,
) (mds *InMemoryDataset, err error) {
	mds = &InMemoryDataset{
		backend:               backend,
		randomNumberGenerator: rand.New(rand.NewSource(time.Now().UnixNano())),
		gatherExec:            MustNewExec(backend, gatherFromDataTensorsGraph),
		name:                  name,
		shortName:             name[:3],
		inputsAndLabelsData:   make([]*tensors.Tensor, 0, len(inputs)+len(labels)),
		numInputsTensors:      len(inputs),
	}

	// Parse inputs + labels in one go.
	errMsgFn := func(ii int) string {
		if ii < mds.numInputsTensors {
			return fmt.Sprintf("parsing inputs[%d]", ii)
		}
		return fmt.Sprintf("parsing labels[%d]", ii-mds.numInputsTensors)
	}
	for ii, value := range append(inputs, labels...) {
		var valueT *tensors.Tensor
		err = TryCatch[error](func() { valueT = tensors.FromAnyValue(value) })
		if err != nil {
			err = errors.WithMessage(err, errMsgFn(ii))
			return
		}
		if valueT.Shape().IsScalar() {
			err = errors.Errorf("cannot use scalars when %s", errMsgFn(ii))
			return
		}
		if ii == 0 {
			mds.numExamples = valueT.Shape().Dimensions[0]
		} else if mds.numExamples != valueT.Shape().Dimensions[0] {
			err = errors.Errorf("inputs[0] has %d examples, but got %d examples when %s -- all must have the same number",
				mds.batchSize, valueT.Shape().Dimensions[0], errMsgFn(ii))
			return
		}
		mds.inputsAndLabelsData = append(mds.inputsAndLabelsData, valueT)
	}
	return
}

// isEqualButBatchDimension compares shapes s1 and s2 but for the batch dimension, the leading axis. The batch
// dimension might differ, typically in the last batch of a dataset (if there are not enough examples to fill
// a batch).
func isEqualButBatchDimension(s1, s2 shapes.Shape) bool {
	if s1.DType != s2.DType || s1.Rank() != s2.Rank() {
		return false
	}
	// Compare all dimensions, skipping the first, the batch dimension.
	for ii := 1; ii < s1.Rank(); ii++ {
		if s1.Dimensions[ii] != s2.Dimensions[ii] {
			return false
		}
	}
	return true
}

// readDataset and generate concatenated tensors with all the data.
func (mds *InMemoryDataset) readDataset(ds train.Dataset, dsIsBatched bool) (err error) {
	// allData: for each element in `(inputs, labels)`, a slice with all the tensors returned by ds.Yield.
	var allData [][]*tensors.Tensor
	var inputsAndLabelsShapes []shapes.Shape
	var numLabelsTensors int
	mds.numExamples = 0

	// getElementDesc is used to point to the element when formatting error messages
	getElementDesc := func(ii int) (elementDesc string) {
		if ii < mds.numInputsTensors {
			elementDesc = fmt.Sprintf("inputs[%d]", ii)
		} else {
			elementDesc = fmt.Sprintf("labels[%d]", ii-mds.numInputsTensors)
		}
		return
	}

	count := 0
	for {
		spec, inputs, labels, newErr := ds.Yield()
		err = newErr
		if err == io.EOF {
			break
		}
		inputsAndLabels := append(inputs, labels...)
		if mds.numExamples == 0 {
			// First yield:
			mds.spec = spec
			mds.numInputsTensors = len(inputs)
			numLabelsTensors = len(labels)
			inputsAndLabelsShapes = make([]shapes.Shape, 0, len(inputsAndLabels))
			for _, t := range inputsAndLabels {
				inputsAndLabelsShapes = append(inputsAndLabelsShapes, t.Shape())
			}
			// Initialize allData.
			allData = make([][]*tensors.Tensor, len(inputsAndLabels))

		} else {
			// Check that the inputs/labels are consistent with previous.
			if len(inputs) != mds.numInputsTensors || len(labels) != numLabelsTensors {
				err = errors.Errorf("number of inputs and/or labels changed: it was len(inputs)=%d / len(labels)=%d, "+
					"but got %d/%d for example #%d", mds.numInputsTensors, numLabelsTensors, len(inputs), len(labels), count)
				return
			}
			for ii, t := range inputsAndLabels {
				// batch size can vary from each example read.
				var validShapes bool
				if dsIsBatched {
					validShapes = isEqualButBatchDimension(inputsAndLabelsShapes[ii], t.Shape())
				} else {
					validShapes = inputsAndLabelsShapes[ii].Equal(t.Shape())
				}
				if !validShapes {
					err = errors.Errorf("shape of %s incompatible: other examples shape was %s, for example #%d got shape %s",
						getElementDesc(ii), inputsAndLabelsShapes[ii], count, t.Shape())
					return
				}
			}
		}

		if dsIsBatched {
			// Make sure batch dimension matches: it may be different for different examples read for ds.
			batchSize := inputsAndLabels[0].Shape().Dimensions[0]
			mds.numExamples += batchSize
			for ii, t := range inputsAndLabels {
				elementBatchSize := t.Shape().Dimensions[0]
				if batchSize != elementBatchSize {
					err = errors.Errorf("batch sizes don't match: inputs[0] batch size is %d, but %s batch size is %d",
						batchSize, getElementDesc(ii), elementBatchSize)
					return
				}
			}
		} else {
			// One element at a time.
			mds.numExamples++
		}

		// Append data.
		for ii, t := range inputsAndLabels {
			allData[ii] = append(allData[ii], t)
		}
		count++
	}
	if mds.numExamples == 0 {
		err = errors.Errorf("dataset was empty, nothing mds")
		return
	}
	err = nil

	// Graph building function to concatenate results.
	alreadyBatched := dsIsBatched // Changes after the first round of concatenating.
	concatenateFn := func(parts []*Node) *Node {
		if !alreadyBatched {
			newParts := make([]*Node, 0, len(parts))
			for _, input := range parts {
				newParts = append(newParts, InsertAxes(input, 0))
			}
			parts = newParts
		}
		if len(parts) == 1 {
			return parts[0]
		}
		// Batch dimension is by convention the very first.
		return Concatenate(parts, 0)
	}
	concatenateExec := MustNewExec(mds.backend, concatenateFn)
	// Configure a large cache, since there can be quite a few cycles times the number of elements.
	concatenateExec.SetMaxCache(512)

	// Concatenate each element of inputsAndLabels across all examples: this is done by consecutively
	// concatenating at most `MaxElementsToConcat` elements at a time -- XLA doesn't handle well very large
	// computation graphs.
	const MaxExamplesToConcat = 16
	mds.inputsAndLabelsData = make([]*tensors.Tensor, 0, len(allData))
	convertToAny := func(t *tensors.Tensor) any { return t }
	for concatenationLoopCount := 0; len(allData[0]) > 1; concatenationLoopCount++ {
		newAllData := make([][]*tensors.Tensor, len(allData))
		for inputsAndLabelsIdx, allExamples := range allData {
			numConcatenations := (len(allExamples) + MaxExamplesToConcat - 1) / MaxExamplesToConcat
			newAllExamples := make([]*tensors.Tensor, numConcatenations)
			for jj := range numConcatenations {
				// Take MaxExamplesToConcat examples at a time.
				start := jj * MaxExamplesToConcat
				end := min(start+MaxExamplesToConcat, len(allExamples))
				examplesSlice := allExamples[start:end]
				examplesAsAny := xslices.Map(examplesSlice, convertToAny)
				err = TryCatch[error](func() { newAllExamples[jj] = concatenateExec.MustExec(examplesAsAny...)[0] })
				if err != nil {
					err = errors.WithMessagef(
						err,
						"while concatenating %s examples into large tensor",
						getElementDesc(inputsAndLabelsIdx),
					)
					return
				}
			}
			// Free immediately intermediary resources no longer needed.
			if concatenationLoopCount > 0 {
				// Notice we don't do this on the first loop, because the same tensor could be used
				// in different parts of the inputs (let's say the label is also passed as an input),
				// freeing it here would destroy its value for future iterations on the `inputsAndLabelsIdx`
				// loop. This means for the original read tensors, in CPU memory, we have to wait for the
				// garbage collection to collect them.
				for _, t := range allExamples {
					t.MustFinalizeAll()
				}
			}
			newAllData[inputsAndLabelsIdx] = newAllExamples
		}
		allData = newAllData
		alreadyBatched = true // If input was not already batched, now it is.
	}

	// Fully concatenated is in the last remaining batch for each element.
	for _, allExamples := range allData {
		mds.inputsAndLabelsData = append(mds.inputsAndLabelsData, allExamples[0])
	}
	return
}

// Memory returns an approximation of the memory being used.
func (mds *InMemoryDataset) Memory() uintptr {
	var mem uintptr
	for _, t := range mds.inputsAndLabelsData {
		mem += t.Shape().Memory()
	}
	return mem
}

// NumExamples cached.
func (mds *InMemoryDataset) NumExamples() int {
	return mds.numExamples
}

// Copy returns a copy of the dataset. It uses the same underlying data -- so very little memory is used.
//
// The copy comes configured by default with sequential reading (not random sampling), non-looping, and reset.
func (mds *InMemoryDataset) Copy() *InMemoryDataset {
	return &InMemoryDataset{
		backend:               mds.backend,
		name:                  mds.name,
		spec:                  mds.spec,
		inputsAndLabelsData:   mds.inputsAndLabelsData,
		numInputsTensors:      mds.numInputsTensors,
		numExamples:           mds.numExamples,
		gatherExec:            mds.gatherExec,
		takeN:                 mds.takeN,
		randomNumberGenerator: rand.New(rand.NewSource(time.Now().UnixNano())),
	}
}

// Name implements `train.Dataset`
func (mds *InMemoryDataset) Name() string {
	return mds.name
}

// ShortName implements `train.HasShortName`
func (mds *InMemoryDataset) ShortName() string {
	return mds.shortName
}

// SetName sets the name of the dataset and optionally its ShortName, and returns the updated dataset.
func (mds *InMemoryDataset) SetName(name string, shortName ...string) *InMemoryDataset {
	mds.name = name
	if len(shortName) > 0 {
		mds.shortName = shortName[0]
	} else {
		mds.shortName = name[:3]
	}
	return mds
}

// Reset implements `train.Dataset`
func (mds *InMemoryDataset) Reset() {
	mds.muSampling.Lock()
	defer mds.muSampling.Unlock()

	mds.next = 0
	if mds.shuffle != nil {
		mds.shuffleLocked()
	}
}

// indicesNextYield retrieve the indices for the next Yield call. This needs to be done protected by `muSampling`,
// but the gathering of the returned indices can be parallelized.
func (mds *InMemoryDataset) indicesNextYield() (indices []int) {
	mds.muSampling.Lock()
	defer mds.muSampling.Unlock()
	if mds.next == -1 {
		return // dataset already exhausted.
	}
	n := mds.batchSize
	if n <= 0 {
		n = 1
	}
	indices = make([]int, 0, n)
	for mds.next < mds.numExamples && len(indices) < n {
		if len(mds.shuffle) > 0 {
			indices = append(indices, mds.shuffle[mds.next])
		} else if mds.randomWithReplacement {
			indices = append(indices, mds.randomNumberGenerator.Intn(mds.numExamples))
		} else {
			indices = append(indices, mds.next)
		}
		mds.next++
	}
	if len(indices) < n && mds.dropIncompleteBatch {
		// Drop the incomplete batch.
		indices = nil
	}
	if mds.next >= mds.numExamples {
		mds.next = -1
	}
	if mds.takeN > 0 && mds.next >= mds.takeN*n {
		mds.next = -1
	}
	return
}

// gatherFromDataTensorsGraph will gather the indices from each data input. The `indicesAndDataTensors` first element
// is `indices`, the others are all data values. It returns one gathered value for each data value
// (`len(indicesAndData) - 1` tensors).
func gatherFromDataTensorsGraph(indicesAndData []*Node) (gathered []*Node) {
	indices := indicesAndData[0]
	dataNodes := indicesAndData[1:]
	gathered = make([]*Node, 0, len(dataNodes))
	for _, n := range dataNodes {
		gathered = append(gathered, Gather(n, indices))
	}
	return
}

// Yield implements `train.Dataset`.
//
// Returns next batch's inputs and labels or single example if BatchSize is set to 0.
func (mds *InMemoryDataset) Yield() (spec any, inputs []*tensors.Tensor, labels []*tensors.Tensor, err error) {
	if len(mds.inputsAndLabelsData) == 0 {
		err = errors.Errorf("InMemoryDataset is empty, maybe it has been finalized?")
		return
	}
	for _, data := range mds.inputsAndLabelsData {
		if !data.Ok() {
			err = errors.Errorf("InMemoryDataset data has been been finalized?")
			return
		}
	}
	indices := mds.indicesNextYield()
	if len(indices) == 0 {
		if !mds.infinite {
			// Dataset is already exhausted.
			err = io.EOF
			return
		}

		// If looping infinitely, automatically Reset and pull new indices.
		mds.Reset()
		indices = mds.indicesNextYield()
		if len(indices) == 0 {
			log.Printf("InMemoryDataset configured for infinite loop, but Reset failed to generate new examples!?")
			err = io.EOF
			return
		}
	}

	// Gather the elements (inputs and labels) all in one call, given the indices.
	inputsAndLabels := make([]*tensors.Tensor, len(mds.inputsAndLabelsData))
	indicesAndData := make([]any, 0, len(mds.inputsAndLabelsData)+1)
	var indicesT *tensors.Tensor
	if mds.batchSize == 0 {
		// Index should be a scalar.
		indicesAndData = append(indicesAndData, indices[0])
	} else {
		// Indices are shaped [batch_size, 1].
		indicesT = tensors.FromFlatDataAndDimensions(indices, len(indices), 1)
		defer indicesT.MustFinalizeAll() // Free immediately after use.
		indicesAndData = append(indicesAndData, indicesT)
	}
	for _, data := range mds.inputsAndLabelsData {
		indicesAndData = append(indicesAndData, data)
	}
	inputsAndLabels, err = mds.gatherExec.Exec(indicesAndData...)
	if err != nil {
		err = errors.WithMessagef(err, "failed gathering examples from mds data, indices=%v", indices)
		return
	}

	// Prepare the return values.
	spec = mds.spec
	if mds.numInputsTensors > 0 {
		inputs = inputsAndLabels[:mds.numInputsTensors]
	}
	numLabels := len(mds.inputsAndLabelsData) - mds.numInputsTensors
	if numLabels > 0 {
		labels = inputsAndLabels[mds.numInputsTensors:]
	}
	return
}

// RandomWithReplacement configures the InMemoryDataset to return random elements with replacement.
// If this is configured, Shuffle is canceled.
//
// It returns the modified InMemoryDataset, so calls can be cascaded if one wants.
func (mds *InMemoryDataset) RandomWithReplacement() *InMemoryDataset {
	mds.muSampling.Lock()
	defer mds.muSampling.Unlock()
	mds.randomWithReplacement = true
	mds.shuffle = nil
	return mds
}

// Shuffle configures the InMemoryDataset to shuffle the order of the data. It returns random elements
// without replacement. If this is configured, RandomWithReplacement is canceled.
//
// At each call to Reset() it is reshuffled. It happens automatically if dataset is configured to Loop.
//
// It returns the modified InMemoryDataset, so calls can be cascaded if one wants.
func (mds *InMemoryDataset) Shuffle() *InMemoryDataset {
	mds.muSampling.Lock()
	defer mds.muSampling.Unlock()
	mds.randomWithReplacement = false
	mds.shuffleLocked()
	return mds
}

// shuffleLocked shuffles dataset yield order. It assumed muSampling is locked.
func (mds *InMemoryDataset) shuffleLocked() {
	if mds.shuffle == nil {
		mds.shuffle = make([]int, mds.numExamples)
	}
	for ii := 0; ii < mds.numExamples; ii++ {
		newPos := rand.Intn(ii + 1)
		if newPos == ii {
			mds.shuffle[ii] = ii
		} else {
			// Swap position with the new example.
			mds.shuffle[newPos], mds.shuffle[ii] = ii, mds.shuffle[newPos]
		}
	}
}

// BatchSize configures the InMemoryDataset to return batches of the given size. dropIncompleteBatch is set to true,
// it will simply drop examples if there are not enough to fill a batch -- this can only happen on the last
// batch of an epoch. Otherwise, it will return a partially filled batch.
//
// If `n` is set to 0, it reverts back to yielding one example at a time.
//
// It returns the modified InMemoryDataset, so calls can be cascaded if one wants.
func (mds *InMemoryDataset) BatchSize(n int, dropIncompleteBatch bool) *InMemoryDataset {
	mds.muSampling.Lock()
	defer mds.muSampling.Unlock()
	mds.batchSize = n
	mds.dropIncompleteBatch = dropIncompleteBatch
	return mds
}

// WithRand sets the random number generator (RNG) for shuffling or random sampling. This allows for repeatable
// deterministic random sampling, if one wants. The default is to use an RNG initialized with the current
// nanosecond time.
//
// If dataset is configured with Shuffle, this re-shuffles the dataset immediately.
//
// It returns the modified InMemoryDataset, so calls can be cascaded if one wants.
func (mds *InMemoryDataset) WithRand(rng *rand.Rand) *InMemoryDataset {
	mds.muSampling.Lock()
	defer mds.muSampling.Unlock()
	mds.randomNumberGenerator = rng
	if mds.shuffle != nil {
		mds.shuffleLocked()
	}
	return mds
}

// WithSpec sets the `spec` that is returned in Yield. The default is to use the one read from the
// original dataset passed to InMemory call. This allows one to set to something different.
//
// It returns the modified InMemoryDataset, so calls can be cascaded if one wants.
func (mds *InMemoryDataset) WithSpec(spec any) *InMemoryDataset {
	mds.muSampling.Lock()
	defer mds.muSampling.Unlock()
	mds.spec = spec
	return mds
}

// Infinite sets whether the dataset should loop indefinitely. The default is `infinite = false`, which
// causes the dataset to going through the data only once before returning io.EOF.
//
// It returns the modified InMemoryDataset, so calls can be cascaded if one wants.
func (mds *InMemoryDataset) Infinite(infinite bool) *InMemoryDataset {
	mds.muSampling.Lock()
	defer mds.muSampling.Unlock()
	mds.infinite = infinite
	return mds
}

// TakeN configures dataset to only take N examples before returning io.EOF.
// If set to 0 or -1, it takes as many as there is data.
// If configured, it automatically disables InMemoryDataset.Infinite
func (mds *InMemoryDataset) TakeN(n int) *InMemoryDataset {
	if n > 0 {
		mds.Infinite(false)
	}
	mds.muSampling.Lock()
	defer mds.muSampling.Unlock()
	mds.takeN = n
	return mds
}

// FinalizeAll will immediately free all the underlying data (and not wait for the garbage collector).
// This invalidates not only this InMemoryDataset, but also all other copies that use the same data (created
// with Copy).
//
// This is not concurrency safe: if there are concurrent calls to sampling, this may lead to an undefined
// state or errors.
func (mds *InMemoryDataset) FinalizeAll() {
	for _, data := range mds.inputsAndLabelsData {
		data.MustFinalizeAll()
	}
	mds.inputsAndLabelsData = nil
}

// GobSerialize in-memory content to the encoder.
//
// Only the underlying data is serialized. The graph.Backend or the sampling configuration is not serialized.
// The contents of the `spec` (see WithSpec) is also not serialized.
func (mds *InMemoryDataset) GobSerialize(encoder *gob.Encoder) (err error) {
	enc := func(data any) {
		if err != nil {
			return
		}
		err = encoder.Encode(data)
		if err != nil {
			err = errors.Wrapf(err, "failed to Serialize InMemoryDataset")
		}
	}
	numInputsAndLabels := int32(len(mds.inputsAndLabelsData))
	enc(mds.name)
	enc(mds.numExamples)
	enc(numInputsAndLabels)
	enc(mds.numInputsTensors)
	if err != nil {
		return
	}

	for _, data := range mds.inputsAndLabelsData {
		hasLocal := data.IsLocal()
		err = data.GobSerialize(encoder)
		if err != nil {
			return err
		}
		if !hasLocal && data.IsLocal() {
			// Free the local storage copy of the tensor created, to avoid using too much space.
			data.FinalizeLocal()
		}
	}
	return
}

// GobDeserializeInMemoryToDevice dataset from the decoder. It requires a `graph.Backend` and the deviceNum where the data
// is going to be stored -- it drops the local storage copy of the values.
//
// No sampling configuration is recovered, and the InMemoryDataset created is sequential (no random sampling)
// that reads through only one epoch. The random number generator is also newly initialized (see
// InMemoryDataset.WithRand).
func GobDeserializeInMemoryToDevice(
	backend backends.Backend,
	deviceNum backends.DeviceNum,
	decoder *gob.Decoder,
) (mds *InMemoryDataset, err error) {
	dec := func(data any) {
		if err != nil {
			return
		}
		err = decoder.Decode(data)
		if err != nil {
			err = errors.Wrapf(err, "failed to DeserializeInMemory")
		}
	}
	mds = &InMemoryDataset{
		backend:               backend,
		randomNumberGenerator: rand.New(rand.NewSource(time.Now().UnixNano())),
		gatherExec:            MustNewExec(backend, gatherFromDataTensorsGraph),
	}

	var numInputsAndLabels int32
	dec(&mds.name)
	dec(&mds.numExamples)
	dec(&numInputsAndLabels)
	dec(&mds.numInputsTensors)
	if err != nil {
		return
	}
	mds.inputsAndLabelsData = make([]*tensors.Tensor, 0, numInputsAndLabels)

	var tensor *tensors.Tensor
	for range numInputsAndLabels {
		tensor, err = tensors.GobDeserializeToDevice(decoder, backend, deviceNum)
		if err != nil {
			return
		}
		mds.inputsAndLabelsData = append(mds.inputsAndLabelsData, tensor)
	}
	return
}
