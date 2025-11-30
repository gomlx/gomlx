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

// Package imdb contains code to download and prepare datasets with IMDB Dataset of 50k Movie Reviews.
//
// This can be used to train models, but this library has no library per se. See a demo model training
// in sub-package `demo`.
package imdb

import (
	"bytes"
	"encoding/gob"
	"fmt"
	"io"
	"math/rand"
	"os"
	"path"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gomlx/gomlx/examples/downloader"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gomlx/pkg/ml/train"
	"github.com/gomlx/gomlx/pkg/support/fsutil"
	"github.com/pkg/errors"
)

const (
	DownloadURL  = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
	LocalTarFile = "aclImdb_v1.tar.gz"
	TarHash      = `c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe`
	LocalDir     = "aclImdb"
	BinaryFile   = "aclImdb.bin"
)

var (
	// IncludeSeparators indicates whether when parsing files it should create tokens out of the
	// separators (commas, dots, etc).
	IncludeSeparators = false

	// CaseSensitive indicates whether token collection should be case-sensitive.
	CaseSensitive = false

	// LoadedVocab is materialized after calling Download.
	LoadedVocab *Vocab

	// LoadedExamples is materialized after calling Download. It is based on LoadedVocab.
	// Once loaded it should remain immutable.
	LoadedExamples []*Example

	// reWords captures what are considered tokens.
	reWords = regexp.MustCompile("[[:word:]]+")
)

// Download IMDB reviews dataset to current directory, un-tar it, parses all individual files and
// saves the binary file version.
//
// The vocabulary and examples loaded are set to LoadedVocab and LoadedExamples.
//
// If it's already downloaded, simply load binary file version.
func Download(baseDir string) error {
	baseDir = fsutil.MustReplaceTildeInDir(baseDir)
	loaded, err := loadBinary(baseDir)
	if err != nil {
		return err
	}
	if loaded {
		fmt.Printf("Loaded data from %q: %d examples, %d unique tokens, %d tokens in total.\n",
			BinaryFile, len(LoadedExamples), len(LoadedVocab.ListEntries), LoadedVocab.TotalCount)
		return nil
	}

	if err := downloader.DownloadAndUntarIfMissing(DownloadURL, baseDir, LocalTarFile, LocalDir, TarHash); err != nil {
		return errors.Wrapf(err, "imdb.Download failed")
	}
	LoadedVocab, LoadedExamples, err = LoadIndividualFiles(baseDir)
	if err != nil {
		return err
	}

	if err := saveBinary(baseDir); err != nil {
		return err
	}
	return nil
}

func loadBinary(baseDir string) (loaded bool, err error) {
	f, err := os.Open(path.Join(baseDir, BinaryFile))
	if os.IsNotExist(err) {
		return false, nil
	}
	if err != nil {
		return false, errors.Wrapf(err, "failed loadBinary(%q) while opening file", BinaryFile)
	}
	defer func() {
		_ = f.Close()
	}()

	// Check that the configuration matches.
	dec := gob.NewDecoder(f)
	var loadedIncludeSeparators, loadedCaseSensitive bool
	if err := dec.Decode(&loadedIncludeSeparators); err != nil {
		return false, errors.Wrapf(err, "failed loadBinary(%q) while reading", BinaryFile)
	}
	if err := dec.Decode(&loadedCaseSensitive); err != nil {
		return false, errors.Wrapf(err, "failed loadBinary(%q) while reading", BinaryFile)
	}
	if (loadedIncludeSeparators != IncludeSeparators) || (loadedCaseSensitive != CaseSensitive) {
		// Configuration is different from the one saved on BinaryFile, consider it not loaded,
		// to force regeneration.
		return false, nil
	}

	fmt.Println("> Loading previously generated preprocessed binary file.")
	if err := dec.Decode(&LoadedVocab); err != nil {
		return false, errors.Wrapf(err, "failed loadBinary(%q) while reading", BinaryFile)
	}
	if err := dec.Decode(&LoadedExamples); err != nil {
		return false, errors.Wrapf(err, "failed loadBinary(%q) while reading", BinaryFile)
	}
	return true, nil
}

func saveBinary(baseDir string) error {
	fmt.Println("> Saving preprocessed binary file.")
	f, err := os.Create(path.Join(baseDir, BinaryFile))
	if err != nil {
		return errors.Wrapf(err, "failed to saveBinary(%q)", BinaryFile)
	}
	closed := false
	defer func() {
		if !closed {
			_ = f.Close()
		}
	}()

	// Save configuration.
	enc := gob.NewEncoder(f)
	if err := enc.Encode(IncludeSeparators); err != nil {
		return errors.Wrapf(err, "failed saveBinary(%q) while writing", BinaryFile)
	}
	if err := enc.Encode(CaseSensitive); err != nil {
		return errors.Wrapf(err, "failed saveBinary(%q) while writing", BinaryFile)
	}

	// Save vocabulary.
	if err := enc.Encode(LoadedVocab); err != nil {
		return errors.Wrapf(err, "failed saveBinary(%q) while writing", BinaryFile)
	}
	if err := enc.Encode(LoadedExamples); err != nil {
		return errors.Wrapf(err, "failed saveBinary(%q) while writing", BinaryFile)
	}

	// Report back result of close.
	err = f.Close()
	closed = true
	return err
}

// VocabEntry include the Token and its count.
type VocabEntry struct {
	Token string
	Count TokenId
}

type TokenId = int32

// Vocab stores vocabulary information for the whole corpus.
type Vocab struct {
	ListEntries []VocabEntry
	MapTokens   map[string]TokenId
	TotalCount  int64
}

// NewVocab creates a new vocabulary, with the first token set to "<INVALID>", usually a placeholder
// for padding, and the second token set to "<START>" to indicate start of sentence.
func NewVocab() *Vocab {
	v := &Vocab{
		MapTokens:   make(map[string]TokenId),
		ListEntries: []VocabEntry{{"<INVALID>", 0}, {"<START>", 1}},
	}
	for ii, entry := range v.ListEntries {
		v.MapTokens[entry.Token] = TokenId(ii)
	}
	return v
}

// SortByFrequency sorts the vocabs by their frequency, and returns a map to convert the
// token ids from before the sorting to their new values.
//
// Special tokens "<INVALID>" and "<START>" remain unchanged.
func (v *Vocab) SortByFrequency() (oldIDtoNewID map[TokenId]TokenId) {
	subSlice := v.ListEntries[2:] // "<INVALID>" and "<START>" remain unchanged.
	sort.Slice(subSlice, func(i, j int) bool {
		return subSlice[i].Count > subSlice[j].Count
	})

	// Create new map of tokens to its id.
	newMapTokens := make(map[string]TokenId, len(v.MapTokens))
	for ii, entry := range v.ListEntries {
		newMapTokens[entry.Token] = TokenId(ii)
	}

	// Create conversion map.
	oldIDtoNewID = make(map[TokenId]TokenId, len(v.MapTokens))
	for token, oldId := range v.MapTokens {
		newID := newMapTokens[token]
		oldIDtoNewID[oldId] = newID
	}
	v.MapTokens = newMapTokens
	return
}

// RegisterToken returns the index for the token, and increments the count for the token.
func (v *Vocab) RegisterToken(token string) (idx TokenId) {
	v.TotalCount++
	var found bool
	idx, found = v.MapTokens[token]
	if !found {
		v.MapTokens[token] = TokenId(len(v.ListEntries))
		v.ListEntries = append(v.ListEntries, VocabEntry{token, 1})
	} else {
		v.ListEntries[idx].Count++
	}
	return idx
}

// DatasetType refers to either a train or test example(s).
type DatasetType int

const (
	TypeTrain DatasetType = iota
	TypeTest
)

// Example encapsulates all the information of one example in the IMDB 50k dataset. The fields are:
//
//   - Set can be 0 or 1 for "test", train".
//   - Label is 0, 1 or 2 for negative/positive/unlabeled examples.
//   - Rating is a value from 1 to 10 in imdb. For unlabeled examples they are marked all as 0.
//   - Length is the length (in # of tokens) of the content.
//   - Content are the tokens of the IMDB entry -- there should be a vocabulary associated to
//     the dataset.
type Example struct {
	Set           DatasetType
	Label, Rating int8
	Length        int
	Content       []TokenId
}

// NewExample parses an IMDB content file, tokenize it using the given Vocab and returns the
// parsed example.
//
// It doesn't fill the SetIdx, Label and Rating attributes.
func NewExample(contents []byte, vocab *Vocab) *Example {
	e := &Example{}
	// Remove line breaks <br/>.
	contents = bytes.ReplaceAll(contents, []byte("<br />"), []byte(" "))
	partsIndices := reWords.FindAllIndex(contents, -1)
	appendTokenFn := func(token string) {
		id := vocab.RegisterToken(token)
		e.Content = append(e.Content, id)
	}
	last := 0
	for idx := range partsIndices {
		start, end := partsIndices[idx][0], partsIndices[idx][1]
		if IncludeSeparators && start > last {
			// Get separator token, between last word.
			sep := string(contents[last:start])
			if sep != " " {
				appendTokenFn(sep)
			}
		}
		// Append token.
		token := string(contents[start:end])
		if !CaseSensitive {
			token = strings.ToLower(token)
		}
		appendTokenFn(token)
		last = end
	}
	e.Length = len(e.Content)
	return e
}

func (e *Example) String(vocab *Vocab) string {
	parts := make([]string, 0, len(e.Content))
	for _, tokenId := range e.Content {
		parts = append(parts, vocab.ListEntries[tokenId].Token)
	}
	return "[" + strings.Join(parts, "] [") + "]"
}

func LoadIndividualFiles(baseDir string) (vocab *Vocab, examples []*Example, err error) {
	setSlices := [][]string{
		{"neg", "pos", "unsup"},
		{"neg", "pos"},
	}

	vocab = NewVocab()

	for setIdx, setDir := range []string{"train", "test"} {
		for label, sliceDir := range setSlices[setIdx] {
			dir := path.Join(baseDir, LocalDir, setDir, sliceDir)
			var files []os.DirEntry
			files, err = os.ReadDir(dir)
			if err != nil {
				err = errors.Wrapf(err, "failed to read examples from %s", dir)
				return
			}
			for _, f := range files {
				if f.IsDir() {
					continue
				}
				if !strings.HasSuffix(f.Name(), ".txt") {
					// Skip non-text files.
					continue
				}

				// Get rating from filename.
				var rating int
				name := f.Name()
				name = name[:len(name)-4] // trim ".txt"
				fileNameParts := strings.Split(name, "_")
				if len(fileNameParts) == 2 {
					// Try to convert: if failed, that's fine, we just keep 0 rating.
					rating, _ = strconv.Atoi(fileNameParts[1])
				}

				// Read file.
				var contents []byte
				contents, err = os.ReadFile(path.Join(dir, f.Name()))
				if err != nil {
					err = errors.Wrapf(err, "failed to read example %s from %s", f.Name(), dir)
					return
				}
				_ = contents

				// Create new example.
				e := NewExample(contents, vocab)
				e.Set = DatasetType(setIdx)
				e.Label = int8(label)
				e.Rating = int8(rating)
				examples = append(examples, e)
				if e.Length == 2931 {
					fmt.Printf("%s\n", contents)
				}
			}
		}
	}

	// Sort token ids by their frequencies.
	oldIDToNewID := vocab.SortByFrequency()
	for _, e := range examples {
		for ii, oldID := range e.Content {
			e.Content[ii] = oldIDToNewID[oldID]
		}
	}
	return
}

// Dataset implements train.Dataset. It allows for concurrent Yield calls,
// so one can feed it to ParallelizedDataset.
//
// Yield:
//
//   - inputs[0] (TokenId)[batch_size, ds.MaxLen] will hold the first ds.MaxLen tokens of the example.
//     If the example is shorter than that, the rest is filled with 0s at the start: meaning, the empty space comes first.
//     If there is enough space, the first token will be 1 (the "<START>" token).
//   - labels[0] (int8)[batch_size] labels are 0, 1 or 2 for negative/positive/unlabeled examples.
type Dataset struct {
	name             string
	DatasetType      DatasetType
	MaxLen, MaxVocab int
	BatchSize        int

	// muIndices protects the indices, the mutable part of the Dataset, to allow
	// for concurrent calls to Yield.
	muIndices       sync.Mutex
	ExamplesIndices []int32
	Pos             int
	Infinite        bool
	Shuffler        *rand.Rand
}

// Assert *Dataset implements train.Dataset
var _ train.Dataset = &Dataset{}

// NewDataset creates a labeled Dataset. See Dataset for details.
//
// - name passed along for debugging and metrics naming.
// - dsType: dataset of TypeTrain or TypeTest.
// - maxLen: max length of the content kept per test. Since GoMLX/XLA only works with fixed size tensors, the memory used grows linear with this.
// - infinite: data loops forever (if shuffler is set, it reshuffles at every end of epoch).
func NewDataset(name string, dsType DatasetType, maxLen, batchSize int, infinite bool) *Dataset {
	ds := &Dataset{
		name:        name,
		DatasetType: dsType,
		MaxLen:      maxLen,
		BatchSize:   batchSize,
		Infinite:    infinite,
	}
	ds.createExamplesIndices()
	return ds
}

func (ds *Dataset) createExamplesIndices() {
	ds.ExamplesIndices = make([]int32, 0, 25000)
	for idx, ex := range LoadedExamples {
		if ex.Set == ds.DatasetType && ex.Label != 2 {
			ds.ExamplesIndices = append(ds.ExamplesIndices, int32(idx))
		}
	}
}

// Shuffle marks dataset to yield shuffled results.
func (ds *Dataset) Shuffle() *Dataset {
	ds.Shuffler = rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	ds.Reset()
	return ds
}

// NewUnsupervisedDataset with the DatasetType assumed to be TypeTrain.
func NewUnsupervisedDataset(name string, maxLen, batchSize int, infinite bool) *Dataset {
	set := TypeTrain
	ds := &Dataset{
		name:        name,
		DatasetType: set,
		MaxLen:      maxLen,
		BatchSize:   batchSize,
		Infinite:    infinite,
	}
	ds.createExamplesIndices()
	return ds
}

// Name implements train.Dataset interface.
func (ds *Dataset) Name() string { return ds.name }

// Yield implements train.Dataset interface. If not infinite, return io.EOF at the end of the dataset.
//
// It trims the examples to ds.MaxLen tokens, taken from the end.
//
// It returns `spec==nil` always, since `inputs` and `labels` have always the same type of content.
//
// It can be called concurrently.
func (ds *Dataset) Yield() (spec any, inputs, labels []*tensors.Tensor, err error) {
	// Lock only while selecting the indices for the batch.
	ds.muIndices.Lock()
	if ds.Infinite {
		// Infinite: we only take batches of the specified size, and drop the last ones if not enough to fit the batch.
		if ds.Pos+ds.BatchSize > len(ds.ExamplesIndices) {
			// Infinite: needs a reshuffle.
			ds.resetLocked()
		}
	} else {
		// 1 epoch: take the last batch even if there are only one element.
		if ds.Pos >= len(ds.ExamplesIndices) {
			ds.muIndices.Unlock()
			return nil, nil, nil, io.EOF
		}
	}

	// Effective batch size may be smaller than requested.
	batchSize := min(ds.BatchSize, len(ds.ExamplesIndices)-ds.Pos)
	batchIndices := ds.ExamplesIndices[ds.Pos : ds.Pos+batchSize]
	ds.Pos += batchSize
	ds.muIndices.Unlock()

	// From now on ds is immutable, and it can be run concurrently.

	// Build input tensor.
	input := tensors.FromScalarAndDimensions(TokenId(0), batchSize, ds.MaxLen)
	labelsData := make([]int8, batchSize)
	tensors.MustMutableFlatData(input, func(inputData []TokenId) {
		for batchIdx, exampleIdx := range batchIndices {
			ex := LoadedExamples[exampleIdx]
			labelsData[batchIdx] = int8(ex.Label)
			exInput := inputData[batchIdx*ds.MaxLen:]
			content := ex.Content
			if len(content) > ds.MaxLen {
				content = content[len(content)-ds.MaxLen:]
			}
			copy(exInput[ds.MaxLen-len(content):], content) // Copy at most ds.MaxLen.
			if len(content) < ds.MaxLen {
				exInput[ds.MaxLen-len(content)-1] = 1 // Token "<START>"
			}
			labelsData[batchIdx] = ex.Label
		}
	})
	inputs = []*tensors.Tensor{input}
	labels = []*tensors.Tensor{tensors.FromAnyValue(labelsData)}
	return
}

// Reset restarts the dataset from the beginning. Can be called after io.EOF is reached,
// for instance when running another evaluation on a test dataset.
func (ds *Dataset) Reset() {
	ds.muIndices.Lock()
	defer ds.muIndices.Unlock()
	ds.resetLocked()
}

// resetLocked implements Reset, when Dataset.muIndices is already locked.
func (ds *Dataset) resetLocked() {
	if ds.Shuffler != nil {
		for ii := range ds.ExamplesIndices {
			jj := ds.Shuffler.Intn(len(ds.ExamplesIndices))
			ds.ExamplesIndices[ii], ds.ExamplesIndices[jj] = ds.ExamplesIndices[jj], ds.ExamplesIndices[ii]
		}
	}
	ds.Pos = 0
}

// InputToString returns a string rendered content of one row (pointed to by batchIdx) of an input.
// The input is assumed to be a batch created by a Dataset object.
func InputToString(input *tensors.Tensor, batchIdx int) string {
	if batchIdx < 0 || batchIdx >= input.Shape().Dimensions[0] {
		return fmt.Sprintf("invalid batch idx %d: input shape is %s", batchIdx, input.Shape())
	}
	maxLen := input.Shape().Dimensions[1]
	parts := make([]string, 0, maxLen)
	tensors.MustConstFlatData(input, func(inputData []TokenId) {
		start := batchIdx * maxLen
		for _, tokenId := range inputData[start : start+maxLen] {
			if tokenId == 0 {
				continue
			}
			parts = append(parts, LoadedVocab.ListEntries[tokenId].Token)
		}
	})
	return strings.Join(parts, " ")
}

// Assert that *Dataset implements train.Dataset.
var _ train.Dataset = &Dataset{}
