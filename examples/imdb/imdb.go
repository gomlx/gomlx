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
	"github.com/gomlx/gomlx/ml/data"
	"github.com/gomlx/gomlx/ml/train"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/slices"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
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
)

const (
	DownloadURL  = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
	LocalTarFile = "aclImdb_v1.tar.gz"
	TarHash      = "c40f74a18d3b61f90feba1e17730e0d38e8b97c05fde7008942e91923d1658fe"
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
	baseDir = data.ReplaceTildeInDir(baseDir)
	loaded, err := loadBinary(baseDir)
	if err != nil {
		return err
	}
	if loaded {
		fmt.Printf("Loaded data from %q: %d examples, %d unique tokens, %d tokens in total.\n",
			BinaryFile, len(LoadedExamples), len(LoadedVocab.ListEntries), LoadedVocab.TotalCount)
		return nil
	}

	if err := data.DownloadAndUntarIfMissing(DownloadURL, baseDir, LocalTarFile, LocalDir, TarHash); err != nil {
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
	Count int
}

// Vocab stores vocabulary information for the whole corpus.
type Vocab struct {
	ListEntries []VocabEntry
	MapTokens   map[string]int
	TotalCount  int
}

// NewVocab creates a new vocabulary, with the first token set to "<INVALID>", usually a placeholder
// for padding, and the second token set to "<START>" to indicate start of sentence.
func NewVocab() *Vocab {
	v := &Vocab{
		MapTokens:   make(map[string]int),
		ListEntries: []VocabEntry{{"<INVALID>", 0}, {"<START>", 1}},
	}
	for ii, entry := range v.ListEntries {
		v.MapTokens[entry.Token] = ii
	}
	return v
}

// SortByFrequency sorts the vocabs by their frequency, and returns a map to convert the
// token ids from before the sorting to their new values.
//
// Special tokens "<INVALID>" and "<START>" remain unchanged.
func (v *Vocab) SortByFrequency() (oldIDtoNewID map[int]int) {
	subSlice := v.ListEntries[2:] // "<INVALID>" and "<START>" remain unchanged.
	sort.Slice(subSlice, func(i, j int) bool {
		return subSlice[i].Count > subSlice[j].Count
	})

	// Create new map of tokens to its id.
	newMapTokens := make(map[string]int, len(v.MapTokens))
	for ii, entry := range v.ListEntries {
		newMapTokens[entry.Token] = ii
	}

	// Create conversion map.
	oldIDtoNewID = make(map[int]int, len(v.MapTokens))
	for token, oldId := range v.MapTokens {
		newID := newMapTokens[token]
		oldIDtoNewID[oldId] = newID
	}
	v.MapTokens = newMapTokens
	return
}

// RegisterToken returns the index for the token, and increments the count for the token.
func (v *Vocab) RegisterToken(token string) (idx int) {
	v.TotalCount++
	var found bool
	idx, found = v.MapTokens[token]
	if !found {
		v.MapTokens[token] = len(v.ListEntries)
		v.ListEntries = append(v.ListEntries, VocabEntry{token, 1})
	} else {
		v.ListEntries[idx].Count++
	}
	return idx
}

// SetType refers to either a train or test example(s).
type SetType int

const (
	Train SetType = iota
	Test
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
	Set           SetType
	Label, Rating int
	Length        int
	Content       []int
}

// NewExample parses an IMDB content file, tokenize it using the given Vocab and returns the
// parsed example.
//
// It doesn't fill the SetIdx, Label and Rating attributes.
func NewExample(contents []byte, vocab *Vocab) *Example {
	e := &Example{}
	// Remove line breaks <br/>.
	contents = bytes.Replace(contents, []byte("<br />"), []byte(" "), -1)
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
				e.Set = SetType(setIdx)
				e.Label = label
				e.Rating = rating
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
type Dataset struct {
	name             string
	SetType          SetType
	LabelDType       shapes.DType
	MaxLen, MaxVocab int
	BatchSize        int
	Examples         []*Example

	// muIndices protects the indices, the mutable part of the Dataset, to allow
	// for concurrent calls to Yield.
	muIndices                 sync.Mutex
	Pos                       int
	Infinite, WithReplacement bool
	Shuffle                   *rand.Rand
}

// Assert *Dataset implements train.Dataset
var _ train.Dataset = &Dataset{}

// NewDataset creates a labeled Dataset.
func NewDataset(name string, set SetType, maxLen, batchSize int, labelDType shapes.DType, infinite bool, shuffle *rand.Rand) *Dataset {
	if shuffle == nil {
		shuffle = rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	}
	ds := &Dataset{
		name:       name,
		SetType:    set,
		LabelDType: labelDType,
		MaxLen:     maxLen,
		BatchSize:  batchSize,
		Infinite:   infinite,
		Shuffle:    shuffle,
	}
	ds.Examples = make([]*Example, 0, 25000)
	for _, ex := range LoadedExamples {
		if ex.Set == set && ex.Label != 2 {
			ds.Examples = append(ds.Examples, ex)
		}
	}
	ds.Reset()
	return ds
}

// NewUnsupervisedDataset with the SetType assumed to be Train.
func NewUnsupervisedDataset(name string, maxLen, batchSize int, labelDType shapes.DType, infinite bool, shuffle *rand.Rand) *Dataset {
	if shuffle == nil {
		shuffle = rand.New(rand.NewSource(time.Now().UTC().UnixNano()))
	}
	set := Train
	ds := &Dataset{
		name:       name,
		SetType:    set,
		LabelDType: labelDType,
		MaxLen:     maxLen,
		BatchSize:  batchSize,
		Infinite:   infinite,
		Shuffle:    shuffle,
	}
	ds.Examples = make([]*Example, 0, 25000)
	for _, ex := range LoadedExamples {
		if ex.Set == set && ex.Label == 2 {
			ds.Examples = append(ds.Examples, ex)
		}
	}
	ds.Reset()
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
func (ds *Dataset) Yield() (spec any, inputs, labels []tensor.Tensor, err error) {
	// Lock only while selecting the indices for the batch.
	ds.muIndices.Lock()
	if !ds.Infinite && ds.Pos+ds.BatchSize > len(ds.Examples) {
		ds.muIndices.Unlock()
		return nil, nil, nil, io.EOF
	}
	if ds.Pos+ds.BatchSize > len(ds.Examples) {
		// Infinite: needs a reshuffle.
		ds.resetLocked()
	}

	// Enumerate examples to yield.
	var examplesIdx []int
	if ds.Infinite && ds.WithReplacement {
		examplesIdx = make([]int, ds.BatchSize)
		for ii := 0; ii < ds.BatchSize; ii++ {
			examplesIdx[ii] = ds.Shuffle.Intn(len(ds.Examples))
		}
	} else {
		examplesIdx = slices.Iota(ds.Pos, ds.BatchSize)
		ds.Pos += ds.BatchSize
	}
	ds.muIndices.Unlock()
	// From now on ds is immutable, and it can be run concurrently.

	// Build input tensor.
	input := tensor.FromScalarAndDimensions(0, ds.BatchSize, ds.MaxLen)
	inputRef := input.AcquireData()
	defer inputRef.Release()
	inputData := tensor.FlatFromRef[int](inputRef)
	labelsData := make([]int, ds.BatchSize)
	for batchIdx, datasetIdx := range examplesIdx {
		ex := ds.Examples[datasetIdx]
		labelsData[batchIdx] = ex.Label
		exInput := inputData[batchIdx*ds.MaxLen:]
		content := ex.Content
		if len(content) > ds.MaxLen {
			content = content[len(content)-ds.MaxLen:]
		}
		copy(exInput[ds.MaxLen-len(content):], content) // Copy at most ds.MaxLen.
		if len(content) < ds.MaxLen {
			exInput[ds.MaxLen-len(content)-1] = 1 // Token "<START>"
		}
	}
	inputs = []tensor.Tensor{input}
	labels = []tensor.Tensor{tensor.FromAnyValue(shapes.CastAsDType(labelsData, ds.LabelDType))}
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
	if ds.Infinite && ds.WithReplacement {
		return
	}
	for ii := range ds.Examples {
		jj := ds.Shuffle.Intn(len(ds.Examples))
		ds.Examples[ii], ds.Examples[jj] = ds.Examples[jj], ds.Examples[ii]
	}
	ds.Pos = 0
}

// InputToString returns a string rendered content of one row (pointed to by batchIdx) of an input.
// The input is assumed to be a batch created by a Dataset object.
func InputToString(input tensor.Tensor, batchIdx int) string {
	if batchIdx < 0 || batchIdx >= input.Shape().Dimensions[0] {
		return fmt.Sprintf("invalid batch idx %d: input shape is %s", batchIdx, input.Shape())
	}
	maxLen := input.Shape().Dimensions[1]
	localRef := input.Local().AcquireData()
	defer localRef.Release()
	inputData := tensor.FlatFromRef[int](localRef)
	start := batchIdx * maxLen
	parts := make([]string, 0, maxLen)
	for _, tokenId := range inputData[start : start+maxLen] {
		if tokenId == 0 {
			continue
		}
		parts = append(parts, LoadedVocab.ListEntries[tokenId].Token)
	}
	return strings.Join(parts, " ")
}

// Assert that *Dataset implements train.Dataset.
var _ train.Dataset = &Dataset{}
