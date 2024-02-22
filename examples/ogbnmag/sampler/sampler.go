package sampler

import (
	"encoding/gob"
	"fmt"
	humanize "github.com/dustin/go-humanize"
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"github.com/pkg/errors"
	"math"
	"os"
	"sort"
	"strings"
)

// PaddingIndex is used for all sampling not fulfilled.
// Notice 0 is also valid node index. One should always use
// the mask returned by the Sampler to check whether a value
// is padding or not.
const PaddingIndex = 0

// Sampler can be used to dynamically sample a Graph to be used in GNNs.
// It implements the train.Dataset interface.
//
// It always samples nodes with the same size, padding whenever there is not enough elements
// to sample from. This way the resulting tensors will always be the same shape -- required
// by XLA.
//
// There are 3 phases when using the Sampler:
//
// (1) Specify the full graph data: define node type and edge types, for example for the [OGBN-MAG dataset]:
//
//	sampler := sampler.New()
//	sampler.AddNodeType("papers", mag.NumPapers)
//	sampler.AddNodeType("authors", mag.NumAuthors)
//	sampler.AddEdgeType("writes", "authors", "papers", mag.EdgesWrites, /* reverse= */ false)
//	sampler.AddEdgeType("writtenBy", "authors", "papers", mag.EdgesWrites, /* reverse= */ true)
//	sampler.AddEdgeType("cites", "papers", "papers", mag.EdgesCites, /*reverse=*/ false)
//	sampler.AddEdgeType("citedBy", "papers", "papers", mag.EdgesCites, /*reverse=*/ true)
//
// (2) Create and specify sampling strategy: sampling generates always a tree of elements, with fixed shaped tensors.
// It uses padding if sampling something that doesn't have enough examples to sample. Example:
//
//	trainStrategy := sampler.NewStrategy()
//	seeds := trainStrategy.NodesFromSubset("seeds", "papers", batchSize, /* subset= */TrainSplits)
//	citedBy := seeds.FromEdgesRandomWithoutReplacement(/* name= */ "citedBy", /* edgeSet= */ "citedBy", 5)
//	authors := seeds.SampleFromEdgesRandomWithoutReplacement(/* name= */ "authors", /* edgeSet= */ "writtenBy", 5)
//	coauthoredPapers := authors.SampleFromEdgesRandomWithoutReplacement(/* name= */ "coauthoredPapers", /* edgeSet= */ "writes", 5)
//	citingAuthors := citedBy.SampleFromEdgesRandomWithoutReplacement(/* name= */ "citingAuthors", /* edgeSet= */ "writtenBy", 5)
//
// (3) Create a dataset and use it. The `spec` returned by `Yield` is a pointer to the [Strategy] object,
// and can be used to create a [GraphSample] by providing it the inputs and labels lists. Example:
//
//	  trainDataset := trainStrategy.Dataset()
//	  for {
//	  	spec, inputs, labels, err = trainDataset.Yield()
//	  	samplerStrategy := spec.(*mag.Strategy)
//		  	sample := samplerStrategy.Parse(inputs, labels)
//	  }
//
// Each registration of an edge type creates a corresponding structure to store the edges, that
// will be used for sampling.
//
// # After registration, one can
//
// [OGBN-MAG dataset]: https://github.com/gomlx/gomlx/tree/main/examples/ogbnmag
type Sampler struct {
	d samplerData
}

type samplerData struct {
	EdgeTypes        map[string]*edgeType
	NodeTypesToCount map[string]int32
	Frozen           bool // When true, it can no longer be changed.
}

// edgeType information used by the sampler.
type edgeType struct {
	// SourceNodeType, TargetNodeType of the edges.
	Name, SourceNodeType, TargetNodeType string

	// Starts has one entry for each source node (shifted by 1): it points to the start of the list of
	// target nodes (edges) that this source node is connected.
	//
	// So for source node `i`, the list of edges start at `Starts[i-1]` and ends at `Starts[i]`,
	// except if `i == 0` in which case the start is at 0.
	// It's normal to be 0 if the source node has no target nodes.
	//
	// The number of sources is given by `len(Starts)`.
	Starts []int32

	// List of target nodes ordered by source nodes.
	// The source node for each edge is given by `Starts` above.
	EdgeTargets []int32
}

// New creates a new empty sampler.
//
// After creating it, use AddNodeType and AddEdgeType to define where to
// sample from.
func New() *Sampler {
	return &Sampler{
		d: samplerData{
			EdgeTypes:        make(map[string]*edgeType),
			NodeTypesToCount: make(map[string]int32),
		},
	}
}

// AddNodeType adds the node with the given name and count to the collection of known nodes.
// This assumes this is a dense representation of the node type -- all indices are valid from `0` to `count-1`
//
// A sparse node type (e.g.: indices are random numbers from 0 to MAXINT-1 or strings) is not supported.
func (s *Sampler) AddNodeType(name string, count int) {
	if s.d.Frozen {
		Panicf("Sampler is Frozen, that is, a strategy was already created with NewStrategy() and hence can no longer be modified.")
	}
	if count > math.MaxInt32 {
		Panicf("Sampler uses int32, but node type %q count of %d given is bigger than the max possible.", name, count)
	} else if count <= 0 {
		Panicf("count of %d for node type %q invalid, it must be > 0", count, name)
	}
	s.d.NodeTypesToCount[name] = int32(count)
	if count <= 0 {
		Panicf("Sampler.AddNodeType(name=%q, count=%d): count must be > 0", name, count)
	}
}

// AddEdgeType adds the edge type to the list of known edges.
// It takes the node types names (must have been added with AddNodeType), and
// the `edges` given as pairs (source node, target node).
//
// If `reverse` is true, it reverts the direction of the sampling. Note that
// `sourceNodeType` and `targetNodeType` are given before reversing the direction of the
// edges. So if `reverse` is true, the source is interpreted as the target and vice-versa.
// Same as the values of `edges`.
//
// The `edges` tensor must have shape `(Int32)[N, 2]`. It's contents are changed in place
// -- they are sorted by the source node type (or target if reversed).
// But the edges information themselves are not lost.
func (s *Sampler) AddEdgeType(name, sourceNodeType, targetNodeType string, edges tensor.Tensor, reverse bool) {
	if s.d.Frozen {
		Panicf("Sampler is Frozen, that is, a strategy was already created with NewStrategy() and hence can no longer be modified.")
	}
	if edges.Rank() != 2 || edges.DType() != shapes.Int32 ||
		edges.Shape().Dimensions[1] != 2 || edges.Shape().Dimensions[0] == 0 {
		Panicf("invalid edge shape %s for AddEdgeType(): it must be shaped like (Int32)[N, 2]",
			edges.Shape())
	}
	countSource := s.d.NodeTypesToCount[sourceNodeType]
	countTarget := s.d.NodeTypesToCount[targetNodeType]
	columnSrc, columnTgt := 0, 1
	if reverse {
		columnSrc, columnTgt = 1, 0
		countSource, countTarget = countTarget, countSource
		sourceNodeType, targetNodeType = targetNodeType, sourceNodeType
	}

	// Sort edges according to the source column.
	edgesRef := edges.Local().AcquireData()
	defer edgesRef.Release()
	pairsToSort := pairsToSort{
		data:       edgesRef.Flat().([]int32),
		sortColumn: columnSrc,
	}
	sort.Sort(&pairsToSort)

	// Store edge-lists per source in new SamplerEdgeInfo.
	numEdges := int32(edges.Shape().Dimensions[0])
	samplerEdges := &edgeType{
		Name:           name,
		SourceNodeType: sourceNodeType,
		TargetNodeType: targetNodeType,
		Starts:         make([]int32, countSource),
		EdgeTargets:    make([]int32, numEdges),
	}
	currentSourceIdx := int32(0)
	for row := 0; row < int(numEdges); row++ {
		sourceIdx, targetIdx := pairsToSort.data[row<<1+columnSrc], pairsToSort.data[row<<1+columnTgt]
		if sourceIdx < 0 || sourceIdx >= countSource {
			Panicf("edge type %q has an edge whose source (node type %q) is %d, but node type %q only has a max of %d elements registered (with AddNodeType())",
				name, sourceNodeType, sourceIdx, sourceNodeType, countSource)
		}
		if targetIdx < 0 || targetIdx >= countTarget {
			Panicf("edge type %q has an edge whose target (node type %q) is %d, but node type %q only has a max of %d elements registered (with AddNodeType())",
				name, targetNodeType, targetIdx, targetNodeType, countTarget)
		}
		samplerEdges.EdgeTargets[row] = targetIdx
		for currentSourceIdx < sourceIdx {
			samplerEdges.Starts[currentSourceIdx] = int32(row)
			currentSourceIdx++
		}
	}
	for ; currentSourceIdx < countSource; currentSourceIdx++ {
		samplerEdges.Starts[currentSourceIdx] = int32(numEdges)
	}
	s.d.EdgeTypes[name] = samplerEdges
}

// NumEdges of the edgeType.
func (e *edgeType) NumEdges() int {
	return len(e.EdgeTargets)
}

// EdgeTargetsForSourceIdx returns a slice with the target nodes for the given source nodes.
// Don't modify the returned slice, it's in use by the Sampler -- make a copy if you need to modify.
func (e *edgeType) EdgeTargetsForSourceIdx(srcIdx int32) []int32 {
	if srcIdx < 0 || int(srcIdx) >= len(e.Starts) {
		Panicf("invalid source node (%q) index %d for edge type %q (only %d source nodes)", e.SourceNodeType, srcIdx, e.Name, len(e.Starts))
	}
	var start int32
	if srcIdx > 0 {
		start = e.Starts[srcIdx-1]
	}
	end := e.Starts[srcIdx]
	return e.EdgeTargets[start:end]
}

type pairsToSort struct {
	data       []int32
	sortColumn int
}

func (p *pairsToSort) Len() int { return len(p.data) / 2 }
func (p *pairsToSort) Less(i, j int) bool {
	return p.data[i<<1+p.sortColumn] < p.data[j<<1+p.sortColumn]
}
func (p *pairsToSort) Swap(i, j int) {
	for column := 0; column < 2; column++ {
		p.data[i<<1+column], p.data[j<<1+column] = p.data[j<<1+column], p.data[i<<1+column]
	}
}

// NewStrategy yields a new Strategy object, based on the graph data definitions of the Sampler object.
//
// Once a strategy is created, the Sampler can no longer be changed -- but multiple strategies can be created
// based on the same sampler.
func (s *Sampler) NewStrategy() *Strategy {
	s.d.Frozen = true
	return &Strategy{
		sampler: s,
	}
}

// String returns a multi-line informative description of the sampler data specification.
func (s *Sampler) String() string {
	parts := make([]string, 0, 1+len(s.d.NodeTypesToCount)+len(s.d.EdgeTypes))
	var frozenDesc string
	if s.d.Frozen {
		frozenDesc = ", Frozen"
	}
	parts = append(parts, fmt.Sprintf("Sampler: %d node types, %d edge types%s",
		len(s.d.NodeTypesToCount), len(s.d.EdgeTypes), frozenDesc))
	for name, count := range s.d.NodeTypesToCount {
		parts = append(parts, fmt.Sprintf(
			"\tNodeType %q: %s items", name, humanize.Comma(int64(count))))
	}
	for name, edge := range s.d.EdgeTypes {
		parts = append(parts, fmt.Sprintf(
			"\tEdgeType %q: [%q]->[%q], %s edges",
			name, edge.SourceNodeType, edge.TargetNodeType, humanize.Comma(int64(edge.NumEdges()))))
	}
	return strings.Join(parts, "\n")
}

func initGob() {
	gob.Register(&edgeType{})
	gob.Register(&samplerData{})
}

// Save sampler: it will include the edges indices, so it can be reloaded and ready to go.
func (s *Sampler) Save(filePath string) (err error) {
	initGob()
	f, err := os.Create(filePath)
	if err != nil {
		err = errors.Wrapf(err, "creating %q to save Sampler", filePath)
		return
	}
	enc := gob.NewEncoder(f)
	err = enc.Encode(&s.d)
	if err != nil {
		err = errors.WithMessagef(err, "encoding Sampler to save to %q", filePath)
		return
	}
	err = f.Close()
	if err != nil {
		err = errors.Wrapf(err, "close file %q, where tensor was saved", filePath)
		return
	}
	return
}

// Load previously saved sampler.
// If filePath doesn't exist, it returns an error that can be checked with [os.IsNotExist]
func Load(filePath string) (s *Sampler, err error) {
	initGob()
	f, err := os.Open(filePath)
	if err != nil {
		if os.IsNotExist(err) {
			return
		}
		err = errors.Wrapf(err, "trying to load Sampler from %q", filePath)
		return
	}
	dec := gob.NewDecoder(f)
	s = &Sampler{}
	err = dec.Decode(&s.d)
	if err != nil {
		s = nil
		err = errors.Wrapf(err, "trying to decode Sampler from %q", filePath)
		return
	}
	_ = f.Close()
	return
}
