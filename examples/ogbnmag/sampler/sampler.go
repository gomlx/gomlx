package sampler

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"

	humanize "github.com/dustin/go-humanize"
	. "github.com/gomlx/gomlx/internal/exceptions"
	"github.com/gomlx/gomlx/pkg/core/tensors"
	"github.com/gomlx/gopjrt/dtypes"
	"github.com/pkg/errors"
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
// to sample from. This way the resulting tensors will always be the same Shape -- required
// by XLA.
//
// There are 3 phases when using the Sampler:
//
// (1) Specify the full graph data: define node type and edge types, for example for the [OGBN-MAG dataset]:
//
//	Sampler := Sampler.New()
//	Sampler.AddNodeType("papers", mag.NumPapers)
//	Sampler.AddNodeType("authors", mag.NumAuthors)
//	Sampler.AddEdgeType("writes", "authors", "papers", mag.EdgesWrites, /* reverse= */ false)
//	Sampler.AddEdgeType("writtenBy", "authors", "papers", mag.EdgesWrites, /* reverse= */ true)
//	Sampler.AddEdgeType("cites", "papers", "papers", mag.EdgesCites, /*reverse=*/ false)
//	Sampler.AddEdgeType("citedBy", "papers", "papers", mag.EdgesCites, /*reverse=*/ true)
//
// (2) Create and specify sampling strategy: sampling generates always a tree of elements, with fixed shaped tensors.
// It uses padding if sampling something that doesn't have enough examples to sample. Example:
//
//	trainStrategy := Sampler.NewStrategy()
//	Seeds := trainStrategy.NodesFromSet("Seeds", "papers", batchSize, /* subset= */TrainSplits)
//	citedBy := Seeds.FromEdges(/* Name= */ "citedBy", /* EdgeType= */ "citedBy", 5)
//	authors := Seeds.SampleFromEdgesRandomWithoutReplacement(/* Name= */ "authors", /* edgeSet= */ "writtenBy", 5)
//	coauthoredPapers := authors.SampleFromEdgesRandomWithoutReplacement(/* Name= */ "coauthoredPapers", /* edgeSet= */ "writes", 5)
//	citingAuthors := citedBy.SampleFromEdgesRandomWithoutReplacement(/* Name= */ "citingAuthors", /* edgeSet= */ "writtenBy", 5)
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
// All the information kept by Sampler is available for reading, but avoid changing it
// directly, and instead use the provided methods.
//
// Example usage:
//
// [OGBN-MAG dataset]: https://github.com/gomlx/gomlx/tree/main/examples/ogbnmag
type Sampler struct {
	EdgeTypes        map[string]*EdgeType
	NodeTypesToCount map[string]int32
	Frozen           bool // When true, it can no longer be changed.
}

// EdgeType information used by the Sampler.
type EdgeType struct {
	// SourceNodeType, TargetNodeType of the edges.
	Name, SourceNodeType, TargetNodeType string
	numTargetNodes                       int

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

// NumSourceNodes for the source node type -- total number of nodes, even if they are not used by the edges.
func (et *EdgeType) NumSourceNodes() int { return len(et.Starts) }

// NumTargetNodes for the source node type -- total number of nodes, even if they are not used by the edges.
func (et *EdgeType) NumTargetNodes() int { return et.numTargetNodes }

// NumEdges for this type.
func (et *EdgeType) NumEdges() int { return len(et.EdgeTargets) }

// EdgeTargetsForSourceIdx returns a slice with the target nodes for the given source nodes.
// Don't modify the returned slice, it's in use by the Sampler -- make a copy if you need to modify.
func (et *EdgeType) EdgeTargetsForSourceIdx(srcIdx int32) []int32 {
	if srcIdx < 0 || int(srcIdx) >= len(et.Starts) {
		Panicf(
			"invalid source node (%q) index %d for edge type %q (only %d source nodes)",
			et.SourceNodeType,
			srcIdx,
			et.Name,
			len(et.Starts),
		)
	}
	var start int32
	if srcIdx > 0 {
		start = et.Starts[srcIdx-1]
	}
	end := et.Starts[srcIdx]
	return et.EdgeTargets[start:end]
}

// EdgePairTensor creates new tensors for the source and target indices.
// It can be used by LayerWise inference.
func (et *EdgeType) EdgePairTensor() EdgePair[*tensors.Tensor] {
	sources := make([]int32, et.NumEdges())
	current := int32(0)
	for srcIdx, last := range et.Starts {
		for ii := current; ii < last; ii++ {
			sources[ii] = int32(srcIdx)
		}
		current = last
	}
	return EdgePair[*tensors.Tensor]{
		SourceIndices: tensors.FromValue(sources),
		TargetIndices: tensors.FromValue(et.EdgeTargets),
	}
}

// New creates a new empty Sampler.
//
// After creating it, use AddNodeType and AddEdgeType to define where to
// sample from.
func New() *Sampler {
	return &Sampler{
		EdgeTypes:        make(map[string]*EdgeType),
		NodeTypesToCount: make(map[string]int32),
	}
}

// AddNodeType adds the node with the given Name and Count to the collection of known nodes.
// This assumes this is a dense representation of the node type -- all indices are valid from `0` to `Count-1`
//
// A sparse node type (e.g.: indices are random numbers from 0 to MAXINT-1 or strings) is not supported.
func (s *Sampler) AddNodeType(name string, count int) {
	if s.Frozen {
		Panicf(
			"Sampler is Frozen, that is, a strategy was already created with NewStrategy() and hence can no longer be modified.",
		)
	}
	if count > math.MaxInt32 {
		Panicf("Sampler uses int32, but node type %q Count of %d given is bigger than the max possible.", name, count)
	} else if count <= 0 {
		Panicf("Count of %d for node type %q invalid, it must be > 0", count, name)
	}
	s.NodeTypesToCount[name] = int32(count)
	if count <= 0 {
		Panicf("Sampler.AddNodeType(Name=%q, Count=%d): Count must be > 0", name, count)
	}
}

// AddEdgeType adds the edge type to the list of known edges.
// It takes the node types names (must have been added with AddNodeType), and
// the `edges` given as pairs (source node, target node).
//
// If `reverse` is true, it reverts the direction of the sampling. Note that
// `sourceNodeType` and `targetNodeType` are given before reversing the direction of the
// edges. So if `reverse` is true, the source is interpreted as the target and vice versa.
// Same as the values of `edges`.
//
// The `edges` tensor must have Shape `(Int32)[N, 2]`. It's contents are changed in place
// -- they are sorted by the source node type (or target if reversed).
// But the edges information themselves are not lost.
func (s *Sampler) AddEdgeType(name, sourceNodeType, targetNodeType string, edges *tensors.Tensor, reverse bool) {
	if s.Frozen {
		Panicf(
			"Sampler is frozen, that is, a strategy was already created with NewStrategy() and hence can no longer be modified.",
		)
	}
	if edges.Rank() != 2 || edges.DType() != dtypes.Int32 ||
		edges.Shape().Dimensions[1] != 2 || edges.Shape().Dimensions[0] == 0 {
		Panicf("invalid edge Shape %s for AddEdgeType(): it must be shaped like (Int32)[N, 2]",
			edges.Shape())
	}
	countSource := s.NodeTypesToCount[sourceNodeType]
	countTarget := s.NodeTypesToCount[targetNodeType]
	columnSrc, columnTgt := 0, 1
	if reverse {
		columnSrc, columnTgt = 1, 0
		countSource, countTarget = countTarget, countSource
		sourceNodeType, targetNodeType = targetNodeType, sourceNodeType
	}

	tensors.MustMutableFlatData[int32](edges, func(edgesData []int32) {
		// Sort edges according to the source column.
		pairsToSort := pairsToSort{
			data:       edgesData,
			sortColumn: columnSrc,
		}
		sort.Sort(&pairsToSort)

		// Store edge-lists per source in new SamplerEdgeInfo.
		numEdges := int32(edges.Shape().Dimensions[0])
		samplerEdges := &EdgeType{
			Name:           name,
			SourceNodeType: sourceNodeType,
			TargetNodeType: targetNodeType,
			numTargetNodes: int(countTarget),
			Starts:         make([]int32, countSource),
			EdgeTargets:    make([]int32, numEdges),
		}
		currentSourceIdx := int32(0)
		for row := 0; row < int(numEdges); row++ {
			sourceIdx, targetIdx := edgesData[row<<1+columnSrc], pairsToSort.data[row<<1+columnTgt]
			if sourceIdx < 0 || sourceIdx >= countSource {
				Panicf(
					"edge type %q has an edge whose source (node type %q) is %d, but node type %q only has a max of %d elements registered (with AddNodeType())",
					name,
					sourceNodeType,
					sourceIdx,
					sourceNodeType,
					countSource,
				)
			}
			if targetIdx < 0 || targetIdx >= countTarget {
				Panicf(
					"edge type %q has an edge whose target (node type %q) is %d, but node type %q only has a max of %d elements registered (with AddNodeType())",
					name,
					targetNodeType,
					targetIdx,
					targetNodeType,
					countTarget,
				)
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
		s.EdgeTypes[name] = samplerEdges
	})
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
// based on the same Sampler.
func (s *Sampler) NewStrategy() *Strategy {
	s.Frozen = true
	return &Strategy{
		Sampler: s,
		Rules:   make(map[string]*Rule),
	}
}

// String returns a multi-line informative description of the Sampler data specification.
func (s *Sampler) String() string {
	parts := make([]string, 0, 1+len(s.NodeTypesToCount)+len(s.EdgeTypes))
	var frozenDesc string
	if s.Frozen {
		frozenDesc = ", Frozen"
	}
	parts = append(parts, fmt.Sprintf("Sampler: %d node types, %d edge types%s",
		len(s.NodeTypesToCount), len(s.EdgeTypes), frozenDesc))
	for name, count := range s.NodeTypesToCount {
		parts = append(parts, fmt.Sprintf(
			"\tNodeType %q: %s items", name, humanize.Comma(int64(count))))
	}
	for name, edge := range s.EdgeTypes {
		parts = append(parts, fmt.Sprintf(
			"\tEdgeType %q: [%q]->[%q], %s edges",
			name, edge.SourceNodeType, edge.TargetNodeType, humanize.Comma(int64(edge.NumEdges()))))
	}
	return strings.Join(parts, "\n")
}

func initGob() {
	gob.Register(&EdgeType{})
	gob.Register(&Sampler{})
}

// Save Sampler: it will include the edges indices, so it can be reloaded and ready to go.
func (s *Sampler) Save(filePath string) (err error) {
	initGob()
	f, err := os.Create(filePath)
	if err != nil {
		err = errors.Wrapf(err, "creating %q to save Sampler", filePath)
		return
	}
	enc := gob.NewEncoder(f)
	err = enc.Encode(s)
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

// Load previously saved Sampler.
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
	err = dec.Decode(s)
	if err != nil {
		s = nil
		err = errors.Wrapf(err, "trying to decode Sampler from %q", filePath)
		return
	}
	_ = f.Close()
	return
}

// NameForNodeDependentDegree returns the name of the input field that contains the degree of the given rule node,
// with respect to the dependent rule node.
func NameForNodeDependentDegree(ruleName, dependentName string) string {
	return fmt.Sprintf("[%s->%s].degree", ruleName, dependentName)
}
