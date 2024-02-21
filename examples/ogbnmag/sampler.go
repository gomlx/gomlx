package ogbnmag

import (
	. "github.com/gomlx/exceptions"
	"github.com/gomlx/gomlx/types/shapes"
	"github.com/gomlx/gomlx/types/tensor"
	"math"
	"sort"
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
// Before starting sampling, one has to register nodes and edge types.
//
// Example, for OGBN-MAG:
//
//	sampler := mag.NewSampler()
//	sampler.AddNodeType("papers", mag.NumPapers)
//	sampler.AddNodeType("authors", mag.NumAuthors)
//	sampler.AddEdgeType("writes", "authors", "papers").Pairs(mag.EdgesWrites).Done()
//	sampler.AddEdgeType("written_by", "papers", "authors").Pairs(mag.EdgesWrites).Reverse().Done()
//
// Each registration of an edge type creates a corresponding structure to store the edges, that
// will be used for sampling.
//
// After registration, one can
type Sampler struct {
	EdgeTypes        map[string]*SamplerEdgeType
	NodeTypesToCount map[string]int32
}

// SamplerEdgeType information used by the sampler.
type SamplerEdgeType struct {
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

// NewSampler creates a new empty sampler.
//
// After creating it, use AddNodeType and AddEdgeType to define where to
// sample from.
func NewSampler() *Sampler {
	return &Sampler{
		EdgeTypes:        make(map[string]*SamplerEdgeType),
		NodeTypesToCount: make(map[string]int32),
	}
}

// AddNodeType adds the node with the given name and count to the collection of known nodes.
func (s *Sampler) AddNodeType(name string, count int) {
	if count > math.MaxInt32 {
		Panicf("Sampler uses int32, but node type %q count of %d given is bigger than the max possible.", name, count)
	} else if count <= 0 {
		Panicf("count of %d for node type %q invalid, it must be > 0", count, name)
	}
	s.NodeTypesToCount[name] = int32(count)
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
func (s *Sampler) AddEdgeType(name, sourceNodeType, targetNodeType string, edges *tensor.Local, reverse bool) {
	if edges.Rank() != 2 || edges.DType() != shapes.Int32 ||
		edges.Shape().Dimensions[1] != 2 || edges.Shape().Dimensions[0] == 0 {
		Panicf("invalid edge shape %s for AddEdgeType(): it must be shaped like (Int32)[N, 2]",
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

	// Sort edges according to the source column.
	edgesRef := edges.AcquireData()
	defer edgesRef.Release()
	pairsToSort := pairsToSort{
		data:       edgesRef.Flat().([]int32),
		sortColumn: columnSrc,
	}
	sort.Sort(&pairsToSort)

	// Store edge-lists per source in new SamplerEdgeInfo.
	numEdges := int32(edges.Shape().Dimensions[0])
	samplerEdges := &SamplerEdgeType{
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

	s.EdgeTypes[name] = samplerEdges
}

// EdgeTargetsForSourceIdx returns a slice with the target nodes for the given source nodes.
// Don't modify the returned slice, it's in use by the Sampler -- make a copy if you need to modify.
func (e *SamplerEdgeType) EdgeTargetsForSourceIdx(srcIdx int32) []int32 {
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
