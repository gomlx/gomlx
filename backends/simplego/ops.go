// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package simplego

import (
	"slices"

	"github.com/gomlx/gomlx/backends"
)

// nodeParameter data.
type nodeParameter struct {
	name     string
	inputIdx int
}

// EqualNodeData implements nodeDataComparable for nodeParameter.
func (n *nodeParameter) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*nodeParameter)
	return n.name == o.name && n.inputIdx == o.inputIdx
}

type gatherNode struct {
	indexVectorAxis                                                  int
	offsetOutputAxes, collapsedSlicesAxes, startIndexMap, sliceSizes []int
	indicesAreSorted                                                 bool
}

// EqualNodeData implements nodeDataComparable for gatherNode.
func (g *gatherNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*gatherNode)
	if g.indexVectorAxis != o.indexVectorAxis || g.indicesAreSorted != o.indicesAreSorted {
		return false
	}
	return slices.Equal(g.offsetOutputAxes, o.offsetOutputAxes) &&
		slices.Equal(g.collapsedSlicesAxes, o.collapsedSlicesAxes) &&
		slices.Equal(g.startIndexMap, o.startIndexMap) &&
		slices.Equal(g.sliceSizes, o.sliceSizes)
}

// scatterNode is attached to the Node.data field for ScatterMax, ScatterMin, ScatterSum.
type scatterNode struct {
	indexVectorAxis                                                int
	updateWindowAxes, insertedWindowAxes, scatterAxesToOperandAxes []int
	indicesAreSorted, uniqueIndices                                bool
}

// EqualNodeData implements nodeDataComparable for scatterNode.
func (s *scatterNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*scatterNode)
	if s.indexVectorAxis != o.indexVectorAxis ||
		s.indicesAreSorted != o.indicesAreSorted ||
		s.uniqueIndices != o.uniqueIndices {
		return false
	}
	return slices.Equal(s.updateWindowAxes, o.updateWindowAxes) &&
		slices.Equal(s.insertedWindowAxes, o.insertedWindowAxes) &&
		slices.Equal(s.scatterAxesToOperandAxes, o.scatterAxesToOperandAxes)
}

// sliceNode is attached to the Node.data field for Slice.
type sliceNode struct {
	starts, limits, strides []int
}

// EqualNodeData implements nodeDataComparable for sliceNode.
func (s *sliceNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*sliceNode)
	return slices.Equal(s.starts, o.starts) &&
		slices.Equal(s.limits, o.limits) &&
		slices.Equal(s.strides, o.strides)
}

type argMinMaxNode struct {
	axis  int
	isMin bool
}

// EqualNodeData implements nodeDataComparable for argMinMaxNode.
func (a *argMinMaxNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*argMinMaxNode)
	return a.axis == o.axis && a.isMin == o.isMin
}

type reduceWindowNode struct {
	reductionType                                             backends.ReduceOpType
	windowDimensions, strides, baseDilations, windowDilations []int
	paddings                                                  [][2]int
}

// EqualNodeData implements nodeDataComparable for reduceWindowNode.
func (r *reduceWindowNode) EqualNodeData(other nodeDataComparable) bool {
	o := other.(*reduceWindowNode)
	if r.reductionType != o.reductionType {
		return false
	}
	return slices.Equal(r.windowDimensions, o.windowDimensions) &&
		slices.Equal(r.strides, o.strides) &&
		slices.Equal(r.baseDilations, o.baseDilations) &&
		slices.Equal(r.windowDilations, o.windowDilations) &&
		slices.Equal(r.paddings, o.paddings)
}
