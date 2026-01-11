/*
 *	Copyright 2024 Jan Pfeifer
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

package attention

import (
	. "github.com/gomlx/gomlx/pkg/core/graph"
	"github.com/gomlx/gomlx/pkg/ml/layers/attention/pos"
)

// RoPE applies rotary position embeddings to the last dimension (even length).
// Shapes: x [..., seq_len, head_dim|embed_dim]. Reference: RoFormer.
//
// Deprecated: Use pos.NewRoPE(baseFreq).Apply(x, startPos) instead.
// This function is kept for backward compatibility.
func RoPE(x *Node, startPos int, baseFreq float64) *Node {
	g := x.Graph()
	startPosNode := Const(g, []int32{int32(startPos)})
	return pos.NewRoPE(baseFreq).Apply(x, startPosNode)
}

// RoPEWithCustomDim applies RoPE to x[..., dimStart:dimEnd] (even length).
//
// Deprecated: Use pos.NewRoPEWithDimRange(baseFreq, dimStart, dimEnd).Apply(x, startPos) instead.
// This function is kept for backward compatibility.
func RoPEWithCustomDim(x *Node, startPos int, baseFreq float64, dimStart, dimEnd int) *Node {
	g := x.Graph()
	startPosNode := Const(g, []int32{int32(startPos)})
	return pos.NewRoPEWithDimRange(baseFreq, dimStart, dimEnd).Apply(x, startPosNode)
}
