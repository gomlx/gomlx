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

package context

// Aliases to GoMLX basic types.

import (
	graph "github.com/gomlx/gomlx/graph"
	types "github.com/gomlx/gomlx/types/shapes"
)

type Manager = graph.Manager
type Graph = graph.Graph
type Node = graph.Node
type Shape = types.Shape

var (
	makeShape = types.Make
	f32       = types.Float32
	f64       = types.Float64
)
