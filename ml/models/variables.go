package models

// Variable (or weights) of a model, typically learned during training, but can also be used as large constants.
//
// They have different "views":
//
//   - Outside a computation graph, they have a shape, and if already initialized, a concrete *tensor.Tensor value.
//   - Within a graph building function, they have a graph node that represents them. One per computation graph,
//     that's why the Variable.ValueGraph() method takes a graph parameter as input.
//
// You can create them during the graph building function, or outside of it.
//
// Always use it by reference (pointer), never by value, to keep all the various views in sync.
type Variable struct{}
