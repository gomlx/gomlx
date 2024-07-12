# Bare-Bones Graph Compilation

If one wants complete control on how graphs are being executed, or just to understand what is going
on under the `Exec` object, here is how one compiles and executes a graph the "bare-bones" way:

```go
package main

import (
	"fmt"
	. "gomlx/graph"
	"log"
)

func main() {
	manager := BuildManager().Platform("Host").MustDone()
	g := manager.NewGraph().WithName("sum")
	one := Const(g, 1)
	sum := SumGraph(one, one)
	if !g.Ok() {
		log.Fatalf("Failed to create graph: %+v", g.Error())
	}
	g.Compile(sum)
	fmt.Printf("1+1=%s\n", g.Run(nil))
}
```

Remember, this is the bare-bones way of doing things, see the [tutorial](../examples/tutorial/tutorial.ipynb)
for the simpler version. But it's worth going through it to understand what happens behind the scenes. 
The full code is in [examples/tutorial/graph/main.go](../examples/tutorial/graph/main.go)

One line at a time, this is what is happening:

* `BuildManager().<options setting>.MustDone()` creates a `Manager` object, which connects to an accelerator
  nad manages graph creation and execution for that platform. Usually one creates one at the beginning of
  the program and passes it around.
    * The *platform* is where to execute the computation graphs, either the CPU ("Host" as in this case), or
      an accelerator (e.g: "CUDA" or "TPU"). One can list available and supported platforms with
      `GetPlatforms()`.
* `manager.NewGraph().WithName("sum")`: creates an empty new computation graph, that we are going to build.
* `C(graph, 1)`: creates a new constant node initialized with the value of `int(1)` in the g.
  It returns a `*Node` type. Notice that our graphs only support a few data types
  (called `types.DType`), as of now only `Int64`, `Float32` and `Float64`, which maps to the
  Go types `int`, `float32` and `float64`. More will be supported later.
* `if !g.Ok() { ... g.Error() ...}`: each operation on the graph can potentially
  return an error. Checking it at every step would be cumbersome, so instead we record the
  first error (and stack trace) that happened during a graph creation, and it can be checked
  only once at the end, like here. In practical terms, this is all there is to it. But for
  a more detailed discussion, see [error_handling.md](docs/error_handling.md).
* `g.Compile(sum)` compiles the g. After it is compiled it can no longer be changed. It
  takes as input a list of `*Node` that are the output of the graph execution.
* `g.Run(nil)` executes the graph passing `nil` as a map of the graph parameters -- more on that
  on next section. Here it outputs whatever is evaluated for the `sum` node, since that was the node
  passed as output for `g.Compile`.



## Graph Parameters and Shapes

In this section we create parameters in the graph, so that we can run the graph with different input
parameters. The code is in [examples/tutorial/euclidean1/main.go](../examples/tutorial/euclidean1/main.go)

To exemplify this let's first create a graph function to calculate the euclidean distance:

```go
func EuclideanGraph(a, b *Node) *Node {
	diff := Sub(a, b)
	return Sqrt(Mul(diff, diff))
}
```

This is inline with what we presented on the previous section. But now let's compile this graph in a way that
we can feed different values, as opposed to hard-code it with constants. Note: this is still the **bare-bones**
way of doing this, the simpler way will be explained in the next session.

```go
func main() {
	manager := BuildManager().Platform("Host").MustDone()
	g := manager.NewGraph().WithName("euclidean")

	vectorShape := types.MakeShape(types.Float64, 3) // 3D vectors.
	a := g.Parameter("a", vectorShape)
	b := g.Parameter("b", vectorShape)
	if !g.Ok() {
		log.Fatalf("Failed to create graph: %+v", g.Error())
	}
	output := EuclideanGraph(a, b)
	g.Compile(output)

	fmt.Printf("EuclideanDistance((1, 1, 1), (0, 0, 0)) = %s\n",
		g.Run(ParamsMap{a: []float64{0, 0, 0}, b: []float64{1, 1, 1}}))    
}
```

The output we get is:

```
EuclideanDistance((1, 1, 1), (0, 0, 0)) = (Float64)[]: 1.7320508075688772
```

The creation of the `Manager` and the `Graph` are as described before. New in this snippet are:

* `vectorShape := types.MakeShape(types.Float64, 3)`: Every node has an associated `types.Shape`,
  generally described by its underlying data type (`types.DType`) and the size of each dimension. Most graph
  operations have constraints on the shapes. E.g: general arithmetic operations only work with shapes of the
  same *dtype*. Shapes can also be Tuples, in which case they recursively define the shapes of its elements.
  This line in the code creates a **rank** 1 shape, with `vectorShape.Dimensions[0]==3`, that is, we want
  3D vectors.
* `a := g.Parameter("a", vectorShape)`: This creates a graph "parameter", a special node whose
  value needs to be fed to the graph at the time of the graph execution (`g.Run()`). It must be specified
  with a **static** shape, meaning the graph will be compiled for inputs of that shape, and won't work
  with different shapes. It is also given a name. The following line does the same for the "b" parameter.
* `g.Run(ParamsMap{a: []float64{0, 0, 0}, b: []float64{1, 1, 1}}))`: the compilation of the
  graph was as before, but now when we run the graph we need to provide the values for the parameters.
  The `Graph.Run()` method take as input a map of the parameters to the values they will take during
  execution, in this case two vectors of shape `Float64[3]`. All the parameters to the graph must
  be given, it will return an error if any parameter is missing.

Something worth stressing is that the graph is compiled for one specific set of input shapes
for its parameters, and won't work with anything else. For example if we do:

```go
	result := g.Run(ParamsMap{a: []float64{0, 0}, b: []float64{1, 1}})
	fmt.Printf("\nEuclideanDistance((1, 1), (0, 0)) = %s\n", result)
```

We get an error message, directly from the underlying XLA machinery:

```
EuclideanDistance((1, 1), (0, 0)) = tensor.Device.error=failed Graph.Run(): C++ error INVALID_ARGUMENT, (3): "Argument does not match host shape or layout of computation parameter 0: want f64[3]{0}, got f64[2]{0}"
```

(One can get a stack strace in `result.Error()`, useful to pinpoint the line that made to call)

Same as if we try to run with a different DType, let's say `types.Float32`:

```go
	result = g.Run(ParamsMap{a: []float32{0, 0, 0}, b: []float32{1, 1, 1}})
    fmt.Printf("\nEuclideanDistance(float32(1, 1, 1), float32(0, 0, 0)) = %s\n", result)
```

We get the error message:

```
EuclideanDistance(float32(1, 1, 1), float32(0, 0, 0)) = tensor.Device.error=failed Graph.Run(): C++ error INVALID_ARGUMENT, (3): "Argument does not match host shape or layout of computation parameter 0: want f64[3]{0}, got f32[3]{0}"
```

Yes, in cases where we want to execute the graph for different shapes it's cumbersome having to build and
compile a graph per shape we are going to use. In the section **Exec: Computation Graph Execution Made Easy**
below we address this.
