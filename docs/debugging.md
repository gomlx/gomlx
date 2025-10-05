# Debugging

Unfortunately, the computers just "don't get it", and they do what we told them to do, and not what
we wanted them to do, and programs fail or crash. GoMLX provides different ways to track down various
types of errors. The most commonly used below:

## Good old "printf"

It's convenient and because of Go fast compilation, often a valid way of developing by just logging
results to the stdout. During graph building development, often one prints the shape of the `Node`
being operated to confirm (or not) one's expectations.

## Delayed Errors

Errors during the building of the graph are reported to the `Graph` or the `Context` or both. They,
as well as `Node` and tensors, implement the methods `Ok()` and `Error()` to check if there has been
an error, and what it is. The errors always include a stack-trace -- print error with `"%+v"` to get
full stack-trace output.

To avoid checking at every step (it would make the code to cumbersome), the idea is to check for errors
only sporadically, maybe in the start and end of a graph function. If an error happens in between, all
operations and layers are able to handle invalid `Node`, by returning invalid nodes themselves. The
error stored is always the first one that happened.

This scheme has proven very effective during the development of the various operations.

Tensors also support delayed error, and can be similarly checked. `Exec` objects report failure
in execution through

More discussion on [error handling here](error_handling.go).

## Graph Execution Logging

Every `Node` of the graph can be annotated with by any graph operation can be marked with `SetLogged(msg)`.
The executor (`Exec`) will at the end of the execution log all these values. The default logger
(set with `Exec.SetLogger`) will simply print the message `msg` along with the value of the `Node` of
interest. In package  `gomlx/ml/train/plotdata` there is also a specialized logger that will collect
these values for plotting later. See [dogsandcats]() example to see a plot of the gradient amplitude
during training. Creating a new logger of any type is trivial.

## `Node` with Stack-Traces

When writing gradient of new operations (or any other debugging) sometimes it's useful to know
exactly where a `Node` was created. Use `Graph.SetTraced(true)` to enable all new nodes to include
its stack-trace. And `Node.Trace()` to access it for printing or debugging.

## `NanLogger`

The [`nanlogger.NanLogger`](https://pkg.go.dev/github.com/gomlx/gomlx/pkg/core/graph/nanlogger) allows one 
to select "traces" on the model, which will panic with a stack trace (and some optional arbitrary
scope message) whenever a `NaN` or `±Inf` is seen -- or it can be handled by an arbitrary function,
with relatively low cost to the model.

It has made it relatively easy to track those pesky nans that show up during training -- so far at least.

## TODO / Future Work

### Adding "Scope" to `Graph` and its `Nodes`

To make it easy to associate a node to a layer for instance. This may come in handy for profiling.
