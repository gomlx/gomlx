# Developing

Below is a list of usual low level implementation tasks:

## Adding support to a new node type in gomlx that maps to an op in XLA

Carefully add in `xla/node.go` and `c/gomlx/node.h` the new constant for the op. These two lists
must have the same order (same values) -- TODO: generate one from the other. Also, after editing these
files, go to `xla/` directory and run `$ go generate`, this will regenerate the strings for the
constants -- otherwise it will complain.

Check whether parameters for the new Op can be fit into `xla.SerializedNode` and the C corresponding
`C.SerializedNode` defined in `xla/nodeC.go` and `c/gomlx/node.h` respectively. If yes, that makes
it simpler. If not, you'll need to edit `xla.SerializedToC()` and `xla.DeleteCSerializedNode()`
to accommodate new features.

Create similarly named function in `computation/node.Go`. In most cases it should simply follow suit with
others with the following formula:

1. Get and validate the graph and node inputs with `computation.validateGraphFromInputs()`.
2. Package the parameters in a `xla.SerializedNode` object.
3. Return the results from `computation.newNode()`.
4. Add documentation to the new function. If it's a 1:1 mapping to a XLA op, check out the
   documentation in [TensorFlow/XLA page](https://www.tensorflow.org/xla/operation_semantics).

Then comes the implementation in `c/gomlx/computation.cpp`. Within the function `ComputationAddOp` there
is a large switch statement for each node type. Add one for the new op: typically a simple call
with the parameters already available in `C.SerializedNode`.

Then add a test in `computation/node_test.go`. For node types that have an equivalent in Go, and take
one of two node parameters, all one needs to do is add them on the respective tables. Otherwise,
just follow the examples in the file.

Finally, one needs to recompile the `libgomlx_xla.so` and headers. See instructions on how to build
it below. One can source (as in bash `source`) the script `setup.sh` to setup CGO to use the `.h`
files directly from `c/gomlx` as well as the library from `c/bazel-bin/gomlx`. This is nice
because it makes it simpler to develop in C++ without having to install the compiled C library
every time. Notice the compilation time the first time can take quite long, since it pulls and
recompiles the whole tensorflow. Hopefully it will improve once [OpenXLA](https://github.com/openxla/xla)
is set up.

This is all needed for a forward pass. For a simple op, assuming TensorFlow has already been
compiled once, this can be done in 5 to 10 minutes once one gets the hang of it.

Now the potentially harder part is writing the VJP (vector times gradient of the output with respect to
its inputs) function for backpropagation. XLA doesn't provide it (for most ops), and one needs to
implement it from scratch (or using Jax as a model). This is done in `computation/rev_autodiff.go`
and it is nice to add a test in `computation/rev_autodiff_test.go`. There is a mapping of echo
node type to the corresponding VJP function in `comptutation.VJPRegistration`: just add it there.

## Adding support for a new DType

We tried to minimize the locations that need to change for this, but it's not perfect. A non-exhaustive list
of places to modify: (Please add more if you find out other places)

* The `shapes` package: most of the tools that handle types specially were put there, when possible.
* Tests in package `tensor`: while new types won't break it, they probably should be tested there. Some types (bp16?)
  may need special treatment ?

Next dtypes planned (desired) for support: boolean, bfloat16
