# Error Handling

## Compile time, Runtime, **Graph Building Time** and **Graph Execution Time**

Strong type checking is a major plus for a programming language. It allows catching mistakes much earlier:
in compile time or even IDEs can catch them, while editing. And in some cases it allows catching errors
that would go unseen by testing (odd combination of values uncovered by test). 

Unfortunately it is not possible to encode Graph type checking in compile time, it would require extensions
to the compiler, and in some cases it actually requires running the code, since the output of some ops tensors
have shapes that are dynamically generated.

This slows down development, since it requires running the programs to check the validity their computation
graph types and values.

One recommendation here is to execute the graph building (and not necessarily execution) early in the 
program execution -- before data preprocessing for instance. So this can be iterated fast.

Now there are issues that happen (and can be caught) during graph building
and others that happen only during graph execution (e.g.: NaNs in computation) later on. 
One can think that the graph building code is being executed in two different modes, 
and for convenience we often split the concept of "runtime" for those
into **"graph building time"** and **"graph execution time"**.

> **Note**: This is not as big an issue in practice as some characterize it. As an anecdote TensorFlow 2.0
> adoption of "eager execution mode", to hide the graph building behind the scenes, caused more problems than
> it solved -- developing models in TensorFlow 1.0 was generally simpler and faster
> (in terms of developer speed). But this is a longer topic (and confounded by other aspects that were
> improved in TensorFlow 2.0/Keras).

## Exceptions

`graph.Graph`, `context.Context` and `tensors.Tensor` objects report errors using "exceptions" (`panic` with errors 
with a stack trace), using the very simple [`github.com/gomlx/exceptions`](https://github.com/gomlx/exceptions) library.

This is not aligned with the "idiomatic" Go strategy of checking errors at every step. 
Unfortunately checking for error at every operation would seriously hinder readability of the code building 
computation graphs.

Some libraries, like `ml/train` catch exceptions and return them as errors instead. So, outside those specific
libraries, we are back to normal Go error handling.
