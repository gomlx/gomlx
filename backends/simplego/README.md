# SimpleGo Backend

SimpleGo implements a simple, and not very fast, but very portable backend for GoMLX.

The priority is to make something that will work everywhere and is "ergonomic" (doesn't require
installing any associated C/C++/Rust library, or special packages other than Go itself).

A second priority is having a very short dependency list: this aimed at being safe, very low
dependency library.

See `capabilities.go` file to see operations that are implemented.

## To Do's

This can be split into 2 parts: implement missing ops, and optimizations.

### Missing Ops/Functionality

Too many to list now :) But feel free to add a list of priority ops/dtypes to add support to.

### Optimizations

The initial implementation had almost no optimizations, and was focused on portability and getting it to work.

But there are many relatively "low hanging fruits" for optimization, a few obvious items:

* Pre-calculate the temporary memory needed -- it is known in graph compiling time -- and use memory pool for these blocks.
* Do a proper matrix multiplication.
* Eliminate common sub-expressions.
* Pre-calculate constant sub-expressions.
* Fuse unary ops: it's much faster (for larger data blocks) to loop over the data only once and apply various functions than
  loop over the data many times, each time applying the unary function.
* Fuse binary/unary ops: perform unary functions while traversing the data for binary functions. Again to save
  memory accesses.
* Parallelization: in-operation, and across operations.
* Use intrinsics on platforms that allow it.
