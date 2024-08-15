# Developing

Below is a list of usual low level implementation tasks:

## Updating `coverage.out` file

This is not done as a GitHub actions because it would take too long to download the datasets, etc.
Instead, we do it manually using the `cmd/run_coverage.sh` simple script. 

It takes some 10-20 minutes to run, and updates the file `docs/converage.out`.
Once the file is submitted, a GitHub action will update the coverage badge.

## Adding new backend operation

Here we are referring to new operations defined in the backend, as opposed to new operations that are
combinations of what already exist.

The main backend is `xla`, provided by the `github.com/gomlx/gopjrt` repository. 
You want to add the op there, first, then in the `xla` package. 
See the various generators under `cmd/` (configured with `go:generate`): you want to either use them,
or exclude the new op from using them. It's always a good practice to re-generate (`go generate ./...`)
and see the difference.

## Adding support for a new DType

**GoMLX** uses dtypes defined in `gopjrt` repository.
The `gopjrt/dtypes` package provides lots of support functionality (lots of generics) in order to
make it simple to add new data types. 

If adding new data types, consider adding tests also in the package `tensors`, to make sure the conversions back and
forth are working.
