# Performance Tests 

To run performance tests, use the `-tags=perf` flag, otherwise they are ignored by default.
They take too long to run otherwise.

For example:

```bash
$ GOMLX_BACKEND=xla:cuda go test -tags=perf ./internal/perf -test.run=TestDotGeneral_PerformanceTable -dg_perf_names="NoBatch-Tiny,NoBatch-Small" -perf_duration=500ms
```

See each test file for description of extra flags, and selection of cases to run.

They all support the `-markdown` flag to print the results in markdown format, which is useful for inclusion in reports, and README files.