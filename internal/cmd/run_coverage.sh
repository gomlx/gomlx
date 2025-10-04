#!/bin/bash

# Exit at the first error.
set -e

# Simple script to generate the coverage for GoMLX.
# You should run this from the root of the repository.

# Normal tests:
PACKAGE_COVERAGE_1="./graph/...,./ml/...,./pkg/...,./types/...,./backends,./backends/xla/...,./backends/shapeinference/...,./backends/simplego/..."
go test -cover -coverprofile docs/coverage1.out -coverpkg "${PACKAGE_COVERAGE_1}" ./... -test.count=1

# Tests using the stablehlo backend.
GOMLX_BACKEND=stablehlo:cpu go test -tags=stablehlo -cover -coverprofile docs/coverage2.out -coverpkg "./graph,./backends/stablehlo" ./graph -test.count=1

# Combine and generate final report.
cat docs/coverage1.out <(tail -n +2 docs/coverage2.out) > docs/merged.coverage.out
go tool cover -func docs/merged.coverage.out -o docs/coverage.out

# Remove old files.
rm -f docs/coverage1.out docs/coverage2.out docs/merged.coverage.out
