#!/bin/bash

# Exit at the first error.
set -e

# Simple script to generate the coverage for GoMLX.
# You should run this from the root of the repository.

# PACKAGE_COVERAGE lists the packages that we monitor and are considered in the coverage.
# It excludes, for instance, the examples/ and ui/ directories. They may also be tested, just their coverage is not included in the report.
PACKAGE_COVERAGE="./pkg/...,./backends,./backends/xla/...,./backends/shapeinference/...,./backends/simplego/..."

go test -cover -coverprofile docs/coverage1.out -coverpkg "${PACKAGE_COVERAGE}" \
  ./pkg/... ./backends/... ./examples/... -test.count=1
go tool cover -func docs/coverage1.out -o docs/coverage.out

# Remove old files.
rm -f docs/coverage1.out
