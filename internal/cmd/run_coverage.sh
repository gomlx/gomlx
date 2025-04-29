#!/bin/bash

# Simple script to generate the coverage for GoMLX.
# You should run this from the root of the repository.

PACKAGE_COVERAGE="./graph/...,./ml/...,./models/...,./types/...,./backends,./backends/xla/...,./backends/shapeinference/...,./backends/simplego/..."
go test -cover -coverprofile docs/coverage.out -coverpkg "${PACKAGE_COVERAGE}" ./... -test.count=1
go tool cover -func docs/coverage.out -o docs/coverage.out
