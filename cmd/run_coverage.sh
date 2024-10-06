#!/bin/bash

# Simple script to generate the coverage for GoMLX.
# You should run this from the root of the repository.

PACKAGE_COVERAGE="./graph/...,./ml/...,./models/...,./types/...,./backends/..."
go test -v -cover -coverprofile docs/coverage.out -coverpkg "${PACKAGE_COVERAGE}" ./...
go tool cover -func docs/coverage.out -o docs/coverage.out
