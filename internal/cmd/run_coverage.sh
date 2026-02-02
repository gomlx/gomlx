#!/bin/bash
set -e

# 1. Define the packages we want to MEASURE coverage for.
# Use full import paths.
TARGET_MODULE="github.com/gomlx/gomlx"
PACKAGE_COVERAGE="${TARGET_MODULE}/pkg/...,${TARGET_MODULE}/backends/..."

# File to accumulate coverage data
COV_FINAL="docs/coverage.out"
COV_RAW="coverage.raw"
COV_TMP="coverage.tmp"

# Initialize the final coverage file
echo "mode: atomic" > "${COV_RAW}"

# --- FIX: CREATE TEMPORARY WORKSPACE ---
# This forces all sub-modules to use the LOCAL version of the root module,
# fixing the "file not found" error caused by version mismatch.
echo "🛠️  Setting up temporary workspace..."
go work init
# Find all directories with go.mod and add them to the workspace
find . -name "go.mod" -print0 | xargs -0 -n1 dirname | xargs go work use

# Ensure we clean up the go.work file even if the script fails/exits
trap 'rm -f go.work go.work.sum' EXIT
# ---------------------------------------

echo "🔍 Finding modules to test..."
# We search for directories containing go.mod
find . -name "go.mod" -print0 | while IFS= read -r -d '' mod_file; do
    dir=$(dirname "${mod_file}")
    
    echo "------------------------------------------------"
    echo "🧪 Testing module in: ${dir}"
    
    # Run tests using the workspace context.
    # Note: We do NOT need to 'cd' into the directory anymore because
    # we are using a workspace at the root! We can run tests from the root
    # by pointing to the directory.
    
    # If the directory is ".", run standard local tests
    if [ "$dir" == "." ]; then
        go test -cover -coverpkg="${PACKAGE_COVERAGE}" \
            -coverprofile="${COV_TMP}" ./... -test.count=1
    else
        # Run tests for the sub-module from the root
        # This keeps the workspace context active.
        go test -cover -coverpkg="${PACKAGE_COVERAGE}" \
            -coverprofile="${COV_TMP}" ./"$dir"/... -test.count=1
    fi

    # Merge results (strip first line)
    if [ -f "${COV_TMP}" ]; then
        tail -n +2 "${COV_TMP}" >> "${COV_RAW}"
        rm "${COV_TMP}"
    fi
done

echo "------------------------------------------------"
echo "📊 Generating report..."
go tool cover -func "${COV_RAW}" > "${COV_FINAL}"
tail -n 1 "${COV_FINAL}"
rm -f "${COV_RAW}"
