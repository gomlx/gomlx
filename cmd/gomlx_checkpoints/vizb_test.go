// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/csv"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"
)

func parseCSV(t *testing.T, s string) [][]string {
	t.Helper()
	records, err := csv.NewReader(strings.NewReader(s)).ReadAll()
	require.NoError(t, err)
	return records
}

func TestWriteLinesAsCSV_AlignedSteps(t *testing.T) {
	lines := []*plotLineInfo{
		{short: "Train", steps: []float64{0, 100, 200}, values: []float64{0.9, 0.5, 0.2}},
		{short: "Eval", steps: []float64{0, 100, 200}, values: []float64{0.85, 0.45, 0.25}},
	}
	var buf bytes.Buffer
	require.NoError(t, writeLinesAsCSV(&buf, lines))

	records := parseCSV(t, buf.String())
	require.Equal(t, []string{"step", "col0", "col1"}, records[0])
	require.Len(t, records, 4) // header + 3 rows
	require.Equal(t, []string{"0", "0.9", "0.85"}, records[1])
	require.Equal(t, []string{"100", "0.5", "0.45"}, records[2])
	require.Equal(t, []string{"200", "0.2", "0.25"}, records[3])
}

// TestWriteLinesAsCSV_MisalignedSteps covers the case where train is logged
// every step but eval only every 100 -- the two lines don't share the same
// set of steps, so the pivoted CSV must have blank cells, not an error or a
// silently wrong join.
func TestWriteLinesAsCSV_MisalignedSteps(t *testing.T) {
	lines := []*plotLineInfo{
		{short: "Train", steps: []float64{0, 50, 100, 150, 200}, values: []float64{0.9, 0.7, 0.5, 0.3, 0.2}},
		{short: "Eval", steps: []float64{0, 100, 200}, values: []float64{0.85, 0.45, 0.25}},
	}
	var buf bytes.Buffer
	require.NoError(t, writeLinesAsCSV(&buf, lines))

	records := parseCSV(t, buf.String())
	require.Equal(t, []string{"step", "col0", "col1"}, records[0])
	require.Len(t, records, 6) // header + union of 5 distinct steps

	// step 50: only Train has a value, Eval's cell must be blank.
	require.Equal(t, []string{"50", "0.7", ""}, records[2])
	// step 100: both lines have a value.
	require.Equal(t, []string{"100", "0.5", "0.45"}, records[3])
	// step 150: only Train again.
	require.Equal(t, []string{"150", "0.3", ""}, records[4])
}

func TestWriteLinesAsCSV_SingleLine(t *testing.T) {
	lines := []*plotLineInfo{
		{short: "Train", steps: []float64{0, 10}, values: []float64{1.0, 0.5}},
	}
	var buf bytes.Buffer
	require.NoError(t, writeLinesAsCSV(&buf, lines))

	records := parseCSV(t, buf.String())
	require.Equal(t, []string{"step", "col0"}, records[0])
	require.Len(t, records, 3)
}

func TestWriteLinesAsCSV_NoLines(t *testing.T) {
	var buf bytes.Buffer
	require.NoError(t, writeLinesAsCSV(&buf, nil))

	records := parseCSV(t, buf.String())
	require.Len(t, records, 1) // header only: just "step"
	require.Equal(t, []string{"step"}, records[0])
}

func TestWriteLinesAsCSV_StepsOutOfOrderInSource(t *testing.T) {
	// Points aren't guaranteed to arrive pre-sorted by step; the writer must
	// sort them regardless of input order.
	lines := []*plotLineInfo{
		{short: "Train", steps: []float64{200, 0, 100}, values: []float64{0.2, 0.9, 0.5}},
	}
	var buf bytes.Buffer
	require.NoError(t, writeLinesAsCSV(&buf, lines))

	records := parseCSV(t, buf.String())
	require.Equal(t, []string{"0", "0.9"}, records[1])
	require.Equal(t, []string{"100", "0.5"}, records[2])
	require.Equal(t, []string{"200", "0.2"}, records[3])
}

// --- Integration tests against the real vizb binary ---
// These skip cleanly (not fail) if vizb isn't installed, e.g. in CI -- --plot
// is a soft dependency, and these tests reflect that.

func TestRunVizbLine_RealVizb(t *testing.T) {
	if err := checkVizbAvailable(); err != nil {
		t.Skip("vizb not installed, skipping integration test:", err)
	}
	tmpDir := t.TempDir()
	lines := []*plotLineInfo{
		{short: "Train", steps: []float64{0, 100, 200}, values: []float64{0.9, 0.5, 0.2}},
		{short: "Eval", steps: []float64{0, 100, 200}, values: []float64{0.85, 0.45, 0.25}},
	}

	jsonPath, err := runVizbLine(tmpDir, "loss", lines)
	require.NoError(t, err)
	require.FileExists(t, jsonPath)

	data, err := os.ReadFile(jsonPath)
	require.NoError(t, err)
	content := string(data)
	require.Contains(t, content, "Train")
	require.Contains(t, content, "Eval")
	// All 3 steps must be present -- i.e. not aggregated away (the exact
	// failure mode of vizb's --group mode, which this recipe deliberately avoids).
	require.Contains(t, content, `"xAxis":"0"`)
	require.Contains(t, content, `"xAxis":"100"`)
	require.Contains(t, content, `"xAxis":"200"`)
}

func TestMergeAndRenderVizb_MultipleTypes(t *testing.T) {
	if err := checkVizbAvailable(); err != nil {
		t.Skip("vizb not installed, skipping integration test:", err)
	}
	tmpDir := t.TempDir()

	lossJSON, err := runVizbLine(tmpDir, "loss",
		[]*plotLineInfo{{short: "Train", steps: []float64{0, 100}, values: []float64{0.9, 0.5}}})
	require.NoError(t, err)

	accJSON, err := runVizbLine(tmpDir, "accuracy",
		[]*plotLineInfo{{short: "Train", steps: []float64{0, 100}, values: []float64{0.5, 0.9}}})
	require.NoError(t, err)

	outPath := filepath.Join(tmpDir, "final.html")
	err = mergeAndRenderVizb(tmpDir, []string{lossJSON, accJSON}, outPath)
	require.NoError(t, err)
	require.FileExists(t, outPath)

	content, err := os.ReadFile(outPath)
	require.NoError(t, err)
	require.Greater(t, len(content), 1000) // a real HTML file, not an empty stub
}

func TestMergeAndRenderVizb_SingleType(t *testing.T) {
	if err := checkVizbAvailable(); err != nil {
		t.Skip("vizb not installed, skipping integration test:", err)
	}
	tmpDir := t.TempDir()

	lossJSON, err := runVizbLine(tmpDir, "loss",
		[]*plotLineInfo{{short: "Train", steps: []float64{0, 100}, values: []float64{0.9, 0.5}}})
	require.NoError(t, err)

	outPath := filepath.Join(tmpDir, "final.html")
	// Special case (plan §6 Step 3 / guide Milestone 4): a single dataset
	// should skip `vizb merge` entirely and go straight to `vizb ui`.
	err = mergeAndRenderVizb(tmpDir, []string{lossJSON}, outPath)
	require.NoError(t, err)
	require.FileExists(t, outPath)
}

func TestCheckVizbAvailable_MissingBinary(t *testing.T) {
	t.Setenv("PATH", "") // simulate vizb not being installed
	err := checkVizbAvailable()
	require.Error(t, err)
	require.Contains(t, err.Error(), "vizb")
	require.Contains(t, err.Error(), "vizb.goptics.org")
}
