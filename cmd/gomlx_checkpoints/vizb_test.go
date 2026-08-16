// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"bytes"
	"encoding/csv"
	"encoding/json"
	"os"
	"path/filepath"
	"strconv"
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

// TestRunVizbLine_AppliesLineColorsAsTheme locks in the --theme wiring (see Plot): confirmed by
// hand that vizb embeds a bare comma-separated hex list into the DataSet JSON's "themes" field,
// in the exact order given, including repeating the same hex at two positions when two lines
// should share a color -- this is what lets the chart's own line colors match the sidebar's
// per-metric swatches instead of being an unrelated, internally-assigned palette.
func TestRunVizbLine_AppliesLineColorsAsTheme(t *testing.T) {
	if err := checkVizbAvailable(); err != nil {
		t.Skip("vizb not installed, skipping integration test:", err)
	}
	tmpDir := t.TempDir()
	lines := []*plotLineInfo{
		{short: "Alpha", steps: []float64{0, 1}, values: []float64{0.1, 0.2}},
		{short: "Beta", steps: []float64{0, 1}, values: []float64{0.3, 0.4}},
		{short: "Gamma", steps: []float64{0, 1}, values: []float64{0.5, 0.6}},
	}
	// Alpha and Gamma deliberately share a color, Beta gets a different one -- exercises both the
	// "two lines, one color" case (e.g. one metric shared by two models) and a genuinely distinct
	// color in the same call.
	colors := []string{"#2a78d6", "#eb6834", "#2a78d6"}

	jsonPath, err := runVizbLine(tmpDir, "loss", lines, colors)
	require.NoError(t, err)

	data, err := os.ReadFile(jsonPath)
	require.NoError(t, err)
	var doc struct {
		Themes []struct {
			Colors []string `json:"colors"`
		} `json:"themes"`
	}
	require.NoError(t, json.Unmarshal(data, &doc))
	require.Len(t, doc.Themes, 1)
	require.Equal(t, colors, doc.Themes[0].Colors)
}

func TestRunVizbLine_RealVizb(t *testing.T) {
	if err := checkVizbAvailable(); err != nil {
		t.Skip("vizb not installed, skipping integration test:", err)
	}
	tmpDir := t.TempDir()
	lines := []*plotLineInfo{
		{short: "Train", steps: []float64{0, 100, 200}, values: []float64{0.9, 0.5, 0.2}},
		{short: "Eval", steps: []float64{0, 100, 200}, values: []float64{0.85, 0.45, 0.25}},
	}

	jsonPath, err := runVizbLine(tmpDir, "loss", lines, nil)
	require.NoError(t, err)
	require.FileExists(t, jsonPath)

	data, err := os.ReadFile(jsonPath)
	require.NoError(t, err)
	content := string(data)
	require.Contains(t, content, "Train")
	require.Contains(t, content, "Eval")
	// All 3 steps must be present -- i.e. not aggregated away. Group mode's
	// aggregation only sums rows sharing the same key; since each (step,
	// series) pair is unique here, there's nothing to sum, so nothing is lost.
	require.Contains(t, content, `"xAxis":"0"`)
	require.Contains(t, content, `"xAxis":"100"`)
	require.Contains(t, content, `"xAxis":"200"`)
}

// dataSetEntry mirrors the shape of one entry in vizb's group-mode DataSet JSON "data" array
// (--col-axis y): {"xAxis":"<step>","yAxis":"<series>","stats":[{"value":<v>}]}.
type dataSetEntry struct {
	XAxis string `json:"xAxis"`
	YAxis string `json:"yAxis"`
}

func parseDataSetEntries(t *testing.T, jsonPath string) []dataSetEntry {
	t.Helper()
	data, err := os.ReadFile(jsonPath)
	require.NoError(t, err)
	var doc struct {
		Data []dataSetEntry `json:"data"`
	}
	require.NoError(t, json.Unmarshal(data, &doc))
	return doc.Data
}

// TestRunVizbLine_SeriesOnYAxis characterizes the group-mode shape (-g step -p x --col-axis y)
// runVizbLine uses: every line's name lands on "yAxis", not as a per-series "type" key the way
// solo mode did. This is what lets vizb's chart renderer overlay every series of one metric type
// on a single chart with a shared, clickable legend -- solo mode's DataSet JSON also grouped
// every line under one dataset, but vizb's `-c line` chart type rendered that as one chart *per
// series* instead (confirmed both by asking vizb's maintainers directly and by hands-on browser
// testing against real multi-series data).
func TestRunVizbLine_SeriesOnYAxis(t *testing.T) {
	if err := checkVizbAvailable(); err != nil {
		t.Skip("vizb not installed, skipping integration test:", err)
	}
	tmpDir := t.TempDir()
	lines := []*plotLineInfo{
		{short: "Train", steps: []float64{0, 100}, values: []float64{0.9, 0.5}},
		{short: "Eval", steps: []float64{0, 100}, values: []float64{0.85, 0.45}},
	}

	jsonPath, err := runVizbLine(tmpDir, "loss", lines, nil)
	require.NoError(t, err)
	entries := parseDataSetEntries(t, jsonPath)

	seriesNames := make(map[string]bool)
	for _, e := range entries {
		seriesNames[e.YAxis] = true
	}
	require.Equal(t, map[string]bool{"Train": true, "Eval": true}, seriesNames)
	require.Len(t, entries, 4) // 2 series x 2 steps, none missing.
}

// TestRunVizbLine_MisalignedStepsOmitMissingSeries verifies, against the real vizb binary (not
// just writeLinesAsCSV's own unit tests), that a step where one series has no value (a blank CSV
// cell -- see writeLinesAsCSV) is correctly omitted from that series' data rather than being
// turned into a zero or causing an error. This was verified by hand first, against the real
// fixture data that originally motivated the group-mode switch, before being locked in here.
func TestRunVizbLine_MisalignedStepsOmitMissingSeries(t *testing.T) {
	if err := checkVizbAvailable(); err != nil {
		t.Skip("vizb not installed, skipping integration test:", err)
	}
	tmpDir := t.TempDir()
	// Train logged every step, Eval only at step 100 -- step 50's cell for Eval is blank.
	lines := []*plotLineInfo{
		{short: "Train", steps: []float64{50, 100}, values: []float64{0.7, 0.5}},
		{short: "Eval", steps: []float64{100}, values: []float64{0.45}},
	}

	jsonPath, err := runVizbLine(tmpDir, "loss", lines, nil)
	require.NoError(t, err)
	entries := parseDataSetEntries(t, jsonPath)

	var step50Series []string
	for _, e := range entries {
		if e.XAxis == "50" {
			step50Series = append(step50Series, e.YAxis)
		}
	}
	require.Equal(t, []string{"Train"}, step50Series)
}

// TestRunVizbLine_PreservesNarrowRangePrecision guards against a `vizb line` regression: older
// versions (< v0.18.0) rounded every value to a fixed 2 decimal places during CSV->JSON
// conversion, silently collapsing genuinely different, closely-spaced values -- e.g. a converging
// loss curve -- into an identical, flat-looking value. Fixed upstream, confirmed in
// https://github.com/goptics/vizb/issues/336#issuecomment-5191941764. Uses the exact values that
// first exposed the bug (all 5 used to round to 0.91).
func TestRunVizbLine_PreservesNarrowRangePrecision(t *testing.T) {
	if err := checkVizbAvailable(); err != nil {
		t.Skip("vizb not installed, skipping integration test:", err)
	}
	tmpDir := t.TempDir()
	values := []float64{0.914273581, 0.913891247, 0.912456839, 0.910128475, 0.909983221}
	lines := []*plotLineInfo{
		{short: "Train", steps: []float64{1, 2, 3, 4, 5}, values: values},
	}

	jsonPath, err := runVizbLine(tmpDir, "loss", lines, nil)
	require.NoError(t, err)

	data, err := os.ReadFile(jsonPath)
	require.NoError(t, err)
	var doc struct {
		Data []struct {
			XAxis string `json:"xAxis"`
			Stats []struct {
				Value float64 `json:"value"`
			} `json:"stats"`
		} `json:"data"`
	}
	require.NoError(t, json.Unmarshal(data, &doc))
	require.Len(t, doc.Data, 5)

	seen := make(map[float64]bool)
	for _, entry := range doc.Data {
		require.Len(t, entry.Stats, 1)
		seen[entry.Stats[0].Value] = true
	}
	// On the buggy version, all 5 collapse to the single rounded value 0.91.
	require.Greaterf(t, len(seen), 1, "all 5 distinct values collapsed to the same rounded value: %v", seen)

	// Values should pass through essentially untouched (full float64 round-trip via CSV text).
	for i, entry := range doc.Data {
		step, err := strconv.Atoi(entry.XAxis)
		require.NoError(t, err)
		want := values[step-1]
		require.InDelta(t, want, entry.Stats[0].Value, 1e-9, "point %d", i)
	}
}

func TestMergeAndRenderVizb_MultipleTypes(t *testing.T) {
	if err := checkVizbAvailable(); err != nil {
		t.Skip("vizb not installed, skipping integration test:", err)
	}
	tmpDir := t.TempDir()

	lossJSON, err := runVizbLine(tmpDir, "loss",
		[]*plotLineInfo{{short: "Train", steps: []float64{0, 100}, values: []float64{0.9, 0.5}}}, nil)
	require.NoError(t, err)

	accJSON, err := runVizbLine(tmpDir, "accuracy",
		[]*plotLineInfo{{short: "Train", steps: []float64{0, 100}, values: []float64{0.5, 0.9}}}, nil)
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
		[]*plotLineInfo{{short: "Train", steps: []float64{0, 100}, values: []float64{0.9, 0.5}}}, nil)
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
