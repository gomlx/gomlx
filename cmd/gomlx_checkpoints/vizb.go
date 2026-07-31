// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"strconv"

	"github.com/pkg/errors"
)

// checkVizbAvailable reports a clear, actionable error if the `vizb` CLI tool
// is not on PATH. --plot is a soft dependency on vizb: every other flag works
// regardless of whether it's installed.
func checkVizbAvailable() error {
	if _, err := exec.LookPath("vizb"); err != nil {
		return errors.New("--plot requires the `vizb` CLI tool, which was not found on your PATH.\n" +
			"Install it with:\n\n\tgo install github.com/goptics/vizb@latest\n\n" +
			"or see https://vizb.goptics.org/ for other install options.")
	}
	return nil
}

// writeLinesAsCSV pivots a set of lines (sharing an x-axis of "step") into a
// wide CSV: one row per distinct step (ascending), one column per line.
// Steps missing for a given line are left blank -- vizb's CSV parser is
// expected to skip blank numeric cells for that series/point.
func writeLinesAsCSV(w io.Writer, lines []*plotLineInfo) error {
	// Collect the sorted union of all Step values across all lines: they may
	// not be aligned (e.g. eval logged every 100 steps, train every step).
	stepSet := make(map[float64]bool)
	for _, line := range lines {
		for _, step := range line.steps {
			stepSet[step] = true
		}
	}
	steps := make([]float64, 0, len(stepSet))
	for step := range stepSet {
		steps = append(steps, step)
	}
	slices.Sort(steps)

	// Per-line lookup: step -> value.
	lineStepValue := make([]map[float64]float64, len(lines))
	for i, line := range lines {
		m := make(map[float64]float64, len(line.steps))
		for j, step := range line.steps {
			m[step] = line.values[j]
		}
		lineStepValue[i] = m
	}

	cw := csv.NewWriter(w)
	header := make([]string, 1+len(lines))
	header[0] = "step"
	for i := range lines {
		header[i+1] = fmt.Sprintf("col%d", i)
	}
	if err := cw.Write(header); err != nil {
		return errors.Wrap(err, "failed to write CSV header")
	}

	row := make([]string, 1+len(lines))
	for _, step := range steps {
		row[0] = strconv.FormatFloat(step, 'f', -1, 64)
		for i := range lines {
			if value, ok := lineStepValue[i][step]; ok {
				row[i+1] = strconv.FormatFloat(value, 'f', -1, 64)
			} else {
				row[i+1] = ""
			}
		}
		if err := cw.Write(row); err != nil {
			return errors.Wrap(err, "failed to write CSV row")
		}
	}
	cw.Flush()
	if err := cw.Error(); err != nil {
		return errors.Wrap(err, "failed to flush CSV writer")
	}
	return nil
}

// runVizbLine shapes one metric type's lines into a CSV, invokes
// `vizb line ... --select ...`, and returns the path to the resulting
// DataSet JSON file (written inside tmpDir).
func runVizbLine(tmpDir, metricType string, lines []*plotLineInfo) (jsonPath string, err error) {
	csvPath := filepath.Join(tmpDir, metricType+".csv")
	f, err := os.Create(csvPath)
	if err != nil {
		return "", errors.Wrapf(err, "failed to create CSV file for metric type %q", metricType)
	}
	writeErr := writeLinesAsCSV(f, lines)
	closeErr := f.Close()
	if writeErr != nil {
		return "", errors.Wrapf(writeErr, "failed to write CSV for metric type %q", metricType)
	}
	if closeErr != nil {
		return "", errors.Wrapf(closeErr, "failed to close CSV file for metric type %q", metricType)
	}

	jsonPath = filepath.Join(tmpDir, metricType+".json")
	args := []string{"line", csvPath, "-o", jsonPath, "-n", metricType}
	for i, line := range lines {
		args = append(args, "--select", fmt.Sprintf("step,col%d{%s}", i, line.short))
	}

	cmd := exec.Command("vizb", args...)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", errors.Wrapf(err, "vizb line failed for metric type %q:\n%s", metricType, string(output))
	}
	return jsonPath, nil
}

// mergeAndRenderVizb combines one or more per-metric-type DataSet JSON files
// (from runVizbLine) into a single self-contained HTML file at outputPath.
func mergeAndRenderVizb(tmpDir string, jsonPaths []string, outputPath string) error {
	if len(jsonPaths) == 0 {
		return errors.New("mergeAndRenderVizb: no JSON datasets to render")
	}

	renderInput := jsonPaths[0]
	if len(jsonPaths) > 1 {
		mergedPath := filepath.Join(tmpDir, "merged.json")
		args := append([]string{"merge"}, jsonPaths...)
		args = append(args, "-o", mergedPath)
		cmd := exec.Command("vizb", args...)
		output, err := cmd.CombinedOutput()
		if err != nil {
			return errors.Wrapf(err, "vizb merge failed:\n%s", string(output))
		}
		renderInput = mergedPath
	}

	cmd := exec.Command("vizb", "ui", renderInput, "-o", outputPath, "-c", "line")
	output, err := cmd.CombinedOutput()
	if err != nil {
		return errors.Wrapf(err, "vizb ui failed:\n%s", string(output))
	}
	return nil
}
