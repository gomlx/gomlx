// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

// withPlotOutput temporarily sets *flagPlotOutput for the duration of the
// test, restoring it afterward -- flagPlotOutput is a package-level flag var,
// shared across tests, so it must always be restored.
func withPlotOutput(t *testing.T, value string) {
	t.Helper()
	old := *flagPlotOutput
	*flagPlotOutput = value
	t.Cleanup(func() { *flagPlotOutput = old })
}

// resetPlotsSessionState resets the package-level, process-lifetime state
// resolveOutputFilePath/BuildPlots rely on (cachedOutputPath, browserOpenedOnce,
// hasRenderedOnce, lastRenderModTime), both before and after the test -- these
// are deliberately global (real session state, not per-call), so tests that
// touch them must isolate themselves from each other explicitly.
func resetPlotsSessionState(t *testing.T) {
	t.Helper()
	reset := func() {
		cachedOutputPath = ""
		browserOpenedOnce = false
		hasRenderedOnce = false
		lastRenderModTime = time.Time{}
	}
	reset()
	t.Cleanup(reset)
}

func TestResolveOutputFilePath_DefaultIsStableWithinASession(t *testing.T) {
	resetPlotsSessionState(t)
	withPlotOutput(t, "")
	checkpointPaths := []string{"/home/user/mnist_data/checkpoint"}

	// Calling it twice must return the exact same path -- that's the whole
	// point (see BuildPlots' doc comment): a stable path lets repeated -loop
	// iterations overwrite the same file instead of a new one accumulating
	// every time. It does NOT need to be reproducible across separate
	// process runs -- just stable within one (per maintainer feedback).
	path1 := resolveOutputFilePath(checkpointPaths)
	path2 := resolveOutputFilePath(checkpointPaths)
	require.Equal(t, path1, path2)
	require.True(t, filepath.IsAbs(path1))
}

func TestResolveOutputFilePath_CacheIgnoresCheckpointPathsArgument(t *testing.T) {
	resetPlotsSessionState(t)
	withPlotOutput(t, "")

	// Once cached, the *same session* reuses the same path regardless of what
	// checkpointPaths is passed on a later call -- correct, because in real
	// usage one process is invoked with one fixed set of checkpoint paths for
	// its entire lifetime; the cache is intentionally not keyed by them.
	pathA := resolveOutputFilePath([]string{"/checkpoints/model-a"})
	pathB := resolveOutputFilePath([]string{"/checkpoints/model-b"})
	require.Equal(t, pathA, pathB)
}

func TestResolveOutputFilePath_ExplicitRelativePath(t *testing.T) {
	resetPlotsSessionState(t)
	withPlotOutput(t, "plot.html")
	got := resolveOutputFilePath([]string{"/checkpoints/model-a"})
	require.Equal(t, "/checkpoints/model-a/plot.html", got)
}

func TestResolveOutputFilePath_ExplicitAbsolutePath(t *testing.T) {
	resetPlotsSessionState(t)
	// t.TempDir(), not a hardcoded "/tmp/..." literal: a Unix-style leading-slash
	// path isn't absolute by Windows' rules (filepath.IsAbs requires a drive
	// letter there), so a hardcoded literal passes on Unix and fails on the
	// Windows CI runner -- t.TempDir() is a real absolute path on whichever OS
	// the test is running on.
	absPath := filepath.Join(t.TempDir(), "my-plot.html")
	withPlotOutput(t, absPath)
	got := resolveOutputFilePath([]string{"/checkpoints/model-a"})
	require.Equal(t, absPath, got)
}

func TestLatestMetricsModTime_NoFilesReturnsZero(t *testing.T) {
	got := latestMetricsModTime([]string{filepath.Join(t.TempDir(), "does-not-exist")})
	require.True(t, got.IsZero())
}

func TestLatestMetricsModTime_ReturnsMostRecent(t *testing.T) {
	dirA, dirB := t.TempDir(), t.TempDir()
	older := time.Now().Add(-time.Hour)
	newer := time.Now()

	writePoints := func(dir string, modTime time.Time) {
		filePath := filepath.Join(dir, "training_plot_points.json")
		require.NoError(t, os.WriteFile(filePath, []byte("{}\n"), 0644))
		require.NoError(t, os.Chtimes(filePath, modTime, modTime))
	}
	writePoints(dirA, older)
	writePoints(dirB, newer)

	got := latestMetricsModTime([]string{dirA, dirB})
	require.WithinDuration(t, newer, got, time.Second)
}

func TestLatestMetricsModTime_SkipsMissingFilesAmongMultiple(t *testing.T) {
	dirWithFile := t.TempDir()
	modTime := time.Now()
	filePath := filepath.Join(dirWithFile, "training_plot_points.json")
	require.NoError(t, os.WriteFile(filePath, []byte("{}\n"), 0644))
	require.NoError(t, os.Chtimes(filePath, modTime, modTime))

	dirWithoutFile := filepath.Join(t.TempDir(), "no-training-data-yet")

	got := latestMetricsModTime([]string{dirWithoutFile, dirWithFile})
	require.WithinDuration(t, modTime, got, time.Second)
}

func TestInjectAutoRefresh_InsertsTag(t *testing.T) {
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath, []byte("<!DOCTYPE html><html><head><title>x</title></head><body></body></html>"), 0644))

	require.NoError(t, injectAutoRefresh(htmlPath, 30*time.Second))

	content, err := os.ReadFile(htmlPath)
	require.NoError(t, err)
	require.Contains(t, string(content), `<meta http-equiv="refresh" content="30">`)
	// Must land right after <head>, before the rest of the original content.
	require.Contains(t, string(content), `<head><meta http-equiv="refresh" content="30"><title>x</title>`)
}

func TestInjectAutoRefresh_SubSecondPeriodClampsToOne(t *testing.T) {
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath, []byte("<head></head>"), 0644))

	require.NoError(t, injectAutoRefresh(htmlPath, 200*time.Millisecond))

	content, err := os.ReadFile(htmlPath)
	require.NoError(t, err)
	require.Contains(t, string(content), `content="1"`)
}

func TestInjectAutoRefresh_NoHeadTagReturnsError(t *testing.T) {
	dir := t.TempDir()
	htmlPath := filepath.Join(dir, "plot.html")
	require.NoError(t, os.WriteFile(htmlPath, []byte("<html><body>no head here</body></html>"), 0644))

	err := injectAutoRefresh(htmlPath, 30*time.Second)
	require.Error(t, err)
}
