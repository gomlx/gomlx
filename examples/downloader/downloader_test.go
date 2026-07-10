// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package downloader

import (
	"crypto/sha256"
	"encoding/hex"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"

	"github.com/stretchr/testify/require"
)

var testBody = []byte("fake dataset content for downloader test")

func testHash(body []byte) string {
	sum := sha256.Sum256(body)
	return hex.EncodeToString(sum[:])
}

func writeTestFile(t *testing.T, path string, body []byte) {
	t.Helper()
	require.NoError(t, os.WriteFile(path, body, 0666))
}

func testServer(t *testing.T, body []byte) (url string, hits *int) {
	t.Helper()
	count := 0
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		count++
		_, _ = w.Write(body)
	}))
	t.Cleanup(srv.Close)
	return srv.URL, &count
}

func TestDownloadIfMissing_GoodFinal(t *testing.T) {
	dir := t.TempDir()
	final := filepath.Join(dir, "data.bin")
	writeTestFile(t, final, testBody)
	url, hits := testServer(t, testBody)
	err := DownloadIfMissing(url, final, testHash(testBody))

	require.NoError(t, err)
	require.Equal(t, 0, *hits)
	require.NoFileExists(t, final+".part")
	require.FileExists(t, final)
}

func TestDownloadIfMissing_BadFinal(t *testing.T) {
	dir := t.TempDir()
	final := filepath.Join(dir, "data.bin")
	writeTestFile(t, final, []byte("corrupt"))
	url, hits := testServer(t, testBody)
	err := DownloadIfMissing(url, final, testHash(testBody))

	require.NoError(t, err)
	require.Equal(t, 1, *hits)
	require.FileExists(t, final)
	require.NoFileExists(t, final+".part")
}

func TestDownloadIfMissing_NoFinal(t *testing.T) {
	dir := t.TempDir()
	final := filepath.Join(dir, "data.bin")
	url, hits := testServer(t, testBody)
	err := DownloadIfMissing(url, final, testHash(testBody))

	require.NoError(t, err)
	require.Equal(t, 1, *hits)
	require.FileExists(t, final)
	require.NoFileExists(t, final+".part")
}

func TestDownloadIfMissing_BadChecksum(t *testing.T) {
	dir := t.TempDir()
	final := filepath.Join(dir, "data.bin")
	url, hits := testServer(t, []byte("wrong bytes"))
	err := DownloadIfMissing(url, final, testHash(testBody))

	require.Error(t, err)
	require.Equal(t, 1, *hits)
	require.NoFileExists(t, final)
	require.NoFileExists(t, final+".part")
}

func TestDownloadIfMissing_DownloadFails(t *testing.T) {
	dir := t.TempDir()
	final := filepath.Join(dir, "data.bin")
	err := DownloadIfMissing("http://127.0.0.1:1", final, testHash(testBody))

	require.Error(t, err)
	require.NoFileExists(t, final)
	require.NoFileExists(t, final+".part")
}
