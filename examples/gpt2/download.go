package main

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"time"
)

// downloadFile downloads a file from URL to destPath if it doesn't exist.
func downloadFile(url, destPath string) error {
	// Check if already exists
	if _, err := os.Stat(destPath); err == nil {
		return nil // File already exists
	}

	// Create HTTP client with longer timeout for large files
	client := &http.Client{
		Timeout: 10 * time.Minute,
	}

	// Create request
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("failed to download: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("download failed with status: %s", resp.Status)
	}

	// Create destination file
	destFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer destFile.Close()

	// Copy data with progress
	size := resp.ContentLength
	if size > 0 {
		fmt.Printf("    Downloading %.1f MB...\n", float64(size)/(1024*1024))
	}

	written, err := io.Copy(destFile, resp.Body)
	if err != nil {
		return fmt.Errorf("failed to write file: %w", err)
	}

	fmt.Printf("    \u2713 Downloaded %.1f MB\n", float64(written)/(1024*1024))
	return nil
}
