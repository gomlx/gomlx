// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"flag"
	"fmt"
	"io/fs"
	"log"
	"os"
	"path/filepath"
	"strings"
)

var (
	flagProject = flag.String("project", "GoMLX", "Project name to use in the copyright header")
)

func main() {
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "Usage: %s [flags] [path ...]\n", os.Args[0])
		fmt.Fprintf(os.Stderr, "\nEnumerates Go files and adds a copyright header if missing.\n")
		fmt.Fprintf(os.Stderr, "Default path is current directory.\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	header := fmt.Sprintf("// Copyright 2023-2026 The %s Authors. SPDX-License-Identifier: Apache-2.0\n\n", *flagProject)

	roots := flag.Args()
	if len(roots) == 0 {
		roots = []string{"."}
	}

	for _, root := range roots {
		err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return err
			}

			// Skip hidden directories and vendor.
			if d.IsDir() {
				if strings.HasPrefix(d.Name(), ".") && d.Name() != "." {
					return filepath.SkipDir
				}
				if d.Name() == "vendor" {
					return filepath.SkipDir
				}
				return nil
			}

			// Process only Go files.
			if !strings.HasSuffix(d.Name(), ".go") {
				return nil
			}

			// Skip generated files.
			if strings.HasPrefix(d.Name(), "gen_") {
				return nil
			}

			return processFile(path, header)
		})
		if err != nil {
			log.Fatalf("Error walking path %q: %v", root, err)
		}
	}
}

func processFile(path, header string) error {
	content, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("failed to read %q: %w", path, err)
	}

	lines := strings.Split(string(content), "\n")

	// Check for existing copyright header in the first 50 lines.
	// Also find the last build tag line.
	lastBuildTagIndex := -1
	for i, line := range lines {
		if i > 50 {
			break
		}
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(trimmed, "// Copyright") {
			return nil
		}
		if strings.HasPrefix(trimmed, "//go:build") || strings.HasPrefix(trimmed, "// +build") {
			lastBuildTagIndex = i
		}
	}

	log.Printf("Adding header to %s", path)

	if lastBuildTagIndex != -1 {
		// Insert after the last build tag.
		// We want an empty line between build tags and copyright.
		// And the header itself has 2 newlines at the end.

		// If the line after the build tag is not empty, we add a newline separator.
		prefix := strings.Join(lines[:lastBuildTagIndex+1], "\n")
		suffix := strings.Join(lines[lastBuildTagIndex+1:], "\n")

		// Add an empty line after build tags if not present in the suffixes start?
		// Actually, let's just force the structure:
		// [Build Tags]
		// <Empty Line>
		// [Header]
		// [Rest] (The header already ends with \n\n, so we just need one \n separator from build tags)

		newContent := prefix + "\n\n" + header + suffix
		// Warning: if suffix started with empty lines, we might have too many.
		// But simplicity first.

		// Wait, header is "// Copyright ...\n\n".
		// So prefix + "\n\n" + header + suffix might be:
		// //go:build foo\n\n// Copyright ...\n\npackage bar
		// That looks perfect.

		if err := os.WriteFile(path, []byte(newContent), 0644); err != nil {
			return fmt.Errorf("failed to write %q: %w", path, err)
		}
	} else {
		// No build tags, prepend at top.
		newContent := append([]byte(header), content...)
		if err := os.WriteFile(path, newContent, 0644); err != nil {
			return fmt.Errorf("failed to write %q: %w", path, err)
		}
	}

	return nil
}
