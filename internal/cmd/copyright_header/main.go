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

	// Check if header exists.
	// We check for "// Copyright" prefix.
	if strings.HasPrefix(string(content), "// Copyright") {
		return nil
	}

	log.Printf("Adding header to %s", path)

	newContent := append([]byte(header), content...)
	if err := os.WriteFile(path, newContent, 0644); err != nil {
		return fmt.Errorf("failed to write %q: %w", path, err)
	}

	return nil
}
