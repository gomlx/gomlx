// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"path/filepath"
	"strings"
)

// MinimalUniquePaths takes a list of file paths and returns a list of minimal unique identifiers
// that distinguish each path from others using the minimum necessary path parts.
func MinimalUniquePaths(paths ...string) []string {
	if len(paths) <= 1 {
		return paths
	}

	// Split all paths into components
	splitPaths := make([][]string, len(paths))
	for i, path := range paths {
		splitPaths[i] = strings.Split(filepath.Clean(path), string(filepath.Separator))
	}

	// Find minimal unique representations
	result := make([]string, len(paths))
	for i, components := range splitPaths {
		diffIndexes := []int{}

		// Compare with all other paths
		for j, otherComponents := range splitPaths {
			if i == j {
				continue
			}

			// Find different components
			minLen := len(components)
			if len(otherComponents) < minLen {
				minLen = len(otherComponents)
			}

			for k := 0; k < minLen; k++ {
				if components[k] != otherComponents[k] {
					found := false
					for _, idx := range diffIndexes {
						if idx == k {
							found = true
							break
						}
					}
					if !found {
						diffIndexes = append(diffIndexes, k)
					}
				}
			}
		}

		// Create minimal representation
		if len(diffIndexes) == 0 {
			result[i] = components[len(components)-1]
		} else if len(diffIndexes) == 1 {
			result[i] = components[diffIndexes[0]]
		} else {
			// Multiple differences - use ellipsis
			first := components[diffIndexes[0]]
			last := components[diffIndexes[len(diffIndexes)-1]]
			result[i] = first + "..." + last
		}
	}

	return result
}
