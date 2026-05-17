package main

import (
	"flag"
	"fmt"
	"io/fs"
	"log"
	"path/filepath"
	"strings"

	"github.com/gomlx/gomlx/internal/importrefactor"
)

var (
	flagDir = flag.String("dir", ".", "directory to search for .go files")
)

func main() {
	flag.Parse()

	rules := importrefactor.RewriteRules{
		ImportPathMap: map[string]string{
			"github.com/gomlx/gomlx/backends":                  "github.com/gomlx/compute",
			"github.com/gomlx/gomlx/backends/simplego":         "github.com/gomlx/compute/gobackend",
			"github.com/gomlx/gomlx/backends/xla":              "github.com/gomlx/go-xla/compute/xla",
			"github.com/gomlx/gomlx/pkg/core/dtypes":           "github.com/gomlx/compute/dtypes",
			"github.com/gomlx/gomlx/pkg/core/shapes":           "github.com/gomlx/compute/shapes",
			"github.com/gomlx/gomlx/pkg/core/distributed":      "github.com/gomlx/compute/distributed",
			"github.com/gomlx/gomlx/pkg/core/tensors/images":   "github.com/gomlx/gomlx/core/tensors/images",
			"github.com/gomlx/gomlx/pkg/core/graph/bucketing":  "github.com/gomlx/gomlx/core/graph/bucketing",
			"github.com/gomlx/gomlx/pkg/ml/nn":                 "github.com/gomlx/gomlx/ml/nn",
			"github.com/gomlx/gomlx/pkg/ml/context":            "github.com/gomlx/gomlx/ml/model",
			"github.com/gomlx/gomlx/pkg/ml/context/checkpoints": "github.com/gomlx/gomlx/ml/model/checkpoints",
			"github.com/gomlx/gomlx/pkg/ml/context/ctxtest":     "github.com/gomlx/gomlx/ml/model/ctxtest",
			"github.com/gomlx/gomlx/pkg/ml/context/initializers": "github.com/gomlx/gomlx/ml/model/initializers",
		},
		PackageNameMap: map[string]string{
			"backends": "compute",
			"context":  "model",
		},
	}

	err := filepath.WalkDir(*flagDir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" || d.Name() == "vendor" {
				return filepath.SkipDir
			}
			return nil
		}

		if !strings.HasSuffix(d.Name(), ".go") {
			return nil
		}

		modified, err := importrefactor.RefactorFile(path, rules)
		if err != nil {
			log.Printf("Failed to refactor %s: %v", path, err)
			return nil // Continue with other files
		}

		if modified {
			fmt.Printf("Refactored %s\n", path)
		}

		return nil
	})

	if err != nil {
		log.Fatalf("Error walking directory: %v", err)
	}
}
