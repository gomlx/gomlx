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
			"github.com/gomlx/gomlx/backends":                               "github.com/gomlx/compute",
			"github.com/gomlx/gomlx/backends/simplego":                      "github.com/gomlx/compute/gobackend",
			"github.com/gomlx/gomlx/backends/xla":                           "github.com/gomlx/go-xla/compute/xla",
			"github.com/gomlx/gomlx/pkg/core/dtypes":                        "github.com/gomlx/compute/dtypes",
			"github.com/gomlx/gomlx/pkg/core/shapes":                        "github.com/gomlx/compute/shapes",
			"github.com/gomlx/gomlx/pkg/core/distributed":                   "github.com/gomlx/compute/distributed",
			"github.com/gomlx/gomlx/pkg/core/tensors/images":                "github.com/gomlx/gomlx/core/tensors/images",
			"github.com/gomlx/gomlx/pkg/core/graph/bucketing":               "github.com/gomlx/gomlx/core/graph/bucketing",
			"github.com/gomlx/gomlx/pkg/ml/datasets":                        "github.com/gomlx/gomlx/ml/dataset",
			"github.com/gomlx/gomlx/pkg/ml/decode":                          "github.com/gomlx/gomlx/ml/decode",
			"github.com/gomlx/gomlx/pkg/ml/ggml":                            "github.com/gomlx/gomlx/ml/ggml",
			"github.com/gomlx/gomlx/pkg/ml/layers":                          "github.com/gomlx/gomlx/ml/layers",
			"github.com/gomlx/gomlx/pkg/ml/layers/regularizers":             "github.com/gomlx/gomlx/ml/layers/regularizer",
			"github.com/gomlx/gomlx/pkg/ml/layers/fnn":                      "github.com/gomlx/gomlx/ml/layers/fnn",
			"github.com/gomlx/gomlx/pkg/ml/layers/lstm":                     "github.com/gomlx/gomlx/ml/layers/lstm",
			"github.com/gomlx/gomlx/pkg/ml/layers/batchnorm":                "github.com/gomlx/gomlx/ml/layers/batchnorm",
			"github.com/gomlx/gomlx/pkg/ml/layers/bsplines":                 "github.com/gomlx/gomlx/ml/layers/bsplines",
			"github.com/gomlx/gomlx/pkg/ml/layers/activations":              "github.com/gomlx/gomlx/ml/layers/activation",
			"github.com/gomlx/gomlx/pkg/ml/layers/vnn":                      "github.com/gomlx/gomlx/ml/layers/vnn",
			"github.com/gomlx/gomlx/pkg/ml/layers/rational":                 "github.com/gomlx/gomlx/ml/layers/rational",
			"github.com/gomlx/gomlx/pkg/ml/layers/kan":                      "github.com/gomlx/gomlx/ml/layers/kan",
			"github.com/gomlx/gomlx/pkg/ml/layers/attention":                "github.com/gomlx/gomlx/ml/layers/attention",
			"github.com/gomlx/gomlx/pkg/ml/layers/attention/pos":            "github.com/gomlx/gomlx/ml/layers/attention/pos",
			"github.com/gomlx/gomlx/pkg/ml/train":                           "github.com/gomlx/gomlx/ml/train",
			"github.com/gomlx/gomlx/pkg/ml/train/losses":                    "github.com/gomlx/gomlx/ml/train/loss",
			"github.com/gomlx/gomlx/pkg/ml/train/metrics":                   "github.com/gomlx/gomlx/ml/train/metrics",
			"github.com/gomlx/gomlx/pkg/ml/train/optimizers":                "github.com/gomlx/gomlx/ml/train/optimizer",
			"github.com/gomlx/gomlx/pkg/ml/train/optimizers/cosineschedule": "github.com/gomlx/gomlx/ml/train/optimizer/cosineschedule",
			"github.com/gomlx/gomlx/pkg/ml/model/transformer":               "github.com/gomlx/gomlx/ml/zoo/transformer",
			"github.com/gomlx/gomlx/pkg/ml/nn":                              "github.com/gomlx/gomlx/ml/nn",
			"github.com/gomlx/gomlx/pkg/ml/context":                         "github.com/gomlx/gomlx/ml/model",
			"github.com/gomlx/gomlx/pkg/ml/context/checkpoints":             "github.com/gomlx/gomlx/ml/model/checkpoint",
			"github.com/gomlx/gomlx/pkg/ml/context/ctxtest":                 "github.com/gomlx/gomlx/ml/model/modeltest",
			"github.com/gomlx/gomlx/pkg/ml/context/initializers":            "github.com/gomlx/gomlx/ml/model/initializer",
		},
		PackageNameMap: map[string]string{
			"backends": "compute",
			"context":  "model",
		},
		TypeNameMap: map[string]string{
			"model.Context": "Scope",
		},
		FunctionNameMap: map[string]string{
			"commandline.CreateContextSettingsFlag":     "CreateSettingsFlag",
			"commandline.ParseContextSettings":          "ParseSettings",
			"commandline.SprintContextSettings":         "SprintSettings",
			"commandline.SprintModifiedContextSettings": "SprintModifiedSettings",
		},
		MethodNameMap: map[string]string{
			"model.Scope.VariableWithValueGraph": "VariableWithNodeValue",
			"model.Variable.ValueGraph":          "NodeValue",
			"model.Variable.SetValueGraph":       "SetNodeValue",
		},
		VariableNameMap: map[string]importrefactor.VariableRename{
			"ctx": {NewName: "scope", TypeName: "model.Context"},
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
