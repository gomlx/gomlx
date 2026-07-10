package importrefactor

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRefactorFile(t *testing.T) {
	tempDir, err := os.MkdirTemp("", "importrefactor_test")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tempDir)

	filePath := filepath.Join(tempDir, "test.go")
	inputCode := `package test

import (
	"github.com/gomlx/compute/dtypes"
)

func Foo[T dtypes.Supported]() {
	var a dtypes.Number
	var b dtypes.NumberNotComplex
	var c dtypes.NumberComplex
	var d dtypes.NumberHalfPrecision
	var e dtypes.GoFloat
	var f dtypes.HalfPrecision[float32]
}
`
	err = os.WriteFile(filePath, []byte(inputCode), 0644)
	if err != nil {
		t.Fatal(err)
	}

	rules := RewriteRules{
		TypeNameMap: map[string]string{
			"dtypes.Supported":            "gotype.Supported",
			"dtypes.Number":               "gotype.Numeric",
			"dtypes.NumberNotComplex":     "gotype.NumericNotComplex",
			"dtypes.NumberComplex":        "gotype.Complex",
			"dtypes.NumberHalfPrecision":  "gotype.AnyHalfPrecision",
			"dtypes.GoFloat":              "gotype.Float",
			"dtypes.HalfPrecision":        "gotype.HalfPrecision",
			"dtypes.HalfPrecisionPtr":      "gotype.HalfPrecisionPtr",
		},
	}

	modified, err := RefactorFile(filePath, rules)
	if err != nil {
		t.Fatalf("RefactorFile failed: %v", err)
	}
	if !modified {
		t.Fatal("expected file to be modified")
	}

	outputBytes, err := os.ReadFile(filePath)
	if err != nil {
		t.Fatal(err)
	}
	outputCode := string(outputBytes)

	// Check if all expected replacements are present
	expectedPatterns := []string{
		`"github.com/gomlx/compute/dtypes/gotype"`,
		`func Foo[T gotype.Supported]()`,
		`var a gotype.Numeric`,
		`var b gotype.NumericNotComplex`,
		`var c gotype.Complex`,
		`var d gotype.AnyHalfPrecision`,
		`var e gotype.Float`,
		`var f gotype.HalfPrecision[float32]`,
	}

	for _, pattern := range expectedPatterns {
		if !strings.Contains(outputCode, pattern) {
			t.Errorf("expected output to contain %q, but got:\n%s", pattern, outputCode)
		}
	}
}
