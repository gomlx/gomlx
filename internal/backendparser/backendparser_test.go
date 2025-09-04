package backendparser

import (
	"fmt"
	"testing"
)

func TestParseBuilder(t *testing.T) {
	methods, err := ParseBuilder()
	if err != nil {
		t.Fatalf("ParseBuilder failed: %v", err)
	}

	// Check Add method
	var foundAdd, foundConstant bool
	for _, method := range methods {
		if method.Name == "Add" {
			if method.Comment == "" {
				t.Errorf("Add method has no comment: %+v", method)
			}
			fmt.Printf("Add: %+v\n", method)
			foundAdd = true
		} else if method.Name == "Constant" {
			if method.Comment == "" {
				t.Errorf("Constant method has no comment: %+v", method)
			}
			fmt.Printf("Constant: %+v\n", method)
			foundConstant = true
		}
	}
	if !foundAdd {
		t.Error("Add method not found")
	}
	if !foundConstant {
		t.Error("Constant method not found")
	}
}
