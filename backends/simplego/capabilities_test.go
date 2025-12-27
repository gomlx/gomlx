package simplego

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestCapabilities_SupportsDynamicShapes(t *testing.T) {
	// Test that SimpleGo backend reports SupportsDynamicShapes as true
	assert.True(t, Capabilities.SupportsDynamicShapes,
		"SimpleGo backend should support dynamic shapes since graph creation is cheap")
}

func TestCapabilities_Clone(t *testing.T) {
	// Test that Clone copies the SupportsDynamicShapes field
	cloned := Capabilities.Clone()
	assert.Equal(t, Capabilities.SupportsDynamicShapes, cloned.SupportsDynamicShapes,
		"Clone should copy SupportsDynamicShapes field")

	// Verify other fields are also cloned
	assert.Equal(t, len(Capabilities.Operations), len(cloned.Operations),
		"Clone should copy Operations map")
	assert.Equal(t, len(Capabilities.DTypes), len(cloned.DTypes),
		"Clone should copy DTypes map")
}
