package humanize

import (
	"testing"
)

func TestUnderscores(t *testing.T) {
	tests := []struct {
		input    int
		expected string
	}{
		{0, "0"},
		{1, "1"},
		{12, "12"},
		{123, "123"},
		{1234, "1_234"},
		{1234567, "1_234_567"},
		{1000000000, "1_000_000_000"},
		{-1, "-1"},
		{-12, "-12"},
		{-123, "-123"},
		{-1234, "-1_234"},
		{-1234567, "-1_234_567"},
	}

	for _, tc := range tests {
		got := Underscores(tc.input)
		if got != tc.expected {
			t.Errorf("Underscores(%d) = %q, want %q", tc.input, got, tc.expected)
		}
	}

	// Test uint64 with very large numbers
	var maxUint uint64 = 18446744073709551615
	expectedUint := "18_446_744_073_709_551_615"
	if got := Underscores(maxUint); got != expectedUint {
		t.Errorf("Underscores(%d) = %q, want %q", maxUint, got, expectedUint)
	}
}

func TestBytes(t *testing.T) {
	tests := []struct {
		input    int64
		expected string
	}{
		{0, "0 B"},
		{100, "100 B"},
		{500, "500 B"},
		{1024, "1.0 KiB"},
		{-1024, "-1.0 KiB"},
		{1048576, "1.0 MiB"}, // 1024 * 1024
		{1610612736, "1.5 GiB"}, // 1.5 * 1024 * 1024 * 1024
	}

	for _, tc := range tests {
		got := Bytes(tc.input)
		if got != tc.expected {
			t.Errorf("Bytes(%d) = %q, want %q", tc.input, got, tc.expected)
		}
	}
}
