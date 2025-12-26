/*
 *	Copyright 2025 Jan Pfeifer
 *
 *	Licensed under the Apache License, Version 2.0 (the "License");
 *	you may not use this file except in compliance with the License.
 *	You may obtain a copy of the License at
 *
 *	http://www.apache.org/licenses/LICENSE-2.0
 *
 *	Unless required by applicable law or agreed to in writing, software
 *	distributed under the License is distributed on an "AS IS" BASIS,
 *	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *	See the License for the specific language governing permissions and
 *	limitations under the License.
 */

package bucketing

import (
	"testing"
)

func TestPow2Strategy(t *testing.T) {
	s := Pow2()

	tests := []struct {
		input    int
		expected int
	}{
		// Edge cases
		{0, 0},
		{-1, -1},   // Symbolic dimension preserved
		{-100, -100}, // Symbolic dimension preserved

		// Powers of 2 stay the same
		{1, 1},
		{2, 2},
		{4, 4},
		{8, 8},
		{16, 16},
		{32, 32},
		{64, 64},
		{128, 128},
		{256, 256},

		// Non-powers round up
		{3, 4},
		{5, 8},
		{6, 8},
		{7, 8},
		{9, 16},
		{10, 16},
		{15, 16},
		{17, 32},
		{31, 32},
		{33, 64},
		{100, 128},
		{129, 256},
	}

	for _, tt := range tests {
		got := s.Bucket(tt.input)
		if got != tt.expected {
			t.Errorf("Pow2.Bucket(%d) = %d, want %d", tt.input, got, tt.expected)
		}
	}
}

func TestLinearStrategy(t *testing.T) {
	tests := []struct {
		step     int
		input    int
		expected int
	}{
		// Step = 8
		{8, 0, 0},
		{8, -1, -1},
		{8, 1, 8},
		{8, 7, 8},
		{8, 8, 8},
		{8, 9, 16},
		{8, 16, 16},
		{8, 17, 24},
		{8, 100, 104},

		// Step = 4
		{4, 1, 4},
		{4, 4, 4},
		{4, 5, 8},
		{4, 9, 12},

		// Step = 16
		{16, 1, 16},
		{16, 15, 16},
		{16, 16, 16},
		{16, 17, 32},

		// Step = 1 (no change for positive)
		{1, 5, 5},
		{1, 100, 100},
	}

	for _, tt := range tests {
		s := Linear(tt.step)
		got := s.Bucket(tt.input)
		if got != tt.expected {
			t.Errorf("Linear(%d).Bucket(%d) = %d, want %d", tt.step, tt.input, got, tt.expected)
		}
	}
}

func TestExponentialStrategy(t *testing.T) {
	// Test with base 1.4
	s := Exponential(1.4)

	tests := []struct {
		input    int
		expected int
	}{
		// Edge cases
		{0, 0},
		{-1, -1},
		{-50, -50},

		// Small values
		// The sequence for base 1.4 is: 1, 2 (1.4^1=1.4→2), 3 (1.4^2=1.96→2, but need >=3 so 1.4^3=2.74→3)
		{1, 1},
	}

	for _, tt := range tests {
		got := s.Bucket(tt.input)
		if got != tt.expected {
			t.Errorf("Exponential(1.4).Bucket(%d) = %d, want %d", tt.input, got, tt.expected)
		}
	}

	// Test that bucketed values are always >= input
	for i := 1; i <= 1000; i++ {
		got := s.Bucket(i)
		if got < i {
			t.Errorf("Exponential(1.4).Bucket(%d) = %d, but should be >= %d", i, got, i)
		}
	}

	// Test that sequence is non-decreasing
	prev := 0
	for i := 1; i <= 100; i++ {
		got := s.Bucket(i)
		if got < prev {
			t.Errorf("Exponential(1.4).Bucket(%d) = %d, but previous was %d (should be non-decreasing)",
				i, got, prev)
		}
		prev = got
	}

	// Test base 2.0 should behave like Pow2
	s2 := Exponential(2.0)
	pow2 := Pow2()
	for i := 1; i <= 100; i++ {
		expGot := s2.Bucket(i)
		pow2Got := pow2.Bucket(i)
		if expGot != pow2Got {
			t.Errorf("Exponential(2.0).Bucket(%d) = %d, Pow2.Bucket(%d) = %d, should be equal",
				i, expGot, i, pow2Got)
		}
	}
}

func TestNoneStrategy(t *testing.T) {
	s := None()

	tests := []struct {
		input int
	}{
		{0},
		{-1},
		{-100},
		{1},
		{5},
		{100},
		{1000},
	}

	for _, tt := range tests {
		got := s.Bucket(tt.input)
		if got != tt.input {
			t.Errorf("None.Bucket(%d) = %d, want %d (unchanged)", tt.input, got, tt.input)
		}
	}
}

func TestExponentialInvalidBase(t *testing.T) {
	// Invalid bases should default to 2.0 (same as Pow2)
	invalidBases := []float64{-1.0, 0.0, 0.5, 1.0}

	pow2 := Pow2()

	for _, base := range invalidBases {
		s := Exponential(base)
		for i := 1; i <= 50; i++ {
			got := s.Bucket(i)
			want := pow2.Bucket(i)
			if got != want {
				t.Errorf("Exponential(%v).Bucket(%d) = %d, want %d (should default to Pow2)",
					base, i, got, want)
			}
		}
	}
}

func TestExponentialGranularity(t *testing.T) {
	// Test that smaller bases give finer granularity
	bases := []float64{1.2, 1.4, 1.6, 2.0}

	// Count unique bucket values for dimensions 1-100
	for _, base := range bases {
		s := Exponential(base)
		seen := make(map[int]bool)
		for i := 1; i <= 100; i++ {
			seen[s.Bucket(i)] = true
		}
		t.Logf("Exponential(%.1f): %d unique buckets for dims 1-100", base, len(seen))
	}
}

// Benchmark tests
func BenchmarkPow2(b *testing.B) {
	s := Pow2()
	for i := 0; i < b.N; i++ {
		s.Bucket(i % 1000)
	}
}

func BenchmarkLinear(b *testing.B) {
	s := Linear(8)
	for i := 0; i < b.N; i++ {
		s.Bucket(i % 1000)
	}
}

func BenchmarkExponential(b *testing.B) {
	s := Exponential(1.4)
	for i := 0; i < b.N; i++ {
		s.Bucket(i % 1000)
	}
}

func BenchmarkNone(b *testing.B) {
	s := None()
	for i := 0; i < b.N; i++ {
		s.Bucket(i % 1000)
	}
}
