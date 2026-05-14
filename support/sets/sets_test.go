// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package sets

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSet(t *testing.T) {
	// Sets are created empty.
	s := Make[int](10)
	assert.Len(t, s, 0)

	// Check inserting and recovery.
	s.Insert(3, 7)
	assert.Len(t, s, 2)
	assert.True(t, s.Has(3))
	assert.True(t, s.Has(7))
	assert.False(t, s.Has(5))

	s2 := MakeWith(5, 7)
	assert.Len(t, s2, 2)
	assert.True(t, s2.Has(5))
	assert.True(t, s2.Has(7))
	assert.False(t, s2.Has(3))

	s3 := s.Sub(s2)
	assert.Len(t, s3, 1)
	assert.True(t, s3.Has(3))

	delete(s, 7)
	assert.Len(t, s, 1)
	assert.True(t, s.Has(3))
	assert.False(t, s.Has(7))
	assert.True(t, s.Equal(s3))
	assert.False(t, s.Equal(s2))
	s4 := MakeWith(-3)
	assert.False(t, s.Equal(s4))
}
