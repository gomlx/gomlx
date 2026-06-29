// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package testutil

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/stretchr/testify/require"
)

// TestGetOfficialBackendSkipsWhenAbsent pins that GetOfficialBackend returns the named backend when
// present (the CPU "go" backend always is) and a nil for an absent name (caller skips). CPU-only.
func TestGetOfficialBackendSkipsWhenAbsent(t *testing.T) {
	BuildTestBackend()
	require.NotNil(t, GetOfficialBackend("go"), "go backend should be available")
	require.Nil(t, GetOfficialBackend("definitely-not-a-backend"))
}
