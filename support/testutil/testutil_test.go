// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package testutil

import (
	"testing"

	_ "github.com/gomlx/gomlx/backends/default"
	"github.com/stretchr/testify/require"
)

// TestGetOfficialBackendSkipsWhenAbsent pins that GetOfficialBackend returns the named backend when
// present and a nil for an absent name (caller skips). The present-name comes from
// OfficialTestBackendNames[0] (the backend BuildTestBackend selects), so the assertion holds
// regardless of a GOMLX_BACKEND override instead of hardcoding "go".
func TestGetOfficialBackendSkipsWhenAbsent(t *testing.T) {
	BuildTestBackend()
	name := OfficialTestBackendNames[0]
	require.NotNilf(t, GetOfficialBackend(name), "%s backend should be available", name)
	require.Nil(t, GetOfficialBackend("definitely-not-a-backend"))
}
