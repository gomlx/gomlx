package xla

import (
	"github.com/gomlx/gomlx/backends"
	"github.com/pkg/errors"
)

// This file implements other ops, not covered by github.com/gomlx/gopjrt/xlabuilder.

// Clamp returns the element-wise clamping operation.
//
// The values max and min can either be a scalar or have the same shape as x.
func (b *Builder) Clamp(min, x, max backends.Op) (backends.Op, error) {
	clamped, err := b.Max(min, x)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed Clamp", BackendName)
	}
	clamped, err = b.Min(clamped, max)
	if err != nil {
		return nil, errors.WithMessagef(err, "Backend %q: failed Clamp", BackendName)
	}
	return clamped, nil
}
