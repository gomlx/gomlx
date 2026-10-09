// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package layers

import (
	"cmp"
	"slices"
	"sync"

	. "github.com/gomlx/gomlx/core/graph"
	"github.com/gomlx/gomlx/ml/model"
)

// PostLinearHook is called after a linear transformation (e.g. dense layer in FNN, attention, etc.).
// It receives the layer's scope, the input to the linear projection (before transformation),
// and the output of the linear projection (after base weights and bias, but before activation/dropout/norm).
// It returns the updated output node.
type PostLinearHook func(scope *model.Scope, input, output *Node) *Node

// Standard hook priorities. Lower values execute first.
const (
	HookPriorityEarly   = -100
	HookPriorityDefault = 0
	HookPriorityLate    = 100
)

type prioritizedHook struct {
	priority int
	order    int // insertion order for stable tie-breaking
	hook     PostLinearHook
}

var (
	hooksMu      sync.RWMutex
	linearHooks  []prioritizedHook
	hookSeqOrder int
)

// RegisterPostLinearHook registers a hook called after every linear projection,
// ordered by priority (lower priority values are called first).
//
// It is thread-safe and can be called from init() functions or at runtime.
func RegisterPostLinearHook(priority int, hook PostLinearHook) {
	hooksMu.Lock()
	defer hooksMu.Unlock()
	hookSeqOrder++
	linearHooks = append(linearHooks, prioritizedHook{
		priority: priority,
		order:    hookSeqOrder,
		hook:     hook,
	})
	slices.SortFunc(linearHooks, func(a, b prioritizedHook) int {
		if c := cmp.Compare(a.priority, b.priority); c != 0 {
			return c
		}
		return cmp.Compare(a.order, b.order)
	})
}

// ClearPostLinearHooks removes all registered post-linear hooks.
// This is primarily intended for testing.
func ClearPostLinearHooks() {
	hooksMu.Lock()
	defer hooksMu.Unlock()
	linearHooks = nil
	hookSeqOrder = 0
}

// ApplyPostLinearHooks executes all registered post-linear hooks in priority order.
func ApplyPostLinearHooks(scope *model.Scope, input, output *Node) *Node {
	hooksMu.RLock()
	hooks := slices.Clone(linearHooks)
	hooksMu.RUnlock()

	for _, h := range hooks {
		output = h.hook(scope, input, output)
	}
	return output
}
