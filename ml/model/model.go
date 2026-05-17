// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

// Package model implements a model's `Store` object (to store variables and
// hyperparameters), `Scope`s (scope, like a directory "path") within a store,
// and `Exec`, an executor that takes as an extra parameter a `Store` and will
// handle passing the variables automatically as extra inputs and outputs to the
// graph being built. `Exec` simplifies building models using or updateing the
// variables of its `Store`.
//
// # Example
package model

//go:generate go run ../../internal/cmd/constraints_generator -model
