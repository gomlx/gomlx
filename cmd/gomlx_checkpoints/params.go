// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"
	"slices"

	"github.com/gomlx/gomlx/pkg/ml/context"
	"github.com/gomlx/gomlx/pkg/support/sets"
	"golang.org/x/exp/maps"
)

func Params(ctxs, scopedCtxs []*context.Context, names []string) {
	numCheckpoints := len(names)
	numCols := numCheckpoints + 3

	fmt.Println(titleStyle.Render("Hyperparameters"))
	table := newPlainTableWithReds(true)

	// Build the headers row.
	headers := make([]string, 0, numCols)
	headers = append(headers, []string{"Scope", "Name", "Type"}...)
	if numCheckpoints == 1 {
		headers = append(headers, "Value")
	} else {
		for i := 0; i < numCheckpoints; i++ {
			headers = append(headers, names[i])
		}
	}
	table.Table.Headers(headers...)

	// List params set on all models.
	type scopeKey struct{ Scope, Key string }
	scopeKeySet := sets.Make[scopeKey]()
	for _, ctx := range ctxs {
		ctx.EnumerateParams(func(scope, key string, value any) {
			scopeKeySet.Insert(scopeKey{Scope: scope, Key: key})
		})
	}
	scopeKeys := maps.Keys(scopeKeySet)
	slices.SortFunc(scopeKeys, func(a scopeKey, b scopeKey) int {
		if a.Scope < b.Scope {
			return -1
		}
		if a.Scope > b.Scope {
			return 1
		}
		if a.Key < b.Key {
			return -1
		}
		if a.Key > b.Key {
			return 1
		}
		return 0
	})

	for _, pair := range scopeKeys {
		row := make([]string, numCols)
		scope, key := pair.Scope, pair.Key
		row[0] = scope
		row[1] = key
		for ii, ctx := range ctxs {
			if scope != "/" {
				ctx = ctx.In(scope)
			}
			value, found := ctx.GetParam(key)
			if !found {
				continue
			}
			if row[2] == "" {
				// Set the type of the value.
				row[2] = fmt.Sprintf("%T", value)
			}
			row[3+ii] = fmt.Sprintf("%v", value)
		}
		table.Row(!isAllEqual(row[3:]), row...)
	}
	fmt.Println(table.Table.Render())
}
