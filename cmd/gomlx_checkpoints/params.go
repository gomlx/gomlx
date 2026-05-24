// Copyright 2023-2026 The GoMLX Authors. SPDX-License-Identifier: Apache-2.0

package main

import (
	"fmt"

	"github.com/gomlx/compute/support/xslices"
	"github.com/gomlx/gomlx/ml/model"
	"github.com/gomlx/gomlx/support/sets"
)

func Params(stores []*model.Store, scopes []*model.Scope, names []string) {
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
	paramPaths := sets.Make[string]()
	for _, store := range stores {
		for paramPath := range store.IterParams() {
			paramPaths.Insert(paramPath)
		}
	}
	for _, fullPath := range xslices.SortedKeys(paramPaths) {
		row := make([]string, numCols)
		scope, key := model.SplitPath(fullPath)
		row[0] = scope
		row[1] = key
		for ii, store := range stores {
			value, found := store.GetParam(key)
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
