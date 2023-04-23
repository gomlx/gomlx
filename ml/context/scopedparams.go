/*
 *	Copyright 2023 Jan Pfeifer
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

package context

import (
	"github.com/gomlx/gomlx/types/slices"
	"strings"
)

// ScopedParams provides a mapping from string to any data type that is "scoped":
//
//   - For every scope there is a map of string to data.
//   - Accessing a key triggers a search from the current scope up to the root scope, the
//     first result found is returned.
//
// Example: let's say the current ScopedParams hold:
//
//	Scope: "/": { "x":10, "y": 20, "z": 40 }
//	Scope: "/a": { "y": 30 }
//	Scope: "/a/b": { "x": 100 }
//
//	ScopedParams.Get("/a/b", "x") -> 100
//	ScopedParams.Get("/a/b", "y") -> 30
//	ScopedParams.Get("/a/b", "z") -> 40
//	ScopedParams.Get("/a/b", "w") -> Not found.
//
// Notice that "/" (== ScopeSeparator constant) separates parts of the scope path, and the root
// scope is referred to as "/". There is no "empty" scope, and every scope name must start with
// a ScopeSeparator.
type ScopedParams struct {
	scopeToMap map[string]map[string]any
}

// NewScopedParams create an empy ScopedParams.
func NewScopedParams() *ScopedParams {
	return &ScopedParams{
		scopeToMap: make(map[string]map[string]any),
	}
}

// Set sets the value for the given key, in the given scope.
func (p *ScopedParams) Set(scope, key string, value any) {
	dataMap, found := p.scopeToMap[scope]
	if found && dataMap != nil {
		dataMap[key] = value
	} else {
		dataMap := make(map[string]any)
		dataMap[key] = value
		p.scopeToMap[scope] = dataMap
	}
}

// Get retrieves the value for the given key in the given scope or any parent scope.
// E.g: Get("/a/b", "myKey") will search for "myKey" in scopes "/a/b", "/a" and "/"
// consecutively until "myKey" is found.
//
// It returns the first value found if any, and whether some value was found.
func (p *ScopedParams) Get(scope, key string) (value any, found bool) {
	scopeParts := strings.Split(scope, ScopeSeparator)
	for ii := len(scopeParts) - 1; ii >= 0; ii-- {
		var dataMap map[string]any
		dataMap, found = p.scopeToMap[scope]
		if found && dataMap != nil {
			value, found = dataMap[key]
			if found {
				return
			}
		}
		scope = scope[:len(scope)-len(scopeParts[ii])]
		if ii > 1 {
			// Remove tailing separator, except for the root scope ("/").
			scope = scope[:len(scope)-len(ScopeSeparator)]
		}
	}
	return nil, false
}

// Enumerate enumerates all parameters stored in the ScopedParams structure and calls the given closure with
// them.
func (p *ScopedParams) Enumerate(fn func(scope, key string, value any)) {
	scopes := slices.SortedKeys(p.scopeToMap)
	for _, scope := range scopes {
		keyValues := p.scopeToMap[scope]
		keys := slices.SortedKeys(keyValues)
		for _, key := range keys {
			value := keyValues[key]
			fn(scope, key, value)
		}
	}
}
