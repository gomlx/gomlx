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
	"github.com/gomlx/gomlx/types/xslices"
	"strings"
)

// scopedParams provides a mapping from string to any data type that is "scoped":
//
//   - For every scope there is a map of string to data.
//   - Accessing a key triggers a search from the current scope up to the root scope, the
//     first result found is returned.
//
// Example: let's say the current scopedParams hold:
//
//	Scope: "/": { "x":10, "y": 20, "z": 40 }
//	Scope: "/a": { "y": 30 }
//	Scope: "/a/b": { "x": 100 }
//
//	scopedParams.Get("/a/b", "x") -> 100
//	scopedParams.Get("/a/b", "y") -> 30
//	scopedParams.Get("/a/b", "z") -> 40
//	scopedParams.Get("/a/b", "w") -> Not found.
//
// Notice that "/" (== ScopeSeparator constant) separates parts of the scope path, and the root
// scope is referred to as "/". There is no "empty" scope, and every scope name must start with
// a ScopeSeparator.
//
// The Context object uses scopedParams to store the normal hyperparameters (see `Context.GetParam` and `Context.SetParam`)
// and to store the graph hyperparameters (see `Context.GetGraphParam` and `Context.SetGraphParam`).
//
// Usually there will be no need for the end user to use this.
type scopedParams struct {
	scopeToMap map[string]map[string]any
}

// newScopedParams create an empy scopedParams.
func newScopedParams() *scopedParams {
	return &scopedParams{
		scopeToMap: make(map[string]map[string]any),
	}
}

// Clone returns a deep copy of the scopedParams.
func (p *scopedParams) Clone() *scopedParams {
	newScopedParams := newScopedParams()
	for scope, dataMap := range p.scopeToMap {
		newScopedParams.scopeToMap[scope] = make(map[string]any)
		for key, value := range dataMap {
			newScopedParams.scopeToMap[scope][key] = value
		}
	}
	return newScopedParams
}

// Set sets the value for the given key, in the given scope.
func (p *scopedParams) Set(scope, key string, value any) {
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
func (p *scopedParams) Get(scope, key string) (value any, found bool) {
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

// Enumerate enumerates all parameters stored in the scopedParams structure and calls the given closure with
// them.
func (p *scopedParams) Enumerate(fn func(scope, key string, value any)) {
	scopes := xslices.SortedKeys(p.scopeToMap)
	for _, scope := range scopes {
		keyValues := p.scopeToMap[scope]
		keys := xslices.SortedKeys(keyValues)
		for _, key := range keys {
			value := keyValues[key]
			fn(scope, key, value)
		}
	}
}
