// Package _default includes the default backends, namely XLA and SimpleGo.
//
// To use it simply include:
//
//	import _ "github.com/gomlx/gomlx/backends/default"
//
// If you add the tag `noxla` it will not include xla -- useful if you don't have the corresponding libraries installed.
package _default

import (
	_ "github.com/gomlx/gomlx/backends/simplego"
)
