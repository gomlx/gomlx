// Package types is mostly a top level directory for GoMLX important types. See
// sub-packages `shapes`, `tensor` and `slices`.
//
// This package also provides the types: Set.
package types

// Set implements a Set for the key type T.
type Set[T comparable] map[T]struct{}

// MakeSet returns an empty Set of the given type. Size is optional, and if given
// will reserve the expected size.
func MakeSet[T comparable](size ...int) Set[T] {
	if len(size) == 0 {
		return make(Set[T])
	}
	return make(Set[T], size[0])
}

// Has returns true if Set s has the given key.
func (s Set[T]) Has(key T) bool {
	_, found := s[key]
	return found
}

// Insert key into set.
func (s Set[T]) Insert(key T) {
	s[key] = struct{}{}
}
