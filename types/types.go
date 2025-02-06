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

// SetWith creates a Set[T] with the given elements inserted.
func SetWith[T comparable](elements ...T) Set[T] {
	s := MakeSet[T](len(elements))
	for _, element := range elements {
		s.Insert(element)
	}
	return s
}

// Has returns true if Set s has the given key.
func (s Set[T]) Has(key T) bool {
	_, found := s[key]
	return found
}

// Insert keys into set.
func (s Set[T]) Insert(keys ...T) {
	for _, key := range keys {
		s[key] = struct{}{}
	}
}

// Sub returns `s - s2`, that is, all elements in `s` that are not in `s2`.
func (s Set[T]) Sub(s2 Set[T]) Set[T] {
	sub := MakeSet[T]()
	for k := range s {
		if !s2.Has(k) {
			sub.Insert(k)
		}
	}
	return sub
}

// Equal returns whether s and s2 have the exact same elements.
func (s Set[T]) Equal(s2 Set[T]) bool {
	if len(s) != len(s2) {
		return false
	}
	for k := range s {
		if !s2.Has(k) {
			return false
		}
	}
	return true
}
