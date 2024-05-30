// Package keepalive provides a simple Acquire and Release mechanism to
// make sure data is kept alive in between.
//
// This is used to access data that is not managed by Go, and hence the GC
// has no way of knowing if its in use.
//
// Example: the code guarantees that `resource` is being kept alive, and `data`
// is valid until released, and hence `data` (presumably Go GC doesn't know about it)
// can be safely accessed.
//
//	ref := keeplive.Acquire(resource)
//	defer ref.Release()
//	data := resource.C_data
//	...
//
// The package also provides ListAcquired method to investigate leaks (resources
// allocated but not released).
package keepalive

import "sync"

var (
	// allRefs is the global pool of all acquired references being kept alive.
	//
	// It's indexed by KeepAlive values.
	//
	// For the slots not using pointing to any reference, it stores a KeepAlive
	// value of the next available slot, forming a linked list.
	allRefs []any

	// nextFree implements the next available free entry in allRefs, or -1 if
	// nothing available.
	nextFree KeepAlive

	// Protects for concurrent use.
	muRefs sync.Mutex
)

// KeepAlive is a pointer to the reference being kept alive.
type KeepAlive int

const InitialFreeSlots = 128

const EndOfList = KeepAlive(-1)

// Initialize allRefs with InitialFreeSlots, and link them.
func init() {
	allRefs = make([]any, InitialFreeSlots)
	for ii := 0; ii < len(allRefs)-1; ii++ {
		allRefs[ii] = KeepAlive(ii + 1)
	}
	allRefs[len(allRefs)-1] = EndOfList
	nextFree = 0
}

func Acquire(reference any) KeepAlive {
	muRefs.Lock()
	defer muRefs.Unlock()
	if nextFree == EndOfList {
		// No available slots, we append a new one.
		allRefs = append(allRefs, reference)
		return KeepAlive(len(allRefs) - 1)
	}

	// Take next free in list of available.
	acquired := nextFree
	nextFree = allRefs[nextFree].(KeepAlive)
	allRefs[acquired] = reference
	return acquired
}

func (k KeepAlive) Release() {
	muRefs.Lock()
	defer muRefs.Unlock()

	allRefs[k] = nextFree
	nextFree = k
}
