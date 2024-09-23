// Package xsync implements some extra synchronization tools.
package xsync

import "sync"

// Latch implements a "latch" synchronization mechanism.
//
// A Latch is a signal that can be waited for until it is triggered.
// Once triggered it never changes state, it's forever triggered.
type Latch struct {
	muTrigger sync.Mutex
	wait      chan struct{}
}

// NewLatch returns an un-triggered latch.
func NewLatch() *Latch {
	return &Latch{
		wait: make(chan struct{}),
	}
}

// Trigger latch.
func (l *Latch) Trigger() {
	l.muTrigger.Lock()
	defer l.muTrigger.Unlock()

	if l.Test() {
		// Already triggered, discard value.
		return
	}
	close(l.wait)
}

// Wait waits for the latch to be triggered.
func (l *Latch) Wait() {
	<-l.wait
}

// Test checks whether the latch has been triggered.
func (l *Latch) Test() bool {
	select {
	case <-l.wait:
		return true
	default:
		return false
	}
}

// WaitChan returns the channel that one can use on a `select` to check when
// the latch triggers.
// The returned channel is closed when the latch is triggered.
func (l *Latch) WaitChan() <-chan struct{} {
	return l.wait
}

// LatchWithValue implements a "latch" synchronization mechanism, with a value associated with the
// triggering of the latch.
//
// A LatchWithValue is a signal that can be waited for until it is triggered. Once triggered it never
// changes state, it's forever triggered.
type LatchWithValue[T any] struct {
	value T
	latch *Latch
}

// NewLatchWithValue returns an un-triggered latch.
func NewLatchWithValue[T any]() *LatchWithValue[T] {
	return &LatchWithValue[T]{
		latch: NewLatch(),
	}
}

// Trigger latch and saves the associated value.
func (l *LatchWithValue[T]) Trigger(value T) {
	l.latch.muTrigger.Lock()
	defer l.latch.muTrigger.Unlock()

	if l.latch.Test() {
		// Already triggered, discard value.
		return
	}
	l.value = value
	close(l.latch.wait)
}

// Wait waits for the latch to be triggered.
func (l *LatchWithValue[T]) Wait() T {
	l.latch.Wait()
	return l.value
}

// Test checks whether the latch has been triggered.
func (l *LatchWithValue[T]) Test() bool {
	return l.latch.Test()
}

// TrySend tries to send value through the channel.
// It returns false if it failed, presumably because the channel is closed.
func TrySend[T any](c chan T, value T) (ok bool) {
	defer func() {
		exception := recover()
		ok = exception == nil
	}()
	c <- value
	return
}

// SendNoBlock tries to send value through the channel.
// It returns 0 if the value was sent, 1 if sending it would block (channel buffer full)
// or 2 if the channel `c` was closed.
func SendNoBlock[T any](c chan T, value T) (status int) {
	defer func() {
		exception := recover()
		if exception != nil {
			status = 2
		}
	}()
	select {
	case c <- value:
		status = 0
	default:
		status = 1
	}
	return
}

// Semaphore that allows dynamic resizing.
//
// It uses a sync.Cond, to allow dynamic resizing, so it will be slower than a pure channel version
// of a semaphore, with a fixed capacity. This shouldn't matter for more coarse resource control.
type Semaphore struct {
	cond              sync.Cond
	capacity, current int // Tracks capacity and current usage.
}

// NewSemaphore returns a Semaphore that allows at most capacity simultaneous acquisitions.
// If capacity <= 0, there is no limit on acquisitions.
//
// FIFO ordering may be lost during resizes (Semaphore.Resize) to larger capacity, but otherwise it is respected.
func NewSemaphore(capacity int) *Semaphore {
	return &Semaphore{
		cond:     sync.Cond{L: &sync.Mutex{}},
		capacity: capacity,
	}
}

// Acquire resource observing current semaphore capacity.
// It must be matched by exactly one call to Semaphore.Release after the reservation is no longer needed.
func (s *Semaphore) Acquire() {
	s.cond.L.Lock()
	defer s.cond.L.Unlock()
	for {
		if s.capacity <= 0 || s.current < s.capacity {
			// No limits.
			s.current++
			return
		}
		s.cond.Wait()
	}
}

// Release resource previously allocated with Semaphore.Acquire.
func (s *Semaphore) Release() {
	s.cond.L.Lock()
	defer s.cond.L.Unlock()
	s.current--
	if s.capacity == 0 || s.current < s.capacity-1 {
		return
	}
	s.cond.Signal()
}

// Resize number of available resources in the Semaphore.
//
// If newCapacity is larger than previous one, this may immediately allow pending Semaphore.Acquire to proceed.
// Notice since all waiting Semaphore.Acquire are awoken (broadcast), the queue order may be lost.
//
// If newCapacity is smaller than previous one, it doesn't have any effect on current acquisitions. So if the Semaphore
// is being used to control a worker pool, reducing its size won't stop workers currently executing.
func (s *Semaphore) Resize(newCapacity int) {
	s.cond.L.Lock()
	defer s.cond.L.Unlock()
	if newCapacity == s.capacity {
		return // No change needed.
	}
	if (newCapacity > 0 && newCapacity < s.capacity) || s.capacity == 0 {
		// Capacity is shrinking, no Semaphore.Acquire will be released.
		s.capacity = newCapacity
		return
	}

	// Wake-up everyone -- to preserve the queue order we would need to call s.cond.Signal() for the amount of
	// increased capacity, but that would make this call O(capacity), potentially slow for large capacities.
	s.capacity = newCapacity
	s.cond.Broadcast()
}
