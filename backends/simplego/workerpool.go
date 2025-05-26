package simplego

// StartWorker runs fn in a separate goroutine, if there is enough workers left.
// It returns true if it found workers to run the function, false otherwise.
//
// It's up to the client to synchronize the end of the function execution.
func (b *Backend) StartWorker(fn func()) bool {
	if b.maxParallelism > 0 && b.currentWorkers.Load() >= int32(b.maxParallelism) {
		return false
	}
	b.currentWorkers.Add(1)
	go func() {
		fn()
		b.currentWorkers.Add(-1)
	}()
	return true
}
