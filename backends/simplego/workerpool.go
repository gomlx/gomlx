package simplego

// startWorker runs fn in a separate goroutine, if there is enough workers left.
// It returns true if it found workers to run the function, false otherwise.
//
// It's up to the client to synchronize the end of the function execution.
func (b *Backend) startWorker(fn func()) bool {
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

// workerIsAsleep indicates it is waiting for other workers, and temporarily decrease the
// number of current workers active.
//
// Call workerRestarted when the worker is ready to run again.
func (b *Backend) workerIsAsleep() {
	b.currentWorkers.Add(-1)
}

// workerRestarted indicates the worker is ready to run again.
// It should only be called after workerIsAsleep.
//
// It increases the number of current workers active.
// Notice this may lead temporarily to having more workers active than maxParallelism.
func (b *Backend) workerRestarted() {
	b.currentWorkers.Add(1)
}
