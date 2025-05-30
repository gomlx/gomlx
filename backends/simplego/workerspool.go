package simplego

import (
	"runtime"
	"sync"
	"sync/atomic"
)

type workersPool struct {
	// maxParallelism is a soft target on the limit of parallel work to do.
	// The actual number of goroutines is higher than that -- because of waits and such.
	maxParallelism int
	mu             sync.Mutex
	cond           sync.Cond // Should be signaled whenever numRunning is decreased.
	numRunning     int

	// extraParallelism is temporarily increased when a worked goes to sleep.
	extraParallelism atomic.Int32
}

// Initialize should be called before use.
func (w *workersPool) Initialize() {
	w.maxParallelism = runtime.NumCPU()
	w.cond = sync.Cond{L: &w.mu}
}

// IsEnabled returns whether parallelism is enabled (maxParallelism is != 0)
func (w *workersPool) IsEnabled() bool {
	return w.maxParallelism != 0
}

// IsUnlimited returns whether parallelism is unlimited (maxParallelism < 0)
func (w *workersPool) IsUnlimited() bool {
	return w.maxParallelism < 0
}

// MaxParallelism is a soft-target for parallelism (the limit of goroutines is higher that this).
// If set to 0 parallelism is disabled.
// If set to -1 parallelism is unlimited.
func (w *workersPool) MaxParallelism() int {
	return w.maxParallelism
}

// SetMaxParallelism sets the maxParallelism.
//
// You should only change the parallelism before any workers start running. If changed during the execution
// the behavior is undefined.
func (w *workersPool) SetMaxParallelism(maxParallelism int) {
	w.maxParallelism = maxParallelism
}

const goroutineToParallelismRatio = 2

// lockedIsFull returns whether all available workers are in use.
//
// It must be called with workerPool.mu acquired.
func (w *workersPool) lockedIsFull() bool {
	if w.maxParallelism == 0 {
		return true
	} else if w.maxParallelism < 0 {
		return false
	}
	return w.numRunning >= goroutineToParallelismRatio*w.maxParallelism+int(w.extraParallelism.Load())
}

// WaitToStart waits until there is a worker available to run the task.
//
// If parallelism is disabled (maxParallelism is 0), it runs the task inline and returns when it is finished.
// This is risky if one is relying on concurrency, and it can lead to deadlocks.
// Avoid using this function if the parallelism is disabled.
func (w *workersPool) WaitToStart(task func()) {
	if w.IsUnlimited() {
		go task()
		return

	} else if w.maxParallelism == 0 {
		// No parallelism, run inline -- better avoided.
		task()
		return
	}

	w.mu.Lock()
	defer w.mu.Unlock()
	for w.lockedIsFull() {
		w.cond.Wait()
	}
	w.lockedRunTaskInGoroutine(task)
}

// lockedRunTaskInGoroutine and keep tabs on w.numRunning.
//
// It must be called with workerPool.mu acquired.
func (w *workersPool) lockedRunTaskInGoroutine(task func()) {
	w.numRunning++
	go func() {
		task()
		w.mu.Lock()
		w.numRunning--
		w.cond.Signal()
		w.mu.Unlock()
	}()
}

// StartIfAvailable runs the task in a separate goroutine, if there are enough workers left.
// It returns true if it found workers to run the function, false otherwise.
//
// It's up to the client to synchronize the end of the function execution.
func (w *workersPool) StartIfAvailable(task func()) bool {
	if w.IsUnlimited() {
		go task()
		return true
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.lockedIsFull() {
		return false
	}
	w.lockedRunTaskInGoroutine(task)
	return true
}

// WorkerIsAsleep indicates the worker (the one that called the method) is going to sleep waiting
// for other workers, and temporarily increases the available number of workers.
//
// Call WorkerRestarted when the worker is ready to run again.
func (w *workersPool) WorkerIsAsleep() {
	w.extraParallelism.Add(1)
}

// WorkerRestarted indicates the worker (the one that called the method) is ready to run again.
// It should only be called after WorkerIsAsleep.
//
// It returns the temporary number of extra available workers.
func (w *workersPool) WorkerRestarted() {
	w.extraParallelism.Add(-1)
}
